"""Reference-shaped multi-level CTC model on Wav2Vec2-BERT features."""

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel
from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import Wav2Vec2BertSelfAttention
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel

from .configuration import QuranMultiLevelCTCConfig


@dataclass
class QuranMultiLevelCTCOutput(ModelOutput):
    """Model outputs. `logits` maps every level to (batch, frames, classes)."""

    loss: torch.FloatTensor | None = None
    logits: dict[str, torch.FloatTensor] | None = None
    output_lengths: torch.LongTensor | None = None
    encoder_last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass(frozen=True)
class CheckpointLoadReport:
    """Explicit key and shape audit for an upstream checkpoint load."""

    checkpoint: str
    matched: tuple[tuple[str, str], ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_mismatches: tuple[str, ...]
    loaded_parameter_count: int

    @property
    def is_complete(self):
        return not self.missing and not self.unexpected and not self.shape_mismatches


class CheckpointCompatibilityError(RuntimeError):
    """A checkpoint did not exactly match the model's parameters."""

    def __init__(self, report: CheckpointLoadReport):
        self.report = report
        super().__init__(
            "Checkpoint parameter mismatch: "
            f"{len(report.missing)} missing, {len(report.unexpected)} unexpected, "
            f"{len(report.shape_mismatches)} shape mismatches"
        )


class _SDPAWav2Vec2BertSelfAttention(Wav2Vec2BertSelfAttention):
    """Wav2Vec2-BERT self-attention using PyTorch SDPA when weights are not requested.

    The upstream Transformers 5.8.1 implementation materializes both attention
    scores and softmax probabilities. For this checkpoint's ``relative_key``
    positions, the positional term is an additive attention bias and can be
    passed to scaled-dot-product attention without changing the score equation.
    """

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        relative_position_embeddings=None,
        output_attentions=False,
    ):
        # Preserve the upstream attention map convention and its less common
        # relative-position implementation exactly when callers request weights.
        if output_attentions or self.position_embeddings_type == "relative":
            return super().forward(
                hidden_states,
                attention_mask=attention_mask,
                relative_position_embeddings=relative_position_embeddings,
                output_attentions=output_attentions,
            )

        batch_size, sequence_length, _ = hidden_states.size()
        query_key_states = hidden_states
        if self.position_embeddings_type == "rotary":
            if relative_position_embeddings is None:
                raise ValueError("relative_position_embeddings are required for rotary attention")
            query_key_states = self._apply_rotary_embedding(
                query_key_states, relative_position_embeddings
            )

        query = self.linear_q(query_key_states).view(
            batch_size, sequence_length, self.num_heads, self.head_size
        ).transpose(1, 2)
        key = self.linear_k(query_key_states).view(
            batch_size, sequence_length, self.num_heads, self.head_size
        ).transpose(1, 2)
        value = self.linear_v(hidden_states).view(
            batch_size, sequence_length, self.num_heads, self.head_size
        ).transpose(1, 2)

        attention_bias = attention_mask
        if self.position_embeddings_type == "relative_key":
            query_length, key_length = query.shape[2], key.shape[2]
            position_ids_l = torch.arange(
                query_length, dtype=torch.long, device=hidden_states.device
            ).view(-1, 1)
            position_ids_r = torch.arange(
                key_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_r - position_ids_l
            distance = torch.clamp(
                distance, -self.left_max_position_embeddings,
                self.right_max_position_embeddings,
            )
            positional_embedding = self.distance_embedding(
                distance + self.left_max_position_embeddings
            ).to(dtype=query.dtype)
            relative_bias = torch.einsum(
                "bhld,lrd->bhlr", query, positional_embedding
            ) / (self.head_size ** 0.5)
            attention_bias = relative_bias if attention_bias is None else attention_bias + relative_bias

        context = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_bias,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        context = context.transpose(1, 2).reshape(
            batch_size, sequence_length, self.num_heads * self.head_size
        )
        return self.linear_out(context), None


class QuranMultiLevelCTC(PreTrainedModel):
    """Eleven CTC heads over the original Wav2Vec2-BERT encoder.

    The forward input is the feature extractor's `input_features` (B,T,160).
    Raw waveforms are accepted by `forward_waveforms` when a HF feature
    extractor is supplied, following the original inference pipeline.
    """

    config_class = QuranMultiLevelCTCConfig
    base_model_prefix = "encoder"
    main_input_name = "input_features"
    supports_gradient_checkpointing = True

    def __init__(self, config: QuranMultiLevelCTCConfig):
        super().__init__(config)
        self.encoder = Wav2Vec2BertModel(config.make_encoder_config())
        self.set_attention_implementation(config.attention_implementation)
        self.dropout = nn.Dropout(config.final_dropout)
        output_dim = config.encoder_config.get("output_hidden_size") or config.encoder_config["hidden_size"]
        self.level_to_lm_head = nn.ModuleDict({
            level: nn.Linear(output_dim, size)
            for level, size in config.level_to_vocab_size.items()
        })
        self.post_init()

    def set_attention_implementation(self, implementation: str) -> None:
        """Select eager or SDPA attention without changing parameter names or shapes."""
        if implementation not in {"eager", "sdpa"}:
            raise ValueError("attention implementation must be 'eager' or 'sdpa'")
        for layer in self.encoder.encoder.layers:
            attention = layer.self_attn
            position_type = attention.position_embeddings_type
            if implementation == "sdpa" and position_type == "relative":
                raise ValueError(
                    "SDPA is not implemented for Wav2Vec2-BERT position_embeddings_type='relative'"
                )
            if implementation == "sdpa":
                attention.__class__ = _SDPAWav2Vec2BertSelfAttention
            elif isinstance(attention, _SDPAWav2Vec2BertSelfAttention):
                attention.__class__ = Wav2Vec2BertSelfAttention
        self.config.attention_implementation = implementation

    def attention_implementation_status(self) -> dict[str, int | str]:
        sdpa_layers = sum(
            isinstance(layer.self_attn, _SDPAWav2Vec2BertSelfAttention)
            for layer in self.encoder.encoder.layers
        )
        return {
            "configured": self.config.attention_implementation,
            "sdpa_layers": sdpa_layers,
            "eager_layers": len(self.encoder.encoder.layers) - sdpa_layers,
        }

    @classmethod
    def from_pretrained_checkpoint(
        cls, checkpoint, *, revision=None, local_files_only=False, device="cpu"
    ):
        """Load and audit the original Muaalem checkpoint, including all CTC heads.

        The source encoder prefix ``wav2vec2_bert.`` is mapped explicitly to this
        model's ``encoder.`` prefix. Safetensors are read tensor-by-tensor, so a
        second complete checkpoint-sized state dictionary is not held in memory.
        Returns ``(model, report)``. Loading raises with the complete report on
        any missing, unexpected, or shape-incompatible tensor; partial/random
        initialization is never returned as a successful checkpoint load.
        """
        checkpoint_label = str(checkpoint)
        local = Path(checkpoint)
        if local.is_dir():
            config_path = local / "config.json"
            weights_path = local / "model.safetensors"
        else:
            config_path = Path(hf_hub_download(
                repo_id=checkpoint, filename="config.json", revision=revision,
                local_files_only=local_files_only,
            ))
            weights_path = Path(hf_hub_download(
                repo_id=checkpoint, filename="model.safetensors", revision=revision,
                local_files_only=local_files_only,
            ))
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        defaults = Wav2Vec2BertConfig().to_dict()
        encoder_config = Wav2Vec2BertConfig.from_dict({
            key: value for key, value in raw_config.items() if key in defaults
        })
        config = QuranMultiLevelCTCConfig(
            encoder_config=encoder_config,
            level_to_vocab_size=raw_config["level_to_vocab_size"],
            level_to_loss_weight=raw_config.get("level_to_loss_weight"),
            final_dropout=raw_config.get("final_dropout", 0.1),
            pad_token_id=raw_config.get("pad_token_id", 0),
            ctc_loss_reduction=raw_config.get("ctc_loss_reduction", "mean"),
            ctc_zero_infinity=raw_config.get("ctc_zero_infinity", False),
        )

        # Build only tensor metadata first. No random 660M-parameter allocation
        # is needed unless the key/shape audit has passed.
        with torch.device("meta"):
            model = cls(config)
        target_state = model.state_dict()
        target_shapes = {key: tuple(value.shape) for key, value in target_state.items()}
        source_to_target = {}
        source_shapes = {}
        with safe_open(str(weights_path), framework="pt", device="cpu") as source:
            source_keys = source.keys()
            for source_key in source_keys:
                if source_key.startswith("wav2vec2_bert."):
                    target_key = "encoder." + source_key.removeprefix("wav2vec2_bert.")
                elif source_key.startswith("level_to_lm_head."):
                    target_key = source_key
                else:
                    target_key = None
                if target_key is not None:
                    source_to_target[source_key] = target_key
                    source_shapes[source_key] = tuple(source.get_slice(source_key).get_shape())

        expected = set(target_shapes)
        mapped_targets = set(source_to_target.values())
        missing = tuple(sorted(expected - mapped_targets))
        unexpected = tuple(sorted(set(source_keys) - set(source_to_target)))
        mismatches = tuple(sorted(
            f"{source_key} -> {target_key}: checkpoint {source_shapes[source_key]}, "
            f"model {target_shapes[target_key]}"
            for source_key, target_key in source_to_target.items()
            if source_shapes[source_key] != target_shapes[target_key]
        ))
        matched_pairs = tuple(sorted(
            (source_key, target_key)
            for source_key, target_key in source_to_target.items()
            if source_shapes[source_key] == target_shapes[target_key]
        ))
        loaded_count = sum(
            int(torch.tensor(target_shapes[target_key]).prod().item())
            for _, target_key in matched_pairs
        )
        report = CheckpointLoadReport(
            checkpoint=checkpoint_label, matched=matched_pairs, missing=missing,
            unexpected=unexpected, shape_mismatches=mismatches,
            loaded_parameter_count=loaded_count,
        )
        if not report.is_complete:
            raise CheckpointCompatibilityError(report)

        target_device = torch.device(device)
        model.to_empty(device=target_device)
        model_state = model.state_dict()
        with safe_open(str(weights_path), framework="pt", device="cpu") as source:
            with torch.no_grad():
                for source_key, target_key in matched_pairs:
                    model_state[target_key].copy_(source.get_tensor(source_key))
        return model, report

    def _output_lengths(self, input_features, attention_mask):
        if attention_mask is None:
            feature_lengths = torch.full(
                (input_features.shape[0],), input_features.shape[1],
                dtype=torch.long, device=input_features.device,
            )
        else:
            if attention_mask.ndim != 2 or attention_mask.shape != input_features.shape[:2]:
                raise ValueError("attention_mask must have shape (batch, feature_frames)")
            feature_lengths = attention_mask.to(torch.long).sum(dim=-1)
        return self.encoder._get_feat_extract_output_lengths(feature_lengths).to(torch.long)

    def forward(
        self, input_features, attention_mask=None, input_lengths=None, labels=None,
        target_lengths=None, label_level_mask=None,
        output_attentions=None, output_hidden_states=None, return_dict=True,
    ):
        """Run all heads and optionally compute CTC loss for supplied levels.

        `labels` may contain any non-empty subset of configured levels, matching
        the original implementation. Missing levels still produce logits but
        do not contribute loss; configured loss weights are not renormalized.
        """
        if input_features is None or input_features.ndim != 3:
            raise ValueError("input_features must be a (batch, frames, feature_dim) tensor")
        expected_dim = self.config.encoder_config["feature_projection_input_dim"]
        if input_features.shape[-1] != expected_dim:
            raise ValueError(f"Expected {expected_dim} feature channels; got {input_features.shape[-1]}")
        if input_lengths is not None:
            provided_input_lengths = torch.as_tensor(input_lengths, device=input_features.device)
            actual_input_lengths = (
                attention_mask.to(torch.long).sum(dim=-1)
                if attention_mask is not None
                else torch.full_like(provided_input_lengths, input_features.shape[1])
            )
            if provided_input_lengths.shape != actual_input_lengths.shape or not torch.equal(
                provided_input_lengths.to(torch.long), actual_input_lengths
            ):
                raise ValueError("input_lengths must match the feature-frame attention_mask lengths")
        if labels is not None:
            if not isinstance(labels, dict) or not labels:
                raise ValueError("labels must be a non-empty level-to-target mapping")
            unknown = set(labels) - set(self.config.level_to_vocab_size)
            if unknown:
                raise ValueError(f"Unknown label levels: {sorted(unknown)}")
            if label_level_mask is not None:
                unknown_masks = set(label_level_mask) - set(labels)
                if unknown_masks:
                    raise ValueError(f"Level masks supplied without labels: {sorted(unknown_masks)}")
            for level, target in labels.items():
                if target.ndim != 2 or target.shape[0] != input_features.shape[0]:
                    raise ValueError(f"Labels for {level} must have shape (batch, target_length)")
                valid = target[target != self.config.label_pad_token_id]
                if valid.numel() and (valid.min() < 1 or valid.max() >= self.config.level_to_vocab_size[level]):
                    raise ValueError(f"Valid {level} target IDs must be in [1, {self.config.level_to_vocab_size[level] - 1}]")
                supervised = self._level_supervision_mask(level, label_level_mask, target.shape[0], target.device)
                if torch.any((~supervised) & (target != self.config.label_pad_token_id).any(dim=-1)):
                    raise ValueError(f"Unsupervised {level} rows must contain only ignored padding")
                if target_lengths is not None and level in target_lengths:
                    supplied_lengths = torch.as_tensor(target_lengths[level], device=target.device).to(torch.long)
                    actual_lengths = (target != self.config.label_pad_token_id).sum(dim=-1)
                    if supplied_lengths.shape != actual_lengths.shape or not torch.equal(supplied_lengths, actual_lengths):
                        raise ValueError(f"target_lengths[{level!r}] do not match non-padding labels")
            if target_lengths is not None and set(target_lengths) - set(labels):
                raise ValueError("target_lengths contains a level without labels")
            if not any(self._level_supervision_mask(level, label_level_mask, input_features.shape[0], input_features.device).any() for level in labels):
                raise ValueError("At least one level and example must have supervised labels")

        enc = self.encoder(
            input_features=input_features, attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden = self.dropout(enc.last_hidden_state)
        logits = {name: head(hidden) for name, head in self.level_to_lm_head.items()}
        lengths = self._output_lengths(input_features, attention_mask)
        loss = None
        if labels is not None:
            loss = hidden.new_zeros((), dtype=torch.float32)
            for level, target in labels.items():
                supervised = self._level_supervision_mask(level, label_level_mask, target.shape[0], target.device)
                if not supervised.any():
                    continue
                target = target[supervised]
                valid = target != self.config.label_pad_token_id
                target_lengths = valid.sum(dim=-1).to(torch.long)
                flat_targets = target.masked_select(valid).to(torch.long)
                log_probs = logits[level][supervised].log_softmax(dim=-1, dtype=torch.float32).transpose(0, 1)
                with torch.backends.cudnn.flags(enabled=False):
                    level_loss = nn.functional.ctc_loss(
                        log_probs, flat_targets, lengths[supervised], target_lengths,
                        blank=self.config.pad_token_id,
                        reduction=self.config.ctc_loss_reduction,
                        zero_infinity=self.config.ctc_zero_infinity,
                    )
                loss = loss + self.config.level_to_loss_weight[level] * level_loss
        if not return_dict:
            values = (logits, enc.hidden_states, enc.attentions)
            return ((loss,) + values) if loss is not None else values
        return QuranMultiLevelCTCOutput(
            loss=loss, logits=logits, output_lengths=lengths,
            encoder_last_hidden_state=enc.last_hidden_state,
            hidden_states=enc.hidden_states, attentions=enc.attentions,
        )

    @staticmethod
    def _level_supervision_mask(level, label_level_mask, batch_size, device):
        if label_level_mask is None or level not in label_level_mask:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        mask = torch.as_tensor(label_level_mask[level], dtype=torch.bool, device=device)
        if mask.ndim != 1 or mask.shape[0] != batch_size:
            raise ValueError(f"label_level_mask[{level!r}] must have shape ({batch_size},)")
        return mask

    def forward_waveforms(self, waveforms, sampling_rate, feature_extractor, **forward_kwargs):
        """Run Stage 5 float waveform batches through the reference HF extractor."""
        if sampling_rate != 16000:
            raise ValueError("The reference feature extractor expects 16 kHz waveforms")
        batch = feature_extractor(waveforms, sampling_rate=sampling_rate, return_tensors="pt")
        device = next(self.parameters()).device
        features = batch["input_features"].to(device=device)
        mask = batch.get("attention_mask")
        if mask is not None:
            mask = mask.to(device=device)
        return self(input_features=features, attention_mask=mask, **forward_kwargs)


__all__ = ["QuranMultiLevelCTC", "QuranMultiLevelCTCOutput"]
