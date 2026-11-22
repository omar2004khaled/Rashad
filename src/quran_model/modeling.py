"""Reference-shaped multi-level CTC model on Wav2Vec2-BERT features."""

from dataclasses import dataclass

import torch
from torch import nn
from transformers import Wav2Vec2BertModel
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


class QuranMultiLevelCTC(PreTrainedModel):
    """Eleven CTC heads over the original Wav2Vec2-BERT encoder.

    The forward input is the feature extractor's `input_features` (B,T,160).
    Raw waveforms are accepted by `forward_waveforms` when a HF feature
    extractor is supplied, following the original inference pipeline.
    """

    config_class = QuranMultiLevelCTCConfig
    base_model_prefix = "encoder"
    main_input_name = "input_features"

    def __init__(self, config: QuranMultiLevelCTCConfig):
        super().__init__(config)
        self.encoder = Wav2Vec2BertModel(config.make_encoder_config())
        self.dropout = nn.Dropout(config.final_dropout)
        output_dim = config.encoder_config.get("output_hidden_size") or config.encoder_config["hidden_size"]
        self.level_to_lm_head = nn.ModuleDict({
            level: nn.Linear(output_dim, size)
            for level, size in config.level_to_vocab_size.items()
        })
        self.post_init()

    @classmethod
    def from_pretrained_encoder(cls, checkpoint, config=None, **kwargs):
        """Initialize this model then load only its Wav2Vec2-BERT encoder weights."""
        if config is None:
            from transformers import Wav2Vec2BertConfig
            try:
                enc_config = Wav2Vec2BertConfig.from_pretrained(checkpoint)
            except ValueError:
                # The published Muaalem checkpoint identifies itself as a custom
                # model type. Its encoder architecture is Wav2Vec2-BERT; use the
                # base checkpoint config unless the caller passes the exact one.
                enc_config = Wav2Vec2BertConfig.from_pretrained(
                    "facebook/wav2vec2-bert-rel-pos-large"
                )
            config = QuranMultiLevelCTCConfig(encoder_config=enc_config)
        model = cls(config)
        model.encoder = Wav2Vec2BertModel.from_pretrained(
            checkpoint, config=config.make_encoder_config(), **kwargs
        )
        return model

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
        self, input_features, attention_mask=None, labels=None,
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
        if labels is not None:
            unknown = set(labels) - set(self.config.level_to_vocab_size)
            if unknown:
                raise ValueError(f"Unknown label levels: {sorted(unknown)}")
            for level, target in labels.items():
                if target.ndim != 2 or target.shape[0] != input_features.shape[0]:
                    raise ValueError(f"Labels for {level} must have shape (batch, target_length)")
                valid = target[target != self.config.label_pad_token_id]
                if valid.numel() and (valid.min() < 1 or valid.max() >= self.config.level_to_vocab_size[level]):
                    raise ValueError(f"Valid {level} target IDs must be in [1, {self.config.level_to_vocab_size[level] - 1}]")

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
                valid = target != self.config.label_pad_token_id
                target_lengths = valid.sum(dim=-1).to(torch.long)
                flat_targets = target.masked_select(valid).to(torch.long)
                log_probs = logits[level].log_softmax(dim=-1, dtype=torch.float32).transpose(0, 1)
                with torch.backends.cudnn.flags(enabled=False):
                    level_loss = nn.functional.ctc_loss(
                        log_probs, flat_targets, lengths, target_lengths,
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
