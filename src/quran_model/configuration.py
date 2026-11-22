"""Configuration for the reference-compatible Wav2Vec2-BERT CTC model."""

from copy import deepcopy

from transformers import PretrainedConfig, Wav2Vec2BertConfig


REFERENCE_LEVEL_TO_VOCAB_SIZE = {
    "phonemes": 43,
    "hams_or_jahr": 3,
    "shidda_or_rakhawa": 4,
    "tafkheem_or_taqeeq": 4,
    "itbaq": 3,
    "safeer": 3,
    "qalqla": 3,
    "tikraar": 3,
    "tafashie": 3,
    "istitala": 3,
    "ghonna": 3,
}

REFERENCE_LEVEL_TO_LOSS_WEIGHT = {
    "ghonna": 0.059875,
    "hams_or_jahr": 0.059875,
    "istitala": 0.059875,
    "itbaq": 0.059875,
    "phonemes": 0.4,
    "qalqla": 0.059875,
    "safeer": 0.059875,
    "shidda_or_rakhawa": 0.0605,
    "tafashie": 0.059875,
    "tafkheem_or_taqeeq": 0.0605,
    "tikraar": 0.059875,
}

REFERENCE_ENCODER_CONFIG = {
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "intermediate_size": 4096,
    "feature_projection_input_dim": 160,
    "hidden_act": "swish",
    "hidden_dropout": 0.0,
    "activation_dropout": 0.0,
    "attention_dropout": 0.0,
    "feat_proj_dropout": 0.0,
    "layerdrop": 0.0,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-5,
    "apply_spec_augment": True,
    "mask_time_prob": 0.0,
    "mask_time_length": 10,
    "mask_time_min_masks": 2,
    "mask_feature_prob": 0.0,
    "mask_feature_length": 10,
    "mask_feature_min_masks": 0,
    "ctc_loss_reduction": "mean",
    "ctc_zero_infinity": False,
    "add_adapter": True,
    "adapter_kernel_size": 3,
    "adapter_stride": 2,
    "num_adapter_layers": 1,
    "adapter_act": "relu",
    "use_intermediate_ffn_before_adapter": False,
    "output_hidden_size": 1024,
    "position_embeddings_type": "relative_key",
    "rotary_embedding_base": 10000,
    "max_source_positions": 5000,
    "left_max_position_embeddings": 64,
    "right_max_position_embeddings": 8,
    "conv_depthwise_kernel_size": 31,
    "conformer_conv_dropout": 0.1,
}


class QuranMultiLevelCTCConfig(PretrainedConfig):
    """Head configuration plus a separately serialized Wav2Vec2-BERT config."""

    model_type = "quran_multi_level_ctc"

    def __init__(
        self,
        encoder_config=None,
        level_to_vocab_size=None,
        level_to_loss_weight=None,
        final_dropout=0.1,
        pad_token_id=0,
        label_pad_token_id=-100,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=False,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        if encoder_config is None:
            encoder_config = Wav2Vec2BertConfig(**REFERENCE_ENCODER_CONFIG).to_dict()
        elif isinstance(encoder_config, Wav2Vec2BertConfig):
            encoder_config = encoder_config.to_dict()
        self.encoder_config = deepcopy(encoder_config)
        self.level_to_vocab_size = dict(level_to_vocab_size or REFERENCE_LEVEL_TO_VOCAB_SIZE)
        if not self.level_to_vocab_size or "phonemes" not in self.level_to_vocab_size:
            raise ValueError("level_to_vocab_size must include the phonemes level")
        if any(int(size) < 2 for size in self.level_to_vocab_size.values()):
            raise ValueError("Each CTC level needs blank ID 0 and at least one label class")
        if pad_token_id != 0:
            raise ValueError("The reference model uses CTC blank ID 0")
        self.label_pad_token_id = int(label_pad_token_id)
        if self.label_pad_token_id != -100:
            raise ValueError("Reference training batches use -100 for ignored label padding")
        self.final_dropout = float(final_dropout)
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = bool(ctc_zero_infinity)
        weights = dict(level_to_loss_weight or REFERENCE_LEVEL_TO_LOSS_WEIGHT)
        total = sum(weights.values())
        if total > 1.0 + 1e-8:
            raise ValueError("The sum of level_to_loss_weight must not exceed 1")
        missing = [level for level in self.level_to_vocab_size if level not in weights]
        if missing:
            share = (1.0 - total) / len(missing)
            weights.update({level: share for level in missing})
        unknown = set(weights) - set(self.level_to_vocab_size)
        if unknown:
            raise ValueError(f"Loss weights name unknown levels: {sorted(unknown)}")
        self.level_to_loss_weight = weights

    def make_encoder_config(self):
        """Return a fresh HF encoder config, keeping encoder and heads independent."""
        return Wav2Vec2BertConfig.from_dict(deepcopy(self.encoder_config))
