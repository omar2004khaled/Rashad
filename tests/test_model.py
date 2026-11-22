import torch
import pytest
from transformers import Wav2Vec2BertConfig

from quran_model import QuranMultiLevelCTC, QuranMultiLevelCTCConfig
from quran_model.configuration import REFERENCE_LEVEL_TO_VOCAB_SIZE
from quran_training_data.encoding import LabelCodec


def tiny_config(add_adapter=False):
    encoder = Wav2Vec2BertConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        feature_projection_input_dim=160,
        conv_depthwise_kernel_size=3,
        conformer_conv_dropout=0.0,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        feat_proj_dropout=0.0,
        apply_spec_augment=False,
        layerdrop=0.0,
        left_max_position_embeddings=8,
        right_max_position_embeddings=4,
        add_adapter=add_adapter,
        adapter_kernel_size=3,
        adapter_stride=2,
    )
    return QuranMultiLevelCTCConfig(encoder_config=encoder, final_dropout=0.0)


def test_init_and_forward_shapes_all_eleven_heads():
    model = QuranMultiLevelCTC(tiny_config()).eval()
    features = torch.randn(2, 12, 160)
    result = model(features)
    assert len(result.logits) == 11
    assert {k: v.out_features for k, v in model.level_to_lm_head.items()} == REFERENCE_LEVEL_TO_VOCAB_SIZE
    assert all(logits.shape == (2, 12, classes) for name, logits in result.logits.items()
               for classes in [REFERENCE_LEVEL_TO_VOCAB_SIZE[name]])
    assert result.output_lengths.tolist() == [12, 12]
    assert LabelCodec().level_vocab_sizes == REFERENCE_LEVEL_TO_VOCAB_SIZE


def test_default_config_matches_published_checkpoint_encoder():
    config = QuranMultiLevelCTCConfig()
    assert config.ctc_loss_reduction == "mean"
    assert config.encoder_config["hidden_size"] == 1024
    assert config.encoder_config["num_hidden_layers"] == 24
    assert config.encoder_config["num_attention_heads"] == 16
    assert config.encoder_config["feature_projection_input_dim"] == 160
    assert config.encoder_config["add_adapter"] is True
    assert config.encoder_config["adapter_stride"] == 2
    assert config.level_to_loss_weight["phonemes"] == 0.4
    assert config.level_to_loss_weight["shidda_or_rakhawa"] == 0.0605


def test_explicit_ctc_reduction_is_preserved():
    config = QuranMultiLevelCTCConfig(ctc_loss_reduction="sum")
    assert config.ctc_loss_reduction == "sum"


def test_model_passes_configured_ctc_reduction_to_loss(monkeypatch):
    config = tiny_config()
    config.ctc_loss_reduction = "sum"
    model = QuranMultiLevelCTC(config)
    reductions = []
    original_ctc_loss = torch.nn.functional.ctc_loss

    def recording_ctc_loss(*args, **kwargs):
        reductions.append(kwargs["reduction"])
        return original_ctc_loss(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "ctc_loss", recording_ctc_loss)
    out = model(torch.randn(1, 12, 160), labels={"phonemes": torch.tensor([[1, 2]])})
    assert torch.isfinite(out.loss)
    assert reductions == ["sum"]


def test_attention_mask_lengths_for_padded_feature_batch():
    model = QuranMultiLevelCTC(tiny_config()).eval()
    features = torch.randn(2, 12, 160)
    mask = torch.tensor([[1] * 12, [1] * 8 + [0] * 4])
    result = model(features, attention_mask=mask)
    assert result.output_lengths.tolist() == [12, 8]
    assert all(x.shape[:2] == (2, 12) for x in result.logits.values())


def test_adapter_downsampling_lengths_match_encoder():
    model = QuranMultiLevelCTC(tiny_config(add_adapter=True)).eval()
    features = torch.randn(2, 20, 160)
    mask = torch.tensor([[1] * 20, [1] * 16 + [0] * 4])
    result = model(features, attention_mask=mask)
    assert result.logits["phonemes"].shape == (2, 10, 43)
    assert result.output_lengths.tolist() == [10, 8]


def test_ctc_blank_and_ignored_label_padding_and_gradients():
    model = QuranMultiLevelCTC(tiny_config())
    x = torch.randn(2, 12, 160, requires_grad=True)
    labels = {"phonemes": torch.tensor([[1, 2], [3, -100]])}
    out = model(x, labels=labels)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert model.level_to_lm_head["phonemes"].weight.grad is not None
    assert model.encoder.feature_projection.projection.weight.grad is not None
    assert x.grad is not None
    assert model.config.pad_token_id == 0
    assert model.config.label_pad_token_id == -100


def test_partial_level_labels_compute_loss_only_for_supplied_heads():
    model = QuranMultiLevelCTC(tiny_config())
    out = model(
        torch.randn(1, 12, 160),
        labels={"phonemes": torch.tensor([[1, 2]])},
    )
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert model.level_to_lm_head["phonemes"].weight.grad is not None
    assert model.level_to_lm_head["ghonna"].weight.grad is None


def test_rejects_blank_as_target_and_wrong_feature_dimension():
    model = QuranMultiLevelCTC(tiny_config())
    with pytest.raises(ValueError, match="Valid phonemes target IDs"):
        model(torch.randn(1, 12, 160), labels={"phonemes": torch.tensor([[0]])})
    with pytest.raises(ValueError, match="Expected 160 feature channels"):
        model(torch.randn(1, 12, 80))


def test_config_serialization_round_trip(tmp_path):
    config = tiny_config()
    model = QuranMultiLevelCTC(config)
    model.save_pretrained(tmp_path, safe_serialization=True)
    restored = QuranMultiLevelCTC.from_pretrained(tmp_path)
    assert restored.config.level_to_vocab_size == REFERENCE_LEVEL_TO_VOCAB_SIZE
    assert restored.config.encoder_config["hidden_size"] == 16
    assert restored.config.pad_token_id == 0
    assert restored.config.ctc_loss_reduction == "mean"
    assert restored(torch.randn(1, 9, 160)).logits["ghonna"].shape == (1, 9, 3)


def test_waveform_adapter_uses_stage5_float_audio():
    class FakeExtractor:
        def __call__(self, waves, sampling_rate, return_tensors):
            assert sampling_rate == 16000
            assert return_tensors == "pt"
            return {"input_features": torch.randn(len(waves), 10, 160)}

    model = QuranMultiLevelCTC(tiny_config()).eval()
    result = model.forward_waveforms([torch.zeros(1000), torch.ones(900)], 16000, FakeExtractor())
    assert len(result.logits) == 11
    with pytest.raises(ValueError, match="16 kHz"):
        model.forward_waveforms([torch.zeros(1000)], 22050, FakeExtractor())
