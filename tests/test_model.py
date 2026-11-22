import torch
import pytest
import json
from safetensors.torch import save_file
from transformers import Wav2Vec2BertConfig

from quran_model import QuranMultiLevelCTC, QuranMultiLevelCTCConfig
from quran_model.modeling import CheckpointCompatibilityError
from quran_model.configuration import REFERENCE_LEVEL_TO_VOCAB_SIZE
from quran_training_data.encoding import LabelCodec


def tiny_config(add_adapter=False, attention_implementation="sdpa"):
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
    return QuranMultiLevelCTCConfig(
        encoder_config=encoder, final_dropout=0.0,
        attention_implementation=attention_implementation,
    )


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
    assert config.attention_implementation == "sdpa"


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


def test_gradient_checkpointing_can_be_enabled():
    model = QuranMultiLevelCTC(tiny_config())
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert model.encoder.encoder.gradient_checkpointing is True
    assert all(layer.gradient_checkpointing for layer in model.encoder.encoder.layers)


def test_sdpa_attention_matches_eager_outputs_losses_and_gradients():
    torch.manual_seed(12)
    eager = QuranMultiLevelCTC(tiny_config(attention_implementation="eager")).train()
    sdpa = QuranMultiLevelCTC(tiny_config(attention_implementation="sdpa")).train()
    sdpa.load_state_dict(eager.state_dict())
    assert eager.attention_implementation_status() == {
        "configured": "eager", "sdpa_layers": 0, "eager_layers": 1,
    }
    assert sdpa.attention_implementation_status() == {
        "configured": "sdpa", "sdpa_layers": 1, "eager_layers": 0,
    }
    inputs = torch.randn(1, 12, 160)
    targets = {"phonemes": torch.tensor([[1, 2, 3]])}
    torch.manual_seed(44)
    eager_output = eager(inputs, labels=targets)
    eager_output.loss.backward()
    eager_encoder_grad = eager.encoder.feature_projection.projection.weight.grad.detach().clone()
    eager_head_grad = eager.level_to_lm_head["phonemes"].weight.grad.detach().clone()

    torch.manual_seed(44)
    sdpa_output = sdpa(inputs, labels=targets)
    sdpa_output.loss.backward()
    assert torch.isfinite(eager_output.loss) and torch.isfinite(sdpa_output.loss)
    torch.testing.assert_close(sdpa_output.logits["phonemes"], eager_output.logits["phonemes"],
                               rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(sdpa_output.loss, eager_output.loss, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(
        sdpa.encoder.feature_projection.projection.weight.grad, eager_encoder_grad,
        rtol=3e-4, atol=3e-5,
    )
    torch.testing.assert_close(
        sdpa.level_to_lm_head["phonemes"].weight.grad, eager_head_grad,
        rtol=3e-4, atol=3e-5,
    )
    assert torch.isfinite(sdpa.encoder.feature_projection.projection.weight.grad).all()


def test_sdpa_attention_falls_back_when_attention_maps_requested():
    model = QuranMultiLevelCTC(tiny_config(attention_implementation="sdpa")).eval()
    result = model(torch.randn(1, 10, 160), output_attentions=True)
    assert result.attentions is not None
    assert len(result.attentions) == 1
    assert result.attentions[0].shape == (1, 4, 10, 10)


def test_attention_backend_config_serializes_and_can_be_changed(tmp_path):
    model = QuranMultiLevelCTC(tiny_config(attention_implementation="sdpa"))
    state_keys = set(model.state_dict())
    model.set_attention_implementation("eager")
    assert set(model.state_dict()) == state_keys
    model.save_pretrained(tmp_path, safe_serialization=True)
    restored = QuranMultiLevelCTC.from_pretrained(tmp_path)
    assert restored.config.attention_implementation == "eager"
    assert restored.attention_implementation_status()["sdpa_layers"] == 0


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


def _write_synthetic_original_checkpoint(path, *, omit_key=None, mismatch_key=None, add_unexpected=False):
    source_model = QuranMultiLevelCTC(tiny_config()).eval()
    source = {}
    for key, tensor in source_model.state_dict().items():
        if key == omit_key:
            continue
        source_key = "wav2vec2_bert." + key.removeprefix("encoder.") if key.startswith("encoder.") else key
        value = tensor.detach().cpu().contiguous()
        if key == mismatch_key:
            value = value.flatten()[:1].clone()
        source[source_key] = value
    if add_unexpected:
        source["unexpected.weight"] = torch.zeros(1)
    save_file(source, str(path / "model.safetensors"))
    raw = dict(source_model.encoder.config.to_dict())
    raw.update({
        "level_to_vocab_size": source_model.config.level_to_vocab_size,
        "level_to_loss_weight": source_model.config.level_to_loss_weight,
        "final_dropout": source_model.config.final_dropout,
        "pad_token_id": 0,
        "ctc_loss_reduction": source_model.config.ctc_loss_reduction,
        "ctc_zero_infinity": False,
    })
    (path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    return source_model


def test_original_checkpoint_parameter_mapping_loads_encoder_and_heads(tmp_path):
    source_model = _write_synthetic_original_checkpoint(tmp_path)
    loaded, report = QuranMultiLevelCTC.from_pretrained_checkpoint(tmp_path)
    assert report.is_complete
    assert not report.missing and not report.unexpected and not report.shape_mismatches
    assert report.loaded_parameter_count == sum(x.numel() for x in source_model.state_dict().values())
    matched = dict(report.matched)
    assert matched["wav2vec2_bert.feature_projection.projection.weight"] == "encoder.feature_projection.projection.weight"
    assert ("level_to_lm_head.phonemes.weight", "level_to_lm_head.phonemes.weight") in report.matched
    for key, value in source_model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])


@pytest.mark.parametrize("kwargs, report_field", [
    ({"omit_key": "encoder.feature_projection.projection.weight"}, "missing"),
    ({"mismatch_key": "encoder.feature_projection.projection.weight"}, "shape_mismatches"),
    ({"mismatch_key": "level_to_lm_head.phonemes.weight"}, "shape_mismatches"),
    ({"add_unexpected": True}, "unexpected"),
])
def test_checkpoint_loader_reports_incompatibilities(tmp_path, kwargs, report_field):
    _write_synthetic_original_checkpoint(tmp_path, **kwargs)
    with pytest.raises(CheckpointCompatibilityError) as exc:
        QuranMultiLevelCTC.from_pretrained_checkpoint(tmp_path)
    assert getattr(exc.value.report, report_field)


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
