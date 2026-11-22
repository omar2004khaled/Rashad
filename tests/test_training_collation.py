import numpy as np
import pytest
import torch

from quran_training_data import IGNORE_INDEX, QuranTrainingCollator
from quran_model import QuranMultiLevelCTC, QuranMultiLevelCTCConfig
from transformers import Wav2Vec2BertConfig


def tiny_config():
    encoder = Wav2Vec2BertConfig(
        hidden_size=16, num_hidden_layers=1, num_attention_heads=4,
        intermediate_size=32, feature_projection_input_dim=160,
        conv_depthwise_kernel_size=3, conformer_conv_dropout=0.0,
        hidden_dropout=0.0, attention_dropout=0.0, activation_dropout=0.0,
        feat_proj_dropout=0.0, apply_spec_augment=False, layerdrop=0.0,
    )
    return QuranMultiLevelCTCConfig(encoder_config=encoder, final_dropout=0.0)


def test_collator_pads_variable_features_and_partial_labels():
    collator = QuranTrainingCollator()
    examples = [
        {
            "input_features": np.ones((5, 160), dtype=np.float32),
            "labels": {"phonemes": [1, 2], "hams_or_jahr": [1]},
        },
        {
            "input_features": np.full((3, 160), 2, dtype=np.float32),
            "labels": {"phonemes": [3], "ghonna": [2, 1]},
        },
    ]
    batch = collator(examples)
    assert batch["input_features"].shape == (2, 5, 160)
    assert batch["input_lengths"].tolist() == [5, 3]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]
    assert batch["labels"]["phonemes"].tolist() == [[1, 2], [3, IGNORE_INDEX]]
    assert batch["labels"]["hams_or_jahr"].tolist() == [[1], [IGNORE_INDEX]]
    assert batch["label_level_mask"]["hams_or_jahr"].tolist() == [True, False]
    assert batch["label_level_mask"]["ghonna"].tolist() == [False, True]
    assert batch["target_lengths"]["phonemes"].tolist() == [2, 1]
    assert batch["target_lengths"]["ghonna"].tolist() == [0, 2]
    # Collation leaves the source arrays and annotations untouched.
    assert np.all(examples[0]["input_features"] == 1)
    assert examples[0]["labels"]["phonemes"] == [1, 2]


def test_collated_partial_supervision_runs_model_loss():
    batch = QuranTrainingCollator()([
        {"input_features": np.ones((8, 160), dtype=np.float32),
         "labels": {"phonemes": [1, 2]}},
        {"input_features": np.ones((6, 160), dtype=np.float32),
         "labels": {"hams_or_jahr": [2]}},
    ])
    model = QuranMultiLevelCTC(tiny_config())
    out = model(**batch)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert model.level_to_lm_head["phonemes"].weight.grad is not None
    assert model.level_to_lm_head["hams_or_jahr"].weight.grad is not None
    assert model.level_to_lm_head["ghonna"].weight.grad is None


def test_collator_extracts_variable_waveforms_and_checks_rate():
    class FakeFeatureExtractor:
        def __call__(self, waveform, sampling_rate, return_tensors):
            assert sampling_rate == 16000
            assert return_tensors == "pt"
            frames = len(waveform) // 2
            return {"input_features": torch.full((1, frames, 160), float(frames))}

    batch = QuranTrainingCollator(FakeFeatureExtractor())([
        {"audio_samples": np.zeros(10, dtype=np.float32), "sampling_rate": 16000,
         "labels": {"phonemes": [1]}},
        {"audio_samples": np.zeros(6, dtype=np.float32), "sampling_rate": 16000,
         "labels": {"phonemes": [2]}},
    ])
    assert batch["input_features"].shape == (2, 5, 160)
    assert batch["input_lengths"].tolist() == [5, 3]
    with pytest.raises(ValueError, match="Expected 16000 Hz"):
        QuranTrainingCollator(FakeFeatureExtractor())([
            {"audio_samples": np.zeros(10), "sampling_rate": 22050,
             "labels": {"phonemes": [1]}}
        ])


def test_collator_rejects_invalid_features_or_label_ids():
    collator = QuranTrainingCollator()
    with pytest.raises(ValueError, match=r"shape \(frames, 160\)"):
        collator([{"input_features": np.ones((3, 80)), "labels": {"phonemes": [1]}}])
    with pytest.raises(ValueError, match="target IDs"):
        collator([{"input_features": np.ones((3, 160)), "labels": {"phonemes": [0]}}])
