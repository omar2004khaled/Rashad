"""Feature extraction and padding for variable-length CTC training examples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from quran_model.configuration import REFERENCE_LEVEL_TO_VOCAB_SIZE

from .encoding import IGNORE_INDEX


class QuranTrainingCollator:
    """Convert Stage 5 examples or feature records into a padded model batch.

    Stage 5 ``TrainingExample`` objects contain mono waveforms. A matching HF
    feature extractor converts these to Wav2Vec2-BERT features. Tests and
    cached-feature workflows can instead supply each row's ``input_features``.
    Missing labels are explicitly tracked with `label_level_mask`; their padded
    `-100` values do not become empty CTC targets.
    """

    def __init__(
        self,
        feature_extractor=None,
        *,
        feature_dim: int = 160,
        sampling_rate: int = 16000,
        level_to_vocab_size: Mapping[str, int] | None = None,
    ):
        self.feature_extractor = feature_extractor
        self.feature_dim = int(feature_dim)
        self.sampling_rate = int(sampling_rate)
        self.level_to_vocab_size = dict(level_to_vocab_size or REFERENCE_LEVEL_TO_VOCAB_SIZE)

    @staticmethod
    def _get(example: Any, name: str, default=None):
        if isinstance(example, Mapping):
            return example.get(name, default)
        return getattr(example, name, default)

    def _features_for_example(self, example: Any) -> torch.Tensor:
        precomputed = self._get(example, "input_features")
        if precomputed is not None:
            features = torch.as_tensor(precomputed, dtype=torch.float32)
            if features.ndim == 3 and features.shape[0] == 1:
                features = features[0]
            return features

        if self.feature_extractor is None:
            raise ValueError("A feature_extractor is required for waveform examples")
        waveform = self._get(example, "audio_samples")
        sample_rate = self._get(example, "sampling_rate")
        audio_obj = self._get(example, "audio")
        if waveform is None and audio_obj is not None:
            waveform = getattr(audio_obj, "samples", None)
        if sample_rate is None and audio_obj is not None:
            sample_rate = getattr(audio_obj, "sampling_rate", None)
        if waveform is None:
            raise ValueError("Example must provide input_features or audio_samples")
        if sample_rate != self.sampling_rate:
            raise ValueError(f"Expected {self.sampling_rate} Hz waveform, got {sample_rate}")
        waveform = np.asarray(waveform)
        if waveform.ndim == 2 and waveform.shape[1] == 1:
            waveform = waveform[:, 0]
        if waveform.ndim != 1:
            raise ValueError(f"Expected mono waveform, got shape {waveform.shape}")
        encoded = self.feature_extractor(
            waveform, sampling_rate=sample_rate, return_tensors="pt"
        )
        features = encoded["input_features"]
        if features.ndim != 3 or features.shape[0] != 1:
            raise ValueError("Feature extractor must return a single (1, frames, channels) input")
        return features[0].to(dtype=torch.float32)

    def __call__(self, examples: Sequence[Any]) -> dict[str, Any]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        feature_rows = [self._features_for_example(example) for example in examples]
        for row in feature_rows:
            if row.ndim != 2 or row.shape[1] != self.feature_dim or row.shape[0] == 0:
                raise ValueError(
                    f"Each feature sequence must have shape (frames, {self.feature_dim})"
                )
            if not torch.isfinite(row).all():
                raise ValueError("Feature sequence contains non-finite values")
        input_lengths = torch.tensor([row.shape[0] for row in feature_rows], dtype=torch.long)
        max_frames = int(input_lengths.max().item())
        input_features = torch.zeros((len(examples), max_frames, self.feature_dim), dtype=torch.float32)
        attention_mask = torch.zeros((len(examples), max_frames), dtype=torch.long)
        for index, row in enumerate(feature_rows):
            input_features[index, : row.shape[0]] = row
            attention_mask[index, : row.shape[0]] = 1

        label_rows = [self._get(example, "labels", {}) for example in examples]
        if any(not isinstance(labels, Mapping) for labels in label_rows):
            raise TypeError("Each example's labels must be a mapping from level to token IDs")
        for labels in label_rows:
            unknown = set(labels) - set(self.level_to_vocab_size)
            if unknown:
                raise ValueError(f"Unknown label levels: {sorted(unknown)}")
        levels = [name for name in self.level_to_vocab_size if any(name in row for row in label_rows)]
        if not levels:
            raise ValueError("At least one example must contain a supervised label level")

        batched_labels = {}
        label_level_mask = {}
        target_lengths = {}
        for level in levels:
            supervised = torch.tensor([level in row for row in label_rows], dtype=torch.bool)
            sequences = []
            for row, is_supervised in zip(label_rows, supervised.tolist()):
                sequence = list(row[level]) if is_supervised else []
                if any(not isinstance(token, (int, np.integer)) for token in sequence):
                    raise TypeError(f"{level} target IDs must be integers")
                if any(token < 1 or token >= self.level_to_vocab_size[level] for token in sequence):
                    raise ValueError(f"{level} target IDs must be in [1, {self.level_to_vocab_size[level] - 1}]")
                sequences.append(sequence)
            lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
            padded = torch.full((len(examples), int(lengths.max().item())), IGNORE_INDEX, dtype=torch.long)
            for index, sequence in enumerate(sequences):
                if sequence:
                    padded[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            batched_labels[level] = padded
            label_level_mask[level] = supervised
            target_lengths[level] = lengths

        return {
            "input_features": input_features,
            "attention_mask": attention_mask,
            "input_lengths": input_lengths,
            "labels": batched_labels,
            "target_lengths": target_lengths,
            "label_level_mask": label_level_mask,
        }


__all__ = ["QuranTrainingCollator"]
