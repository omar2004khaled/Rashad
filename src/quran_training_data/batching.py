"""Padding helpers for inspecting/assembling CTC label batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .encoding import IGNORE_INDEX


@dataclass(frozen=True)
class PaddedTargets:
    values: np.ndarray
    lengths: np.ndarray
    padding_value: int = IGNORE_INDEX


def pad_sequences(
    sequences: Sequence[Sequence[int]], *, padding_value: int = IGNORE_INDEX
) -> PaddedTargets:
    """Right-pad variable target sequences; padding is ignored by CTC loss."""
    if not sequences:
        raise ValueError("At least one sequence is required")
    lengths = np.asarray([len(seq) for seq in sequences], dtype=np.int64)
    width = int(lengths.max(initial=0))
    values = np.full((len(sequences), width), padding_value, dtype=np.int64)
    for row, seq in enumerate(sequences):
        if seq:
            values[row, : len(seq)] = np.asarray(seq, dtype=np.int64)
    return PaddedTargets(values=values, lengths=lengths, padding_value=padding_value)


def pad_label_batch(
    examples: Sequence[Mapping[str, object]], *, padding_value: int = IGNORE_INDEX
) -> tuple[dict[str, PaddedTargets], dict[str, np.ndarray]]:
    """Pad phoneme and each sifat level independently, preserving their lengths."""
    if not examples:
        raise ValueError("At least one example is required")
    levels = set(examples[0]["labels"])
    if any(set(example["labels"]) != levels for example in examples):
        raise ValueError("Every example must provide the same target levels")
    targets: dict[str, PaddedTargets] = {}
    for level in sorted(levels):
        seqs = [example["labels"][level] for example in examples]
        targets[level] = pad_sequences(seqs, padding_value=padding_value)
    return targets, {level: target.lengths.copy() for level, target in targets.items()}
