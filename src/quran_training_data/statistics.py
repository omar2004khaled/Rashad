"""Streaming counters for preparation outcomes; never retains audio samples."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

import numpy as np

from .preparation import PreparedRecord


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None}
    data = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "min": float(data.min()), "median": float(np.median(data)),
            "p90": float(np.quantile(data, 0.9)), "max": float(data.max())}


class DatasetStatistics:
    """Incremental dataset statistics over included and excluded candidate rows."""

    def __init__(self) -> None:
        self.total = 0
        self.included = 0
        self.excluded = 0
        self.exclusion_reasons: Counter[str] = Counter()
        self.quran_counts: Counter[str] = Counter()
        self.reciters: set[str] = set()
        self.moshafs: set[str] = set()
        self.durations: list[float] = []
        self.phoneme_lengths: list[int] = []
        self.sifat_group_counts: list[int] = []
        self.phoneme_frequency: Counter[str] = Counter()
        self.sifat_frequency: dict[str, Counter[str]] = {}
        self.split_rows: Counter[str] = Counter()
        self.split_reciters: dict[str, set[str]] = {}
        self.split_moshafs: dict[str, set[str]] = {}

    def observe(self, row: Mapping[str, Any], outcome: PreparedRecord) -> None:
        self.total += 1
        self.quran_counts["quran" if row.get("has_quran") is True else "non_quran" if row.get("has_quran") is False else "unknown"] += 1
        if row.get("reciter_id") is not None:
            self.reciters.add(str(row["reciter_id"]))
        if row.get("moshaf_id") is not None:
            self.moshafs.add(str(row["moshaf_id"]))
        if outcome.excluded is not None:
            self.excluded += 1
            self.exclusion_reasons.update(outcome.excluded.reasons)
            return
        example = outcome.example
        assert example is not None
        self.included += 1
        duration = example.audio.original.duration_seconds
        self.durations.append(float(duration))
        self.phoneme_lengths.append(len(example.labels["phonemes"]))
        self.sifat_group_counts.append(len(example.phoneme_groups))
        self.phoneme_frequency.update(symbol for symbol in row["phonemes"] if symbol != " ")
        for group in row["sifat"]:
            for field, value in group.items():
                if field == "phonemes":
                    continue
                self.sifat_frequency.setdefault(field, Counter()).update([str(value)])
        if example.split is not None:
            split = example.split
            self.split_rows[split] += 1
            self.split_reciters.setdefault(split, set())
            self.split_moshafs.setdefault(split, set())
            if row.get("reciter_id") is not None:
                self.split_reciters[split].add(str(row["reciter_id"]))
            if row.get("moshaf_id") is not None:
                self.split_moshafs[split].add(str(row["moshaf_id"]))

    def summary(self, *, rare_count: int = 5) -> dict[str, Any]:
        rare_phonemes = {key: count for key, count in sorted(self.phoneme_frequency.items()) if count <= rare_count}
        rare_sifat = {
            field: {label: count for label, count in sorted(counter.items()) if count <= rare_count}
            for field, counter in sorted(self.sifat_frequency.items())
        }
        return {
            "records": {"total_seen": self.total, "included": self.included, "excluded": self.excluded,
                        "exclusion_reasons": dict(sorted(self.exclusion_reasons.items()))},
            "quran_flags": dict(sorted(self.quran_counts.items())),
            "unique_reciters": len(self.reciters), "unique_moshaf_sets": len(self.moshafs),
            "audio_duration_seconds": _distribution(self.durations),
            "phoneme_sequence_length": _distribution([float(x) for x in self.phoneme_lengths]),
            "sifat_group_count": _distribution([float(x) for x in self.sifat_group_counts]),
            "phoneme_frequency": dict(sorted(self.phoneme_frequency.items())),
            "rare_phonemes_at_or_below_threshold": rare_phonemes,
            "sifat_category_frequency": {field: dict(sorted(counter.items()))
                                         for field, counter in sorted(self.sifat_frequency.items())},
            "rare_sifat_categories_at_or_below_threshold": rare_sifat,
            "split_rows": dict(sorted(self.split_rows.items())),
            "split_unique_reciters": {key: len(value) for key, value in sorted(self.split_reciters.items())},
            "split_unique_moshaf_sets": {key: len(value) for key, value in sorted(self.split_moshafs.items())},
            "rare_threshold": rare_count,
        }
