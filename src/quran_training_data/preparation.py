"""One-record-at-a-time construction of audio/label examples and exclusions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import numpy as np
from quran_transcript import chunck_phonemes

from quran_annotations import validate_record
from quran_audio import AudioProcessingConfig, ProcessedAudio, preprocess_audio
from quran_dataset import ConfiguredRecord, DatasetLoader, decode_audio

from .encoding import LabelCodec, UnknownLabelError
from .splitting import assign_record_split


@dataclass(frozen=True)
class InclusionPolicy:
    """Explicit filtering choices. No duration/score cuts are assumed by default."""

    include_non_quran: bool = True
    expected_sample_rate: int | None = 16000
    require_mono: bool = True
    min_match_ratio: float | None = None
    min_duration_seconds: float | None = None
    max_duration_seconds: float | None = None
    on_missing_annotation: Literal["exclude", "error"] = "exclude"
    on_invalid_annotation: Literal["exclude", "error"] = "exclude"

    def __post_init__(self) -> None:
        if self.expected_sample_rate is not None and self.expected_sample_rate <= 0:
            raise ValueError("expected_sample_rate must be positive or None")
        if self.min_match_ratio is not None and not 0 <= self.min_match_ratio <= 1:
            raise ValueError("min_match_ratio must be between 0 and 1")
        if self.min_duration_seconds is not None and self.min_duration_seconds < 0:
            raise ValueError("min_duration_seconds cannot be negative")
        if self.max_duration_seconds is not None and self.max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be positive")
        if (self.min_duration_seconds is not None and self.max_duration_seconds is not None
                and self.min_duration_seconds > self.max_duration_seconds):
            raise ValueError("minimum duration exceeds maximum duration")


@dataclass(frozen=True)
class TrainingExample:
    source_key: str
    configuration: str
    split: str | None
    audio: ProcessedAudio
    phoneme_text: str
    phoneme_text_for_encoding: str
    phoneme_groups: tuple[str, ...]
    group_phoneme_ids: tuple[tuple[int, ...], ...]
    labels: Mapping[str, tuple[int, ...]]
    metadata: Mapping[str, Any]
    validation_warnings: tuple[str, ...] = ()

    @property
    def audio_samples(self) -> np.ndarray:
        return self.audio.samples

    def batch_mapping(self) -> dict[str, Any]:
        return {"labels": self.labels}


@dataclass(frozen=True)
class ExcludedRecord:
    source_key: str
    configuration: str
    reasons: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedRecord:
    example: TrainingExample | None
    excluded: ExcludedRecord | None

    def __post_init__(self) -> None:
        if (self.example is None) == (self.excluded is None):
            raise ValueError("Exactly one of example or excluded must be set")


class RecordPreparationError(ValueError):
    """A configured policy requested stopping on a malformed record."""


def _source_key(record: Mapping[str, Any], configuration: str) -> str:
    moshaf = record.get("moshaf_id") or configuration.removeprefix("moshaf_")
    segment = record.get("segment_index")
    if segment is None:
        segment = record.get("id", "unknown")
    return f"{configuration}:{moshaf}:{segment}"


def _trace_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "segment_index", "moshaf_id", "moshaf_name", "reciter_id", "reciter_arabic_name",
        "reciter_english_name", "sura_or_aya_index", "index_type", "timestamp_seconds",
        "duration_seconds", "sample_rate", "has_quran", "has_istiaatha", "has_bismillah",
        "has_sadaka", "start_span", "end_span", "match_ratio", "imlaey", "uthmani",
    )
    return {key: record.get(key) for key in keys}


def prepare_record(
    record: Mapping[str, Any],
    configuration: str,
    *,
    audio_config: AudioProcessingConfig,
    policy: InclusionPolicy = InclusionPolicy(),
    codec: LabelCodec | None = None,
    split_manifest: Mapping[str, Any] | None = None,
) -> PreparedRecord:
    """Validate and build one example; source record and annotation are untouched."""
    codec = codec or LabelCodec()
    key = _source_key(record, configuration)
    metadata = _trace_metadata(record)
    reasons: list[str] = []

    phonemes = record.get("phonemes")
    sifat = record.get("sifat")
    missing = []
    if not isinstance(phonemes, str) or not phonemes.strip():
        missing.append("missing_phonemes")
    if not isinstance(sifat, list) or not sifat:
        missing.append("missing_sifat")
    if missing:
        if policy.on_missing_annotation == "error":
            raise RecordPreparationError(f"{key}: {', '.join(missing)}")
        reasons.extend(missing)

    validation = validate_record(record)
    if validation.errors:
        codes = [issue.code for issue in validation.errors]
        if policy.on_invalid_annotation == "error":
            raise RecordPreparationError(f"{key}: invalid annotations {codes}")
        reasons.extend(f"invalid_annotation:{code}" for code in codes)

    if record.get("has_quran") is False and not policy.include_non_quran:
        reasons.append("policy:non_quran_excluded")
    ratio = record.get("match_ratio")
    if policy.min_match_ratio is not None:
        if ratio is None:
            reasons.append("policy:missing_match_ratio")
        elif not isinstance(ratio, (int, float)) or isinstance(ratio, bool) or not isfinite(float(ratio)) or ratio < policy.min_match_ratio:
            reasons.append("policy:low_match_ratio")

    duration = record.get("duration_seconds")
    valid_duration = isinstance(duration, (int, float)) and not isinstance(duration, bool) and isfinite(float(duration)) and duration > 0
    if valid_duration:
        if policy.min_duration_seconds is not None and duration < policy.min_duration_seconds:
            reasons.append("policy:too_short")
        if policy.max_duration_seconds is not None and duration > policy.max_duration_seconds:
            reasons.append("policy:too_long")

    if reasons:
        return PreparedRecord(None, ExcludedRecord(key, configuration, tuple(dict.fromkeys(reasons)), metadata))

    try:
        labels = codec.encode_sifat(sifat)
        phoneme_text_for_encoding = phonemes.replace(" ", "")
        phoneme_ids = codec.encode_phonemes(phonemes, remove_spaces=True)
        groups = tuple(chunck_phonemes(phonemes))
        if len(groups) != len(sifat) or [entry.get("phonemes") for entry in sifat] != list(groups):
            return PreparedRecord(None, ExcludedRecord(key, configuration,
                ("invalid_annotation:sifat_alignment_mismatch",), metadata))
    except (UnknownLabelError, TypeError, AttributeError) as exc:
        return PreparedRecord(None, ExcludedRecord(key, configuration,
            (f"unsupported_target:{type(exc).__name__}:{exc}",), metadata))

    try:
        audio_data = decode_audio(record)
        processed = preprocess_audio(audio_data, audio_config)
    except Exception as exc:
        return PreparedRecord(None, ExcludedRecord(key, configuration,
            (f"invalid_audio:{type(exc).__name__}:{exc}",), metadata))

    if policy.expected_sample_rate is not None and processed.sampling_rate != policy.expected_sample_rate:
        return PreparedRecord(None, ExcludedRecord(key, configuration,
            (f"audio:wrong_sample_rate:{processed.sampling_rate}",), metadata))
    if policy.require_mono and processed.channels != 1:
        return PreparedRecord(None, ExcludedRecord(key, configuration,
            (f"audio:not_mono:{processed.channels}_channels",), metadata))

    # If duration metadata is absent/unusable, use the decoded samples for configured cuts.
    actual_duration = processed.samples.shape[0] / processed.sampling_rate
    if not valid_duration:
        if policy.min_duration_seconds is not None and actual_duration < policy.min_duration_seconds:
            reasons.append("policy:too_short")
        if policy.max_duration_seconds is not None and actual_duration > policy.max_duration_seconds:
            reasons.append("policy:too_long")
    if reasons:
        return PreparedRecord(None, ExcludedRecord(key, configuration, tuple(dict.fromkeys(reasons)), metadata))

    split = assign_record_split(record, split_manifest) if split_manifest is not None else None
    return PreparedRecord(
        TrainingExample(
            source_key=key,
            configuration=configuration,
            split=split,
            audio=processed,
            phoneme_text=phonemes,
            phoneme_text_for_encoding=phoneme_text_for_encoding,
            phoneme_groups=groups,
            group_phoneme_ids=tuple(tuple(codec.encode_phonemes(group)) for group in groups),
            labels={"phonemes": tuple(phoneme_ids), **{name: tuple(ids) for name, ids in labels.items()}},
            metadata=metadata,
            validation_warnings=tuple(issue.code for issue in validation.warnings),
        ),
        None,
    )


class TrainingDataBuilder:
    """Lazy adapter over the Stage 2 loader; no all-row materialization by default."""

    def __init__(
        self,
        loader: DatasetLoader,
        *,
        audio_config: AudioProcessingConfig,
        policy: InclusionPolicy = InclusionPolicy(),
        codec: LabelCodec | None = None,
        split_manifest: Mapping[str, Any] | None = None,
    ) -> None:
        self.loader = loader
        self.audio_config = audio_config
        self.policy = policy
        self.codec = codec or LabelCodec()
        self.split_manifest = split_manifest

    def iter_prepared(
        self,
        configurations: Iterable[str],
        *,
        limit_per_configuration: int | None = 3,
        split: str | None = None,
        exclusion_ledger_path: str | Path | None = None,
    ) -> Iterable[PreparedRecord]:
        ledger = Path(exclusion_ledger_path) if exclusion_ledger_path is not None else None
        if ledger is not None:
            ledger.parent.mkdir(parents=True, exist_ok=True)
        for item in self.loader.iter_configurations(
            configurations, split=split, limit_per_configuration=limit_per_configuration
        ):
            outcome = prepare_record(
                item.record, item.configuration, audio_config=self.audio_config,
                policy=self.policy, codec=self.codec, split_manifest=self.split_manifest,
            )
            if ledger is not None and outcome.excluded is not None:
                with ledger.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(asdict(outcome.excluded), ensure_ascii=False) + "\n")
            yield outcome
