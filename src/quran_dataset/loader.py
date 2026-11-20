"""Streaming dataset access without label or audio transformations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping
import io
import json

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset


class DatasetAccessError(RuntimeError):
    """A configuration or split could not be accessed."""


class InvalidAudioError(ValueError):
    """An audio field is missing or cannot be decoded as supplied."""


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str
    available_configurations: tuple[str, ...]
    recitation_configurations: tuple[str, ...]
    metadata_configurations: Mapping[str, str]
    default_split: str = "train"
    selected_configurations: tuple[str, ...] = ()
    sample_limit: int = 3
    streaming: bool = True
    decode_audio: bool = False
    shuffle_seed: int = 42
    shuffle_buffer_size: int = 1000
    schema_unavailable_via_stage1_first_rows: tuple[str, ...] = ()

    @classmethod
    def from_file(cls, path: str | Path) -> "DatasetConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            dataset_id=raw["dataset_id"],
            available_configurations=tuple(raw["available_configurations"]),
            recitation_configurations=tuple(raw["recitation_configurations"]),
            metadata_configurations=raw["metadata_configurations"],
            default_split=raw.get("default_split", "train"),
            selected_configurations=tuple(raw.get("selected_configurations", ())),
            sample_limit=int(raw.get("sample_limit", 3)),
            streaming=bool(raw.get("streaming", True)),
            decode_audio=bool(raw.get("decode_audio", False)),
            shuffle_seed=int(raw.get("shuffle_seed", 42)),
            shuffle_buffer_size=int(raw.get("shuffle_buffer_size", 1000)),
            schema_unavailable_via_stage1_first_rows=tuple(
                raw.get("schema_unavailable_via_stage1_first_rows", ())
            ),
        )


@dataclass(frozen=True)
class SchemaInspection:
    configuration: str
    split: str
    verified: bool
    features: Any = None
    error: str | None = None


@dataclass(frozen=True)
class ConfiguredRecord:
    """The configuration label plus an unchanged record returned by HF datasets."""

    configuration: str
    record: Mapping[str, Any]


@dataclass(frozen=True)
class EnrichedRecord:
    """An unchanged source record with separate metadata references."""

    record: Mapping[str, Any]
    moshaf_metadata: Mapping[str, Any] | None
    reciter_metadata: Mapping[str, Any] | None


@dataclass(frozen=True)
class AudioData:
    """Decoded samples and properties; samples retain the decoder's values."""

    samples: np.ndarray
    sampling_rate: int
    channels: int
    duration_seconds: float


_CONFIG_LIMIT = object()


class DatasetLoader:
    def __init__(
        self,
        config: DatasetConfig | str | Path,
        *,
        loader_fn: Callable[..., Any] | None = None,
    ) -> None:
        self.config = (
            config if isinstance(config, DatasetConfig) else DatasetConfig.from_file(config)
        )
        self._loader_fn = loader_fn or load_dataset

    def _validate_configuration(self, configuration: str) -> None:
        if configuration not in self.config.available_configurations:
            raise ValueError(
                f"Unknown configuration {configuration!r}. Available configurations: "
                f"{', '.join(self.config.available_configurations)}"
            )

    def load_config(
        self,
        configuration: str,
        *,
        split: str | None = None,
        streaming: bool | None = None,
        decode_audio_fields: bool | None = None,
    ) -> Any:
        """Open one explicit configuration/split; streaming is enabled by default."""
        self._validate_configuration(configuration)
        selected_split = split or self.config.default_split
        use_streaming = self.config.streaming if streaming is None else streaming
        decode_audio = (
            self.config.decode_audio
            if decode_audio_fields is None
            else decode_audio_fields
        )
        try:
            dataset = self._loader_fn(
                self.config.dataset_id,
                name=configuration,
                split=selected_split,
                streaming=use_streaming,
            )
            if not decode_audio and "audio" in getattr(dataset, "features", {}):
                dataset = dataset.cast_column("audio", Audio(decode=False))
            return dataset
        except Exception as exc:
            raise DatasetAccessError(
                f"Could not load {self.config.dataset_id!r}, configuration "
                f"{configuration!r}, split {selected_split!r} "
                f"(streaming={use_streaming}): {exc}"
            ) from exc

    def inspect_schema(
        self, configuration: str, *, split: str | None = None
    ) -> SchemaInspection:
        """Inspect only this configuration's declared features, without fallback."""
        selected_split = split or self.config.default_split
        try:
            # Inspect before casting audio to decode=False so the HF declaration is kept.
            self._validate_configuration(configuration)
            dataset = self._loader_fn(
                self.config.dataset_id,
                name=configuration,
                split=selected_split,
                streaming=True,
            )
            features = getattr(dataset, "features", None)
            if features is None:
                raise RuntimeError("The dataset loader returned no feature declaration")
            return SchemaInspection(configuration, selected_split, True, features=features)
        except Exception as exc:
            return SchemaInspection(
                configuration,
                selected_split,
                False,
                error=f"{type(exc).__name__}: {exc}",
            )

    def iter_samples(
        self,
        configuration: str,
        *,
        split: str | None = None,
        limit: int | None | object = _CONFIG_LIMIT,
        streaming: bool | None = None,
        shuffle: bool = False,
        seed: int | None = None,
        buffer_size: int | None = None,
    ) -> Iterator[Mapping[str, Any]]:
        """Iterate unchanged records; default limit comes from the JSON config."""
        effective_limit = self.config.sample_limit if limit is _CONFIG_LIMIT else limit
        if effective_limit is not None and (not isinstance(effective_limit, int) or effective_limit < 0):
            raise ValueError("limit must be a non-negative integer or None")
        dataset = self.load_config(configuration, split=split, streaming=streaming)
        if shuffle:
            if not hasattr(dataset, "shuffle"):
                raise ValueError("This dataset object does not support deterministic shuffle")
            dataset = dataset.shuffle(
                seed=self.config.shuffle_seed if seed is None else seed,
                buffer_size=self.config.shuffle_buffer_size if buffer_size is None else buffer_size,
            )
        iterator = iter(dataset)
        yield from iterator if effective_limit is None else islice(iterator, effective_limit)

    def get_sample(
        self,
        configuration: str,
        index: int,
        *,
        split: str | None = None,
        streaming: bool | None = None,
    ) -> Mapping[str, Any]:
        """Get a zero-based sample; for streaming data this walks to that index."""
        if not isinstance(index, int) or index < 0:
            raise ValueError("index must be a non-negative integer")
        dataset = self.load_config(configuration, split=split, streaming=streaming)
        try:
            if not (self.config.streaming if streaming is None else streaming):
                return dataset[index]
            return next(islice(iter(dataset), index, index + 1))
        except StopIteration as exc:
            raise IndexError(f"No sample at index {index} in {configuration!r}") from exc
        except (IndexError, KeyError):
            raise
        except Exception as exc:
            raise DatasetAccessError(
                f"Could not read index {index} from configuration {configuration!r}: {exc}"
            ) from exc

    def iter_configurations(
        self,
        configurations: Iterable[str] | None = None,
        *,
        split: str | None = None,
        limit_per_configuration: int | None | object = _CONFIG_LIMIT,
        streaming: bool | None = None,
        shuffle: bool = False,
        seed: int | None = None,
        buffer_size: int | None = None,
    ) -> Iterator[ConfiguredRecord]:
        """Iterate selected configurations, retaining config identity separately."""
        selected = tuple(configurations or self.config.selected_configurations)
        if not selected:
            raise ValueError("Provide configurations or set selected_configurations in config")
        for configuration in selected:
            self._validate_configuration(configuration)
            for record in self.iter_samples(
                configuration,
                split=split,
                limit=limit_per_configuration,
                streaming=streaming,
                shuffle=shuffle,
                seed=seed,
                buffer_size=buffer_size,
            ):
                yield ConfiguredRecord(configuration, record)

    def load_metadata(self, kind: str, *, split: str | None = None) -> list[Mapping[str, Any]]:
        """Read a small metadata configuration (27 or 22 rows in Stage 1)."""
        if kind not in self.config.metadata_configurations:
            raise ValueError(
                f"Unknown metadata kind {kind!r}; choose one of "
                f"{', '.join(self.config.metadata_configurations)}"
            )
        configuration = self.config.metadata_configurations[kind]
        return list(self.iter_samples(configuration, split=split, limit=None))

    def metadata_index(self, *, split: str | None = None) -> "MetadataIndex":
        return MetadataIndex(
            self.load_metadata("moshaf", split=split),
            self.load_metadata("reciters", split=split),
        )


class MetadataIndex:
    """Lookup and join recording/reciter metadata without editing source records."""

    def __init__(
        self,
        moshaf_rows: Iterable[Mapping[str, Any]],
        reciter_rows: Iterable[Mapping[str, Any]],
    ) -> None:
        self._moshaf = {
            str(row["id"]): row for row in moshaf_rows if row.get("id") is not None
        }
        self._reciters = {
            str(row["id"]): row for row in reciter_rows if row.get("id") is not None
        }

    def lookup_moshaf(self, moshaf_id: Any) -> Mapping[str, Any] | None:
        return None if moshaf_id is None else self._moshaf.get(str(moshaf_id))

    def lookup_reciter(self, reciter_id: Any) -> Mapping[str, Any] | None:
        return None if reciter_id is None else self._reciters.get(str(reciter_id))

    def join(self, record: Mapping[str, Any]) -> EnrichedRecord:
        return EnrichedRecord(
            record=record,
            moshaf_metadata=self.lookup_moshaf(record.get("moshaf_id")),
            reciter_metadata=self.lookup_reciter(record.get("reciter_id")),
        )


def candidate_record_id(record: Mapping[str, Any]) -> tuple[str, str] | None:
    """Return (moshaf_id, segment_index) only when both values are present."""
    moshaf_id = record.get("moshaf_id")
    segment_index = record.get("segment_index")
    if moshaf_id is None or segment_index is None:
        return None
    moshaf_id = str(moshaf_id).strip()
    segment_index = str(segment_index).strip()
    if not moshaf_id or not segment_index:
        return None
    return moshaf_id, segment_index


def decode_audio(record: Mapping[str, Any]) -> AudioData:
    """Decode audio without resampling, normalization, denoising, or trimming."""
    if "audio" not in record or record["audio"] is None:
        raise InvalidAudioError("Record has no usable 'audio' field")
    audio = record["audio"]
    try:
        if isinstance(audio, Mapping) and audio.get("array") is not None:
            samples = np.asarray(audio["array"])
            rate = audio.get("sampling_rate")
            if rate is None:
                rate = record.get("sample_rate")
        else:
            payload = audio if isinstance(audio, Mapping) else {"bytes": audio}
            raw = payload.get("bytes")
            path = payload.get("path")
            if raw is not None:
                source: Any = io.BytesIO(raw)
            elif path:
                source = Path(path)
            else:
                raise InvalidAudioError("Audio needs decoded samples, bytes, or a local path")
            samples, rate = sf.read(source, always_2d=True, dtype="float32")
        if rate is None or int(rate) <= 0:
            raise InvalidAudioError("Audio sampling rate is missing or invalid")
        if samples.ndim == 1:
            channels = 1
            frame_count = samples.shape[0]
        elif samples.ndim == 2:
            # soundfile returns (frames, channels); HF audio examples here are mono.
            # For multichannel decoded HF arrays, this reports the trailing dimension.
            channels = samples.shape[1]
            frame_count = samples.shape[0]
        else:
            raise InvalidAudioError(f"Expected 1D or 2D audio, got shape {samples.shape}")
        if frame_count == 0 or channels == 0:
            raise InvalidAudioError("Audio contains no samples or channels")
        return AudioData(
            samples=samples,
            sampling_rate=int(rate),
            channels=int(channels),
            duration_seconds=float(frame_count / int(rate)),
        )
    except InvalidAudioError:
        raise
    except Exception as exc:
        raise InvalidAudioError(f"Could not decode audio: {exc}") from exc


def inspect_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize a row without dumping arrays or encoded audio bytes."""
    audio_field = record.get("audio")
    audio_kind = None
    if isinstance(audio_field, Mapping):
        audio_kind = "decoded" if audio_field.get("array") is not None else "encoded"
    elif audio_field is not None:
        audio_kind = type(audio_field).__name__
    return {
        "fields": list(record.keys()),
        "candidate_id": candidate_record_id(record),
        "audio_kind": audio_kind,
        "duration_seconds": record.get("duration_seconds"),
        "sample_rate": record.get("sample_rate"),
        "has_quran": record.get("has_quran"),
        "phonemes_length": len(record["phonemes"]) if isinstance(record.get("phonemes"), str) else None,
        "sifat_count": len(record["sifat"]) if isinstance(record.get("sifat"), list) else None,
        "imlaey": record.get("imlaey"),
        "uthmani": record.get("uthmani"),
    }
