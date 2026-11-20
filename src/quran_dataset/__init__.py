"""Read-only access helpers for the Quran Muaalem annotated dataset."""

from .loader import (
    AudioData,
    ConfiguredRecord,
    DatasetAccessError,
    DatasetConfig,
    DatasetLoader,
    EnrichedRecord,
    InvalidAudioError,
    MetadataIndex,
    SchemaInspection,
    candidate_record_id,
    decode_audio,
    inspect_record,
)

__all__ = [
    "AudioData",
    "ConfiguredRecord",
    "DatasetAccessError",
    "DatasetConfig",
    "DatasetLoader",
    "EnrichedRecord",
    "InvalidAudioError",
    "MetadataIndex",
    "SchemaInspection",
    "candidate_record_id",
    "decode_audio",
    "inspect_record",
]
