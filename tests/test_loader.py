from __future__ import annotations

import io
import json
import random
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from quran_dataset import (
    DatasetAccessError,
    DatasetConfig,
    DatasetLoader,
    InvalidAudioError,
    MetadataIndex,
    candidate_record_id,
    decode_audio,
    inspect_record,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "dataset.json"


class FakeDataset:
    def __init__(self, rows, features=None):
        self.rows = rows
        self.features = features or {}
        self.cast_calls = []

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def cast_column(self, name, feature):
        self.cast_calls.append((name, feature))
        return self

    def shuffle(self, seed, buffer_size):
        rows = list(self.rows)
        random.Random(seed).shuffle(rows)
        return FakeDataset(rows, self.features)


def make_config(**updates):
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw.update(updates)
    return DatasetConfig(
        dataset_id=raw["dataset_id"],
        available_configurations=tuple(raw["available_configurations"]),
        recitation_configurations=tuple(raw["recitation_configurations"]),
        metadata_configurations=raw["metadata_configurations"],
        default_split=raw.get("default_split", "train"),
        selected_configurations=tuple(raw.get("selected_configurations", ())),
        sample_limit=raw.get("sample_limit", 3),
        streaming=raw.get("streaming", True),
        decode_audio=raw.get("decode_audio", False),
        shuffle_seed=raw.get("shuffle_seed", 42),
        schema_unavailable_via_stage1_first_rows=tuple(
            raw.get("schema_unavailable_via_stage1_first_rows", ())
        ),
    )


def test_config_file_and_configuration_validation():
    config = DatasetConfig.from_file(CONFIG_PATH)
    assert config.dataset_id == "obadx/muaalem-annotated-v3"
    assert config.default_split == "train"
    assert config.streaming is True
    assert len(config.recitation_configurations) == 27

    loader = DatasetLoader(config, loader_fn=lambda *args, **kwargs: FakeDataset([]))
    with pytest.raises(ValueError, match="Unknown configuration"):
        loader.load_config("moshaf_does_not_exist")


def test_load_config_forwards_split_streaming_and_keeps_audio_encoded():
    calls = []
    dataset = FakeDataset([{"id": 1}], features={"audio": object()})

    def fake_load(dataset_id, **kwargs):
        calls.append((dataset_id, kwargs))
        return dataset

    loader = DatasetLoader(make_config(), loader_fn=fake_load)
    loaded = loader.load_config("moshaf_0.1", split="train")
    assert loaded is dataset
    assert calls[0][0] == "obadx/muaalem-annotated-v3"
    assert calls[0][1]["name"] == "moshaf_0.1"
    assert calls[0][1]["split"] == "train"
    assert calls[0][1]["streaming"] is True
    assert dataset.cast_calls[0][0] == "audio"


def test_limited_iteration_and_selected_configurations_preserve_rows():
    rows = [{"segment_index": str(i), "phonemes": "unchanged"} for i in range(5)]
    calls = []

    def fake_load(dataset_id, **kwargs):
        calls.append(kwargs["name"])
        return FakeDataset(rows)

    loader = DatasetLoader(make_config(selected_configurations=("moshaf_0.1",)), loader_fn=fake_load)
    assert list(loader.iter_samples("moshaf_0.1", limit=2)) == rows[:2]
    assert [x.record for x in loader.iter_configurations(["moshaf_0.1", "moshaf_1.0"], limit_per_configuration=1)] == [rows[0], rows[0]]
    assert calls == ["moshaf_0.1", "moshaf_0.1", "moshaf_1.0"]
    assert list(loader.iter_samples("moshaf_0.1", limit=0)) == []
    with pytest.raises(ValueError, match="non-negative"):
        list(loader.iter_samples("moshaf_0.1", limit=-1))


def test_stream_shuffle_uses_reproducible_seed():
    rows = [{"index": i} for i in range(10)]
    loader = DatasetLoader(make_config(), loader_fn=lambda *a, **k: FakeDataset(rows))
    first = list(loader.iter_samples("moshaf_0.1", limit=5, shuffle=True, seed=42))
    second = list(loader.iter_samples("moshaf_0.1", limit=5, shuffle=True, seed=42))
    assert first == second


def test_get_sample_streaming_and_non_streaming():
    rows = [{"index": i} for i in range(3)]
    loader = DatasetLoader(make_config(), loader_fn=lambda *a, **k: FakeDataset(rows))
    assert loader.get_sample("moshaf_0.1", 1)["index"] == 1
    assert loader.get_sample("moshaf_0.1", 1, streaming=False)["index"] == 1
    with pytest.raises(IndexError):
        loader.get_sample("moshaf_0.1", 4)


def test_metadata_lookup_and_join_use_ids_without_mutating_record():
    moshaf = {"id": "0.1", "name": "test moshaf"}
    reciter = {"id": 0, "english_name": "reciter"}
    index = MetadataIndex([moshaf], [reciter])
    record = {"moshaf_id": "0.1", "reciter_id": 0, "segment_index": "001.0000"}
    joined = index.join(record)
    assert joined.moshaf_metadata is moshaf
    assert joined.reciter_metadata is reciter
    assert joined.record is record
    assert set(record) == {"moshaf_id", "reciter_id", "segment_index"}
    assert index.lookup_moshaf("missing") is None
    assert index.lookup_reciter(None) is None


def test_metadata_loading_selects_metadata_configurations():
    rows = {
        "moshaf_metadata": [{"id": "0.1"}],
        "reciters_metadata": [{"id": 0}],
    }
    requested = []

    def fake_load(dataset_id, **kwargs):
        requested.append(kwargs["name"])
        return FakeDataset(rows[kwargs["name"]])

    loader = DatasetLoader(make_config(), loader_fn=fake_load)
    assert loader.load_metadata("moshaf") == rows["moshaf_metadata"]
    assert loader.load_metadata("reciters") == rows["reciters_metadata"]
    assert requested == ["moshaf_metadata", "reciters_metadata"]
    with pytest.raises(ValueError, match="Unknown metadata kind"):
        loader.load_metadata("unknown")


@pytest.mark.parametrize(
    "record,expected",
    [
        ({"moshaf_id": "0.1", "segment_index": "001.0000"}, ("0.1", "001.0000")),
        ({"moshaf_id": "", "segment_index": "001"}, None),
        ({"moshaf_id": "0.1"}, None),
        ({"segment_index": "001"}, None),
    ],
)
def test_candidate_compound_id_checks_missing_and_empty_fields(record, expected):
    assert candidate_record_id(record) == expected


def test_audio_decode_returns_original_rate_channels_duration_and_samples():
    source_samples = np.array([[0.0], [0.25], [-0.5], [0.75]], dtype=np.float32)
    encoded = io.BytesIO()
    sf.write(encoded, source_samples, 16000, format="WAV", subtype="FLOAT")
    raw = encoded.getvalue()
    expected, expected_rate = sf.read(io.BytesIO(raw), always_2d=True, dtype="float32")

    decoded = decode_audio({"audio": {"bytes": raw, "path": None}, "sample_rate": 16000})
    np.testing.assert_array_equal(decoded.samples, expected)
    assert decoded.sampling_rate == expected_rate == 16000
    assert decoded.channels == 1
    assert decoded.duration_seconds == len(source_samples) / 16000
    np.testing.assert_array_equal(source_samples, np.array([[0.0], [0.25], [-0.5], [0.75]], dtype=np.float32))


def test_audio_decode_accepts_already_decoded_samples_without_transforming():
    samples = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    decoded = decode_audio({"audio": {"array": samples, "sampling_rate": 8000}})
    assert decoded.samples is samples
    assert decoded.sampling_rate == 8000
    assert decoded.channels == 1
    assert decoded.duration_seconds == 3 / 8000


@pytest.mark.parametrize(
    "record",
    [
        {},
        {"audio": None},
        {"audio": {"bytes": b"not a wav", "path": None}},
        {"audio": {"array": np.array([], dtype=np.float32), "sampling_rate": 16000}},
        {"audio": {"array": np.zeros((2, 2, 2)), "sampling_rate": 16000}},
        {"audio": {"array": np.ones(3), "sampling_rate": 0}},
    ],
)
def test_invalid_audio_records_raise_clear_error(record):
    with pytest.raises(InvalidAudioError):
        decode_audio(record)


def test_inspection_summary_avoids_audio_arrays():
    summary = inspect_record({"audio": {"bytes": b"huge"}, "phonemes": "abc", "sifat": [1, 2]})
    assert summary["audio_kind"] == "encoded"
    assert summary["phonemes_length"] == 3
    assert summary["sifat_count"] == 2
    assert "samples" not in summary


def test_schema_errors_are_reported_without_schema_substitution():
    requested = []

    def failing_load(dataset_id, **kwargs):
        requested.append(kwargs["name"])
        raise RuntimeError("fixture schema failure")

    loader = DatasetLoader(make_config(), loader_fn=failing_load)
    result = loader.inspect_schema("moshaf_5.0")
    assert result.verified is False
    assert result.features is None
    assert "fixture schema failure" in result.error
    assert requested == ["moshaf_5.0"]
    with pytest.raises(DatasetAccessError, match="moshaf_5.0"):
        loader.load_config("moshaf_5.0")

