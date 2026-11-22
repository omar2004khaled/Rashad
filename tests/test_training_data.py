from copy import deepcopy
from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from quran_transcript import SifaOutput, alphabet, chunck_phonemes

from quran_annotations import validate_record
from quran_audio import AudioProcessingConfig
from quran_dataset import ConfiguredRecord
from quran_training_data import (
    CTC_BLANK_ID, IGNORE_INDEX, PAD_ID, DatasetStatistics, InclusionPolicy,
    LabelCodec, TrainingDataBuilder, UnknownLabelError, assert_no_group_leakage, build_reciter_split_manifest,
    assign_record_split, pad_label_batch, pad_sequences, prepare_record,
    expected_random_segment_group_leakage,
)


def make_row(*, has_quran=True, audio_rate=16000, duration=0.2):
    codec = LabelCodec()
    phonemes = list(asdict(alphabet.phonetics).values())[0]
    groups = chunck_phonemes(phonemes)
    sifat = []
    for group in groups:
        item = {"phonemes": group}
        for name, field in SifaOutput.model_fields.items():
            if name != "phonemes":
                item[name] = field.annotation.__args__[0]
        sifat.append(item)
    buffer = BytesIO()
    sf.write(buffer, np.zeros(int(duration * audio_rate), dtype=np.float32), audio_rate, format="WAV")
    return {
        "audio": {"bytes": buffer.getvalue()}, "sample_rate": audio_rate,
        "duration_seconds": duration, "phonemes": phonemes, "sifat": sifat,
        "imlaey": "test", "uthmani": "test", "has_quran": has_quran,
        "has_istiaatha": not has_quran, "has_bismillah": False, "has_sadaka": False,
        "start_span": {"sura_idx": 1, "aya_idx": 1, "imlaey": 0, "uthmani": 0} if has_quran else None,
        "end_span": {"sura_idx": 1, "aya_idx": 1, "imlaey": 1, "uthmani": 1} if has_quran else None,
        "match_ratio": 1.0, "timestamp_seconds": [0.0, duration],
        "moshaf_id": "0.1", "segment_index": "001.0000", "reciter_id": 0,
        "reciter_english_name": "Sample reciter",
    }


def test_phoneme_ids_round_trip_and_inventory_has_pad_plus_42():
    codec = LabelCodec()
    text = "بِسمِۦۦ"
    assert codec.decode_phonemes(codec.encode_phonemes(text)) == text
    assert codec.phoneme_vocab_size == 43
    assert PAD_ID == CTC_BLANK_ID == 0


def test_phoneme_space_removal_matches_original_phonetizer_contract():
    codec = LabelCodec()
    symbols = list(codec.phoneme_to_id)
    source_text = symbols[0] + " " + symbols[1]
    target_ids = codec.encode_phonemes(source_text, remove_spaces=True)
    assert codec.decode_phonemes(target_ids) == symbols[0] + symbols[1]
    with pytest.raises(UnknownLabelError, match="Unsupported phoneme symbol"):
        codec.encode_phonemes(source_text)


def test_unknown_and_special_tokens_are_not_target_labels():
    codec = LabelCodec()
    with pytest.raises(UnknownLabelError, match="Unsupported phoneme symbol"):
        codec.encode_phonemes("?")
    with pytest.raises(UnknownLabelError, match="Padding/blank"):
        codec.decode_phonemes([PAD_ID])
    with pytest.raises(UnknownLabelError, match="Unknown phoneme id"):
        codec.decode_phonemes([99])


def test_sifat_ids_round_trip_and_use_independent_levels():
    codec = LabelCodec()
    row = make_row()
    encoded = codec.encode_sifat(row["sifat"])
    for level, ids in encoded.items():
        decoded = codec.decode_sifat(level, ids)
        assert decoded == [entry[level] for entry in row["sifat"]]
    assert codec.level_vocab_sizes["shidda_or_rakhawa"] == 4
    assert codec.level_vocab_sizes["phonemes"] == 43
    assert codec.level_vocab_sizes == {
        "phonemes": 43, "hams_or_jahr": 3, "shidda_or_rakhawa": 4,
        "tafkheem_or_taqeeq": 4, "itbaq": 3, "safeer": 3,
        "qalqla": 3, "tikraar": 3, "tafashie": 3, "istitala": 3, "ghonna": 3,
    }


def test_sifat_alignment_is_one_label_per_reference_group():
    row = make_row()
    outcome = prepare_record(row, "moshaf_0.1", audio_config=AudioProcessingConfig(target_sample_rate=None))
    assert outcome.example is not None
    example = outcome.example
    assert len(example.phoneme_groups) == len(example.labels["ghonna"])
    assert len(example.group_phoneme_ids) == len(example.phoneme_groups)
    assert len(example.labels["phonemes"]) == len(row["phonemes"])
    assert validate_record(row).is_valid


def test_multiple_phoneme_groups_survive_space_removal_and_encoding():
    row = make_row()
    symbols = list(asdict(alphabet.phonetics).values())
    first_group = row["sifat"][0]
    second_group = first_group.copy()
    second_group["phonemes"] = symbols[1]
    row["phonemes"] = symbols[0] + " " + symbols[1]
    row["sifat"] = [first_group, second_group]
    outcome = prepare_record(row, "moshaf_0.1", audio_config=AudioProcessingConfig(target_sample_rate=None))
    assert outcome.example is not None
    example = outcome.example
    assert example.phoneme_text == symbols[0] + " " + symbols[1]
    assert example.phoneme_text_for_encoding == symbols[0] + symbols[1]
    assert len(example.labels["phonemes"]) == 2
    assert len(example.labels["ghonna"]) == 2
    assert len(example.phoneme_groups) == 2


def test_padding_uses_ignore_index_not_ctc_blank():
    padded = pad_sequences([[1, 2], [3]])
    assert padded.values.tolist() == [[1, 2], [3, IGNORE_INDEX]]
    assert padded.lengths.tolist() == [2, 1]
    assert padded.padding_value != CTC_BLANK_ID


def test_label_batch_pads_each_output_level_independently():
    examples = [{"labels": {"phonemes": [1, 2], "ghonna": [1]}},
                {"labels": {"phonemes": [3], "ghonna": [2]}}]
    targets, lengths = pad_label_batch(examples)
    assert targets["phonemes"].values.tolist() == [[1, 2], [3, IGNORE_INDEX]]
    assert targets["ghonna"].values.tolist() == [[1], [2]]
    assert lengths["phonemes"].tolist() == [2, 1]


def test_non_quran_policy_is_explicit_and_null_spans_are_allowed():
    row = make_row(has_quran=False)
    accepted = prepare_record(row, "moshaf_0.1", audio_config=AudioProcessingConfig(target_sample_rate=None))
    assert accepted.example is not None
    dropped = prepare_record(row, "moshaf_0.1", audio_config=AudioProcessingConfig(target_sample_rate=None),
                             policy=InclusionPolicy(include_non_quran=False))
    assert dropped.excluded is not None
    assert dropped.excluded.reasons == ("policy:non_quran_excluded",)


def test_invalid_labels_excluded_with_reason_and_source_is_not_mutated():
    row = make_row()
    before = deepcopy(row)
    row["phonemes"] += "?"
    mutated_input = deepcopy(row)
    outcome = prepare_record(row, "moshaf_0.1", audio_config=AudioProcessingConfig(target_sample_rate=None))
    assert outcome.excluded is not None
    assert any("invalid_annotation:unknown_phoneme_symbols" in code for code in outcome.excluded.reasons)
    assert row == mutated_input
    assert before["phonemes"] != row["phonemes"]


def test_missing_annotations_get_individual_reasons():
    row = make_row()
    row["phonemes"] = None
    row["sifat"] = None
    outcome = prepare_record(row, "moshaf_0.1", audio_config=AudioProcessingConfig(target_sample_rate=None))
    assert outcome.excluded is not None
    assert "missing_phonemes" in outcome.excluded.reasons
    assert "missing_sifat" in outcome.excluded.reasons


def test_builder_writes_a_reason_for_each_excluded_record(tmp_path):
    row = make_row()
    row["phonemes"] = None

    class FakeLoader:
        def iter_configurations(self, configurations, **kwargs):
            yield ConfiguredRecord("moshaf_0.1", row)

    builder = TrainingDataBuilder(FakeLoader(), audio_config=AudioProcessingConfig(target_sample_rate=None))
    ledger = tmp_path / "exclusions.jsonl"
    results = list(builder.iter_prepared(["moshaf_0.1"], exclusion_ledger_path=ledger))
    entry = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
    assert results[0].excluded is not None
    assert "missing_phonemes" in entry["reasons"]


def test_duration_and_match_ratio_cuts_are_configurable():
    row = make_row(duration=0.2)
    policy = InclusionPolicy(min_match_ratio=0.9, min_duration_seconds=0.3)
    outcome = prepare_record(row, "moshaf_0.1", audio_config=AudioProcessingConfig(target_sample_rate=None), policy=policy)
    assert outcome.excluded is not None
    assert set(outcome.excluded.reasons) == {"policy:too_short"}
    row["match_ratio"] = 0.2
    outcome = prepare_record(row, "moshaf_0.1", audio_config=AudioProcessingConfig(target_sample_rate=None), policy=policy)
    assert "policy:low_match_ratio" in outcome.excluded.reasons


def test_model_audio_rate_and_mono_requirements_are_explicit():
    stereo = make_row()
    buffer = BytesIO()
    sf.write(buffer, np.zeros((3200, 2), dtype=np.float32), 16000, format="WAV")
    stereo["audio"] = {"bytes": buffer.getvalue()}
    rejected = prepare_record(stereo, "moshaf_0.1", audio_config=AudioProcessingConfig(target_sample_rate=None))
    assert rejected.excluded is not None
    assert "audio:not_mono:2_channels" in rejected.excluded.reasons
    accepted = prepare_record(stereo, "moshaf_0.1",
                              audio_config=AudioProcessingConfig(target_sample_rate=16000, convert_to_mono=True))
    assert accepted.example is not None
    assert accepted.example.audio.channels == 1

    wrong_rate = make_row(audio_rate=8000)
    rejected_rate = prepare_record(wrong_rate, "moshaf_0.1",
                                   audio_config=AudioProcessingConfig(target_sample_rate=None))
    assert rejected_rate.excluded is not None
    assert "audio:wrong_sample_rate:8000" in rejected_rate.excluded.reasons


def test_group_splits_are_fixed_and_leakage_free():
    reciters = [{"id": i, "english_name": str(i), "moshaf_set_ids": [str(i)]} for i in range(8)]
    counts = {str(i): 10 + i for i in range(8)}
    one = build_reciter_split_manifest(reciters, counts, dataset_revision="test", seed=12,
                                       validation_groups=1, test_groups=1)
    two = build_reciter_split_manifest(reciters, counts, dataset_revision="test", seed=12,
                                       validation_groups=1, test_groups=1)
    assert one == two
    rows = []
    for row in reciters:
        split = one["reciter_id_to_split"][str(row["id"])]
        rows.append((str(row["id"]), {"reciter_id": row["id"]}, split))
    assert_no_group_leakage(rows)
    assert len(set(one["reciter_id_to_split"].values())) == 3
    with pytest.raises(ValueError, match="leakage"):
        assert_no_group_leakage([("a", {"reciter_id": 1}, "train"),
                                 ("b", {"reciter_id": 1}, "test")])


def test_random_segment_split_is_expected_to_leak_large_groups():
    report = expected_random_segment_group_leakage({str(i): 8000 for i in range(22)})
    assert report["expected_groups_present_in_each_split"] == {
        "train": 22.0, "validation": 22.0, "test": 22.0,
    }
    assert report["expected_groups_present_in_multiple_splits"] == 22.0


def test_manifest_split_traces_from_both_reciter_and_moshaf():
    manifest = {"reciter_id_to_split": {"4": "validation"},
                "moshaf_id_to_split": {"8.0": "validation"}}
    assert assign_record_split({"reciter_id": 4, "moshaf_id": "8.0"}, manifest) == "validation"


def test_prepared_example_has_traceable_key_and_reproducible_audio_settings():
    row = make_row()
    original = row["audio"]["bytes"]
    outcome = prepare_record(row, "moshaf_0.1", audio_config=AudioProcessingConfig(target_sample_rate=16000))
    assert outcome.example is not None
    assert outcome.example.source_key == "moshaf_0.1:0.1:001.0000"
    assert outcome.example.metadata["reciter_id"] == 0
    assert outcome.example.audio.sampling_rate == 16000
    assert outcome.example.audio.operations[0].name == "to_float32"
    assert row["audio"]["bytes"] == original


def test_dataset_statistics_tracks_exclusions_and_included_classes():
    stats = DatasetStatistics()
    row = make_row()
    outcome = prepare_record(row, "moshaf_0.1", audio_config=AudioProcessingConfig(target_sample_rate=None))
    stats.observe(row, outcome)
    row2 = make_row()
    row2["phonemes"] = "?"
    excluded = prepare_record(row2, "moshaf_0.1", audio_config=AudioProcessingConfig(target_sample_rate=None))
    stats.observe(row2, excluded)
    summary = stats.summary()
    assert summary["records"] == {"total_seen": 2, "included": 1, "excluded": 1,
                                  "exclusion_reasons": {"invalid_annotation:unknown_phoneme_symbols": 1,
                                                        "invalid_annotation:phonemes_not_chunkable": 1,
                                                        "invalid_annotation:sifat_alignment_mismatch": 1}}
    assert " " not in summary["phoneme_frequency"]


def test_metadata_split_manifest_covers_all_published_reciters():
    path = Path("configs/reciter_split_manifest.json")
    if not path.exists():
        pytest.skip("Live split manifest is generated by the bounded metadata script")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["source_row_count_total"] == 286537
    assert len(manifest["reciter_id_to_split"]) == 22
    assert len(manifest["moshaf_id_to_split"]) == 27
    assert set(manifest["actual_source_row_counts"]) == {"train", "validation", "test"}
