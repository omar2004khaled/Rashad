from copy import deepcopy
from dataclasses import asdict

from quran_transcript import SifaOutput, alphabet, chunck_phonemes

from quran_annotations import validate_record


def valid_record():
    phonemes = next(iter(asdict(alphabet.phonetics).values()))
    group = chunck_phonemes(phonemes)[0]
    sifat = {"phonemes": group}
    for name, field in SifaOutput.model_fields.items():
        if name != "phonemes":
            sifat[name] = field.annotation.__args__[0]
    return {
        "phonemes": phonemes,
        "sifat": [sifat],
        "imlaey": "test",
        "uthmani": "test",
        "has_quran": True,
        "has_istiaatha": False,
        "has_bismillah": False,
        "has_sadaka": False,
        "start_span": {"sura_idx": 1, "aya_idx": 1, "imlaey": 0, "uthmani": 0},
        "end_span": {"sura_idx": 1, "aya_idx": 1, "imlaey": 1, "uthmani": 1},
        "match_ratio": 1.0,
        "timestamp_seconds": [0.0, 1.0],
        "duration_seconds": 1.0,
    }


def codes(result):
    return {issue.code for issue in result.errors + result.warnings}


def test_accepts_reference_compatible_annotations():
    assert validate_record(valid_record()).is_valid


def test_non_quran_null_spans_are_valid():
    row = valid_record()
    row.update(has_quran=False, has_istiaatha=True, start_span=None, end_span=None)
    assert validate_record(row).is_valid


def test_unknown_phoneme_symbol_is_reported():
    row = valid_record()
    row["phonemes"] = row["phonemes"] + "?"
    result = validate_record(row)
    assert "unknown_phoneme_symbols" in codes(result)
    assert row["phonemes"].endswith("?")


def test_validation_does_not_mutate_source_record():
    row = valid_record()
    before = deepcopy(row)
    validate_record(row)
    assert row == before


def test_sifat_alignment_mismatch_is_error():
    row = valid_record()
    row["sifat"] = row["sifat"] + [row["sifat"][0].copy()]
    assert "sifat_alignment_mismatch" in codes(validate_record(row))


def test_invalid_sifat_category_is_reported():
    row = valid_record()
    row["sifat"][0]["ghonna"] = "unknown"
    assert "sifat_value_invalid" in codes(validate_record(row))


def test_bad_span_and_timestamps_are_reported():
    row = valid_record()
    row["start_span"]["sura_idx"] = -1
    row["timestamp_seconds"] = [3.0, 2.0]
    result = validate_record(row)
    assert "span_value_invalid" in codes(result)
    assert "timestamp_order_invalid" in codes(result)


def test_non_quran_populated_spans_are_warning():
    row = valid_record()
    row["has_quran"] = False
    assert "non_quran_span_present" in codes(validate_record(row))


def test_empty_text_and_nonchunkable_phonemes_are_errors():
    row = valid_record()
    row.update(imlaey="", phonemes="?")
    result = validate_record(row)
    assert "text_missing_or_empty" in codes(result)
    assert "unknown_phoneme_symbols" in codes(result)
    assert "phonemes_not_chunkable" in codes(result)
