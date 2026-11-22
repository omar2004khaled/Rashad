"""Validate annotation shapes without changing source records.

The allowed phoneme symbols, sifat properties, and phoneme grouping are taken
from the installed ``quran-transcript`` reference package used by the original
project. Validation reports problems; it never repairs a record.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Mapping

from quran_transcript import SifaOutput, alphabet, chunck_phonemes


@dataclass(frozen=True)
class AnnotationIssue:
    severity: str
    code: str
    field: str
    message: str


@dataclass(frozen=True)
class AnnotationValidation:
    errors: tuple[AnnotationIssue, ...]
    warnings: tuple[AnnotationIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": [asdict(issue) for issue in self.errors],
            "warnings": [asdict(issue) for issue in self.warnings],
        }


def _issue(target: list[AnnotationIssue], severity: str, code: str, field: str, message: str) -> None:
    target.append(AnnotationIssue(severity, code, field, message))


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(float(value))


def validate_record(record: Mapping[str, Any]) -> AnnotationValidation:
    """Check record annotation format against the original reference contract."""
    errors: list[AnnotationIssue] = []
    warnings: list[AnnotationIssue] = []

    symbols = set(asdict(alphabet.phonetics).values())
    phonemes = record.get("phonemes")
    groups: list[str] | None = None
    if not isinstance(phonemes, str) or not phonemes.strip():
        _issue(errors, "error", "phonemes_missing_or_empty", "phonemes", "Expected a non-empty phoneme string.")
    else:
        unknown = sorted(set(phonemes) - symbols - {" "})
        if unknown:
            _issue(errors, "error", "unknown_phoneme_symbols", "phonemes", f"Symbols outside quran-transcript alphabet: {unknown!r}.")
        try:
            groups = chunck_phonemes(phonemes)
        except Exception as exc:
            _issue(errors, "error", "phoneme_chunking_failed", "phonemes", f"Reference chunker failed: {type(exc).__name__}: {exc}")
        else:
            if not groups:
                _issue(errors, "error", "phonemes_not_chunkable", "phonemes", "Reference chunker returned no phoneme groups.")

    sifat = record.get("sifat")
    expected_fields = set(SifaOutput.model_fields)
    if not isinstance(sifat, list) or not sifat:
        _issue(errors, "error", "sifat_missing_or_empty", "sifat", "Expected a non-empty list of SifaOutput entries.")
    else:
        for i, entry in enumerate(sifat):
            path = f"sifat[{i}]"
            if not isinstance(entry, Mapping):
                _issue(errors, "error", "sifat_entry_not_mapping", path, "Expected an object with SifaOutput fields.")
                continue
            missing = expected_fields - set(entry)
            extra = set(entry) - expected_fields
            if missing:
                _issue(errors, "error", "sifat_fields_missing", path, f"Missing reference fields: {sorted(missing)}.")
            if extra:
                _issue(warnings, "warning", "sifat_fields_extra", path, f"Fields not defined by reference SifaOutput: {sorted(extra)}.")
            for name, field in SifaOutput.model_fields.items():
                if name not in entry:
                    continue
                value = entry[name]
                if name == "phonemes":
                    if not isinstance(value, str) or not value:
                        _issue(errors, "error", "sifat_phonemes_invalid", f"{path}.phonemes", "Expected a non-empty string.")
                else:
                    allowed = getattr(field.annotation, "__args__", ())
                    if value not in allowed:
                        _issue(errors, "error", "sifat_value_invalid", f"{path}.{name}", f"Expected one of {allowed!r}; got {value!r}.")
        if groups is not None and all(isinstance(x, Mapping) and isinstance(x.get("phonemes"), str) for x in sifat):
            observed = [entry["phonemes"] for entry in sifat]
            if observed != groups:
                _issue(errors, "error", "sifat_alignment_mismatch", "sifat", f"Reference chunk count={len(groups)}; sifat count={len(observed)}; group sequences differ.")

    for field_name in ("imlaey", "uthmani"):
        value = record.get(field_name)
        if not isinstance(value, str) or not value.strip():
            _issue(errors, "error", "text_missing_or_empty", field_name, "Expected a non-empty text string.")

    flag_names = ("has_quran", "has_istiaatha", "has_bismillah", "has_sadaka")
    for name in flag_names:
        if name not in record or not isinstance(record.get(name), bool):
            _issue(warnings, "warning", "flag_missing_or_invalid", name, "Expected a boolean flag; null or missing values are not resolved by the reference contract.")

    start, end = record.get("start_span"), record.get("end_span")
    span_keys = {"sura_idx", "aya_idx", "imlaey", "uthmani"}
    if (start is None) != (end is None):
        _issue(warnings, "warning", "partial_span", "start_span/end_span", "Only one endpoint is null.")
    for name, span in (("start_span", start), ("end_span", end)):
        if span is None:
            continue
        if not isinstance(span, Mapping):
            _issue(errors, "error", "span_not_mapping", name, "Expected null or a span object.")
            continue
        if set(span) != span_keys:
            _issue(errors, "error", "span_keys_invalid", name, f"Expected exactly {sorted(span_keys)}.")
        for key in span_keys & set(span):
            value = span[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                _issue(errors, "error", "span_value_invalid", f"{name}.{key}", "Expected a non-negative integer.")
    if record.get("has_quran") is True and (start is None or end is None):
        _issue(warnings, "warning", "quran_span_missing", "start_span/end_span", "Quran-flagged row has a null endpoint; samples do not establish whether this is always invalid.")
    if record.get("has_quran") is False and (start is not None or end is not None):
        _issue(warnings, "warning", "non_quran_span_present", "start_span/end_span", "Non-Quran-flagged row has span data; semantic consistency is unresolved.")

    ratio = record.get("match_ratio")
    if ratio is not None and (not _finite_number(ratio) or not 0 <= float(ratio) <= 1):
        _issue(errors, "error", "match_ratio_invalid", "match_ratio", "Expected null or a finite number in [0, 1].")
    duration = record.get("duration_seconds")
    if duration is not None and (not _finite_number(duration) or float(duration) <= 0):
        _issue(errors, "error", "duration_invalid", "duration_seconds", "Expected null or a positive finite number.")
    timestamp = record.get("timestamp_seconds")
    if timestamp is not None:
        if not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2 or not all(_finite_number(x) for x in timestamp):
            _issue(errors, "error", "timestamp_shape_invalid", "timestamp_seconds", "Expected null or two finite numeric values.")
        elif timestamp[0] > timestamp[1]:
            _issue(errors, "error", "timestamp_order_invalid", "timestamp_seconds", "The first value exceeds the second.")
    return AnnotationValidation(tuple(errors), tuple(warnings))
