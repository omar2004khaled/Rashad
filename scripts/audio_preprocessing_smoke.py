"""Measure a small, streamed real-data sample and verify safe transforms."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quran_dataset import DatasetConfig, DatasetLoader, decode_audio  # noqa: E402
from quran_audio import (  # noqa: E402
    AudioProcessingConfig,
    ProcessedAudio,
    analyze_audio,
    detect_silence,
    normalize_peak,
    preprocess_audio,
)


def tail_rms_dbfs(samples: np.ndarray, sample_rate: int, seconds: float = 0.2) -> float:
    frames = max(1, int(seconds * sample_rate))
    tail = samples[-frames:].astype(np.float64)
    rms = float(np.sqrt(np.mean(tail**2)))
    return float("-inf") if rms == 0 else float(20.0 * np.log10(rms))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    loader_config = DatasetConfig.from_file(ROOT / "configs" / "dataset.json")
    processing_config = AudioProcessingConfig.from_file(ROOT / "configs" / "audio_preprocessing.json")
    loader = DatasetLoader(loader_config)
    measured = []
    ending_sample = None
    source_checks = []

    for configuration in ["moshaf_0.1", "moshaf_1.0", "moshaf_26.1"]:
        rows = list(loader.iter_samples(configuration, limit=5))
        for index, record in enumerate(rows):
            decoded = decode_audio(record)
            source_copy = decoded.samples.copy()
            signal = ProcessedAudio.from_audio_data(decoded)
            properties = analyze_audio(
                signal,
                silence_threshold_dbfs=processing_config.silence_threshold_dbfs,
                clipping_threshold=processing_config.clipping_threshold,
            )
            measured.append({
                "configuration": configuration,
                "segment_index": record.get("segment_index"),
                "has_quran": record.get("has_quran"),
                "ending_text": record.get("uthmani", "").split()[-1] if record.get("uthmani") else None,
                **properties.as_dict(),
                "tail_200ms_rms_dbfs": tail_rms_dbfs(signal.samples, signal.sampling_rate),
            })

            default_output = preprocess_audio(signal, processing_config)
            source_checks.append({
                "configuration": configuration,
                "segment_index": record.get("segment_index"),
                "default_pipeline_exact_sample_match": bool(np.array_equal(default_output.samples, source_copy)),
                "source_array_unchanged": bool(np.array_equal(decoded.samples, source_copy)),
                "operations": [
                    {"name": op.name, "applied": op.applied, "reason": op.reason}
                    for op in default_output.operations
                ],
            })
            if configuration == "moshaf_1.0" and index == 4:
                ending_sample = (record, signal, default_output, source_copy)

    if ending_sample is None:
        raise RuntimeError("Quiet ending example moshaf_1.0 index 4 was not loaded")
    record, quiet_signal, default_output, original_copy = ending_sample
    normalized = normalize_peak(
        quiet_signal,
        target_dbfs=processing_config.normalize_target_dbfs,
        max_gain_db=processing_config.normalize_max_gain_db,
        minimum_peak_dbfs=processing_config.normalize_minimum_peak_dbfs,
    )
    before_props = analyze_audio(quiet_signal, clipping_threshold=processing_config.clipping_threshold)
    after_props = analyze_audio(normalized, clipping_threshold=processing_config.clipping_threshold)
    before_tail = quiet_signal.samples[-int(0.2 * quiet_signal.sampling_rate):, :]
    after_tail = normalized.samples[-int(0.2 * normalized.sampling_rate):, :]
    quiet_ending = {
        "configuration": "moshaf_1.0",
        "segment_index": record.get("segment_index"),
        "final_text_word": record.get("uthmani", "").split()[-1],
        "tail_window_seconds": 0.2,
        "tail_rms_before_dbfs": tail_rms_dbfs(quiet_signal.samples, quiet_signal.sampling_rate),
        "tail_rms_after_dbfs": tail_rms_dbfs(normalized.samples, normalized.sampling_rate),
        "tail_has_nonzero_samples_before": bool(np.any(before_tail != 0)),
        "tail_has_nonzero_samples_after": bool(np.any(after_tail != 0)),
        "last_sample_nonzero_before": bool(np.any(quiet_signal.samples[-1] != 0)),
        "last_sample_nonzero_after": bool(np.any(normalized.samples[-1] != 0)),
        "duration_before_seconds": before_props.duration_seconds,
        "duration_after_seconds": after_props.duration_seconds,
        "clipped_samples_before": before_props.clipped_sample_count,
        "clipped_samples_after": after_props.clipped_sample_count,
        "peak_after": after_props.peak_amplitude,
        "normalization_applied": normalized.operations[-1].applied,
        "original_unchanged": bool(np.array_equal(quiet_signal.original.samples, original_copy)),
        "trailing_silence_at_minus_60_dbfs_seconds": detect_silence(quiet_signal, threshold_dbfs=-60).trailing_seconds,
    }
    if not all([
        quiet_ending["tail_has_nonzero_samples_before"],
        quiet_ending["tail_has_nonzero_samples_after"],
        quiet_ending["last_sample_nonzero_before"],
        quiet_ending["last_sample_nonzero_after"],
        quiet_ending["clipped_samples_after"] == 0,
        quiet_ending["duration_before_seconds"] == quiet_ending["duration_after_seconds"],
        quiet_ending["original_unchanged"],
    ]):
        raise RuntimeError(f"Quiet ending preservation check failed: {quiet_ending}")
    if not all(item["default_pipeline_exact_sample_match"] and item["source_array_unchanged"] for item in source_checks):
        raise RuntimeError("The default pipeline changed at least one source sample")

    print(json.dumps({
        "configurations": ["moshaf_0.1", "moshaf_1.0", "moshaf_26.1"],
        "records_measured": len(measured),
        "default_pipeline_checks": source_checks,
        "measurements": measured,
        "quiet_ending_normalization_check": quiet_ending,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
