"""Bounded full-fine-tuning pilot over a stratified, reciter-held-out sample."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import random
import re
import time
import unicodedata

import requests
import torch
from transformers import AutoFeatureExtractor

from quran_audio import AudioProcessingConfig
from quran_dataset import DatasetConfig, DatasetLoader
from quran_model import QuranMultiLevelCTC
from quran_training import QuranTrainer, TrainingConfig, TrainingRunConfig
from quran_training_data import (
    InclusionPolicy,
    MAX_DECODED_DURATION_SECONDS,
    QuranTrainingCollator,
    assign_record_split,
    decoded_duration_rejection,
    duration_group,
    duration_rejection_reason,
    prepare_record,
    select_preflight_examples,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "obadx/muaalem-annotated-v3"
MODEL_ID = "obadx/muaalem-model-v3_2"
MODEL_REVISION = "01a1ef9fbe40d144ef845101e89ff924aed3fef5"
ROWS_API = "https://datasets-server.huggingface.co/rows"
TRAIN_CONFIGS = ("moshaf_0.1", "moshaf_2.0", "moshaf_6.0", "moshaf_26.1")
VALIDATION_CONFIGS = ("moshaf_12.0", "moshaf_13.0", "moshaf_24.0")
TRAIN_OFFSETS = tuple(range(0, 3000, 300))
VALIDATION_OFFSETS = {
    "moshaf_12.0": (3500, 4000, 4500),
    "moshaf_13.0": (3500, 4000, 4500),
    "moshaf_24.0": (6000, 6500, 7000),
}
ROWS_PER_WINDOW = 20
MAX_TRAIN_POOL = 128
MAX_PILOT_UPDATES = 6
MAX_PILOT_TRAIN_EXAMPLES = 96
PILOT_EPOCHS = 2
SEED = 42
PREFLIGHT_ROWS_PER_CONFIGURATION = 80
PREFLIGHT_TRAINING_COUNT = 32
PREFLIGHT_VALIDATION_MIN = 1
PREFLIGHT_VALIDATION_MAX = 3


def _duration_bucket(duration: float) -> str | None:
    if duration_rejection_reason(duration):
        return None
    return "short" if duration <= 5 else "medium"


def _span_info(row: dict) -> tuple[set[int], set[tuple[int, int]]]:
    start, end = row.get("start_span"), row.get("end_span")
    if not isinstance(start, dict) or not isinstance(end, dict):
        return set(), set()
    first_sura, last_sura = start.get("sura_idx"), end.get("sura_idx")
    first_aya, last_aya = start.get("aya_idx"), end.get("aya_idx")
    if not all(isinstance(value, int) for value in (first_sura, last_sura, first_aya, last_aya)):
        return set(), set()
    suras = set(range(min(first_sura, last_sura), max(first_sura, last_sura) + 1))
    ayas = set()
    if first_sura == last_sura:
        ayas.update((first_sura, aya) for aya in range(min(first_aya, last_aya), max(first_aya, last_aya) + 1))
    else:
        # Cross-surah rows are uncommon in the sampled schema; retain endpoints conservatively.
        ayas.update({(first_sura, first_aya), (last_sura, last_aya)})
    return suras, ayas


def _normalized_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    return "".join(normalized.split())


def _fetch_page(configuration: str, offset: int) -> tuple[str, int, list[dict]]:
    params = {
        "dataset": DATASET_ID,
        "config": configuration,
        "split": "train",
        "offset": offset,
        "length": ROWS_PER_WINDOW,
    }
    session = requests.Session()
    last_error = None
    for attempt in range(5):
        try:
            response = session.get(ROWS_API, params=params, timeout=45)
            if response.status_code in {429, 502, 503, 504}:
                last_error = RuntimeError(f"Rows API HTTP {response.status_code} for {configuration}@{offset}")
                time.sleep(min(2 ** attempt, 12))
                continue
            response.raise_for_status()
            return configuration, offset, response.json()["rows"]
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < 4:
                time.sleep(min(2 ** attempt, 12))
    raise RuntimeError(f"Could not fetch bounded candidate window {configuration}@{offset}: {last_error}")


def _bounded_candidates(manifest: dict) -> tuple[dict[str, list[dict]], dict[str, list[dict]], dict]:
    current = requests.get(f"https://huggingface.co/api/datasets/{DATASET_ID}", timeout=30)
    current.raise_for_status()
    current_revision = current.json().get("sha")
    if current_revision != manifest["dataset_revision"]:
        raise RuntimeError(
            f"Dataset revision {current_revision} does not match split manifest "
            f"{manifest['dataset_revision']}"
        )
    jobs = [(name, offset) for name in TRAIN_CONFIGS for offset in TRAIN_OFFSETS]
    jobs.extend((name, offset) for name in VALIDATION_CONFIGS for offset in VALIDATION_OFFSETS[name])
    pages = []
    # Low concurrency and bounded row windows to avoid broad dataset scans/rate spikes.
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(lambda job: _fetch_page(*job), jobs):
            pages.append(result)
    train: dict[str, list[dict]] = {name: [] for name in TRAIN_CONFIGS}
    validation: dict[str, list[dict]] = {name: [] for name in VALIDATION_CONFIGS}
    for configuration, offset, rows in pages:
        target = train if configuration in train else validation
        for envelope in rows:
            row = envelope["row"]
            row["_pilot_row_index"] = int(envelope["row_idx"])
            row["_pilot_duration_seconds"] = float(row.get("duration_seconds") or 0.0)
            target[configuration].append(row)
    scan_summary = {
        "dataset_revision": current_revision,
        "row_windows_per_configuration": ROWS_PER_WINDOW,
        "training_candidate_rows_by_configuration": {k: len(v) for k, v in train.items()},
        "validation_candidate_rows_by_configuration": {k: len(v) for k, v in validation.items()},
        "source_dataset_split": "train (the published dataset exposes train only; validation is reciter-manifest-derived)",
        "metadata_only_rows_examined": sum(map(len, train.values())) + sum(map(len, validation.values())),
        "audio_downloads_before_selection": 0,
    }
    return train, validation, scan_summary


def _candidate_fields(row: dict, configuration: str, manifest: dict, expected_split: str) -> dict | None:
    try:
        from quran_training_data import assign_record_split
        split = assign_record_split(row, manifest)
    except (KeyError, ValueError):
        return None
    if split != expected_split or row.get("has_quran") is not True:
        return None
    if duration_rejection_reason(row.get("_pilot_duration_seconds")):
        return None
    suras, ayas = _span_info(row)
    if not suras or not ayas:
        return None
    duration = row["_pilot_duration_seconds"]
    bucket = _duration_bucket(duration)
    if bucket is None:
        return None
    moshaf_id = str(row.get("moshaf_id") or configuration.removeprefix("moshaf_"))
    segment = str(row.get("segment_index") or "")
    if not segment:
        return None
    return {
        "configuration": configuration,
        "source_key": f"{configuration}:{moshaf_id}:{segment}",
        "split": split,
        "duration_seconds": duration,
        "bucket": bucket,
        "suras": suras,
        "ayas": ayas,
        "normalized_text": _normalized_text(row.get("imlaey") or ""),
        "row": row,
    }


def _pick_spread(candidates: list[dict], count: int, seed: int) -> list[dict]:
    ordered = sorted(candidates, key=lambda item: (item["duration_seconds"], item["row"]["_pilot_row_index"]))
    if len(ordered) < count:
        raise RuntimeError(f"Need {count} candidates, found {len(ordered)}")
    if count == len(ordered):
        return ordered
    # Deterministic evenly spaced duration quantiles produce a useful range in each bin.
    offset = random.Random(seed).random()
    chosen = []
    used = set()
    for i in range(count):
        index = min(len(ordered) - 1, int((i + offset) * len(ordered) / count))
        while index in used:
            index = (index + 1) % len(ordered)
        used.add(index)
        chosen.append(ordered[index])
    return chosen


def _select_examples(train_rows: dict, validation_rows: dict, manifest: dict) -> tuple[list[dict], list[dict], dict]:
    grouped = {}
    for configuration in TRAIN_CONFIGS:
        valid = [item for row in train_rows[configuration]
                 if (item := _candidate_fields(row, configuration, manifest, "train")) is not None]
        for bucket in ("short", "medium"):
            group = [item for item in valid if item["bucket"] == bucket]
            if group:
                grouped[(configuration, bucket)] = group

    if len(grouped) < 3:
        raise RuntimeError(f"Need at least three nonempty training configuration/duration groups; found {len(grouped)}")
    target = min(MAX_TRAIN_POOL, sum(map(len, grouped.values())))
    if target < MAX_PILOT_TRAIN_EXAMPLES:
        raise RuntimeError(
            f"Only {target} safe-duration training examples are available; "
            f"the configured pilot consumes {MAX_PILOT_TRAIN_EXAMPLES}"
        )

    # Allocate round-robin across available configuration/duration groups. This
    # keeps all represented reciters/configurations and duration classes in the pool.
    group_keys = sorted(grouped)
    allocations = {key: 0 for key in group_keys}
    for key in group_keys:
        allocations[key] = 1
    remaining = target - len(group_keys)
    while remaining:
        progressed = False
        for key in group_keys:
            if allocations[key] < len(grouped[key]):
                allocations[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            break

    selected_train = []
    for index, key in enumerate(group_keys):
        configuration, bucket = key
        selected_train.extend(_pick_spread(
            grouped[key], allocations[key], SEED + index + len(bucket),
        ))

    train_suras = set().union(*(item["suras"] for item in selected_train))
    train_text = {item["normalized_text"] for item in selected_train if item["normalized_text"]}
    selected_validation = []
    validation_suras = set()
    validation_text = set()
    for configuration in VALIDATION_CONFIGS:
        valid = [item for row in validation_rows[configuration]
                 if (item := _candidate_fields(row, configuration, manifest, "validation")) is not None]
        for bucket in ("short", "medium"):
            eligible = [item for item in valid if item["bucket"] == bucket
                        and not item["suras"].intersection(train_suras | validation_suras)
                        and (not item["normalized_text"] or item["normalized_text"] not in train_text | validation_text)]
            if not eligible:
                available = Counter(item["bucket"] for item in valid)
                nonoverlap = Counter(
                    item["bucket"] for item in valid
                    if not item["suras"].intersection(train_suras | validation_suras)
                )
                raise RuntimeError(
                    f"No leakage-free validation candidate for {configuration}/{bucket}; "
                    f"all bucket counts={dict(available)}, nonoverlap counts={dict(nonoverlap)}, "
                    f"held-out training suras={sorted(train_suras)}, "
                    f"validation suras seen={sorted(validation_suras)}"
                )
            chosen = _pick_spread(eligible, 1, SEED + len(configuration) + len(bucket))[0]
            selected_validation.append(chosen)
            validation_suras.update(chosen["suras"])
            if chosen["normalized_text"]:
                validation_text.add(chosen["normalized_text"])

    if train_suras.intersection(validation_suras):
        raise RuntimeError(f"Selected training/validation Quran surahs overlap: {train_suras & validation_suras}")
    if train_text.intersection(validation_text):
        raise RuntimeError("Selected training and validation Imlaey annotations overlap")
    return selected_train, selected_validation, {
        "training_sura_ids": sorted(train_suras),
        "validation_sura_ids": sorted(validation_suras),
        "sura_sets_disjoint": True,
        "exact_normalized_imlaey_overlap_count": 0,
        "training_counts_by_configuration_and_duration": {
            c: dict(Counter(item["bucket"] for item in selected_train if item["configuration"] == c))
            for c in TRAIN_CONFIGS
        },
        "validation_counts_by_configuration_and_duration": {
            c: dict(Counter(item["bucket"] for item in selected_validation if item["configuration"] == c))
            for c in VALIDATION_CONFIGS
        },
    }


def _download_and_prepare(items: list[dict], manifest: dict, audio_config: AudioProcessingConfig) -> tuple[list, list[dict], list[dict]]:
    prepared = []
    metadata = []
    rejected = []
    policy = InclusionPolicy(
        include_non_quran=False,
        expected_sample_rate=16000,
        require_mono=True,
        on_missing_annotation="error",
        on_invalid_annotation="error",
    )
    for item in items:
        row = dict(item["row"])
        audio_refs = row.get("audio")
        if isinstance(audio_refs, list) and audio_refs:
            audio_ref = audio_refs[0]
        elif isinstance(audio_refs, dict):
            audio_ref = audio_refs
        else:
            raise RuntimeError(f"No audio reference for selected example {item['source_key']}")
        if audio_ref.get("bytes") is not None:
            row["audio"] = {
                "bytes": audio_ref["bytes"],
                "sampling_rate": int(row.get("sample_rate") or 16000),
            }
        else:
            source_url = audio_ref.get("src") or audio_ref.get("url")
            if not source_url:
                rejected.append({"source_key": item["source_key"], "reason": "audio_payload_missing"})
                continue
            try:
                response = requests.get(source_url, timeout=60)
                response.raise_for_status()
                row["audio"] = {
                    "bytes": response.content,
                    "sampling_rate": int(row.get("sample_rate") or 16000),
                }
            except Exception as exc:
                rejected.append({
                    "source_key": item["source_key"],
                    "reason": f"audio_fetch_{type(exc).__name__}: {exc}",
                })
                continue
        try:
            outcome = prepare_record(
                row,
                item["configuration"],
                audio_config=audio_config,
                policy=policy,
                split_manifest=manifest,
            )
        except Exception as exc:
            rejected.append({
                "source_key": item["source_key"],
                "reason": f"stage2_preparation_{type(exc).__name__}: {exc}",
            })
            continue
        if outcome.example is None:
            rejected.append({
                "source_key": item["source_key"],
                "reason": f"stage2_preparation_excluded: {outcome.excluded.reasons if outcome.excluded else 'unknown'}",
            })
            continue
        example = outcome.example
        actual_duration = example.audio.samples.shape[0] / example.audio.sampling_rate
        duration_reason = decoded_duration_rejection(item["duration_seconds"], actual_duration)
        if duration_reason:
            rejected.append({
                "source_key": item["source_key"],
                "metadata_duration_seconds": item["duration_seconds"],
                "decoded_duration_seconds": actual_duration,
                "reason": duration_reason,
            })
            continue
        if abs(actual_duration - item["duration_seconds"]) > 0.15:
            rejected.append({
                "source_key": item["source_key"],
                "metadata_duration_seconds": item["duration_seconds"],
                "decoded_duration_seconds": actual_duration,
                "reason": "metadata_decoded_duration_mismatch_over_0.15_seconds",
            })
            continue
        prepared.append(example)
        metadata.append({
            "source_key": example.source_key,
            "configuration": example.configuration,
            "split": example.split,
            "reciter_id": example.metadata.get("reciter_id"),
            "reciter_english_name": example.metadata.get("reciter_english_name"),
            "moshaf_id": example.metadata.get("moshaf_id"),
            "source_row_index": int(item["row"]["_pilot_row_index"]),
            "duration_seconds_metadata": item["duration_seconds"],
            "duration_seconds_decoded": actual_duration,
            "duration_bucket": item["bucket"],
            "sample_rate": example.audio.sampling_rate,
            "channels": example.audio.channels,
            "sura_ids": sorted(item["suras"]),
            "verse_keys": [list(key) for key in sorted(item["ayas"])],
            "has_quran": True,
            "all_eleven_label_levels": len(example.labels) == 11,
        })
    return prepared, metadata, rejected


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _pilot_run_directory(prefix: str) -> tuple[str, Path]:
    run_id = datetime.now(timezone.utc).strftime(f"{prefix}_%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "runs" / run_id
    suffix = 1
    while run_dir.exists():
        run_dir = ROOT / "runs" / f"{run_id}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir.name, run_dir


def _scan_preflight_candidates(run_dir: Path, manifest: dict) -> tuple[dict, dict, dict]:
    """Bounded Stage 2 streaming scan; no model is constructed and audio is not decoded."""
    config = DatasetConfig.from_file(ROOT / "configs/dataset.json")
    loader = DatasetLoader(config)
    train_rows = {name: [] for name in TRAIN_CONFIGS}
    validation_rows = {name: [] for name in VALIDATION_CONFIGS}
    counts = {
        "by_configuration": {},
        "by_reciter": {},
        "by_duration_group": {},
        "by_split_configuration_duration_group": {},
    }
    exclusions = Counter()
    exclusion_examples = []
    processed = 0
    _write_json(run_dir / "preflight_progress.json", {
        "status": "scanning",
        "source": "Stage 2 DatasetLoader streaming iterator",
        "published_dataset_split": "train",
        "maximum_rows_per_configuration": PREFLIGHT_ROWS_PER_CONFIGURATION,
        "processed_rows": 0,
        "completed_configurations": [],
    })
    jobs = [(name, "train") for name in TRAIN_CONFIGS]
    jobs.extend((name, "validation") for name in VALIDATION_CONFIGS)

    for configuration, expected_split in jobs:
        seen = 0
        eligible = 0
        for row_index, row in enumerate(loader.iter_samples(
            configuration, split="train", limit=PREFLIGHT_ROWS_PER_CONFIGURATION,
            streaming=True,
        )):
            seen += 1
            processed += 1
            row = dict(row)
            row["_pilot_row_index"] = row_index
            row["_pilot_duration_seconds"] = row.get("duration_seconds")
            reason = None
            try:
                assigned_split = assign_record_split(row, manifest)
            except (KeyError, ValueError) as exc:
                assigned_split = None
                reason = f"manifest_split_assignment:{type(exc).__name__}"
            if reason is None and assigned_split != expected_split:
                reason = f"manifest_split_is_{assigned_split}_expected_{expected_split}"
            if reason is None and row.get("has_quran") is not True:
                reason = "not_quran"
            if reason is None:
                reason = duration_rejection_reason(row.get("duration_seconds"))
            if reason is None and not _span_info(row)[0]:
                reason = "missing_or_invalid_sura_span"
            if reason is None and not str(row.get("segment_index") or ""):
                reason = "missing_segment_index"
            if reason is not None:
                exclusions[reason] += 1
                if len(exclusion_examples) < 40:
                    exclusion_examples.append({
                        "configuration": configuration,
                        "row_index": row_index,
                        "source_key": f"{configuration}:{row.get('segment_index', '')}",
                        "reason": reason,
                    })
            else:
                item = _candidate_fields(row, configuration, manifest, expected_split)
                if item is not None:
                    item.update({
                        "reciter_id": row.get("reciter_id"),
                        "reciter_english_name": row.get("reciter_english_name"),
                        "moshaf_id": str(row.get("moshaf_id") or configuration.removeprefix("moshaf_")),
                        "sura_ids": sorted(_span_info(row)[0]),
                    })
                    item["duration_group"] = duration_group(float(item["duration_seconds"]))
                    target = train_rows if expected_split == "train" else validation_rows
                    target[configuration].append(item)
                    eligible += 1
                    counts["by_configuration"][configuration] = counts["by_configuration"].get(configuration, 0) + 1
                    reciter_key = f"{row.get('reciter_id')}:{row.get('reciter_english_name')}"
                    counts["by_reciter"][reciter_key] = counts["by_reciter"].get(reciter_key, 0) + 1
                    group = item["duration_group"]
                    counts["by_duration_group"][group] = counts["by_duration_group"].get(group, 0) + 1
                    split_group = f"{expected_split}|{configuration}|{group}"
                    counts["by_split_configuration_duration_group"][split_group] = (
                        counts["by_split_configuration_duration_group"].get(split_group, 0) + 1
                    )
                else:
                    exclusions["candidate_field_validation_failed"] += 1

            if row_index % 20 == 19:
                _write_json(run_dir / "preflight_progress.json", {
                    "source": "Stage 2 DatasetLoader streaming iterator",
                    "published_dataset_split": "train",
                    "maximum_rows_per_configuration": PREFLIGHT_ROWS_PER_CONFIGURATION,
                    "processed_rows": processed,
                    "current_configuration": configuration,
                    "current_configuration_rows": seen,
                    "current_configuration_eligible": eligible,
                    "candidate_counts": counts,
                    "exclusion_counts": dict(exclusions),
                })
        counts.setdefault("rows_examined_by_configuration", {})[configuration] = seen
        counts.setdefault("eligible_by_configuration", {})[configuration] = eligible
        _write_json(run_dir / "preflight_progress.json", {
            "source": "Stage 2 DatasetLoader streaming iterator",
            "published_dataset_split": "train",
            "maximum_rows_per_configuration": PREFLIGHT_ROWS_PER_CONFIGURATION,
            "processed_rows": processed,
            "completed_configurations": list(counts["rows_examined_by_configuration"]),
            "candidate_counts": counts,
            "exclusion_counts": dict(exclusions),
        })
    counts["total_rows_examined"] = processed
    counts["excluded_counts"] = dict(exclusions)
    counts["exclusion_examples_first_40"] = exclusion_examples
    return train_rows, validation_rows, counts


def _preflight_markdown(run_id: str, status: str, result: dict) -> str:
    lines = [
        "# Stage 7 Part A.3 Pilot Safety Preflight",
        "",
        f"**Status: {status}**",
        "",
        f"Run directory: `runs/{run_id}/`",
        "",
        "This was selection-only. No model was initialized and no optimizer update was run.",
        "",
        "## Selection summary",
        "",
        f"- Duration limit: {MAX_DECODED_DURATION_SECONDS:.1f} seconds, checked in metadata and decoded audio.",
        f"- Training selected: {result.get('training_selected_count', 0)} / {PREFLIGHT_TRAINING_COUNT}.",
        f"- Eligible training pool: {result.get('eligible_training_candidate_count', 0)}; must exceed 32.",
        f"- Validation selected: {result.get('validation_selected_count', 0)} (requested 1–3).",
        f"- Rows examined: {result.get('candidate_scan', {}).get('total_rows_examined', 0)}; maximum {PREFLIGHT_ROWS_PER_CONFIGURATION} per configuration.",
        "",
        "## Candidate counts",
        "",
        "```json",
        json.dumps(result.get("candidate_scan", {}), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Split and leakage checks",
        "",
        "```json",
        json.dumps(result.get("selection_checks", {}), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Rejected selected examples",
        "",
        "```json",
        json.dumps(result.get("rejected_examples", []), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Notes",
        "",
        result.get("message", "Preflight details are in `preflight_result.json` and `selected_examples.json`."),
        "",
    ]
    return "\n".join(lines)


def run_selection_preflight() -> Path:
    run_id, run_dir = _pilot_run_directory("pilot_preflight")
    manifest = json.loads((ROOT / "configs/reciter_split_manifest.json").read_text(encoding="utf-8"))
    result = {
        "run_id": run_id,
        "dataset_id": DATASET_ID,
        "dataset_revision_expected": manifest["dataset_revision"],
        "selection_only": True,
        "training_model_initialized": False,
        "optimizer_updates_run": 0,
        "duration_limit_seconds": MAX_DECODED_DURATION_SECONDS,
        "training_required": PREFLIGHT_TRAINING_COUNT,
        "validation_required_min": PREFLIGHT_VALIDATION_MIN,
        "validation_required_max": PREFLIGHT_VALIDATION_MAX,
        "candidate_scan": {},
        "selection_checks": {},
        "rejected_examples": [],
    }
    status = "INCOMPLETE"
    try:
        train_rows, validation_rows, scan_counts = _scan_preflight_candidates(run_dir, manifest)
        result["candidate_scan"] = scan_counts
        train_candidates = [item for group in train_rows.values() for item in group]
        validation_candidates = [item for group in validation_rows.values() for item in group]
        result["eligible_training_candidate_count"] = len(train_candidates)
        result["eligible_validation_candidate_count"] = len(validation_candidates)
        selected_train, selected_validation, checks = select_preflight_examples(
            train_candidates, validation_candidates,
            training_count=PREFLIGHT_TRAINING_COUNT,
            validation_min=PREFLIGHT_VALIDATION_MIN,
            validation_max=PREFLIGHT_VALIDATION_MAX,
        )
        result["selection_checks"] = checks
        result["eligible_training_candidate_count"] = len(train_candidates)
        result["training_selected_count"] = len(selected_train)
        result["validation_selected_count"] = len(selected_validation)
        audio_config = AudioProcessingConfig.from_file(str(ROOT / "configs/audio_preprocessing.json"))
        train_examples, train_metadata, train_rejected = _download_and_prepare(selected_train, manifest, audio_config)
        validation_examples, validation_metadata, validation_rejected = _download_and_prepare(selected_validation, manifest, audio_config)
        rejected = train_rejected + validation_rejected
        result["rejected_examples"] = rejected
        result["decoded_training_count"] = len(train_examples)
        result["decoded_validation_count"] = len(validation_examples)
        result["selected_examples"] = train_metadata + validation_metadata
        if rejected or len(train_examples) != PREFLIGHT_TRAINING_COUNT:
            result["message"] = (
                "One or more selected audio records failed preparation/duration validation. "
                "No replacement was made; selection is incomplete."
            )
        elif not (PREFLIGHT_VALIDATION_MIN <= len(validation_examples) <= PREFLIGHT_VALIDATION_MAX):
            result["message"] = "Decoded validation count is outside the requested 1–3 range."
        else:
            result["message"] = "Selection, split, duration, and decoded-example checks passed."
            status = "PASS"
        selection_payload = {
            "run_id": run_id,
            "dataset_id": DATASET_ID,
            "dataset_revision": manifest["dataset_revision"],
            "selection_policy": {
                "training_examples_required": PREFLIGHT_TRAINING_COUNT,
                "eligible_candidate_pool_must_be_larger_than_required": True,
                "validation_examples_range": [PREFLIGHT_VALIDATION_MIN, PREFLIGHT_VALIDATION_MAX],
                "maximum_decoded_duration_seconds": MAX_DECODED_DURATION_SECONDS,
                "training_examples": train_metadata,
                "validation_examples": validation_metadata,
                "rejected_selected_examples": rejected,
                "selection_checks": checks,
            },
        }
        _write_json(run_dir / "selected_examples.json", selection_payload)
    except Exception as exc:
        result["message"] = f"Preflight could not complete: {type(exc).__name__}: {exc}"
        result["exception_type"] = type(exc).__name__
        result["exception_message"] = str(exc)
        _write_json(run_dir / "selected_examples.json", {
            "run_id": run_id,
            "selection_only": True,
            "status": status,
            "candidate_scan": result.get("candidate_scan", {}),
            "selection_error": result["message"],
        })
    result["status"] = status
    _write_json(run_dir / "preflight_result.json", result)
    (run_dir / "preflight_report.md").write_text(_preflight_markdown(run_id, status, result), encoding="utf-8")
    print(json.dumps({"status": status, "run_dir": str(run_dir), "message": result["message"]}, indent=2))
    return run_dir


def main():
    started = time.perf_counter()
    manifest = json.loads((ROOT / "configs/reciter_split_manifest.json").read_text(encoding="utf-8"))
    train_rows, validation_rows, scan_summary = _bounded_candidates(manifest)
    selected_train, selected_validation, leakage_summary = _select_examples(
        train_rows, validation_rows, manifest
    )
    audio_config = AudioProcessingConfig.from_file(str(ROOT / "configs/audio_preprocessing.json"))
    train_examples, train_metadata, train_rejected = _download_and_prepare(selected_train, manifest, audio_config)
    validation_examples, validation_metadata, validation_rejected = _download_and_prepare(selected_validation, manifest, audio_config)
    if train_rejected or validation_rejected:
        raise RuntimeError(f"Selected examples rejected by preparation/duration validation: {train_rejected + validation_rejected}")
    if len(train_examples) < MAX_PILOT_TRAIN_EXAMPLES:
        raise RuntimeError("Selected training pool is smaller than the configured example cap")
    if any(not record["all_eleven_label_levels"] for record in train_metadata + validation_metadata):
        raise RuntimeError("Pilot examples do not all provide the eleven expected label levels")

    run_id = datetime.now(timezone.utc).strftime("controlled_pilot_%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    selection_data = {
        "run_id": run_id,
        "dataset_id": DATASET_ID,
        **scan_summary,
        "selection_policy": {
            "training_configurations": list(TRAIN_CONFIGS),
            "validation_configurations": list(VALIDATION_CONFIGS),
            "duration_buckets_seconds": {"short": "0 < duration <= 5", "medium": "5 < duration <= 6.5", "long": "6.5 < duration <= 8"},
            "training_examples_selected": len(train_examples),
            "validation_examples_selected": len(validation_examples),
            "training_examples_consumption_cap": MAX_PILOT_TRAIN_EXAMPLES,
            "maximum_optimizer_updates_including_resume_update": MAX_PILOT_UPDATES,
            "epochs_for_scheduler_step_calculation": PILOT_EPOCHS,
            "non_quran_records_excluded_for_pilot": True,
            "leakage_check": leakage_summary,
        },
        "training_examples": train_metadata,
        "validation_examples": validation_metadata,
    }
    _write_json(run_dir / "selected_examples.json", selection_data)

    training_config = TrainingConfig.from_file(ROOT / "configs/training.json")
    if training_config.optimizer != "adafactor" or training_config.learning_rate != 1e-5:
        raise RuntimeError("Pilot requires the configured Adafactor optimizer at learning_rate=1e-5")
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    collator = QuranTrainingCollator(feature_extractor)
    model, load_report = QuranMultiLevelCTC.from_pretrained_checkpoint(
        MODEL_ID, revision=MODEL_REVISION, device="cpu"
    )
    if not load_report.is_complete or load_report.loaded_parameter_count != 605_754_251:
        raise RuntimeError(f"Original checkpoint compatibility failed: {load_report}")

    main_run = TrainingRunConfig(
        epochs=PILOT_EPOCHS,
        max_optimizer_steps=5,
        max_train_examples=80,
        checkpoint_every_steps=2,
        log_every_steps=1,
        seed=SEED,
        max_gpu_reserved_fraction=0.90,
        max_nvidia_smi_used_mib=5700,
        min_system_available_gib=2.5,
        require_all_parameter_gradients=True,
    )
    trainer = QuranTrainer(
        model=model,
        collator=collator,
        training_config=training_config,
        run_config=main_run,
        output_dir=run_dir,
    )
    main_result = trainer.fit(
        train_examples,
        validation_examples,
        max_validation_examples=9,
    )
    resume_checkpoint = main_result["last_checkpoint"]
    if main_result["optimizer_steps_completed"] != 5 or main_result["examples_seen"] != 80:
        raise RuntimeError(f"Main pilot stopped before its configured cap: {main_result['status']}")

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    resumed_model, resume_load_report = QuranMultiLevelCTC.from_pretrained_checkpoint(
        MODEL_ID, revision=MODEL_REVISION, device="cpu"
    )
    if not resume_load_report.is_complete:
        raise RuntimeError("Fresh model failed original checkpoint key/shape audit before resume")
    resume_run = TrainingRunConfig(
        epochs=PILOT_EPOCHS,
        max_optimizer_steps=MAX_PILOT_UPDATES,
        max_train_examples=MAX_PILOT_TRAIN_EXAMPLES,
        checkpoint_every_steps=2,
        log_every_steps=1,
        seed=SEED,
        max_gpu_reserved_fraction=0.90,
        max_nvidia_smi_used_mib=5700,
        min_system_available_gib=2.5,
        require_all_parameter_gradients=True,
    )
    resumed_trainer = QuranTrainer(
        model=resumed_model,
        collator=collator,
        training_config=training_config,
        run_config=resume_run,
        output_dir=run_dir,
    )
    resume_result = resumed_trainer.fit(
        train_examples,
        validation_examples,
        max_validation_examples=9,
        resume_from=resume_checkpoint,
    )
    if resume_result["optimizer_steps_completed"] != MAX_PILOT_UPDATES:
        raise RuntimeError(f"Resume update did not reach the configured cap: {resume_result['status']}")
    if not resume_result["resume"]["restored_state"] or resume_result["resume"]["restored_state"]["global_step"] != 5:
        raise RuntimeError(f"Resume metadata was not restored at global step 5: {resume_result['resume']}")
    if resume_result["examples_seen"] != MAX_PILOT_TRAIN_EXAMPLES:
        raise RuntimeError("Training example budget accounting mismatch")

    combined = {
        "run_id": run_id,
        "selection": selection_data,
        "checkpoint_load": {
            "complete": load_report.is_complete,
            "matched_tensors": len(load_report.matched),
            "loaded_parameters": load_report.loaded_parameter_count,
        },
        "training_config": training_config.__dict__,
        "pilot_limits": {
            "updates_including_resumed_update": MAX_PILOT_UPDATES,
            "training_examples": MAX_PILOT_TRAIN_EXAMPLES,
            "selected_training_pool": len(train_examples),
            "selected_validation_examples": len(validation_examples),
            "full_dataset_epoch": False,
            "full_dataset_download": False,
        },
        "scheduler": {
            "kind": "linear warmup then linear decay",
            "planned_total_optimizer_steps": main_result["scheduler_total_steps"],
            "warmup_steps": main_result["warmup_steps"],
            "configured_learning_rate": training_config.learning_rate,
            "warmup_ratio": training_config.warmup_ratio,
            "scheduler_lr_trace_main": [
                {"step": row["global_step"], "used": row["learning_rate_used"], "next": row["learning_rate_next"]}
                for row in main_result["step_metrics"]
            ],
            "scheduler_lr_trace_resume": [
                {"step": row["global_step"], "used": row["learning_rate_used"], "next": row["learning_rate_next"]}
                for row in resume_result["step_metrics"]
            ],
        },
        "main_run": main_result,
        "resume_continuation": resume_result,
        "elapsed_wall_seconds": time.perf_counter() - started,
        "status": "completed",
    }
    _write_json(run_dir / "pilot_results.json", combined)
    _write_json(ROOT / "reports" / "controlled_training_pilot_results.json", combined)
    print(json.dumps({
        "run_id": run_id,
        "status": combined["status"],
        "examples_selected": {"train": len(train_examples), "validation": len(validation_examples)},
        "updates": MAX_PILOT_UPDATES,
        "training_examples_consumed": MAX_PILOT_TRAIN_EXAMPLES,
        "checkpoint": resume_result["last_checkpoint"],
        "validation_baseline": main_result["initial_validation"]["mean_ctc_loss"],
        "validation_final": resume_result["final_validation"]["mean_ctc_loss"],
        "memory": resume_result["memory"],
        "results": str(ROOT / "reports" / "controlled_training_pilot_results.json"),
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quran Muaalem controlled pilot tooling")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true", help="Select and validate real data without initializing the model")
    mode.add_argument("--train", action="store_true", help="Run the controlled optimizer pilot")
    arguments = parser.parse_args()
    if arguments.preflight_only:
        run_selection_preflight()
    else:
        main()
