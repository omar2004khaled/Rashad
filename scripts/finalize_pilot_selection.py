"""Bounded final selection and audio validation for the A.3 pilot gate."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from quran_audio import AudioProcessingConfig  # noqa: E402
from quran_dataset import DatasetConfig, DatasetLoader  # noqa: E402
from quran_annotations import validate_record  # noqa: E402
from quran_training_data import LabelCodec  # noqa: E402
from scripts.run_training_pilot import (  # noqa: E402
    DATASET_ID,
    _candidate_fields,
    _download_and_prepare,
    _pick_spread,
    _write_json,
)


SCAN_RUN = ROOT / "runs/candidate_availability_20261122T154412Z"
TRAIN_CONFIGS = ("moshaf_11.0", "moshaf_27.0")
TRAIN_COUNT_PER_CONFIG = 16
VALIDATION_CONFIGS = ("moshaf_12.0", "moshaf_13.0", "moshaf_24.0")
MAX_DURATION_SECONDS = 6.3


def candidate_from_row(row: dict, row_index: int, config: str, manifest: dict, split: str) -> dict | None:
    row = dict(row)
    row["_pilot_row_index"] = int(row_index)
    row["_pilot_duration_seconds"] = float(row.get("duration_seconds") or 0.0)
    item = _candidate_fields(row, config, manifest, split)
    if item is None:
        return None
    validation = validate_record(row)
    codec = LabelCodec()
    if validation.errors:
        return None
    codec.encode_phonemes(row["phonemes"], remove_spaces=True)
    codec.encode_sifat(row["sifat"])
    item["row"] = row
    return item


def main() -> int:
    manifest = json.loads((ROOT / "configs/reciter_split_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((SCAN_RUN / "candidate_availability.json").read_text(encoding="utf-8"))
    if summary["combined_metadata_candidate_count"] <= 32 or summary["new_annotation_valid_candidate_count"] <= 32:
        raise RuntimeError("Saved metadata scan does not establish more than 32 eligible candidates")

    run_id = datetime.now(timezone.utc).strftime("pilot_final_selection_%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    progress = {"run_id": run_id, "metadata_pages_completed": [], "audio_decodes_completed": [],
                "audio_payloads_requested_before_selection": False}
    _write_json(run_dir / "selection_progress.json", progress)

    # Only refetch full row metadata for two known, candidate-rich training configurations.
    cached_candidates = {}
    for page in (SCAN_RUN / "pages").glob("*.json"):
        data = json.loads(page.read_text(encoding="utf-8"))
        if data["configuration"] in TRAIN_CONFIGS:
            for record in data["rows"]:
                if record["eligible_annotation_valid_candidate"]:
                    cached_candidates[record["source_key"]] = record
    train_by_config = {config: [] for config in TRAIN_CONFIGS}
    try:
        loader = DatasetLoader(DatasetConfig.from_file(ROOT / "configs/dataset.json"))
        for config in TRAIN_CONFIGS:
            for row_index, row in enumerate(loader.iter_samples(config, split="train", limit=80)):
                item = candidate_from_row(row, row_index, config, manifest, "train")
                if item is not None and item["source_key"] in cached_candidates:
                    expected = cached_candidates[item["source_key"]]
                    if abs(float(expected["duration_seconds_metadata"]) - item["duration_seconds"]) > 1e-5:
                        raise RuntimeError(f"Dataset revision/duration mismatch at {item['source_key']}")
                    train_by_config[config].append(item)
                if row_index % 20 == 19:
                    progress["metadata_pages_completed"].append({"configuration": config,
                                                                  "offset": row_index - 19, "rows": 20,
                                                                  "audio_bytes_returned_by_stage2": True,
                                                                  "audio_decoded": False})
                    _write_json(run_dir / "selection_progress.json", progress)
            progress["configurations_completed"] = [*progress.get("configurations_completed", []), config]
            _write_json(run_dir / "selection_progress.json", progress)

        if any(len(train_by_config[c]) < TRAIN_COUNT_PER_CONFIG for c in TRAIN_CONFIGS):
            raise RuntimeError(f"Insufficient verified candidates by training configuration: "
                               f"{ {k: len(v) for k, v in train_by_config.items()} }")
        selected_train = []
        for i, config in enumerate(TRAIN_CONFIGS):
            selected_train.extend(_pick_spread(train_by_config[config], TRAIN_COUNT_PER_CONFIG, 42 + i))
        if len({x["source_key"] for x in selected_train}) != 32:
            raise RuntimeError("Training selection contains duplicate example IDs")

        train_reciters = {x["row"].get("reciter_id") for x in selected_train}
        train_moshafs = {str(x["row"].get("moshaf_id")) for x in selected_train}
        train_suras = set().union(*(x["suras"] for x in selected_train))
        train_text = {x["normalized_text"] for x in selected_train if x["normalized_text"]}

        validation_candidates = []
        for config in VALIDATION_CONFIGS:
            for row_index, row in enumerate(loader.iter_samples(config, split="train", limit=80)):
                item = candidate_from_row(row, row_index, config, manifest, "validation")
                if item is not None:
                    candidate_row = item["row"]
                    if (candidate_row.get("reciter_id") not in train_reciters
                            and str(candidate_row.get("moshaf_id")) not in train_moshafs):
                        validation_candidates.append(item)
                if row_index % 20 == 19:
                    progress["metadata_pages_completed"].append({"configuration": config,
                                                                  "offset": row_index - 19, "rows": 20,
                                                                  "audio_bytes_returned_by_stage2": True,
                                                                  "audio_decoded": False})
                    _write_json(run_dir / "selection_progress.json", progress)
            progress["configurations_completed"] = [*progress.get("configurations_completed", []), config]
            _write_json(run_dir / "selection_progress.json", progress)
        if not validation_candidates:
            raise RuntimeError("No candidate met held-out validation reciter/moshaf constraints")
        # Prefer a validation example with disjoint suras and normalized text.
        disjoint = [x for x in validation_candidates
                    if not x["suras"].intersection(train_suras)
                    and (not x["normalized_text"] or x["normalized_text"] not in train_text)]
        selected_validation = _pick_spread(disjoint or validation_candidates, 1, 84)

        selection = {
            "run_id": run_id,
            "dataset_id": DATASET_ID,
            "dataset_revision": manifest["dataset_revision"],
            "candidate_availability_run": str(SCAN_RUN.relative_to(ROOT)),
            "candidate_pool_size_metadata": summary["combined_metadata_candidate_count"],
            "new_annotation_valid_candidate_pool_size": summary["new_annotation_valid_candidate_count"],
            "selection_rule": {"train_count": 32, "train_configs": list(TRAIN_CONFIGS),
                               "train_per_config": TRAIN_COUNT_PER_CONFIG,
                               "validation_count": 1, "validation_configurations": list(VALIDATION_CONFIGS),
                               "maximum_duration_seconds": MAX_DURATION_SECONDS,
                               "prefer_disjoint_suras_and_normalized_text": True,
                               "disjoint_candidate_found": bool(disjoint)},
            "training_candidates_verified": {c: len(v) for c, v in train_by_config.items()},
            "training_examples": [metadata(x) for x in selected_train],
            "validation_examples": [metadata(x) for x in selected_validation],
            "leakage_check": {
                "reciter_sets_disjoint": not train_reciters.intersection(
                    {x["row"].get("reciter_id") for x in selected_validation}),
                "moshaf_sets_disjoint": not train_moshafs.intersection(
                    {str(x["row"].get("moshaf_id")) for x in selected_validation}),
                "sura_sets_disjoint": not train_suras.intersection(selected_validation[0]["suras"]),
                "normalized_text_disjoint": not train_text.intersection(
                    {x["normalized_text"] for x in selected_validation if x["normalized_text"]}),
                "disjoint_validation_candidate_found": bool(disjoint),
                "training_reciter_ids": sorted(train_reciters),
                "validation_reciter_ids": sorted({x["row"].get("reciter_id") for x in selected_validation}),
                "training_moshaf_ids": sorted(train_moshafs),
                "validation_moshaf_ids": sorted({str(x["row"].get("moshaf_id")) for x in selected_validation}),
                "training_sura_ids": sorted(train_suras),
                "validation_sura_ids": sorted(set().union(*(x["suras"] for x in selected_validation))),
            },
        }
        _write_json(run_dir / "selected_metadata.json", selection)

        audio_config = AudioProcessingConfig.from_file(ROOT / "configs/audio_preprocessing.json")
        prepared_train, train_audio, rejected_train = _download_and_prepare(selected_train, manifest, audio_config)
        progress["audio_decodes_completed"].extend(train_audio)
        progress["rejections"] = rejected_train
        _write_json(run_dir / "selection_progress.json", progress)
        prepared_val, val_audio, rejected_val = _download_and_prepare(selected_validation, manifest, audio_config)
        progress["audio_decodes_completed"].extend(val_audio)
        progress["rejections"] = rejected_train + rejected_val
        _write_json(run_dir / "selection_progress.json", progress)

        audio_ok = len(prepared_train) == 32 and len(prepared_val) == 1 and not rejected_train and not rejected_val
        if audio_ok:
            for example in prepared_train + prepared_val:
                if not (0 < len(example.audio.samples) / example.audio.sampling_rate <= MAX_DURATION_SECONDS):
                    audio_ok = False
                if example.audio.sampling_rate != 16000 or example.audio.channels != 1:
                    audio_ok = False
                if not np.isfinite(example.audio.samples).all():
                    audio_ok = False
        selection["decoded_training_count"] = len(prepared_train)
        selection["decoded_validation_count"] = len(prepared_val)
        selection["decoded_training_examples"] = train_audio
        selection["decoded_validation_examples"] = val_audio
        selection["rejected_examples"] = rejected_train + rejected_val
        selection["audio_validation_pass"] = audio_ok
        selection["status"] = "PASS" if audio_ok else "INCOMPLETE_AUDIO_VALIDATION"
        _write_json(run_dir / "selection_report.json", selection)
        (run_dir / "selection_report.md").write_text(render_report(selection), encoding="utf-8")
        np.savez_compressed(
            run_dir / "prepared_audio.npz",
            **{f"train_{i:02d}": x.audio.samples for i, x in enumerate(prepared_train)},
            **{f"validation_{i:02d}": x.audio.samples for i, x in enumerate(prepared_val)},
        )
        _write_json(run_dir / "prepared_labels.json", {
            "training": [{"source_key": x.source_key, "labels": x.labels} for x in prepared_train],
            "validation": [{"source_key": x.source_key, "labels": x.labels} for x in prepared_val],
        })
        print(json.dumps({"status": selection["status"], "run_dir": str(run_dir),
                          "train": len(prepared_train), "validation": len(prepared_val),
                          "rejects": len(rejected_train) + len(rejected_val)}, indent=2))
        return 0 if audio_ok else 2
    except Exception as exc:
        progress["status"] = "INCOMPLETE"
        progress["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(run_dir / "selection_progress.json", progress)
        print(json.dumps({"status": "INCOMPLETE", "run_dir": str(run_dir), "error": progress["error"]}, indent=2))
        return 2


def metadata(item: dict) -> dict:
    row = item["row"]
    return {"source_key": item["source_key"], "configuration": item["configuration"],
            "source_row_index": row["_pilot_row_index"], "reciter_id": row.get("reciter_id"),
            "moshaf_id": row.get("moshaf_id"), "metadata_duration_seconds": item["duration_seconds"],
            "sura_ids": sorted(item["suras"]), "normalized_text": item["normalized_text"],
            "annotation_valid": True, "sample_rate_metadata": row.get("sample_rate")}


def render_report(data: dict) -> str:
    lines = ["# Final Pilot Selection and Audio Validation", "", f"Status: **{data['status']}**", "",
             f"Training examples decoded: {data['decoded_training_count']}/32",
             f"Validation examples decoded: {data['decoded_validation_count']}/1", "",
             "## Rejections", "", "```json", json.dumps(data["rejected_examples"], indent=2), "```", "",
             "## Selection", "", "```json", json.dumps(data["selection_rule"], indent=2), "```", "",
             "## Leakage checks", "", "```json", json.dumps(data["leakage_check"], indent=2), "```", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
