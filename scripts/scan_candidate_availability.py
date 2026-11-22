"""Bounded, resumable metadata-only scan for pilot candidate availability.

This uses the Hugging Face dataset-server rows response, which supplies audio
references but not audio bytes. It never follows those references.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import os
import uuid
import time

import requests

from quran_annotations import validate_record
from quran_training_data import LabelCodec, assign_record_split


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "obadx/muaalem-annotated-v3"
ROWS_API = "https://datasets-server.huggingface.co/rows"
MAX_DURATION_SECONDS = 6.3
PAGE_SIZE = 20
REQUEST_TIMEOUT_SECONDS = 15
MAX_ATTEMPTS_PER_PAGE = 2
PRIOR_RUN = ROOT / "runs" / "pilot_preflight_20261122T151943Z"
PRIOR_TRAIN_CONFIGS = {"moshaf_0.1", "moshaf_2.0", "moshaf_6.0", "moshaf_26.1"}


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    last_error = None
    for attempt in range(4):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(0.15 * (attempt + 1))
    raise last_error


def _source_key(configuration: str, row: dict) -> str:
    return f"{configuration}:{row.get('moshaf_id')}:{row.get('segment_index')}"


def _span_suras(row: dict) -> set[int]:
    start, end = row.get("start_span"), row.get("end_span")
    if not isinstance(start, dict) or not isinstance(end, dict):
        return set()
    values = (start.get("sura_idx"), end.get("sura_idx"), start.get("aya_idx"), end.get("aya_idx"))
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return set()
    return set(range(min(values[0], values[1]), max(values[0], values[1]) + 1))


def _duration_reason(value) -> str | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0:
        return "duration_missing_invalid_or_nonpositive"
    if float(value) > MAX_DURATION_SECONDS:
        return "duration_exceeds_6.3_seconds"
    return None


def classify_row(configuration: str, row_index: int, row: dict, manifest: dict, codec: LabelCodec) -> dict:
    selector_reasons: list[str] = []
    try:
        split = assign_record_split(row, manifest)
    except (KeyError, ValueError) as exc:
        split = None
        selector_reasons.append(f"manifest_assignment:{type(exc).__name__}")
    if split is not None and split != "train":
        selector_reasons.append(f"manifest_split_{split}")
    if row.get("has_quran") is not True:
        selector_reasons.append("not_quran")
    duration_reason = _duration_reason(row.get("duration_seconds"))
    if duration_reason:
        selector_reasons.append(duration_reason)
    suras = _span_suras(row)
    if not suras:
        selector_reasons.append("missing_or_invalid_span")
    if not isinstance(row.get("segment_index"), str) or not row["segment_index"].strip():
        selector_reasons.append("missing_segment_index")

    annotation_errors = []
    validation = validate_record(row)
    annotation_errors.extend(issue.code for issue in validation.errors)
    try:
        codec.encode_phonemes(row.get("phonemes", ""), remove_spaces=True)
        codec.encode_sifat(row.get("sifat"))
    except Exception as exc:
        annotation_errors.append(f"label_codec:{type(exc).__name__}")
    exclusion_reasons = sorted(set(selector_reasons + [f"annotation:{error}" for error in annotation_errors]))

    audio = row.get("audio")
    audio_refs = audio if isinstance(audio, list) else [audio]
    audio_has_bytes = any(isinstance(ref, dict) and ref.get("bytes") is not None for ref in audio_refs)
    if audio_has_bytes:
        raise RuntimeError("Dataset-server rows response unexpectedly contained audio bytes; scan aborted")
    audio_sources = [ref.get("src") for ref in audio_refs if isinstance(ref, dict) and ref.get("src")]
    revision_matches = bool(audio_sources) and all(manifest["dataset_revision"] in src for src in audio_sources)
    if not revision_matches:
        selector_reasons.append("missing_or_wrong_revision_audio_reference")
        exclusion_reasons = sorted(set(selector_reasons + [f"annotation:{error}" for error in annotation_errors]))

    return {
        "configuration": configuration,
        "row_index": row_index,
        "source_key": _source_key(configuration, row),
        "split": split,
        "reciter_id": row.get("reciter_id"),
        "reciter_english_name": row.get("reciter_english_name"),
        "moshaf_id": str(row.get("moshaf_id")) if row.get("moshaf_id") is not None else None,
        "duration_seconds_metadata": row.get("duration_seconds"),
        "duration_group": (
            "short_0_to_5s" if isinstance(row.get("duration_seconds"), (int, float)) and row["duration_seconds"] <= 5
            else "medium_over5_to6.3s" if not duration_reason else None
        ),
        "has_quran": row.get("has_quran"),
        "sura_ids": sorted(suras),
        "annotation_valid": not annotation_errors,
        "annotation_error_codes": sorted(set(annotation_errors)),
        "audio_reference_only": bool(audio_refs),
        "audio_bytes_present": audio_has_bytes,
        "audio_reference_revision_matches_manifest": revision_matches,
        "selector_exclusion_reasons": sorted(set(selector_reasons)),
        "exclusion_reasons": exclusion_reasons,
        "eligible_metadata_candidate": not selector_reasons,
        "eligible_annotation_valid_candidate": not selector_reasons and not annotation_errors,
    }


def _planned_ranges(configurations: list[str]) -> list[dict]:
    plan = []
    for configuration in configurations:
        if configuration in PRIOR_TRAIN_CONFIGS:
            start, stop = 80, 160
        elif configuration == "moshaf_0.0":
            # Row 0 was already inspected in the metadata-only API probe.
            start, stop = 1, 80
        else:
            start, stop = 0, 80
        offset = start
        while offset < stop:
            length = min(PAGE_SIZE, stop - offset)
            plan.append({"configuration": configuration, "offset": offset, "length": length})
            offset += length
    return plan


def _page_path(run_dir: Path, page: dict) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", page["configuration"])
    return run_dir / "pages" / f"{safe}__{page['offset']}__{page['length']}.json"


def _metadata_rows(session: requests.Session, page: dict) -> list[dict]:
    params = {
        "dataset": DATASET_ID,
        "config": page["configuration"],
        "split": "train",
        "offset": page["offset"],
        "length": page["length"],
    }
    last_error = None
    for attempt in range(MAX_ATTEMPTS_PER_PAGE):
        try:
            response = session.get(ROWS_API, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code in {429, 502, 503, 504}:
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                if attempt + 1 < MAX_ATTEMPTS_PER_PAGE:
                    time.sleep(1)
                    continue
                break
            response.raise_for_status()
            result = response.json()
            return result["rows"]
        except (requests.RequestException, ValueError, KeyError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < MAX_ATTEMPTS_PER_PAGE:
                time.sleep(1)
    raise RuntimeError(last_error or "unknown rows API error")


def _aggregate(run_dir: Path, plan: list[dict], manifest: dict, codec: LabelCodec) -> dict:
    by_config = defaultdict(lambda: {"rows": 0, "eligible_metadata_candidates": 0, "eligible_annotation_valid": 0,
                                     "by_duration_group": Counter(), "exclusions": Counter(),
                                     "rows_by_reciter": Counter(), "candidates_by_reciter": Counter()})
    all_rows = 0
    eligible_keys = set()
    api_pages = 0
    for page in plan:
        path = _page_path(run_dir, page)
        if not path.exists():
            continue
        api_pages += 1
        data = json.loads(path.read_text(encoding="utf-8"))
        for summary in data["rows"]:
            all_rows += 1
            bucket = by_config[summary["configuration"]]
            bucket["rows"] += 1
            if summary["reciter_id"] is not None:
                bucket["rows_by_reciter"][str(summary["reciter_id"])] += 1
            for reason in summary["exclusion_reasons"]:
                bucket["exclusions"][reason] += 1
            if summary["eligible_metadata_candidate"]:
                bucket["eligible_metadata_candidates"] += 1
                if summary["reciter_id"] is not None:
                    bucket["candidates_by_reciter"][str(summary["reciter_id"])] += 1
                if summary["duration_group"]:
                    bucket["by_duration_group"][summary["duration_group"]] += 1
                if summary["eligible_annotation_valid_candidate"]:
                    bucket["eligible_annotation_valid"] += 1
                    eligible_keys.add(summary["source_key"])
    prior = json.loads((PRIOR_RUN / "preflight_result.json").read_text(encoding="utf-8"))
    prior_all_counts = prior["candidate_scan"]["eligible_by_configuration"]
    prior_counts = {name: int(prior_all_counts.get(name, 0)) for name in PRIOR_TRAIN_CONFIGS}
    prior_validation_counts = {
        name: int(count) for name, count in prior_all_counts.items() if name not in PRIOR_TRAIN_CONFIGS
    }
    current_counts = {name: int(values["eligible_metadata_candidates"]) for name, values in by_config.items()}
    combined_counts = {name: int(prior_counts.get(name, 0)) + current_counts.get(name, 0)
                       for name in sorted(set(prior_counts) | set(current_counts))}
    prior_rows = sum(prior["candidate_scan"]["rows_examined_by_configuration"].values())
    return {
        "api_pages_completed": api_pages,
        "new_metadata_rows_scanned": all_rows,
        "previous_stage_preflight_rows_not_rescanned": prior_rows,
        "previous_stage_training_metadata_candidates_by_configuration": prior_counts,
        "previous_stage_validation_metadata_candidates_by_configuration": prior_validation_counts,
        "new_scan_by_configuration": {
            name: {
                **{key: value for key, value in values.items()
                   if key not in {"by_duration_group", "exclusions", "rows_by_reciter", "candidates_by_reciter"}},
                "by_duration_group": dict(values["by_duration_group"]),
                "exclusions_by_reason": dict(values["exclusions"]),
                "rows_by_reciter_id": dict(values["rows_by_reciter"]),
                "eligible_candidates_by_reciter_id": dict(values["candidates_by_reciter"]),
            }
            for name, values in sorted(by_config.items())
        },
        "new_metadata_candidates_by_configuration": current_counts,
        "combined_metadata_candidates_by_configuration": combined_counts,
        "combined_metadata_candidate_count": sum(combined_counts.values()),
        "new_annotation_valid_candidate_count": len(eligible_keys),
        "sufficient_training_ready_lower_bound_found": len(eligible_keys) > 32,
        "previous_metadata_probe_rows_not_counted_in_candidate_totals": 1,
        "historical_exclusion_reason_cross_tabulation": "Unavailable: the prior preflight persisted global reason totals, not per-configuration rows. Historical candidate counts are preserved without rescanning those positions.",
        "duration_limit_seconds": MAX_DURATION_SECONDS,
    }


def _write_report(run_dir: Path, result: dict, manifest: dict, plan: list[dict], errors: list[dict]):
    train_groups = [g for g in manifest["groups"] if g["split"] == "train"]
    lines = [
        "# Bounded Candidate Availability Investigation",
        "",
        f"Status: **{result['status']}**",
        "",
        "## Metadata-only behavior",
        "",
        "The Stage 2 `DatasetLoader` casts the audio column to `Audio(decode=False)`, but Hugging Face `datasets` rows still include an `audio` mapping with encoded `bytes` when a streamed row is materialized. A bounded Stage 2 probe confirmed this. The dataset-server `/rows` API returned JSON audio references (`src`, `type`) and row metadata, with no `bytes`; this was checked on a live one-row response. The scanner verifies no audio bytes are present and never follows an audio URL.",
        "",
        "## Configuration coverage and bounds",
        "",
        f"The manifest contains {len(train_groups)} training reciters and {len(result['manifest_training_configurations'])} training moshaf configurations. The prior preflight scanned only four of those train configurations (plus the three validation configurations).",
        f"This metadata-only scan planned {len(plan)} bounded pages and completed {result['api_pages_completed']} pages / {result['new_metadata_rows_scanned']} rows. Each new position is read at most once; prior positions are skipped (the four previously scanned train configs resume at row 80).",
        f"Request bounds: rows API page size {PAGE_SIZE}; per-request timeout {REQUEST_TIMEOUT_SECONDS}s; at most {MAX_ATTEMPTS_PER_PAGE} attempts; row cap 80 new rows per manifest-training configuration (or rows 80–159 for the four prior configurations). No full-dataset iteration is used.",
        "",
        "## Candidate and exclusion counts",
        "",
        f"Combined selector-level training metadata candidates through the scanned windows: **{result['combined_metadata_candidate_count']}**. Of the newly scanned training rows, {result['new_annotation_valid_candidate_count']} also passed annotation schema/codec checks without decoding audio.",
        "",
        "Counts by configuration, duration group, reciter, and exclusion reason are in `candidate_availability.json`.",
        "Historical reason-by-configuration counts are unavailable because the prior preflight retained only global reason totals; prior candidate counts are included without re-fetching its rows.",
        "",
        "## Nine-cell validation question",
        "",
        "The earlier selector's nine cells were three validation configurations crossed with short, medium, and long duration groups; its long bucket began above 6.5 seconds. Under the current hard 6.3-second limit, the long cell is empty by definition, so the nine-cell requirement cannot be met without violating that limit. The current approved preflight requirement is 1–3 validation examples, not nine. The previous metadata scan found 18 validation candidates through its bounded windows (4, 4, and 10 by configuration); exact annotation and decoded-duration validity remain unverified without further metadata scan or audio decode.",
        "",
        "## API errors and limitations",
        "",
        f"Errors recorded: {len(errors)}. See `api_errors.jsonl` and `scan_progress.json` for page offsets and messages.",
        "",
        "The API scan can establish a metadata candidate pool without audio payloads. Metadata duration is not decoded duration, and annotation validity does not establish audio integrity or successful preprocessing. The prior run's historical exclusion breakdown cannot be reconstructed without rescanning rows, which this scan intentionally avoided.",
        "",
    ]
    (run_dir / "candidate_availability_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-run-dir", type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "configs/reciter_split_manifest.json").read_text(encoding="utf-8"))
    manifest_configs = sorted({f"moshaf_{mid}" for group in manifest["groups"] if group["split"] == "train"
                               for mid in group["moshaf_ids"]})
    dataset_config = json.loads((ROOT / "configs/dataset.json").read_text(encoding="utf-8"))
    configured = set(dataset_config["recitation_configurations"])
    unavailable = sorted(set(manifest_configs) - configured)
    if unavailable:
        raise SystemExit(f"Manifest train configurations absent from Stage 2 config: {unavailable}")

    if args.resume_run_dir:
        run_dir = args.resume_run_dir.resolve()
        if not run_dir.exists():
            raise SystemExit(f"Resume directory does not exist: {run_dir}")
        prior_plan = json.loads((run_dir / "scan_plan.json").read_text(encoding="utf-8"))
        plan = prior_plan["pages"]
    else:
        run_id = datetime.now(timezone.utc).strftime("candidate_availability_%Y%m%dT%H%M%SZ")
        run_dir = ROOT / "runs" / run_id
        suffix = 1
        while run_dir.exists():
            run_dir = ROOT / "runs" / f"{run_id}_{suffix}"
            suffix += 1
        run_dir.mkdir(parents=True, exist_ok=False)
        plan = _planned_ranges(manifest_configs)
        _write_json(run_dir / "scan_plan.json", {
            "dataset_id": DATASET_ID,
            "dataset_revision": manifest["dataset_revision"],
            "source_split": "train",
            "manifest_training_configurations": manifest_configs,
            "prior_run": str(PRIOR_RUN.relative_to(ROOT)),
            "prior_scanned_training_configurations": sorted(PRIOR_TRAIN_CONFIGS),
            "pages": plan,
            "maximum_new_rows_per_configuration": 80,
            "audio_payloads_requested": False,
            "audio_decode": False,
        })

    pages_dir = run_dir / "pages"
    pages_dir.mkdir(exist_ok=True)
    progress_path = run_dir / "scan_progress.json"
    errors_path = run_dir / "api_errors.jsonl"
    session = requests.Session()
    codec = LabelCodec()
    errors = []
    completed = 0
    _write_json(progress_path, {
        "status": "scanning",
        "completed_pages": 0,
        "planned_pages": len(plan),
        "completed_row_count": sum(len(json.loads(p.read_text(encoding="utf-8")).get("rows", []))
                                    for page in plan if (p := _page_path(run_dir, page)).exists()),
        "current_page": None,
        "audio_bytes_requested": False,
        "audio_decoded": False,
    })
    for page in plan:
        output_path = _page_path(run_dir, page)
        if output_path.exists():
            completed += 1
            continue
        try:
            envelopes = _metadata_rows(session, page)
            summaries = []
            for envelope in envelopes:
                row = envelope["row"]
                summary = classify_row(page["configuration"], int(envelope["row_idx"]), row, manifest, codec)
                summaries.append(summary)
            _write_json(output_path, {
                "configuration": page["configuration"],
                "offset": page["offset"],
                "requested_length": page["length"],
                "returned_rows": len(summaries),
                "audio_bytes_requested": False,
                "audio_decoded": False,
                "rows": summaries,
            })
            completed += 1
            error = None
        except Exception as exc:
            error = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "configuration": page["configuration"],
                "offset": page["offset"],
                "length": page["length"],
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
            errors.append(error)
            with errors_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(error, ensure_ascii=False) + "\n")
        summary = _aggregate(run_dir, plan, manifest, codec)
        _write_json(progress_path, {
            "status": "incomplete_api_error" if error else "scanning",
            "completed_pages": completed,
            "planned_pages": len(plan),
            "completed_row_count": summary["new_metadata_rows_scanned"],
            "current_page": page,
            "last_error": error,
            "candidate_count_so_far": summary["combined_metadata_candidate_count"],
            "audio_bytes_requested": False,
            "audio_decoded": False,
        })
        if error:
            break

    summary = _aggregate(run_dir, plan, manifest, codec)
    error_rows = []
    if errors_path.exists():
        error_rows = [json.loads(line) for line in errors_path.read_text(encoding="utf-8").splitlines() if line]
    complete = all(_page_path(run_dir, page).exists() for page in plan)
    manifest_reciters = {f"moshaf_{mid}": group["reciter_id"]
                         for group in manifest["groups"] if group["split"] == "train"
                         for mid in group["moshaf_ids"]}
    summary["manifest_training_configurations"] = manifest_configs
    summary["manifest_training_reciters"] = sorted({g["reciter_id"] for g in manifest["groups"] if g["split"] == "train"})
    summary["manifest_configuration_count"] = len(manifest_configs)
    summary["manifest_reciter_count"] = len(summary["manifest_training_reciters"])
    summary["api_error_count"] = len(error_rows)
    summary["all_planned_pages_completed"] = complete
    summary["sufficient_metadata_pool_found"] = summary["combined_metadata_candidate_count"] > 32
    summary["candidate_configs_by_manifest_reciter"] = manifest_reciters
    status = "COMPLETE_METADATA_SCAN" if complete else "INCOMPLETE_INFRASTRUCTURE"
    summary["status"] = status
    _write_json(run_dir / "candidate_availability.json", summary)
    _write_json(progress_path, {
        "status": status,
        "completed_pages": sum(_page_path(run_dir, page).exists() for page in plan),
        "planned_pages": len(plan),
        "completed_row_count": summary["new_metadata_rows_scanned"],
        "candidate_count": summary["combined_metadata_candidate_count"],
        "sufficient_metadata_pool_found": summary["sufficient_metadata_pool_found"],
        "audio_bytes_requested": False,
        "audio_decoded": False,
    })
    _write_report(run_dir, summary, manifest, plan, error_rows)
    print(json.dumps({"status": status, "run_directory": str(run_dir),
                      "new_rows": summary["new_metadata_rows_scanned"],
                      "combined_candidates": summary["combined_metadata_candidate_count"],
                      "sufficient_pool": summary["sufficient_metadata_pool_found"],
                      "api_errors": len(error_rows)}, indent=2))


if __name__ == "__main__":
    main()
