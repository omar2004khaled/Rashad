"""Small, split-aware selection helpers for the Stage 7 pilot preflight."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Sequence


MAX_DECODED_DURATION_SECONDS = 6.3


def duration_rejection_reason(duration_seconds: Any, *, maximum: float = MAX_DECODED_DURATION_SECONDS) -> str | None:
    try:
        duration = float(duration_seconds)
    except (TypeError, ValueError):
        return "duration_missing_or_not_numeric"
    if not math.isfinite(duration) or duration <= 0:
        return "duration_nonpositive_or_nonfinite"
    if duration > maximum:
        return f"duration_exceeds_{maximum:g}_seconds"
    return None


def decoded_duration_rejection(metadata_seconds: Any, decoded_seconds: Any,
                               *, maximum: float = MAX_DECODED_DURATION_SECONDS) -> str | None:
    metadata_reason = duration_rejection_reason(metadata_seconds, maximum=maximum)
    if metadata_reason:
        return f"metadata_{metadata_reason}"
    decoded_reason = duration_rejection_reason(decoded_seconds, maximum=maximum)
    if decoded_reason:
        return f"decoded_{decoded_reason}"
    return None


def duration_group(duration_seconds: float) -> str:
    """Summarize the bounded pilot ranges; there is intentionally no >6.3 s bucket."""
    return "short_0_to_5s" if duration_seconds <= 5 else "medium_over5_to6.3s"


def select_preflight_examples(
    train_candidates: Sequence[dict[str, Any]],
    validation_candidates: Sequence[dict[str, Any]],
    *,
    training_count: int = 32,
    validation_min: int = 1,
    validation_max: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Select a balanced training sample and up to three held-out validation rows.

    Candidates must already be duration-checked and have split values assigned by
    the approved manifest. Invalid-split rows are rejected, never reassigned.
    Quran-text/sura overlap is avoided when available and is reported explicitly.
    """
    if training_count < 1 or validation_min < 1 or validation_max < validation_min:
        raise ValueError("selection sizes must be positive and validation_max >= validation_min")

    rejected = Counter()
    train = []
    validation = []
    for row in train_candidates:
        if row.get("split") != "train":
            rejected["training_candidate_not_manifest_train"] += 1
        elif duration_rejection_reason(row.get("duration_seconds")):
            rejected["training_candidate_duration_invalid"] += 1
        else:
            train.append(row)
    for row in validation_candidates:
        if row.get("split") != "validation":
            rejected["validation_candidate_not_manifest_validation"] += 1
        elif duration_rejection_reason(row.get("duration_seconds")):
            rejected["validation_candidate_duration_invalid"] += 1
        else:
            validation.append(row)

    if len(train) <= training_count:
        raise ValueError(
            f"Need more than {training_count} eligible training candidates; found {len(train)}"
        )
    train_configs = {row.get("configuration") for row in train}
    train_reciters = {row.get("reciter_id") for row in train}
    if len(train_configs) < 2 or len(train_reciters) < 2:
        raise ValueError(
            "Need eligible training candidates from at least two distinct manifest-training "
            f"configurations and reciters; found {len(train_configs)} configurations and {len(train_reciters)} reciters"
        )

    # Use rare validation suras first so training selection can avoid them where possible.
    train_sura_frequency = Counter(
        sura for row in train for sura in set(row.get("sura_ids", ()))
    )
    validation.sort(key=lambda row: (
        sum(train_sura_frequency[sura] for sura in set(row.get("sura_ids", ()))),
        row.get("configuration", ""),
        row.get("source_key", ""),
    ))
    chosen_validation = []
    validation_suras: set[Any] = set()
    validation_text: set[str] = set()
    validation_reciters: set[Any] = set()
    validation_moshafs: set[str] = set()
    for row in validation:
        if len(chosen_validation) >= validation_max:
            break
        reciter = row.get("reciter_id")
        moshaf = str(row.get("moshaf_id"))
        if reciter in train_reciters or moshaf in {str(r.get("moshaf_id")) for r in train}:
            rejected["validation_shares_training_reciter_or_moshaf"] += 1
            continue
        if reciter in validation_reciters or moshaf in validation_moshafs:
            continue
        text = row.get("normalized_text") or ""
        if text and text in validation_text:
            continue
        chosen_validation.append(row)
        validation_reciters.add(reciter)
        validation_moshafs.add(moshaf)
        validation_suras.update(row.get("sura_ids", ()))
        if text:
            validation_text.add(text)

    if len(chosen_validation) < validation_min:
        raise ValueError(
            f"Need at least {validation_min} independent manifest-validation examples; "
            f"found {len(chosen_validation)}"
        )

    # Balance selection over every available configuration/duration group by round robin.
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in train:
        groups.setdefault((str(row["configuration"]), duration_group(float(row["duration_seconds"]))), []).append(row)
    group_keys = sorted(groups)
    allocations = {key: 0 for key in group_keys}
    selected = []
    used_keys = set()
    candidates = {key: sorted(values, key=lambda r: r.get("source_key", "")) for key, values in groups.items()}
    train_suras: set[Any] = set()
    train_text: set[str] = set()
    # Prefer strict sura and exact normalized-text separation. If it cannot fill 32,
    # retain split isolation and state the overlap rather than hiding the constraint relaxation.
    for strict in (True, False):
        progressed = True
        while len(selected) < training_count and progressed:
            progressed = False
            for key in group_keys:
                pool = candidates[key]
                while pool:
                    row = pool.pop(0)
                    row_key = row.get("source_key")
                    if row_key in used_keys:
                        continue
                    suras = set(row.get("sura_ids", ()))
                    text = row.get("normalized_text") or ""
                    if strict and (suras & validation_suras or (text and text in validation_text)):
                        rejected["training_candidate_quran_overlap_with_validation"] += 1
                        continue
                    used_keys.add(row_key)
                    selected.append(row)
                    allocations[key] += 1
                    train_suras.update(suras)
                    if text:
                        train_text.add(text)
                    progressed = True
                    break
                if len(selected) >= training_count:
                    break
        if len(selected) >= training_count:
            break

    if len(selected) != training_count:
        raise ValueError(
            f"Only {len(selected)} training examples satisfy split/duration constraints; "
            f"need {training_count}"
        )
    overlaps = {
        "shared_sura_ids": sorted(train_suras & validation_suras, key=str),
        "shared_normalized_text_count": len(train_text & validation_text),
    }
    return selected, chosen_validation, {
        "eligible_training_candidate_count": len(train),
        "eligible_validation_candidate_count": len(validation),
        "training_selection_by_configuration_and_duration": {
            f"{config}|{group}": count for (config, group), count in allocations.items() if count
        },
        "training_configuration_count": len({row["configuration"] for row in selected}),
        "training_reciter_count": len({row["reciter_id"] for row in selected}),
        "validation_reciter_count": len({row["reciter_id"] for row in chosen_validation}),
        "validation_moshaf_count": len({str(row["moshaf_id"]) for row in chosen_validation}),
        "quran_overlap": overlaps,
        "rejected_candidate_counts": dict(rejected),
    }
