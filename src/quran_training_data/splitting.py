"""Reproducible group-aware splits and leakage assertions."""

from __future__ import annotations

from itertools import combinations
import random
from typing import Any, Iterable, Mapping


def build_reciter_split_manifest(
    reciter_rows: Iterable[Mapping[str, Any]],
    row_counts_by_moshaf: Mapping[str, int],
    *,
    dataset_revision: str,
    seed: int = 42,
    validation_groups: int = 3,
    test_groups: int = 3,
    ratios: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Assign all recording sets for one reciter to one split.

    For the dataset's 22 reciters, enumerate holdout group combinations and
    choose the assignment minimizing squared deviation from requested *row*
    fractions. A seeded shuffle fixes tie order. Train gets all remaining groups.
    """
    target = dict(ratios or {"train": 0.8, "validation": 0.1, "test": 0.1})
    if set(target) != {"train", "validation", "test"} or any(v <= 0 for v in target.values()):
        raise ValueError("ratios must define positive train, validation and test fractions")
    if abs(sum(target.values()) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1")
    groups: dict[str, dict[str, Any]] = {}
    for row in reciter_rows:
        rid = str(row.get("id"))
        set_ids = [str(value) for value in (row.get("moshaf_set_ids") or [])]
        if rid == "None" or not set_ids:
            raise ValueError(f"Reciter row has missing id or moshaf_set_ids: {row!r}")
        unknown = [set_id for set_id in set_ids if set_id not in row_counts_by_moshaf]
        if unknown:
            raise ValueError(f"No row counts for moshaf sets {unknown!r}")
        groups[rid] = {
            "reciter_id": int(row["id"]),
            "english_name": row.get("english_name"),
            "moshaf_ids": sorted(set_ids),
            "row_count": sum(int(row_counts_by_moshaf[s]) for s in set_ids),
        }
    if len(groups) < validation_groups + test_groups + 1:
        raise ValueError("Not enough reciter groups to leave at least one training group")
    total = sum(item["row_count"] for item in groups.values())
    if total <= 0:
        raise ValueError("Row counts must sum to a positive value")

    ordered_ids = list(groups)
    random.Random(seed).shuffle(ordered_ids)
    val_sets = list(combinations(ordered_ids, validation_groups))
    test_sets = list(combinations(ordered_ids, test_groups))
    best: tuple[float, tuple[str, ...], tuple[str, ...]] | None = None
    for validation_ids in val_sets:
        validation_count = sum(groups[rid]["row_count"] for rid in validation_ids)
        validation_set = set(validation_ids)
        for test_ids in test_sets:
            if validation_set.intersection(test_ids):
                continue
            test_count = sum(groups[rid]["row_count"] for rid in test_ids)
            train_count = total - validation_count - test_count
            score = sum(
                ((count / total) - target[split]) ** 2
                for split, count in (("train", train_count), ("validation", validation_count), ("test", test_count))
            )
            if best is None or score < best[0] - 1e-15:
                best = (score, validation_ids, test_ids)
    assert best is not None
    _, validation_ids, test_ids = best
    selected = {**{rid: "validation" for rid in validation_ids},
                **{rid: "test" for rid in test_ids}}
    selected.update({rid: "train" for rid in groups if rid not in selected})
    for rid in groups:
        groups[rid]["split"] = selected[rid]
    set_to_split = {
        set_id: selected[rid]
        for rid, group in groups.items()
        for set_id in group["moshaf_ids"]
    }
    split_sizes = {name: sum(g["row_count"] for g in groups.values() if g["split"] == name)
                   for name in ("train", "validation", "test")}
    return {
        "dataset_id": "obadx/muaalem-annotated-v3",
        "dataset_revision": dataset_revision,
        "seed": seed,
        "grouping_key": "reciter_id",
        "strategy": "reciter-group holdout; exhaustive validation/test group search minimizing squared row-fraction deviation",
        "target_ratios": target,
        "actual_source_row_counts": split_sizes,
        "source_row_count_total": total,
        "groups": sorted(groups.values(), key=lambda x: x["reciter_id"]),
        "reciter_id_to_split": {str(groups[rid]["reciter_id"]): split for rid, split in sorted(selected.items(), key=lambda x: groups[x[0]]["reciter_id"])},
        "moshaf_id_to_split": dict(sorted(set_to_split.items())),
    }


def assign_record_split(record: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    """Resolve a source row to its manifest split, cross-checking both keys."""
    reciter = record.get("reciter_id")
    moshaf = record.get("moshaf_id")
    by_reciter = manifest.get("reciter_id_to_split", {}).get(str(reciter))
    by_moshaf = manifest.get("moshaf_id_to_split", {}).get(str(moshaf))
    if by_reciter is None and by_moshaf is None:
        raise KeyError(f"No split assignment for reciter_id={reciter!r}, moshaf_id={moshaf!r}")
    if by_reciter is not None and by_moshaf is not None and by_reciter != by_moshaf:
        raise ValueError(f"Split manifest disagrees for reciter {reciter!r} and moshaf {moshaf!r}")
    return by_reciter if by_reciter is not None else by_moshaf


def assert_no_group_leakage(
    rows: Iterable[tuple[str, Mapping[str, Any], str]], *, group_field: str = "reciter_id"
) -> None:
    """Raise if any group is assigned to more than one split."""
    assignments: dict[str, set[str]] = {}
    for _, row, split in rows:
        group = row.get(group_field)
        if group is None:
            raise ValueError(f"Cannot check leakage: {group_field!r} is missing")
        assignments.setdefault(str(group), set()).add(split)
    leaked = {group: sorted(splits) for group, splits in assignments.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"Group leakage found for {group_field}: {leaked}")


def expected_random_segment_group_leakage(
    row_counts_by_group: Mapping[str | int, int],
    *,
    ratios: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Expected group overlap if individual rows were independently randomized."""
    proportions = dict(ratios or {"train": 0.8, "validation": 0.1, "test": 0.1})
    if set(proportions) != {"train", "validation", "test"} or any(p < 0 for p in proportions.values()):
        raise ValueError("ratios must contain non-negative train, validation and test probabilities")
    if abs(sum(proportions.values()) - 1.0) > 1e-9:
        raise ValueError("ratios must sum to 1")
    group_counts = {str(group): int(count) for group, count in row_counts_by_group.items()}
    if any(count <= 0 for count in group_counts.values()):
        raise ValueError("group row counts must be positive")
    present_each_split = {
        split: sum(1.0 - (1.0 - probability) ** count for count in group_counts.values())
        for split, probability in proportions.items()
    }
    expected_groups_in_multiple_splits = sum(
        1.0 - sum(probability ** count for probability in proportions.values())
        for count in group_counts.values()
    )
    return {
        "assumption": "independent per-row random assignment with the requested probabilities",
        "expected_groups_present_in_each_split": present_each_split,
        "expected_groups_present_in_multiple_splits": expected_groups_in_multiple_splits,
        "group_count": len(group_counts),
    }
