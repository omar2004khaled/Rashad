import pytest

from quran_training_data import (
    decoded_duration_rejection,
    duration_rejection_reason,
    select_preflight_examples,
)


def candidate(key, *, split, config, reciter, moshaf, sura, text, duration=6.0):
    return {
        "source_key": key,
        "split": split,
        "configuration": config,
        "reciter_id": reciter,
        "moshaf_id": moshaf,
        "duration_seconds": duration,
        "sura_ids": [sura],
        "normalized_text": text,
    }


def test_metadata_duration_includes_boundary_and_rejects_over_limit():
    assert duration_rejection_reason(6.3) is None
    assert duration_rejection_reason(6.300001) == "duration_exceeds_6.3_seconds"
    assert duration_rejection_reason(None) == "duration_missing_or_not_numeric"


def test_decoded_duration_is_checked_independently_of_metadata():
    assert decoded_duration_rejection(6.2, 6.3) is None
    assert decoded_duration_rejection(6.2, 6.3001) == "decoded_duration_exceeds_6.3_seconds"
    assert decoded_duration_rejection(6.3001, 6.0) == "metadata_duration_exceeds_6.3_seconds"


def test_selector_balances_configs_and_keeps_manifest_split_isolation():
    train = []
    for index in range(40):
        if index % 2:
            train.append(candidate(
                f"train-b-{index}", split="train", config="moshaf_2.0", reciter=2,
                moshaf="2.0", duration=4.0 if index % 4 else 5.8,
                sura=1 + index % 4, text=f"train_b_{index}",
            ))
        else:
            train.append(candidate(
                f"train-a-{index}", split="train", config="moshaf_0.1", reciter=0,
                moshaf="0.1", duration=4.0 if index % 4 else 5.8,
                sura=1 + index % 4, text=f"train_a_{index}",
            ))
    validation = [
        candidate("validation-a", split="validation", config="moshaf_12.0", reciter=12,
                  moshaf="12.0", sura=80, text="validation_a"),
        candidate("validation-b", split="validation", config="moshaf_13.0", reciter=13,
                  moshaf="13.0", sura=81, text="validation_b"),
        candidate("validation-c", split="validation", config="moshaf_24.0", reciter=24,
                  moshaf="24.0", sura=82, text="validation_c"),
    ]
    # Wrong-split rows are excluded; they cannot be used to fill either side.
    train.append(candidate(
        "wrong-train-split", split="validation", config="moshaf_12.0", reciter=12,
        moshaf="12.0", sura=90, text="must_not_train",
    ))
    selected_train, selected_validation, checks = select_preflight_examples(train, validation)
    assert len(selected_train) == 32
    assert len(selected_validation) == 3
    assert checks["eligible_training_candidate_count"] == 40
    assert checks["training_configuration_count"] == 2
    assert checks["training_reciter_count"] == 2
    assert {row["split"] for row in selected_train} == {"train"}
    assert {row["split"] for row in selected_validation} == {"validation"}
    assert not ({row["reciter_id"] for row in selected_train}
                & {row["reciter_id"] for row in selected_validation})
    assert not ({row["moshaf_id"] for row in selected_train}
                & {row["moshaf_id"] for row in selected_validation})
    assert not checks["quran_overlap"]["shared_sura_ids"]
    assert checks["rejected_candidate_counts"]["training_candidate_not_manifest_train"] == 1


def test_selector_fails_clearly_when_candidate_pool_is_not_larger_than_selection():
    train = [candidate(
        f"train-{index}", split="train", config="moshaf_0.1", reciter=0,
        moshaf="0.1", sura=1, text=f"train_{index}",
    ) for index in range(32)]
    validation = [candidate(
        "validation", split="validation", config="moshaf_12.0", reciter=12,
        moshaf="12.0", sura=90, text="validation",
    )]
    with pytest.raises(ValueError, match="Need more than 32 eligible training candidates"):
        select_preflight_examples(train, validation)


def test_selector_fails_if_duration_filter_leaves_too_few_candidates():
    train = [candidate(
        f"train-{index}", split="train", config="moshaf_0.1" if index % 2 == 0 else "moshaf_2.0",
        reciter=0 if index % 2 == 0 else 2, moshaf="0.1" if index % 2 == 0 else "2.0",
        duration=6.31 if index >= 30 else 6.0, sura=1, text=f"train_{index}",
    ) for index in range(40)]
    validation = [candidate(
        "validation", split="validation", config="moshaf_12.0", reciter=12,
        moshaf="12.0", sura=90, text="validation",
    )]
    with pytest.raises(ValueError, match="Need more than 32 eligible training candidates"):
        select_preflight_examples(train, validation)
