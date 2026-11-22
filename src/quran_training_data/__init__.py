"""Streaming preparation helpers compatible with the original Muaalem labels."""

from .encoding import CTC_BLANK_ID, IGNORE_INDEX, PAD_ID, LabelCodec, UnknownLabelError
from .preparation import (
    ExcludedRecord,
    InclusionPolicy,
    PreparedRecord,
    TrainingDataBuilder,
    TrainingExample,
    prepare_record,
)
from .splitting import (
    assign_record_split,
    assert_no_group_leakage,
    build_reciter_split_manifest,
    expected_random_segment_group_leakage,
)
from .batching import PaddedTargets, pad_label_batch, pad_sequences
from .statistics import DatasetStatistics

__all__ = [
    "CTC_BLANK_ID", "IGNORE_INDEX", "PAD_ID", "LabelCodec", "UnknownLabelError",
    "ExcludedRecord", "InclusionPolicy", "PreparedRecord", "TrainingDataBuilder",
    "TrainingExample", "prepare_record", "assign_record_split",
    "assert_no_group_leakage", "build_reciter_split_manifest",
    "expected_random_segment_group_leakage",
    "PaddedTargets", "pad_label_batch", "pad_sequences", "DatasetStatistics",
]
