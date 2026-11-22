"""Prepare a bounded set of real WAV-backed rows through the Stage 2 loader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quran_audio import AudioProcessingConfig, analyze_audio  # noqa: E402
from quran_dataset import DatasetConfig, DatasetLoader  # noqa: E402
from quran_training_data import LabelCodec, InclusionPolicy, assert_no_group_leakage, prepare_record  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configurations", nargs="+", default=["moshaf_0.1", "moshaf_1.0", "moshaf_26.1"])
    parser.add_argument("--limit", type=int, default=1, help="Maximum rows per configuration (1–5)")
    args = parser.parse_args()
    if not 1 <= args.limit <= 5:
        parser.error("--limit must be between 1 and 5")
    dataset_config = DatasetConfig.from_file(ROOT / "configs" / "dataset.json")
    loader = DatasetLoader(dataset_config)
    audio_config = AudioProcessingConfig.from_file(str(ROOT / "configs" / "audio_preprocessing.json"))
    policy_raw = json.loads((ROOT / "configs" / "training_data.json").read_text(encoding="utf-8"))["inclusion_policy"]
    policy = InclusionPolicy(**policy_raw)
    manifest = json.loads((ROOT / "configs" / "reciter_split_manifest.json").read_text(encoding="utf-8"))
    codec = LabelCodec()
    outputs = []
    leakage_items = []
    for config in args.configurations:
        for row in loader.iter_samples(config, limit=args.limit):
            source_annotations = {key: row.get(key) for key in ("phonemes", "sifat", "imlaey", "uthmani")}
            result = prepare_record(row, config, audio_config=audio_config, policy=policy,
                                    codec=codec, split_manifest=manifest)
            assert source_annotations == {key: row.get(key) for key in source_annotations}
            if result.excluded:
                outputs.append({"configuration": config, "source_key": result.excluded.source_key,
                                "included": False, "reasons": result.excluded.reasons})
                continue
            example = result.example
            assert example is not None
            assert codec.decode_phonemes(example.labels["phonemes"]) == example.phoneme_text_for_encoding
            assert len(example.phoneme_groups) == len(example.labels["ghonna"])
            assert len(example.phoneme_groups) == len(example.group_phoneme_ids)
            audio = analyze_audio(example.audio)
            outputs.append({
                "configuration": config,
                "source_key": example.source_key,
                "split": example.split,
                "included": True,
                "has_quran": example.metadata["has_quran"],
                "phoneme_source_chars": len(example.phoneme_text),
                "phoneme_target_ids": len(example.labels["phonemes"]),
                "sifat_groups": len(example.phoneme_groups),
                "group_alignment": len(example.phoneme_groups) == len(example.labels["ghonna"]),
                "audio": {"sampling_rate": audio.sampling_rate, "channels": audio.channels,
                          "duration_seconds": audio.duration_seconds, "dtype": str(example.audio.samples.dtype),
                          "operations": [{"name": op.name, "applied": op.applied}
                                         for op in example.audio.operations]},
                "metadata_trace": {key: example.metadata[key] for key in ("moshaf_id", "reciter_id", "segment_index")},
            })
            leakage_items.append((example.source_key, row, example.split or ""))
    # The checked-in manifest is validated across every group/set, while these
    # sampled rows additionally verify runtime row-to-manifest resolution.
    assert_no_group_leakage(leakage_items)
    if not outputs:
        raise RuntimeError("No real records were prepared")
    print(json.dumps({"records_checked": len(outputs), "outputs": outputs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
