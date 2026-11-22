"""Create a small source-revision-pinned split map from public metadata only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quran_training_data.splitting import build_reciter_split_manifest  # noqa: E402

DATASET = "obadx/muaalem-annotated-v3"


def get_json(url: str, *, params: dict | None = None) -> dict:
    for attempt in range(4):
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "configs" / "reciter_split_manifest.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    revision = get_json(f"https://huggingface.co/api/datasets/{DATASET}")["sha"]
    size = get_json("https://datasets-server.huggingface.co/size", params={"dataset": DATASET})["size"]
    row_counts = {
        config["config"].removeprefix("moshaf_"): int(config["num_rows"])
        for config in size["configs"]
        if config["config"].startswith("moshaf_")
        and config["config"] not in {"moshaf_metadata", "moshaf_metadata"}
    }
    metadata = get_json("https://datasets-server.huggingface.co/rows", params={
        "dataset": DATASET, "config": "reciters_metadata", "split": "train", "offset": 0, "length": 100,
    })
    reciters = [item["row"] for item in metadata["rows"]]
    manifest = build_reciter_split_manifest(
        reciters, row_counts, dataset_revision=revision, seed=args.seed,
        validation_groups=3, test_groups=3,
    )
    if manifest["source_row_count_total"] != sum(row_counts.values()):
        raise RuntimeError("reciter metadata does not cover every recitation configuration")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in (
        "dataset_revision", "source_row_count_total", "actual_source_row_counts",
        "reciter_id_to_split", "moshaf_id_to_split",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
