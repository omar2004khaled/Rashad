"""Validate a small, bounded number of public dataset rows across configs."""
from __future__ import annotations

import argparse
import json
import time

import requests

from quran_annotations import validate_record

DATASET = "obadx/muaalem-annotated-v3"


def fetch_rows(configuration: str, offset: int = 0, length: int = 5) -> list[dict]:
    params = {"dataset": DATASET, "config": configuration, "split": "train", "offset": offset, "length": length}
    for attempt in range(4):
        try:
            response = requests.get("https://datasets-server.huggingface.co/rows", params=params, timeout=90)
            response.raise_for_status()
            return [item["row"] for item in response.json()["rows"]]
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configurations", nargs="+", default=["moshaf_0.1", "moshaf_1.0", "moshaf_26.1"])
    parser.add_argument("--limit", type=int, default=4, help="Rows per configuration; bounded, no full download")
    args = parser.parse_args()
    if not 1 <= args.limit <= 100:
        parser.error("--limit must be between 1 and 100")
    failed = 0
    for config in args.configurations:
        rows = fetch_rows(config, length=args.limit)
        for row in rows:
            result = validate_record(row)
            failed += not result.is_valid
            print(json.dumps({
                "configuration": config,
                "segment_index": row.get("segment_index"),
                "has_quran": row.get("has_quran"),
                "has_istiaatha": row.get("has_istiaatha"),
                "has_bismillah": row.get("has_bismillah"),
                "has_sadaka": row.get("has_sadaka"),
                "is_valid": result.is_valid,
                "errors": [issue.code for issue in result.errors],
                "warnings": [issue.code for issue in result.warnings],
                "sifat_count": len(row.get("sifat") or []),
            }, ensure_ascii=False))
    if failed:
        raise SystemExit(f"{failed} real record(s) have validation errors")


if __name__ == "__main__":
    main()
