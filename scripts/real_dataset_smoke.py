"""Fetch a few public records and check audio/metadata access end-to-end."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quran_dataset import DatasetConfig, DatasetLoader, decode_audio, inspect_record  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = DatasetConfig.from_file(ROOT / "configs" / "dataset.json")
    configuration = args.configuration or config.selected_configurations[0]
    limit = config.sample_limit if args.limit is None else args.limit
    if limit < 1:
        parser.error("--limit must be at least 1")

    loader = DatasetLoader(config)
    rows = list(loader.iter_samples(configuration, split=config.default_split, limit=limit))
    if not rows:
        raise RuntimeError("The stream returned no records")

    sample = rows[0]
    schema = loader.inspect_schema(configuration)
    if not schema.verified:
        raise RuntimeError(f"Schema unavailable for {configuration}: {schema.error}")

    audio = decode_audio(sample)
    metadata = loader.metadata_index()
    joined = metadata.join(sample)
    if joined.moshaf_metadata is None or joined.reciter_metadata is None:
        raise RuntimeError("Metadata join did not resolve both source IDs")

    print(json.dumps({
        "configuration": configuration,
        "rows_accessed": len(rows),
        "schema_verified": schema.verified,
        "field_count": len(schema.features),
        "sample": inspect_record(sample),
        "audio": {
            "sampling_rate": audio.sampling_rate,
            "channels": audio.channels,
            "duration_seconds": audio.duration_seconds,
            "shape": list(audio.samples.shape),
        },
        "moshaf_metadata_id": joined.moshaf_metadata.get("id"),
        "reciter_metadata_id": joined.reciter_metadata.get("id"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
