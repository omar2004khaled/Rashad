# Dataset loading pipeline

## Scope

This stage adds a read-only Python access layer for `obadx/muaalem-annotated-v3`. The default behavior is streaming with a small configured sample limit and audio decoding disabled in `datasets`. It preserves the record's annotation fields and does not preprocess audio, edit labels, tokenize, split, or train.

Stage 1 findings remain the factual source for inspected dataset contents; see [dataset_exploration_report.md](dataset_exploration_report.md). In particular, 29 configurations expose `train`; 27 are recitation sets and two are metadata sets. `segment_index` is set-local, so `(moshaf_id, segment_index)` is a candidate key only after checking both fields are present. Six configurations' dataset-server `first-rows` schema requests failed; this loader never copies a schema from another configuration. Stage 2 independently checked those six through the `datasets` streaming backend; all six returned their own 24-field schema during notebook execution.

## Architecture

- `src/quran_dataset/loader.py` contains `DatasetConfig`, `DatasetLoader`, metadata lookup/join types, row inspection, and audio decoding.
- `configs/dataset.json` records the dataset id, available and selected configurations, metadata configuration names, default split, sample limit, streaming/audio settings, shuffle seed/buffer, and Stage 1 schema endpoint limitations.
- `tests/test_loader.py` exercises the code with in-memory fixtures and mocked dataset access; no test needs the full dataset.
- `notebooks/02_dataset_loading.ipynb` performs a small real-dataset demonstration.

The original project `src/quran_muaalem/modeling/multi_level_tokenizer.py` tokenizes a phonetic script and sifat fields for model use. Its `docs/en/training/data-sources.md` and `docs/en/training/pipeline.md` describe how annotations were expected to be produced. This loader stops before those transformation/tokenization stages and returns the published fields intact.

## Installation

Python 3.10 or newer is required. The dependency versions in `pyproject.toml` use compatible version ranges. From the repository root:

```powershell
python -m pip install -e ".[dev]"
```

Core dependencies are `datasets`, `huggingface_hub`, `soundfile`, `numpy`, and `pandas`; `pytest`, `nbclient`, and `nbformat` are in the `dev` extra. The loader disables `datasets` audio decoding by default, so it does not require the optional `torchcodec` decoder.

## Load a few rows

```python
from pathlib import Path
from quran_dataset import DatasetConfig, DatasetLoader, inspect_record

config = DatasetConfig.from_file(Path("configs/dataset.json"))
loader = DatasetLoader(config)

for row in loader.iter_samples("moshaf_0.1", split="train", limit=3):
    print(inspect_record(row))  # never prints the waveform or encoded bytes
```

`load_config(configuration, split, streaming)` opens one explicit configuration. `streaming` defaults to `true`; set `streaming=False` only when the caller intentionally wants a materialized dataset (which can download the selected configuration's shards). No code path automatically loads every configuration. `iter_samples` is deterministic in source order by default. Its omitted `limit` uses the small configured `sample_limit`; pass `limit=None` to iterate the full selected stream. `get_sample(config, index)` supports indexed datasets and walks the stream to the index when streaming is enabled.

Use selected configurations from JSON or pass an explicit sequence:

```python
for item in loader.iter_configurations(
    ["moshaf_0.1", "moshaf_1.0"], limit_per_configuration=2
):
    print(item.configuration, item.record["segment_index"])
```

The wrapper keeps configuration identity separate; `item.record` itself retains the source record without an injected config key. Set `shuffle=True` for reproducible selection from a streaming shuffle buffer; a seed and buffer size are recorded in config. Streaming shuffle is deterministic for the same dataset revision, seed, buffer, and iteration order, but is not a globally uniform permutation.

## Schema inspection and failures

`loader.inspect_schema(configuration)` opens that configuration with streaming and returns `SchemaInspection(verified, features, error)`. On failure, `features` is `None` and the original configuration-specific error is recorded. The loader does not substitute a schema from a different set. Stage 1's HF datasets-server first-row endpoint failed for `moshaf_0.0`, `moshaf_5.0`, `moshaf_11.0`, `moshaf_19.0`, `moshaf_26.0`, and `moshaf_30.0`. Stage 2's notebook successfully queried each of these configurations through `datasets` streaming and saw its own 24-field schema. This distinction matters: a viewer endpoint error did not imply that the streaming schema path was inaccessible in this run. `DatasetAccessError` includes the requested dataset, config, split, and streaming mode when loading fails.

Unknown configuration names are rejected locally using the explicit list in JSON. A split error from Hugging Face is reported with the requested configuration; it does not trigger a retry against another split.

## Metadata lookups and joins

```python
metadata = loader.metadata_index()
record = loader.get_sample("moshaf_0.1", 1)
joined = metadata.join(record)
print(joined.moshaf_metadata)
print(joined.reciter_metadata)
```

`load_metadata("moshaf")` and `load_metadata("reciters")` load the small metadata configurations. `MetadataIndex.lookup_moshaf` keys mushaf rows by metadata `id` and recitation records by `moshaf_id` (both stringified); `lookup_reciter` keys reciter rows by `id`/`reciter_id`. The joined values live alongside the original mapping; the record is not changed. Missing joins return `None`. `candidate_record_id` returns `(moshaf_id, segment_index)` only when both components exist and are non-empty; corpus-wide uniqueness is not asserted.

## Audio access

By default, records contain encoded WAV bytes (as returned by `Audio(decode=False)`) rather than triggering `datasets`' optional audio decoder. Decode on demand:

```python
from quran_dataset import decode_audio

audio = decode_audio(record)
print(audio.sampling_rate, audio.channels, audio.duration_seconds)
print(audio.samples.shape)
```

`decode_audio` returns `AudioData` with samples, rate, channels, and duration. It uses `soundfile` and makes no resampling, normalization, trimming, denoising, or label changes. For encoded files, soundfile returns a `(frames, channels)` array in float32. For already-decoded Hugging Face arrays it preserves the supplied NumPy array and sampling rate; the dataset samples inspected in Stage 1 are mono. Malformed/missing audio raises `InvalidAudioError` rather than yielding fabricated metadata.

## Tests and notebook

Run unit tests:

```powershell
python -m pytest
```

Run the real-data smoke check (three records by default):

```powershell
python scripts/real_dataset_smoke.py --configuration moshaf_0.1 --limit 3
```

Run the loading notebook from the repository root:

```powershell
jupyter nbconvert --to notebook --execute notebooks/02_dataset_loading.ipynb --inplace
```

The smoke script loads at most the requested number of rows from one set, verifies that set's schema, decodes one short WAV, and joins both metadata records. The notebook loads three rows from one selected recitation set, decodes one short WAV, loads the two small metadata tables, joins one sample, and checks each of the six Stage 1 first-row-failure configurations independently. It then demonstrates iteration over a few rows. It reports each network/schema error rather than silently dropping the relevant config.

## Known limitations

- The dataset is about 97.3 GB. Streaming avoids fetching all shards; even streaming must fetch the needed Parquet ranges and the selected audio payloads.
- Hub streaming can be slow or rate-limited, especially without authentication. A `HF_TOKEN` may improve rate limits; it is not required for public data.
- In the standalone Windows smoke run, a Hugging Face/fsspec background request printed a `WinError 10038` retry warning after the successful JSON result. The requested rows, schema, decoded audio, and metadata joins had completed; this looks like a stream/socket cleanup warning, but should be watched on longer-lived runs.
- There is only a `train` split in the inspected dataset metadata. No split creation is performed here.
- The six Stage 1 datasets-server first-row errors may change over time. Their schemas were verified through `datasets` streaming during Stage 2, but use `inspect_schema` per configuration at runtime because availability can change.
- A single streamed index lookup walks preceding records and may be slow. Prefer iteration for small samples.
- Audio sample count and duration are computed from the decoded representation. No corpus-wide check of invalid audio, rates, identifiers, empty annotations, or label alignment has been run.
- Records are not filtered: non-Quran segments and null labels/spans remain as published.

## Stage 2 verification results

- Unit tests: **21 passed** using local fixtures/mocks.
- Real-data smoke command: `python scripts/real_dataset_smoke.py --configuration moshaf_0.1 --limit 3`. It accessed 3 recitation rows, verified a 24-field schema, decoded the first row as `(96960, 1)` float32 samples at 16 kHz (6.06 seconds), and joined `moshaf_id="0.1"` and `reciter_id=0` to metadata rows with matching IDs.
- Notebook: executed end-to-end with no cell errors. It accessed three rows for display, two additional rows for iteration, the two small metadata configurations (27 + 22 rows), and decoded one WAV. It performed schema-only inspection of the six Stage 1 `first-rows` endpoint failures; all six returned 24-field schemas through the `datasets` streaming backend during this run.
- Audio preservation test: decoded samples exactly matched `soundfile.read(..., dtype="float32", always_2d=True)` on the same WAV bytes. The decoded rate and channel count were unchanged; no transformation code is applied.
- No full Parquet shard set or full dataset was requested by the smoke script or notebook. Streaming may fetch Parquet byte ranges needed for the selected rows, and local network/cache behavior depends on the Hugging Face client.
