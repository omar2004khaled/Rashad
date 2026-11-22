# Stage 7 Part A.3 — Bounded Candidate Availability Investigation

## Scope and result

This was a bounded, metadata-only investigation. No training or model initialization occurred, no audio was decoded or fetched, and no dataset/training configuration or model code was changed. Existing run directories were preserved. The scan artifacts are in [`runs/candidate_availability_20261122T154412Z`](../runs/candidate_availability_20261122T154412Z/).

**Result:** the initial 27 candidates did not establish a true dataset shortage. The bounded follow-up found **169 new training rows** that pass the selector metadata conditions and the annotation validator/codec checks, plus the 27 selector candidates recorded by the earlier preflight, for **196 selector-level metadata candidates** across inspected training windows. Thus more than 32 are available in inspected windows. This does not prove decoded audio validity or cover every planned configuration: the API scan stopped with 55 of 84 pages complete due to service errors.

## Why Stage 2 returned audio bytes, and metadata-only API support

In [`src/quran_dataset/loader.py`](../src/quran_dataset/loader.py), `DatasetLoader.load_config()` casts an audio feature to `Audio(decode=False)` when decoding is disabled. That option prevents waveform decoding; it does not strip the encoded audio field from each materialized `datasets` row. A bounded probe through the Stage 2 loader returned an audio mapping containing encoded WAV `bytes`. Therefore, using that loader for metadata-only scanning still retrieves row payloads containing audio bytes.

Metadata-only retrieval is technically supported through Hugging Face’s dataset-server `/rows` API. A live one-row request for `obadx/muaalem-annotated-v3`, configuration `moshaf_0.0`, split `train`, offset 0 returned JSON metadata and audio references (`src`, `type`) without a `bytes` field. The scanner in [`scripts/scan_candidate_availability.py`](../scripts/scan_candidate_availability.py) checks that bytes are absent and never requests/follows the audio references. This verifies the API behavior for the inspected endpoint and dataset revision; it does not establish that every HF API response will always have the same availability.

## Manifest coverage and bounded progress

[`configs/reciter_split_manifest.json`](../configs/reciter_split_manifest.json) defines 16 training reciters and 21 training moshaf configurations. The previous preflight scanned only four of those training configurations (`moshaf_0.1`, `moshaf_2.0`, `moshaf_6.0`, `moshaf_26.1`) and three validation configurations. The new scan plan includes all 21 manifest-training configurations, while avoiding previously inspected positions:

- The four previously scanned training configurations start at row offset 80, covering rows 80–159.
- Other training configurations are limited to offsets 0–79.
- `moshaf_0.0` starts at offset 1 because row 0 was already used for the API behavior probe; that probe row is not classified or counted as a candidate.
- Requests are pages of at most 20 rows, with 15-second timeouts and at most two attempts per page. At most 80 new rows are planned per configuration, except the prior four use the next 80 rows (80–159).

Plan: 84 pages / at most 1,679 new rows. Progress: **55 pages, 1,099 rows**. The final state is `INCOMPLETE_INFRASTRUCTURE`. Completed pages are persisted in `pages/`; resume skips pages already saved. Pending pages are `moshaf_28.0` offset 60; offsets 0, 20, 40, 60 for each of `moshaf_29.0`, `moshaf_3.0`, `moshaf_30.0`, `moshaf_4.0`, `moshaf_5.0`, and `moshaf_8.0`; and offsets 80, 100, 120, 140 for `moshaf_6.0`. Thus 13 configurations have all their *planned* pages complete (with the explicitly excluded `moshaf_0.0` probe row), one is partial, and seven have no new planned page completed. The earlier preflight still covers rows 0–79 for `moshaf_6.0`.

The server returned HTTP 429 and then HTTP 503 for `moshaf_28.0`, offset 60, length 20. The scanner made two finite attempts per invocation and stopped at the error; this failed page was the only page retried on resume. Both errors are recorded in [`api_errors.jsonl`](../runs/candidate_availability_20261122T154412Z/api_errors.jsonl). No further retries were made to avoid repeated requests against a throttled/unavailable service.

## Candidate and exclusion results

Counts below are from new scanned rows. Exclusion reasons are multi-label: a row can contribute to more than one reason. There were 912 `duration_exceeds_6.3_seconds`, 21 `missing_or_invalid_span`, and 21 `not_quran` reason occurrences. Other selector or annotation exclusions did not occur in the resulting counts. Candidate counts shown are metadata eligible; each of these new candidates also passed the annotation checks.

| Configuration | New rows scanned | Eligible candidates | Principal exclusions (reason occurrences) |
|---|---:|---:|---|
| `moshaf_0.0` | 79 | 12 | duration 66; span 1; not Quran 1 |
| `moshaf_0.1` | 80 | 3 | duration 77 |
| `moshaf_0.2` | 80 | 0 | duration 80; span 2; not Quran 2 |
| `moshaf_0.3` | 80 | 12 | duration 66; span 2; not Quran 2 |
| `moshaf_1.0` | 80 | 17 | duration 61; span 2; not Quran 2 |
| `moshaf_11.0` | 80 | 22 | duration 56; span 2; not Quran 2 |
| `moshaf_19.0` | 80 | 17 | duration 62; span 2; not Quran 2 |
| `moshaf_2.0` | 80 | 0 | duration 80 |
| `moshaf_2.1` | 80 | 15 | duration 63; span 2; not Quran 2 |
| `moshaf_22.0` | 80 | 17 | duration 61; span 2; not Quran 2 |
| `moshaf_26.0` | 80 | 17 | duration 60; span 3; not Quran 3 |
| `moshaf_26.1` | 80 | 8 | duration 72 |
| `moshaf_27.0` | 80 | 20 | duration 58; span 2; not Quran 2 |
| `moshaf_28.0` | 60 | 9 | duration 50; span 1; not Quran 1 |

The prior preflight’s training candidate counts are preserved without rescanning: `moshaf_0.1`: 7, `moshaf_2.0`: 0, `moshaf_6.0`: 7, `moshaf_26.1`: 13. Combining those with new windows gives 196 selector-level candidates total. The 169 new candidates all pass `validate_record()` and `LabelCodec` checks. The prior run did not save a per-configuration exclusion-reason cross-tab, so its reason breakdown cannot be supplied without rescanning rows; its saved global reason totals remain in the prior preflight artifacts.

Reasons are calculated by `classify_row()` in the scan script: manifest split must be train; `has_quran` must be true; metadata duration must be positive and at most 6.3 seconds; span and segment index must be valid; audio reference must match the manifest dataset revision; and annotations must pass the existing validator/codec. No selection rule was loosened. Duration is the largest exclusion reason in these bounded windows.

## Validation nine-cell requirement

**The nine-cell validation requirement is not achievable while retaining the 6.3-second maximum if its long-duration cell is defined as greater than 6.5 seconds.** That cell is empty by definition; changing validation examples or scanning more rows cannot satisfy it. The earlier preflight found 18 validation metadata candidates across its three validation configurations (4, 4, and 10), but that is only a bounded metadata count. This follow-up scanned training configurations only. The candidate counts do not verify decoded audio duration or audio integrity. A different validation design would need explicit review; this investigation did not change it.

## Artifacts and reproducibility

The new run directory contains the exact `scan_plan.json`, `scan_progress.json`, `candidate_availability.json`, per-page summaries under `pages/`, API failures in `api_errors.jsonl`, and a run-local summary report. Rows preserve config and row position, source key, reciter/moshaf identifiers, metadata duration, Quran/span and annotation status, audio-reference presence, candidate status, and exclusion reasons. The original audio references are not copied into row summaries and the API audio URLs were never fetched.

To resume later without repeating successful pages, run from the project root:

```powershell
python scripts/scan_candidate_availability.py --resume-run-dir runs/candidate_availability_20261122T154412Z
```

A resume should be considered only after the HF dataset-server rate limit/availability issue has cleared. It remains bounded by the same page plan and retry cap.

## Unknowns and limitations

- The scan is incomplete because of dataset-server 429/503 responses; seven configurations remain unscanned in their planned windows and one configuration is partial.
- Dataset-server `duration_seconds` is metadata. No audio was downloaded, decoded, or checked for actual duration, channels, sample rate, corruption, or endpoint silence.
- The 196 figure is selector-level metadata evidence, not a final training-ready set after audio loading and any training-time checks. The stronger 169 count includes annotation schema/codec validation but still not audio validation.
- No data beyond the bounded windows was inspected. The scan establishes that the earlier count of 27 was insufficiently covered, not that every eligible item in the full dataset has been counted.
- The historical preflight reason counts are available only in aggregate, not by configuration.
- The prior validation candidates do not prove the three-by-three validation matrix is feasible; under the stated >6.5-second long cell and 6.3-second cap, that matrix is formally infeasible.

## Implications for the next stage

The original shortage concern is not supported by the inspected metadata: a pool above 32 was found without audio payload retrieval, and all 169 newly observed candidates have labels that pass current validation/encoding checks. Before a pilot, resolve the validation-cell definition under the unchanged duration cap, wait for API access to recover if full planned coverage is still needed, and perform the already authorized audio-level checks on a deliberately bounded selection. No Stage 7 training run is authorized or performed here.
