# Stage 7 Part A.3 — Pilot Safety and Preflight Report

## Status: INCOMPLETE

The failure-safe monitoring, strict duration checks, test coverage, and selection-only command are implemented. The bounded selection-only preflight ran through the Stage 2 streaming loader and stopped before audio decode because it found only 27 eligible training candidates, fewer than the required pool of 33. No model was initialized, no optimizer update was run, and the prior failed run was left untouched.

## Implemented changes

### Failure-safe training telemetry

`src/quran_training/engine.py` now records phase, global step, example ID, and duration in monitor samples. For any exception in the monitored validation/training/checkpoint path—including `ResourceLimitReached` and `torch.cuda.OutOfMemoryError`—it stops the monitor and atomically writes `failure_report.json` before re-raising the original exception. The report includes the exception, step/context, current and peak PyTorch allocated/reserved bytes, GPU name/index/capacity, configured thresholds, current process and host-memory values, and sampled GPU/process/host telemetry. If telemetry persistence itself fails, that secondary failure is attached as an exception note; the training exception still propagates. Successful-run summaries remain aggregate summaries and do not include the full sample array.

The monitor now queries the configured CUDA device index rather than hard-coding device 0. For CPU runs, NVIDIA polling is disabled.

### Strict duration enforcement

`scripts/run_training_pilot.py` now rejects metadata candidates over 6.3 seconds and independently rejects decoded clips over 6.3 seconds. A metadata/decoded duration mismatch over 0.15 seconds is also rejected. Rejected examples carry source key, both durations when available, and a reason. The selector does not replace a rejected record; an incomplete selected set remains incomplete.

### Selection-only preflight

Run with:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_training_pilot.py --preflight-only
```

This path uses `DatasetLoader.iter_samples()` from `src/quran_dataset/loader.py`, streaming at most 80 rows from each of the seven configured train-split configurations. The approved reciter split manifest assigns each row to its train or validation group. Progress and exclusions are written as each configuration is scanned. The selector requires 32 training rows, a candidate pool larger than 32, at least two manifest-training reciters/configurations, and 1–3 manifest-validation examples. It will not reassign rows or weaken the split/duration checks to fill a quota.

The Stage 2 loader returns encoded `audio["bytes"]` on streamed rows even when audio decoding is disabled. Consequently this bounded scan streamed the encoded payloads for 560 rows, but decoded **zero** examples because the candidate pool was insufficient. It did not load the complete dataset. This is a known cost of the available Stage 2 streaming interface; the preflight avoids any additional audio download and keeps the row scan capped.

## Preflight result

Artifacts are in the new directory [pilot_preflight_20261122T151943Z](C:/Users/OMAR/Downloads/Rashad/runs/pilot_preflight_20261122T151943Z/). The original run `runs/controlled_pilot_20261122T144300Z/` was not changed.

- Scanned 80 rows per configuration, 560 rows total.
- Training candidates at or below 6.3 seconds: **27** (required: more than 32).
  - `moshaf_0.1`, reciter 0: 7
  - `moshaf_2.0`, reciter 2: 0
  - `moshaf_6.0`, reciter 6: 7
  - `moshaf_26.1`, reciter 26: 13
- Validation candidates at or below 6.3 seconds: **18**.
  - `moshaf_12.0`, reciter 12: 4
  - `moshaf_13.0`, reciter 13: 4
  - `moshaf_24.0`, reciter 24: 10
- Candidate duration groups across both splits: 6 at 0–5 seconds and 39 over 5–6.3 seconds.
- Exclusions: 500 exceeded the duration cap; 15 were non-Quran records. No selected examples were decoded because the training-pool prerequisite failed.
- The selector stopped with `Need more than 32 eligible training candidates; found 27`. It did not select or silently substitute examples.

The selection artifact, JSON summary, progress file, and readable scan report are `selected_examples.json`, `preflight_result.json`, `preflight_progress.json`, and `preflight_report.md` in that directory. No final split/leakage check over a selected train/validation set could be completed because no set was selected. Candidate manifest assignments were counted according to the existing reciter-based manifest; all source rows came from the dataset's published `train` split.

## Scheduler calculation and discrepancy

The scheduler implementation and behavior were not changed. `total_optimizer_steps(32, 16, 1)` is exactly 2. With configured `warmup_ratio=0.05`, `ceil(2 * 0.05)` gives one warmup step. Under the existing `LambdaLR` implementation, the learning rates used by the two optimizer steps are:

1. Step 1 uses LR 0 because the scheduler initializes at the start of its one warmup step.
2. After a successful step 1, the scheduler advances to LR `1e-5`; step 2 uses this nonzero LR.
3. After successful step 2, the two-step schedule decays the next LR to 0.

`engine.py` advances the scheduler and global step only when `optimizer_update_succeeded(...)` is true. Focused tests cover scaler overflow detection and this exact two-step LR sequence.

One run-configuration mismatch remains: `scripts/run_training_pilot.py` still declares `PILOT_EPOCHS = 2`. If that value were used with 32 examples and accumulation 16, it would plan 4 updates, not 2. This preflight did not use a scheduler or alter training configuration. A future optimizer experiment that intends exactly two planned updates must explicitly use one run epoch; no such experiment was started here.

## Tests and verification

- Focused tests: **14 passed** (`tests/test_pilot_preflight.py` and `tests/test_training_loop.py`). These cover metadata/decoded duration boundaries, selector split and leakage behavior, insufficient candidates, failure telemetry persistence, preservation of the original exception when telemetry writing fails, and the two-update LR schedule.
- Full project suite: **109 passed** (`pytest -q`).
- `python -m compileall -q src scripts/run_training_pilot.py tests`: passed.
- `git diff --check`: passed except for Git's existing LF-to-CRLF working-copy warnings in previously modified files.
- No training command, resume, model initialization, checkpoint download, or architecture/configuration change was performed.

## Uncertainties and next step

The 80-row-per-configuration scan is intentionally bounded and does not establish that the full dataset lacks a larger qualifying pool. It establishes only that this scan window did not provide enough candidates. If a later preflight is approved, the scan bound and progress/resume approach should be reviewed before increasing it; the current loader streams encoded audio bytes with metadata. Any larger scan should remain capped, preserve progress, and stop clearly on infrastructure failures. Decoded-duration enforcement is covered by synthetic boundary tests but was not exercised on real selected examples in this run because selection stopped before decoding.

**Decision:** INCOMPLETE. No optimizer-update experiment should proceed from this preflight result. Wait for review before expanding the candidate scan or starting training.
