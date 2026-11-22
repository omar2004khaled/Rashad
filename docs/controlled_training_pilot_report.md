# Controlled Full Fine-Tuning Pilot (Stage 7 Part A.3)

## Outcome

The reusable training loop and focused tests are implemented. The bounded real-data pilot **did not complete an optimizer update**: its first backward pass hit the configured GPU memory stop on a 9.68-second clip. The process exited safely before an optimizer step. A shorter retry could not be selected reliably because the Hugging Face rows endpoint began stalling/rate-limiting during the subsequent bounded metadata scan. The A.3 completion criteria are therefore not met, and this report does not recommend proceeding to Part B yet.

No full epoch or full-dataset pass was run. The original checkpoint was not overwritten. The run's initial checkpoint and selected-example metadata are under `runs/controlled_pilot_20261122T144300Z/`.

## Training loop

`src/quran_training/engine.py` implements full-model training with configured Adafactor, gradient accumulation, FP16 autocast/GradScaler, gradient checkpointing, clipping, a linear warmup/decay scheduler, validation, periodic checkpoints, resume state, and metrics. It checks that every model parameter is trainable and, when configured, has a gradient. It logs per-update loss, gradient norm, scaler scale, learning rate, elapsed time, throughput, and memory samples. Validation verifies its split and runs without gradients.

The scheduler derives planned steps as `ceil(selected_examples / accumulation_steps) * epochs`, uses the configured `warmup_ratio`, advances only when GradScaler did not skip the optimizer step, and saves/restores its state with the checkpoint. With the 96-example, two-epoch plan and accumulation 16, the planned schedule is 12 updates with one warmup step. The attempted 80-example main segment planned 10 updates; the run stopped before step 1, so no schedule update was observed in this pilot.

`scripts/run_training_pilot.py` uses the original checkpoint revision `01a1ef9fbe40d144ef845101e89ff924aed3fef5`, requires an exact 605,754,251-parameter checkpoint load, and checks the dataset revision against `configs/reciter_split_manifest.json`. It selects only Quran-containing rows and validates all eleven labels through the Stage 5 preparation path. Training and validation suras and exact normalized Imlaey strings are checked for overlap.

## Selection and attempted run

The completed first selection used four training configurations/reciters and three manifest-held-out validation configurations/reciters. It contained 122 training candidates and 9 validation examples; the consumption cap was 80 examples and five main updates, followed by a planned resume update. Training and validation sura sets were disjoint, with zero exact normalized Imlaey overlaps. The selection manifest records each example's source configuration, reciter, duration, sample rate, channels, sura/verse keys, and eleven-level-label check in `runs/controlled_pilot_20261122T144300Z/selected_examples.json`.

Decoded training durations ranged from 4.28 to 19.975 seconds; validation durations ranged from 7.71 to 16.245 seconds. The first training microbatch included a 9.68-second recording. After forward/backward, the configured reserved-memory guard observed 6,507,462,656 bytes against a 6,441,926,656-byte device total (about 101%). The guard raised `ResourceLimitReached`; no optimizer update completed, no batch was silently skipped, and `metrics.jsonl` remained empty. No A.3 validation loss, optimizer-state resume, update time, gradient norm, or parameter-update result is claimed.

The attempted run selected multiple reciters/configurations and a duration range, but that range proved too risky on this 6 GiB laptop GPU. An offline check of its selected manifest found only 25 of 122 training examples at or below eight seconds (19 at or below seven seconds); only one of the nine validation examples was at or below eight seconds. The planned 96-example resumed pilot therefore cannot be recreated safely from that saved selection alone. A subsequent bounded rows scan stalled after individual requests had succeeded, so it did not establish a new representative shorter selection. No audio was cropped to force fit.

## Effective configuration

The effective project configuration in `configs/training.json` is full fine-tuning, Adafactor, LR `1e-5`, weight decay `0.01`, linear scheduler, warmup ratio `0.05`, accumulation 16, max grad norm 1, FP16 mixed precision, gradient checkpointing enabled, and micro-batch size 1. `src/quran_training/config.py` constructs Adafactor with `relative_step=False` and `scale_parameter=False`, so the configured fixed `1e-5` learning rate is used; weight decay is passed as configured.

The attempted pilot used all model parameters trainable, FP16 autocast with GradScaler, gradient checkpointing, micro-batch 1, and the existing accumulation value. Its safety limits were 90% PyTorch reserved memory, 5,700 MiB NVIDIA sampled use, and a 2.5 GiB minimum available host RAM. The reserved-memory threshold was checked after a microbatch phase, so it stopped after allocation crossed its limit rather than forecasting the peak. This behavior avoided continuing into an optimizer update but did not prevent the transient from exceeding the configured fraction.

## Verification

- The first run loaded the original checkpoint completely and saved `checkpoints/initial.pt` with the initial model/optimizer/scheduler/scaler state.
- The GPU safety guard fired during the first backward, before an update. Therefore encoder/head updates, finite post-update state, and A.3 resume continuation were not verified in a live pilot.
- Existing A.3 unit coverage exercises scheduler step counts, overflow skip behavior, accumulation, validation non-mutation, and checkpoint resume. The focused training tests passed (13 tests).
- Full project test suite: **101 passed** (`pytest -q`).
- The audited A.2 benchmark separately completed 12 updates and tested full checkpoint restoration; those results are historical A.2 evidence, not a successful A.3 pilot.

Commands used: `python scripts/run_training_pilot.py` for the bounded real-data selection and attempted pilot; `pytest tests/test_training_loop.py tests/test_training_checkpointing.py -q` for focused tests; `pytest -q` for the complete project suite; `python -m compileall -q src/quran_training scripts/run_training_pilot.py` and `git diff --check` for syntax/whitespace checks.

## Limitations and next decision

The pilot script's retry selector now caps candidates at eight seconds and allocates across available reciter/configuration and duration groups, with a 128-example selection pool for the 96-example consumption cap. This prevents its earlier hard-coded duration quotas from requiring clips above the safer cap. The retry selector itself could not be exercised against a fresh complete candidate scan because the rows API stalled. The memory guard still checks sampled/current usage at phase boundaries rather than predicting the forward peak; this remains a concrete improvement before another GPU attempt. A safe retry needs a verified, leakage-checked set with enough short/medium examples for multiple updates, plus validation clips compatible with the same memory ceiling. The rows API availability prevented verifying such a replacement selection in this run.

Adafactor and checkpoint recovery passed the separate A.2 benchmark, but Stage A.3 has not yet established sustained end-to-end training, live validation behavior, or resume after actual pilot updates. The current evidence is **not sufficient to approve a longer controlled training pilot or begin Part B**. Resolve the bounded-data selection and safe memory envelope, then rerun the controlled pilot for several successful updates before making that decision.
