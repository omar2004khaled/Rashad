# Stage 7 Part A.3 — RAM Optimization and Controlled Retry

## Final status

**PART A.3 INCOMPLETE**

The single controlled retry completed one genuine optimizer update at nonzero learning rate and changed the encoder projection plus all eleven output heads. It stopped during accumulation for the resumed second update when available system RAM reached 1.97 GiB, below the unchanged 2 GiB floor. The GPU guards remained satisfied. No further training attempt was made, and Part B was not started.

Artifacts are in [`runs/pilot_a3_updates_20261122T173644Z`](../runs/pilot_a3_updates_20261122T173644Z/). The earlier run at `runs/pilot_a3_updates_20261122T164735Z/` remains unchanged.

## Host RAM diagnosis

### Measured host-memory timeline

The new pilot used a 200 ms process/system sampler starting before loading the selected examples. The trainer separately sampled RSS and available RAM every 100 ms during each fit. The values below come from the new run’s `host_memory_telemetry.json`, successful `first_half` telemetry, and `resumed/failure_report.json`.

| Phase | Available RAM minimum | Process RSS peak | Notes |
|---|---:|---:|---|
| Pilot start / selected examples | 10.81 GiB | 0.78 GB | Outer monitor’s first sample |
| Feature extractor load | 10.22 GiB | 0.80 GB | Three outer samples |
| First base checkpoint load | 4.99 GiB | 3.32 GB | Model was placed directly on CUDA; safetensors were read one tensor at a time from CPU |
| First fit, including validation, backward, gradient transfer, optimizer, save and final validation | 2.02 GiB outer monitor; 2.03 GiB trainer monitor | 4.83 GB | One update completed; minimum was close to the 2 GiB floor |
| Release first trainer/model/optimizer | 5.10 GiB | 2.63 GB | Sampled after references were deleted and garbage collection ran |
| Fresh base checkpoint model load for resume | 2.46 GiB | 4.92 GB | This phase needlessly loaded the base weights before the step-1 checkpoint replaced them |
| Resumed validation | 2.03 GiB | 5.38 GB | Validation completed before resumed training |
| Resumed forward/backward and CPU gradient transfer | **1.97 GiB** | **7.89 GB** | RAM guard stopped at global step 1; no second optimizer update |

The host had 31.69 GiB total visible RAM and 10.81 GiB available at the sampler’s first reading. The first run’s minimum available RAM was 2,177,716,224 bytes (2.03 GiB), just above the floor. In the resumed failure report it was 2,118,594,560 bytes (1.97 GiB), with current RSS 7,894,990,848 bytes. The exact failure was `ResourceLimitReached: System available RAM is 1.97 GiB, threshold is 2.00 GiB` during `gradient_offload`, example `moshaf_11.0:11.0:002.0009` (3.305 seconds). The failure report’s last CUDA snapshot records `after_gradient_offload`, and its current allocated/reserved GPU memory had returned to approximately 2.77/2.80 GB. The stop was not a GPU limit or CUDA OOM.

The saved first-run trainer summary does not retain its raw host samples or separate host-memory minima for every inner phase. The outer monitor labels that entire interval `first_update_and_validation`; phase-specific host minima for its forward, backward, transfer, optimizer, validation, and checkpoint-save subphases cannot be recovered. The resumed failure telemetry does retain detailed samples and phase labels.

### Gradient and checkpoint memory accounting

Model parameters are FP32 in this implementation; FP16 autocast does not change their stored parameter dtype. The configured full-fine-tuning model has 605,754,251 parameters, so a complete FP32 gradient set is **2,423,017,004 bytes (2.26 GiB)**. That is code-derived, not directly measured by a CPU allocation counter. The saved telemetry shows the failure occurring in `gradient_offload`; after the gradient transfer, GPU allocated memory fell from about 5.19 GB to 2.77 GB. The likely immediate pressure is therefore the 2.26 GiB CPU gradient accumulator on top of the resumed process’s host working set. This is supported by phase/timing and the exact gradient size, but no heap profiler was attached, so it does not account for every host byte.

Gradient ownership works as follows:

- Each backward creates scaled FP32 parameter gradients on CUDA.
- The helper copies one parameter gradient to CPU, adds into that parameter’s accumulated CPU tensor, then sets the CUDA `.grad` to `None`. There is one complete accumulated CPU set; during each individual copy/add, source and destination for that one parameter coexist briefly. The implementation does not create a second full CPU gradient set.
- Restore now removes each CPU accumulator entry as it copies that parameter’s gradient back to its model device. This avoids keeping the entire CPU accumulator alive after a second complete CUDA gradient set has been restored.
- `optimizer.zero_grad(set_to_none=True)` runs before every accumulation group. The optimizer and scheduler advance only when GradScaler confirms the optimizer step was not skipped.

No microbatch gradients are intentionally discarded. The mathematical test compares ordinary accumulated gradients to offload/restore accumulation; the CUDA test additionally verifies CPU storage during accumulation, CUDA placement after restore, equality within PyTorch tolerance, empty accumulator ownership after restore, and cleared gradients after `zero_grad`.

Checkpoint behavior and remaining memory attribution:

- Resume uses `torch.load(..., mmap=True, map_location="cpu")`. Model and optimizer states are then copied/restored, and the mapped checkpoint object is deleted before forward. This avoids eagerly allocating every checkpoint storage in the CPU heap, although accessed file-backed pages can still enter the process working set.
- The retry still constructed a fresh model by loading the base checkpoint before restoring the step-1 training checkpoint. The measured resume-model-load phase reduced available RAM from 5.10 GiB after first-run cleanup to 2.46 GiB and reached 4.92 GB process RSS. This phase is an avoidable loading pass; it is strongly associated with checkpoint page/tensor staging, but the telemetry cannot distinguish PyTorch staging from Windows file-cache/page residency.
- Validation uses `model.eval()` and `torch.inference_mode()`, checks that parameter version counters did not change, and releases its local batch/output references when `evaluate()` returns. No collection of all validation outputs is retained.
- Training deletes output, batch, and loss references after each backward. The Stage 5 audio NPZ is now opened with a context manager and closed after sample materialization.
- Windows pagefile use and per-process memory for other applications were not sampled during the retry. No historical evidence establishes that paging or another process caused the drop. The recorded process RSS rose from 2.63 GB after cleanup to 7.89 GB during resumed training, so the training process itself accounts for a large part of the observed working-set increase. System-wide available RAM includes other processes; their individual contribution is unknown.

### Changes exercised by the retry

- Load the original safetensors directly into the requested device after the key/shape audit, copying one source tensor at a time instead of first materializing the whole model on CPU.
- Use memory-mapped checkpoint loads and promptly release checkpoint references.
- Release CPU accumulated-gradient entries one parameter at a time during restoration, and delete local transfer references promptly.
- Record outer host-memory telemetry from before selected data/model loading through resumed training, plus detailed trainer phase samples. The outer log is `host_memory_telemetry.json`; the failure telemetry is `resumed/failure_report.json`.

These changes reduced unnecessary checkpoint/gradient copies but did not leave enough RAM to finish the resumed update. A next design would need to avoid a second base-checkpoint load on resume and further reduce the RAM residency of accumulated gradients or establish more available host RAM before a pilot. Neither approach was tested here, and no safety limit should be changed.

## Learning-rate correction

Production settings remain unchanged: `configs/training.json` still sets learning rate `1e-5` and warmup ratio `0.05`. In a two-update schedule, the existing production warmup calculation rounds `ceil(2 × 0.05)` to one warmup step; the first update is therefore assigned LR zero. That behavior explained the previous step-1 checkpoint’s unchanged parameters.

The retry adds a serialized run-level `scheduler_warmup_ratio=0.0` override for this bounded smoke pilot only. It leaves `TrainingConfig.warmup_ratio` and the production JSON untouched. With two planned pilot updates, the expected trace is `1e-5 -> 5e-6 -> 0`; the first two optimizer updates therefore both use nonzero LR. This verifies optimizer/parameter movement in the pilot. Part B must use the approved production warmup policy and a training horizon where its warmup schedule is meaningful; the pilot override is not evidence that production warmup should be removed.

Observed trace in `first_half/metrics.jsonl`:

- Global step 1: optimizer update succeeded, finite mean CTC loss `0.0034154873`, finite gradient norm `0.1658715`, GradScaler scale 65,536, LR used `1e-5`, LR next `5e-6`.
- The resumed step-2 accumulation did not reach an optimizer call. Thus `5e-6` was the restored next LR, not an observed second-update LR.

Read-only comparison of `initial.pt` and `latest.pt` found nonzero FP32 parameter changes:

| Parameter | L2 change |
|---|---:|
| Encoder feature projection weight | 0.000310923 |
| Phoneme head weight | 0.000042215 |
| Each of the ten sifat head weights | 0.0000113–0.0000134 |

The saved `latest.pt` has global step 1, 816 Adafactor state entries, scheduler `last_epoch=1` / `_last_lr=[5e-6]` / total steps 2, and GradScaler scale 65,536 with growth tracker 1. `initial.pt` is step 0 with empty optimizer state and scheduler epoch 0. The resumed trainer loaded the step-1 checkpoint and completed held-out validation before entering resumed training; restoration of model and optimizer state therefore completed. It did not save a step-2 checkpoint. The read-only state and parameter comparison is preserved in [`checkpoint_audit.json`](../runs/pilot_a3_updates_20261122T173644Z/checkpoint_audit.json).

The selected validation example’s initial loss was `0.000747564`; after the genuine update it was `0.000717601`. Both validation evaluations reported parameters unchanged during evaluation. The resumed initial validation completed, but no post-step-2 validation exists.

## GPU evidence

The retry retained all 605,754,251 trainable parameters, all eleven heads, microbatch size 1, accumulation 16, FP16 autocast/GradScaler, Adafactor, gradient checkpointing on all 24 encoder layers, and SDPA on all 24 layers.

| Measurement | First update | Resumed attempt |
|---|---:|---:|
| Peak allocated | 5,235,652,096 B | 5,232,417,792 B |
| Peak reserved | 5,259,657,216 B | 5,387,583,488 B |
| NVIDIA sampled peak | 5,165 MiB / 568 samples | 5,287 MiB / 39 samples |

The reserved peak stayed below 92% of 6,441,926,656 bytes; NVIDIA usage stayed below 5,700 MiB. The run stopped only at the host RAM guard.

## Tests

Before the retry:

- Focused tests (`tests/test_model.py`, `tests/test_training_loop.py`, `tests/test_training_checkpointing.py`): **40 passed**. This includes the CUDA gradient transfer/equivalence/reset test, the pilot-only two-nonzero-LR schedule test, checkpoint reload followed by another Adafactor update, and existing scheduler restoration coverage.
- Full project suite: **116 passed**.
- `python -m compileall -q src scripts tests`: passed.

No training was run before these tests passed. The retry was the single controlled attempt; it was not repeated after the RAM guard stopped update 2.

## Acceptance status

| Criterion | Result |
|---|---|
| Existing 32 training examples plus one held-out validation example reused | Pass |
| 605,754,251 parameters trainable; 11 heads; microbatch 1; accumulation 16 | Pass |
| Finite loss and gradients on completed update | Pass |
| Two successful optimizer updates at nonzero LR | **Fail — update 1 completed; update 2 stopped during gradient offload** |
| Nonzero encoder and all eleven head parameter changes | Pass for update 1 |
| Scheduler/optimizer advancement after successful update | Pass for step 1; step 2 not reached |
| Held-out validation before and after update 1 | Pass |
| Checkpoint save and reload into resumed trainer | Pass; resumed validation and training began |
| Post-resume optimizer update/checkpoint and validation | **Not demonstrated** |
| GPU safety limits | Pass |
| 2 GiB host RAM safety floor | **Fail — minimum 1.97 GiB; guard stopped run** |

**PART A.3 INCOMPLETE.** The memory changes and pilot-only schedule correction produced one real parameter update, but the resumed second update violated the host RAM floor during CPU gradient accumulation. The one allowed controlled retry is complete; do not start Part B or launch another retry under this authorization. A follow-up needs an independently reviewed approach that reduces resumed-process/gradient memory without weakening the safety floor, followed by its own tests and explicitly authorized bounded run.
