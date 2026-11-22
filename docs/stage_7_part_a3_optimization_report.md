# Stage 7 Part A.3 — Optimization and Final Pilot Evidence

## Status

**PART A.3 INCOMPLETE.** The bounded run completed one optimizer step and saved a valid step-1 checkpoint, then stopped during the resumed attempt at update 2 because available system RAM crossed the existing 2 GiB safety floor. The required two successful updates, nonzero encoder/head parameter changes, and a post-update validation after a learning update were not demonstrated. Part B was not started.

The run artifacts are retained in [`runs/pilot_a3_updates_20261122T164735Z`](../runs/pilot_a3_updates_20261122T164735Z/). No second training attempt was made.

## Root cause and memory evidence

The previous A.3 failure was caused by device-resident accumulated gradients, not the duration of the failing clip. Its 3.89-second clip was shorter than the 5.52-second clip that had already passed backward. The encoder and checkpointing configuration were the same. The persistent cost was retaining one full set of FP32 gradients while processing the next microbatch.

The implementation was changed to accumulate scaled gradients in CPU tensors between microbatches, restore them to each parameter immediately before GradScaler unscale, clipping, and optimizer step, and clear device gradients to `None` after offload. The 605,754,251 parameters remain trainable; no layer/head, loss weight, label, or safety threshold was changed. The run diagnostics confirm gradient checkpointing active for all 24 encoder layers and SDPA selected in all 24 layers.

The final run used the exact checkpoint revision `01a1ef9fbe40d144ef845101e89ff924aed3fef5`, loaded all 816 tensors / 605,754,251 parameters, Adafactor, fixed LR 1e-5, weight decay 0.01, FP16 autocast/GradScaler, microbatch 1, accumulation 16, and the existing limits: 92% reserved memory, 5,700 MiB NVIDIA usage, and 2 GiB minimum available RAM.

| Measurement | First update | Resumed attempt at update 2 |
|---|---:|---:|
| PyTorch peak allocated | 5,227,596,800 B | 5,232,417,792 B |
| PyTorch peak reserved | 5,253,365,760 B | 5,387,583,488 B |
| NVIDIA sampled peak | 5,165 MiB (589 samples) | 5,287 MiB (105 samples) |
| Minimum available system RAM | 3,117,068,288 B | 1,799,397,376 B |
| Process RSS peak | 4,965,388,288 B | 5,687,332,864 B |

Both GPU limits remained satisfied. The stop was the RAM guard: `ResourceLimitReached: System available RAM is 1.66 GiB, threshold is 2.00 GiB`, in `train_backward` at global step 1, example `moshaf_27.0:27.0:001.0006` (4.40 seconds). There was no CUDA OOM. The failure report was persisted at [`resumed/failure_report.json`](../runs/pilot_a3_updates_20261122T164735Z/resumed/failure_report.json). The guard fired after backward on the first microbatch of the resumed update and **before** that microbatch's gradients were offloaded to CPU. Thus the new gradient was still on GPU, and no new CPU accumulation buffer had yet been populated. Process RSS peaked at 5.69 GB after fresh model/optimizer/checkpoint restore, but the saved artifacts do not attribute host usage to a specific allocation. No retry followed.

On the first update, after each backward pass the allocated GPU memory returned from roughly 4.87 GB to 2.44 GB after offload. Before the optimizer, restored gradients occupied the device as expected (about 4.87 GB allocated); the optimizer step raised reserved memory to about 5.25 GB. This confirms gradients were restored before optimizer operations. The memory snapshots are retained in `first_half` result data and the resumed failure report.

## Attention and checkpointing checks

The installed Wav2Vec2-BERT self-attention in Transformers 5.8.1 uses eager matrix multiplication and softmax. Its `relative_key` positional term is additive, which allowed a PyTorch `scaled_dot_product_attention` path without changing parameter names or shapes. The implementation keeps eager available and falls back to upstream eager when attention weights are explicitly requested. No Flash Attention dependency or backend was assumed.

The same real failing sample was compared at 3.89 seconds / 62,240 waveform samples / 194 feature frames. Gradient checkpointing was enabled on all 24 layers. Both backends had finite losses and gradients and stayed under safety limits:

- Eager: peak allocated 4,877,607,424 B, peak reserved 4,961,861,632 B, NVIDIA peak 4,877 MiB.
- SDPA: peak allocated 4,877,295,616 B, peak reserved 4,945,084,416 B, NVIDIA peak 4,863 MiB.
- All eleven heads had 100% greedy token agreement. Relative mean absolute logit differences ranged from 0.000219 to 0.000461; cosine similarity was at least 0.99999958. Aggregate loss difference was 0.0000018231.

Thus SDPA passed the real-sample compatibility check but saved only about 17 MiB reserved in this diagnostic. CPU gradient offload addressed the A.3 GPU peak; SDPA is a small additional reduction, not the primary fix.

Gradient checkpointing uses the installed Transformers `GradientCheckpointingLayer` wrapper and non-reentrant checkpoint mode. The actual diagnostic and pilot both report 24/24 layers enabled. The pilot selected the PyTorch SDPA dispatch path; the exact low-level CUDA kernel selected by PyTorch’s dispatcher is not recorded, so this report does not claim Flash Attention.

## Gradient accumulation and update ordering

The regression test `test_cpu_gradient_accumulation_matches_device_resident_sum` compares ordinary two-microbatch accumulated gradients with the offload/restore helper result and passes within PyTorch's default `assert_close` tolerances. It also checks that parameter `.grad` is cleared after each offload. The production loop restores gradients before `scaler.unscale_`, verifies all trainable parameters have gradients, clips, and then calls `scaler.step` / `scaler.update`. The scheduler and global step advance only when `optimizer_update_succeeded` detects that GradScaler did not skip the optimizer update. At the next accumulation group, `optimizer.zero_grad(set_to_none=True)` clears the prior gradients.

The real pilot exercised CPU offload, GPU restoration, clipping, and one successful Adafactor step. It did not complete an update after resume, so cross-step reset behavior on a second real update remains unverified in full-model execution. The focused synthetic test covers repeated offload accumulation and restoration on CPU; it does not independently exercise a CUDA-backed helper test.

## Training state, checkpoint, and validation audit

`first_half/metrics.jsonl` records one successful optimizer call at global step 1 after 16 microbatches, with finite mean CTC loss `0.0032638345`, finite gradient norm `0.1529086`, GradScaler scale 65,536, and learning-rate trace `0.0 -> 1e-5`. Initial and post-first-update held-out validation both completed on the selected validation example with loss `0.00074756396` and no parameter mutation during evaluation.

The scheduler has one warmup step in the two-step schedule, so the first optimizer call used LR **0.0**. Read-only inspection comparing `initial.pt` with `latest.pt` confirms an L2 parameter delta of **0.0** for the encoder feature projection and all eleven output-head weight matrices. Therefore, although global step 1 and optimizer state advancement are real, that first step did not update model weights. The second scheduled update would use LR 1e-5 but did not complete.

Checkpoint inspection:

- `initial.pt`: global step 0; Adafactor state entries 0; scheduler epoch 0; GradScaler scale 65,536.
- `latest.pt` and `final.pt`: both deserialize successfully; global step 1; 816 Adafactor state entries; scheduler epoch 1, total steps 2, warmup steps 1, LR 1e-5 for the next step; GradScaler scale 65,536 with growth tracker 1; 816 model tensors.
- The resumed trainer loaded the step-1 checkpoint into a fresh model/optimizer/scheduler/scaler and completed its initial held-out validation before starting the next training backward. This verifies that checkpoint deserialization and restoration were exercised. No subsequent optimizer update or post-resume checkpoint was completed.

The serialized step-1 checkpoint is structurally intact and contains optimizer/scheduler/scaler state. Resume after an update was attempted and state restoration succeeded sufficiently to enter resumed training, but the acceptance criterion requires a subsequent successful optimizer update; it remains incomplete.

The pilot reused the already validated selection of 32 training examples plus one held-out validation example from `runs/pilot_final_selection_20261122T160812Z`; it did not rescan or download the full dataset. The selected duration cap remained 6.3 seconds. The first update consumed 16 selected examples; the resumed attempt stopped partway through the next group.

## Tests and run evidence

- Focused tests: `pytest tests/test_model.py tests/test_training_loop.py tests/test_training_checkpointing.py -q` — **37 passed**.
- Full suite: `pytest -q` — **113 passed**.
- `python -m compileall -q src scripts tests` — passed.
- Real eager/SDPA comparison: `runs/attention_compare_20261122T164654Z/attention_comparison.json` — PASS for finite values, compatibility criteria, and memory limits. Earlier comparison attempts remain preserved in their own run folders; the final comparison repeated them with dropout disabled and deterministic seeds to compare backend arithmetic rather than different dropout RNG masks.
- Pilot outcome: one successful optimizer invocation, no model parameter changes due to the zero warmup LR, and a resource-guard stop during update 2 before optimizer step.

## Remaining gate

Part A.3 remains incomplete because the bounded experiment did not achieve two successful updates, nonzero encoder/head parameter deltas, post-learning-update validation, or a successful post-update checkpoint resume followed by another optimizer step. The resumed process hit the host RAM floor before the next microbatch gradients reached the CPU accumulator; the specific host-memory cause is not isolated. The one completed update also used LR zero due to the configured two-step warmup schedule.

No thresholds were raised and no limits, splits, model dimensions, vocabulary, heads, or approved loss settings were weakened. The artifacts show that the original GPU reserved-memory failure has been mitigated. The resumed process hit the RAM guard before gradients were offloaded; the host-memory cause is not isolated. Also, the pilot learning-rate schedule made its first update a zero-LR step. A RAM-safe, meaningful bounded update plan is needed before A.3 can pass. This report does not authorize another pilot. Part B must not begin.
