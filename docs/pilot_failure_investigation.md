# Stage 7 Part A.3 Pilot Failure Investigation

## Scope and evidence examined

This is a read-only investigation. No training was run, no dataset audio was downloaded, and no model/configuration code was changed. The checkpoint was inspected with PyTorch `FakeTensorMode` so its metadata could be read without materializing the model's 2.4 GB tensor storage.

Inspected: `docs/controlled_training_pilot_report.md`, `src/quran_training/engine.py`, `scripts/run_training_pilot.py`, `tests/test_training_loop.py`, `configs/training.json`, `configs/reciter_split_manifest.json`, and all files under `runs/controlled_pilot_20261122T144300Z/`. The run folder contains `selected_examples.json` (86,617 bytes), `checkpoints/initial.pt` (2,423,374,915 bytes), and an empty `metrics.jsonl` (0 lines). It has no `pilot_results.json`, failure log, or persisted memory samples.

## 1. Memory failure

### Where the safety check runs

`QuranTrainer.fit()` calls `_check_resources()` in `src/quran_training/engine.py:415` immediately after each microbatch's backward pass and after deleting `output` and `batch`. It calls it again at `:466` after an accumulation group and optimizer/scaler/scheduler processing. It does **not** check before a forward pass or during the forward/backward kernels. The monitor samples system/process memory and `nvidia-smi` in a background thread about every 0.1 seconds, but the Python guard's PyTorch check is at those phase boundaries.

### What memory is checked

The PyTorch check reads **current reserved memory only**:

```python
reserved = torch.cuda.memory_reserved(self.device)
fraction = reserved / torch.cuda.get_device_properties(self.device).total_memory
```

It does not directly test `memory_allocated()` and does not use `max_memory_reserved()` or `max_memory_allocated()` in its stop condition. Separately, it compares the largest sampled `nvidia-smi` device memory-used value to `max_nvidia_smi_used_mib`; this is a device-wide sample, not a PyTorch allocated/reserved measurement. On successful completion, the loop reports PyTorch peak allocated/reserved, but that summary is not written or returned when `ResourceLimitReached` is re-raised.

### Device and units

The run config does not specify `device`, so `QuranTrainer` selects `cuda` when available. On this machine that resolves to CUDA device 0. The safety check queries properties and reserved bytes for `self.device`; `nvidia-smi` is also explicitly queried with `--id=0`. The current device report is an RTX 3060 Laptop GPU with `total_memory=6,441,926,656` bytes, approximately 6.0 GiB or 6,144 MiB. Both operands in the PyTorch ratio are byte counts, so the division has consistent units and uses the intended device/capacity.

The prior report records the guard message as 6,507,462,656 reserved bytes / 6,441,926,656 total bytes = 101.02%. In the attempted run the configured threshold was 90%, so this value triggers the guard. The value above 100% is not evidence of a byte-versus-MiB arithmetic error: the code divides bytes by bytes. `nvidia-smi -q` currently identifies the Windows driver model as **WDDM**, where GPU allocations can be backed through shared/pageable memory and dedicated-VRAM accounting is not a complete measure of process commitment. This makes an allocator reservation above the reported dedicated capacity plausible, but the run has no contemporaneous `nvidia-smi`, allocator, or Windows GPU-memory trace. Therefore the precise mechanism behind the 101% observation cannot be established from saved artifacts.

The longer sequence is a concrete likely contributor, not a proven sole cause: the failed training microbatch is reported as 9.68 seconds, while the audited A.2 Adafactor benchmark's three measured clips were 6.02, 6.06, and 6.28 seconds. Longer inputs increase encoder activations. The attempted backward also occurs before the first Adafactor optimizer step, so the optimizer state had not yet been initialized and cannot explain this particular pre-step observation. The saved run artifacts do not contain a phase breakdown to quantify the contributors.

### Guard versus CUDA OOM

The stop was caused by **`ResourceLimitReached`**, not a CUDA OOM. At `engine.py:415`, the `fraction >= max_gpu_reserved_fraction` branch raises that exception. The exception is re-raised by the distinct `except ResourceLimitReached` block (`:492-493`); `torch.cuda.OutOfMemoryError` has a separate handler (`:489-491`). The earlier report says the thrown message was the 101% reserved-memory check. There is no OOM traceback in the saved artifacts. Since the process console transcript was not saved, the exception type is supported by the code path and prior report, not independently re-read from a persistent run log.

## 2. Training state

The checkpoint contains `global_step=0`, `epoch=0`, `extra.status="initial"`, `next_epoch=0`, `next_example_offset=0`, and `examples_seen=0`. The Adafactor optimizer has one parameter group and **zero state entries**. This is expected before its first `optimizer.step()`. The scheduler state has `last_epoch=0`, `_step_count=1`, `total_steps=16`, `warmup_steps=1`, `_last_lr=[0.0]`, and the optimizer group's current `lr=0.0` (base LR `1e-5`). The GradScaler state is still its initial scale `65,536`, growth tracker 0. No update had occurred to change any of these states.

The checkpoint's run config records two epochs, `max_optimizer_steps=5`, `max_train_examples=80`, accumulation 16, and the 0.90 / 5,700 MiB / 2.5 GiB guards. Its training config records Adafactor, LR `1e-5`, weight decay `0.01`, FP16, gradient checkpointing, and accumulation 16. Because the then-selected training list had 122 examples, `fit()` computed scheduler steps as `ceil(122/16) * 2 = 16`, which matches the saved scheduler state. The prior report's statement that the 80-example main segment itself planned 10 scheduler steps is not what this code/checkpoint did. The run limits capped consumption at 80 and optimizer steps at five; they did not change the scheduler's 16-step calculation.

The checkpoint was created in `fit()` before initial validation and before the training loop. Resuming from it would restore global step 0, example cursor 0, the empty Adafactor state, scheduler step 0, and initial GradScaler state; it would start before the first optimizer update. The checkpoint can restore state structurally, but no resume was attempted in this failed A.3 run.

The control flow implies the initial validation call completed: `fit()` runs validation before opening the training loop, and the reported failure happened on the first training backward. That validation call used `model.eval()` and `torch.inference_mode()`. However, its loss is not in the saved checkpoint or empty metrics file, and the exception prevented `fit()` from returning its result. Thus a successful validation pass is strongly implied by control flow, but its metric cannot be verified from a persisted log.

## 3. Clip selection and split independence

### Saved selection is from the pre-eight-second policy

The saved `selected_examples.json` is from the earlier selector. It records old buckets `0–10`, `10–14`, and `14–20` seconds. Its decoded durations are:

- Training: 122 examples; 2 at or below 5 seconds, 12 from 5–6.5 seconds, 11 from 6.5–8 seconds, and **97 over 8 seconds**; range 4.28–19.975 seconds.
- Validation: 9 examples; none at or below 6.5 seconds, 1 from 6.5–8 seconds, and **8 over 8 seconds**; range 7.71–16.245 seconds.

The actual first microbatch's reported 9.68-second clip is consistent with that saved policy. The saved artifacts do not represent a retry using the later eight-second limit.

### Current selector behavior and feasibility

The current `scripts/run_training_pilot.py` has `_duration_bucket()` capped at 8 seconds (`:48-55`). `_select_examples()` groups safe candidates by training configuration and duration bucket, then distributes selections round-robin over the nonempty groups (`:195-238`). This gives a roughly balanced **count per available configuration-duration group**, subject to each group's candidate count. Each configured training moshaf corresponds to one reciter in the manifest, so it indirectly spreads across the four listed training configurations/reciters. It does not independently optimize or assert equal reciter weights; a reciter with fewer available groups/candidates can contribute fewer examples.

The current selector requires at least 96 safe training candidates and at least three nonempty configuration-duration groups. It selects up to 128. For validation, it requires one eligible example in each of three duration buckets from each of three held-out configurations (nine total), while filtering against training and prior validation suras and exact normalized Imlaey strings. It raises if any of those nine leakage-free candidates is absent. Therefore it can produce enough examples **only if** the fresh metadata scan contains at least 96 eligible training rows distributed across the required groups and all nine validation cells. The failed run did not execute this version against a complete fresh scan, and the saved old selection cannot prove those conditions. Feasibility remains unverified.

### Manifest split check

The saved selection uses four training reciters/configurations: reciters 0, 2, 6, and 26 (`moshaf_0.1`, `2.0`, `6.0`, `26.1`), with config counts 32, 30, 32, and 28. Validation uses reciters 12, 13, and 24 (`moshaf_12.0`, `13.0`, `24.0`), three each. Looking up those IDs in `configs/reciter_split_manifest.json` confirms every training reciter is marked `train` and every validation reciter is marked `validation`. The reciter ID sets and exact moshaf ID sets are disjoint. The selected Quran sura sets are also disjoint, and the selection metadata reports zero exact normalized-Imlaey overlaps. This establishes separation for the saved selection and the manifest-derived split assignment; the dataset server's rows themselves were all fetched from its published `train` split.

## 4. Memory behavior in the implementation

- **Gradient clearing:** `optimizer.zero_grad(set_to_none=True)` runs once before each gradient-accumulation group (`engine.py:397`). Gradients then intentionally accumulate across that group's microbatches. The loss is divided by the actual group size (`:413`), so a shorter final group is averaged by its actual number of examples.
- **Tensor lifetime:** after backward, `output` and `batch` are deleted (`:414`). The Python variable `loss` remains bound until reassigned; after backward PyTorch releases saved backward tensors, but the scalar/autograd wrapper may remain referenced. There is no artifact proving that this small reference caused the reported peak. The current code does not use `empty_cache()` per batch, which is not itself a leak: reserved caching is expected and is exactly what the safety check observes.
- **Validation:** `evaluate()` switches to eval mode and uses `torch.inference_mode()` (`:203-219`), then restores the prior model mode. It checks parameter version counters for mutation. The real run's initial validation is outside the active `MemoryMonitor`; peak stats are reset after it (`:366-374`), so its resource use is not included in training-phase results.
- **Optimizer/scheduler order:** after all microbatch backward passes, gradients are unscaled, optionally checked, clipped, and passed to GradScaler/optimizer (`:417-440`). Scheduler and global step advance only if the scaler indicates an update succeeded (`:434-441`). The failed run stopped before this block.
- **Guard timing and failure evidence:** the reserved check after every backward is later than the peak allocation point and cannot prevent an OOM inside that forward/backward. It uses the allocator's current reserved high-water pool, not a pre-forward estimate. On the raised resource-limit path, `finally` stops the monitor (`:495-497`), but the code re-raises before computing `monitor.summary()` (`:499-506`) or writing a failure record. This explains why the run folder lacks sampled NVIDIA/host/process memory and an exception summary. The empty metrics file is expected because a metrics row is only written after a full accumulation/optimizer step (`:462-465`).

No clear gradient-accumulation or validation-autograd defect is indicated by the inspected code. There is a concrete observability gap on exceptions (monitor data is discarded), and the current script's eight-second filter applies to metadata duration; decoded audio is only checked within a ±0.15-second metadata tolerance (`run_training_pilot.py:327-333`) rather than being explicitly rejected if decoded duration exceeds 8 seconds. Those facts should be addressed or guarded before a retry. No tests were added because this task is an investigation and no implementation was changed.

## 5. Next safe experiment (recommendation only)

Do not resume the saved run as a test of the intended update: its example selection contains 97 clips longer than eight seconds, and its initial checkpoint is before any update. First produce and inspect a new manifest-filtered metadata selection and verify decoded durations for all selected audio before training.

For the smallest useful real-data experiment that can verify an actual nonzero parameter update under the configured warmup:

- **Training:** 32 examples total, from at least two manifest-train reciters/configurations, with a selected pool larger than 32 so the run is not a full pass over the selected pool. Keep accumulation 16, giving two optimizer updates. This matters because a one-step schedule with 5% warmup rounds to one warmup step and starts at LR 0; the saved checkpoint demonstrates this. With two planned updates, the second can exercise the configured nonzero LR and parameter-delta check.
- **Validation:** 1–3 real examples from a manifest-validation reciter/configuration, distinct from all training reciters/moshaf IDs and with disjoint suras/text where possible. Verify the exact selected records and retain their validation result.
- **Clip duration:** cap at **6.3 seconds**, the longest clip in the completed A.2 three-example benchmark (6.28 seconds). Enforce the cap on decoded audio too. This is a bounded first check supported by a benchmark, not evidence that longer clips are safe or that a representative long-duration pilot is complete.
- **Batching and memory stop:** microbatch 1, gradient accumulation 16, FP16/GradScaler and gradient checkpointing as configured. Keep a PyTorch reserved ceiling at 90% of detected device capacity and set a device-wide `nvidia-smi` stop below 5,700 MiB (for example 5,500 MiB), with the existing 2.5 GiB available-host-RAM floor. Persist per-sample memory/failure information and stop at the first threshold. These values are an experiment proposal, not executed or proven safe; note that the PyTorch check currently runs only after backward, so safety must include handling a CUDA OOM and persisting samples from exceptions.

Before calling A.3 complete, require evidence from a real run that includes: verified checkpoint/revision and trainable parameter count; decoded duration and manifest split for each selected row; at least two successful updates with finite loss/gradients and nonzero encoder and supervised-head parameter deltas; scheduler LR trace and GradScaler state; a recorded validation loss on held-out validation data without parameter changes; peak allocated/reserved and NVIDIA/system/process memory including failure-safe samples; successful checkpoint reload/resume after an actual update; a completed run summary; and passing focused plus full test suites. Any later sustained pilot should add longer durations gradually only after this bounded check stays inside the measured envelope.

## Uncertainties

1. There is no saved stdout/stderr or live monitor dump from the failed process. The exception/101% details are preserved in the prior narrative report but cannot be independently cross-checked against a run log.
2. WDDM is confirmed on the current machine; whether WDDM oversubscription/shared-memory backing caused the historical 101% allocator ratio is not proven.
3. The current eight-second selector's candidate sufficiency and its nine validation cells were not verified on a complete fresh dataset-server scan. The saved artifacts predate that selector.
4. The initial validation pass is inferred from control flow and the reported later backward failure; its loss and its own memory use were not persisted.
5. A 6.3-second cap and proposed thresholds are anchored to the separate three-example A.2 benchmark only. They do not establish safe memory behavior over multiple reciters, all sequence shapes, longer audio, or concurrent GPU use.
