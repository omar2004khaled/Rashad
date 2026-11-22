# Stage 7 — Part A.2: Full fine-tuning memory optimization audit

## Scope and result

This audit does not train on the corpus or start Part B. It re-ran the verified original model on three existing real examples from `moshaf_0.1`, with all 605,754,251 parameters trainable, micro-batch size 1, FP16 autocast/GradScaler, and encoder gradient checkpointing. Two warm-up updates were followed by ten timed/measured updates. A restart checkpoint was saved after global step 7 and restored into fresh model/optimizer/scheduler objects; one subsequent update succeeded.

The corrected run completed without OOM, non-finite loss, or non-finite gradient norm. PyTorch peak allocated memory was 5,245,811,200 bytes (4.884 GiB), peak reserved was 5,435,817,984 bytes (5.061 GiB), and `nvidia-smi` sampled 5,331 MiB used out of the device's 6,144 MiB. This supports a longer, controlled pilot with monitoring, but is not proof that long utterances or arbitrary batches fit.

## Training configuration and actual Adafactor behavior

`configs/training.json` selects `"optimizer": "adafactor"`, `learning_rate: 0.00001`, `weight_decay: 0.01`, `scheduler: "linear"`, and `warmup_ratio: 0.05`. The Python default in `src/quran_training/config.py` is also `optimizer="adafactor"`, `learning_rate=1e-5`, and `weight_decay=0.01`. `TrainingConfig.build_optimizer()` constructs **`torch.optim.Adafactor`** and passes those `lr` and `weight_decay` values explicitly. The run's actual optimizer defaults and group LR are recorded in `reports/memory_optimization_benchmark.json`.

The effective Adafactor values observed at runtime were:

| Setting | Effective value / behavior |
|---|---|
| Optimizer | `torch.optim.Adafactor` |
| `lr` passed to optimizer | `1e-5` |
| `relative_step` | PyTorch does not expose this boolean. Its implementation calculates `rho_t = min(lr, 1/sqrt(step))`; with `lr=1e-5`, this remains capped at `1e-5` for ordinary training lengths. |
| `scale_parameter` | PyTorch does not expose this switch. Its update always scales by `max(eps[1], RMS(parameter))`; `eps[1]` defaults to `0.001`. |
| `warmup_init` | Not exposed by this PyTorch implementation. |
| `weight_decay` | `0.01`; PyTorch documents decoupled weight decay using `lr` as the coefficient. |
| Other observed defaults | `beta2_decay=-0.8`, `eps=(None, 0.001)`, `d=1.0`, `foreach=None`, `maximize=False`. |

Thus the configured optimizer `lr` is explicitly 1e-5; it is the maximum relative step-size coefficient in this implementation, with parameter-RMS scaling built in. PyTorch's documented algorithm also uses `lr` to scale decoupled weight decay. See the [official PyTorch Adafactor documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adafactor.html).

The JSON's `scheduler: "linear"` and `warmup_ratio: 0.05` are configuration metadata: there is no project training loop/scheduler builder wired to `TrainingConfig` yet. The benchmark uses a constant `LambdaLR` solely to test scheduler checkpoint save/restore without changing the fixed optimizer group LR. A future training pilot must deliberately implement and verify the intended linear schedule; the current benchmark does not claim to apply it.

## Memory measurement audit

### How each value is computed

In `scripts/benchmark_training.py`:

- **Parameter memory:** `_model_memory()` sums `numel * element_size` for every model parameter, grouped by device and dtype. The model is FP32 and all parameters are trainable in this run.
- **Gradient memory:** the same byte calculation for every non-`None` parameter gradient. `zero_grad(set_to_none=True)` means gradients appear at backward. At the end all 816 tensors (605,754,251 elements) had gradients.
- **Optimizer-state memory:** `_optimizer_memory()` sums tensor bytes in `optimizer.state` by device and dtype. Adafactor lazily creates this state during an initial update, so pre-update state is not a steady-state measurement.
- **Peak allocated:** before each forward, backward, optimizer, or checkpoint phase, the script reads live `torch.cuda.memory_allocated()`, resets CUDA peak statistics, then reads `max_memory_allocated()` after synchronizing at phase end. Run peak is the maximum **absolute phase peak** across all phases, not a sum of phase deltas. The measured-updates peak excludes the two warm-up updates.
- **Peak reserved:** calculated the same way from `memory_reserved()` and `max_memory_reserved()`. Reserved includes CUDA allocator blocks held for reuse; it is not the same as live allocated tensor bytes.
- **Forward transient:** phase peak allocated minus that forward phase's live allocated baseline.
- **Backward transient:** phase peak allocated minus that backward phase's live allocated baseline. Its baseline includes tensors still live after forward; backward allocates gradients and has changing saved-activation lifetimes.
- **Optimizer transient:** phase peak allocated minus the live allocated baseline after backward, unscaling, and gradient clipping. It can include optimizer work buffers and, on the first update, newly initialized optimizer state. The reported steady-state value excludes warm-up.
- **External/system sampling:** `nvidia-smi` and `psutil` poll loops use a 0.1 s wait interval; actual cadence also includes query latency. Peak and sample counts, including per-phase NVIDIA counts, are recorded. These sampled measurements can miss shorter peaks. The dedicated GPU total was 6,441,926,656 bytes.

For every phase, the JSON includes its own baseline allocated/reserved values, absolute peaks, and deltas. A transient delta is meaningful only with its matching phase baseline and peak. Phases overlap in lifetime and are not additive.

### What was wrong with the earlier figures

The original three-update implementation reset PyTorch peak statistics before every phase and every update, then read the counters only once at the end. Consequently, its `peak_allocated_bytes` / `peak_reserved_bytes` described only the last phase of the last update, not a run-wide peak. Separately, its forward/backward/optimizer transient fields took the maximum phase delta across all three updates, including the update where Adafactor state was first allocated. The state/parameter/gradient totals were collected afterward. Those values came from different baselines and sometimes different updates.

In particular, the older figures of **5.174 GB** for parameters+gradients+optimizer state, **5.229 GB** labeled as peak allocated, and **1.286 / 2.371 / 0.361 GB** for forward/backward/optimizer transients are not compatible quantities. The 0.361 GB optimizer delta included lazy state initialization, while 5.229 GB was not a true run maximum. Adding those three transient maxima to 5.174 GB is invalid. The original results remain nested under `previous_results_before_audit` in the JSON for traceability; use `experiments.adafactor` for corrected values.

### Corrected and new Adafactor measurements

The audited command was:

```powershell
$env:PYTHONPATH = "src"
python scripts/benchmark_training.py `
  --scan-limit 5 --examples 3 --steps 10 --warmup-steps 2 `
  --checkpoint-after-updates 7 --optimizers adafactor `
  --output reports/memory_optimization_benchmark.json
```

Examples were `moshaf_0.1:0.1:001.0000`, `001.0001`, and `001.0002`, with durations 6.06, 6.02, and 6.28 seconds, 16 kHz, and 302, 300, and 313 feature frames. The loader excluded zero rows. The benchmark used the original checkpoint `obadx/muaalem-model-v3_2` at revision `01a1ef9fbe40d144ef845101e89ff924aed3fef5`; the load audit matched all 816 tensors and all 605,754,251 parameters.

| Measurement | Corrected run |
|---|---:|
| Parameter memory | 2,423,017,004 B FP32 CUDA |
| Gradient memory | 2,423,017,004 B FP32 CUDA |
| Adafactor state | 327,785,784 B FP32 CUDA + 3,264 B FP32 CPU (1,940 tensors) |
| Parameter + gradient + optimizer state | 5,173,823,056 B (5.174 GB decimal) |
| PyTorch run peak allocated | 5,245,811,200 B (5.246 GB decimal; 4.884 GiB) |
| PyTorch run peak reserved | 5,435,817,984 B (5.436 GB decimal; 5.061 GiB) |
| Highest `nvidia-smi` sample | 5,331 MiB of 6,144 MiB; 234 samples; 0.1 s poll wait interval plus query latency |
| Process RSS peak | 3,154,108,416 B |
| Minimum system available RAM | 8,077,180,928 B; 384 samples |
| Mean measured step time | 1.4187 s over 10 updates, excluding 2 warm-ups |
| Mean forward / backward / optimizer time | 0.2650 / 0.5255 / 0.6282 s |

The corrected **measured-update** transient maxima are below. Each phase's baseline and peak shown are from the same record that produced its maximum delta; they are not combined with other phases.

| Phase | Live allocated baseline | Absolute phase peak | Allocated transient delta | Reserved transient delta |
|---|---:|---:|---:|---:|
| Forward | 2,770,954,752 B | 4,052,579,840 B | 1,281,625,088 B | 0 B |
| Backward | 2,869,952,512 B | 5,242,105,344 B | 2,372,152,832 B | 0 B |
| Optimizer | 5,194,348,032 B | 5,227,943,424 B | 33,595,392 B | 25,165,824 B |
| Checkpoint save/hash | 5,195,435,008 B | 5,195,435,008 B | 0 B | 0 B |

The run-wide PyTorch peak (5,245,811,200 B) is slightly higher than the phase record with the largest backward transient because run peak selects the maximum **absolute** phase peak across all updates. This explains the difference without summing deltas. Forward allocations are largely released or changed by backward; the backward phase baseline contains the live saved tensors. The optimizer baseline includes parameters, gradients, and initialized state. These are different points in the same step. Timings varied between two otherwise identical corrected runs (0.790 s and 1.419 s mean measured step); the reason was not isolated, so the runtime result is noisy. Memory peaks matched both runs.

## Training and resume verification

- 12 full-model optimizer updates: 2 warm-up and 10 measured; optimizer state was initialized during warm-up and was already present for all measured updates.
- 605,754,251/605,754,251 parameters trainable; FP32 parameter and gradient storage; FP16 autocast with enabled GradScaler (scale 65,536); gradient checkpointing active.
- Finite loss and finite gradient norm were verified; clipping is configured to error on a non-finite norm.
- Encoder feature-projection parameters and phoneme-head parameters changed. Gradients were present on all eleven heads.
- The final audited run recorded 234 `nvidia-smi` samples and 384 system/process memory samples. Counts per phase are in the JSON; poll loops use a 0.1-second wait interval plus query latency.
- A temporary real-model checkpoint was saved after global step 7. Fresh objects restored model and optimizer tensor values exactly (SHA-256 digest), scheduler state exactly, and global step 7. The restored model completed another finite-loss update, yielding step 8. The benchmark's scheduler was constant for this state-resume verification; it does not validate an as-yet-unimplemented linear training schedule.

## Historical four-optimizer comparison

The earlier short three-update comparison remains available under `previous_results_before_audit` in the JSON. Its broad optimizer-state byte measurements and update outcomes were useful for choosing Adafactor, but its peak and per-phase memory fields are superseded by the corrected instrumentation described above. Historical claims about measured peak allocation should not be used as corrected peaks. No other optimizer was rerun in this audit.

## Conclusions, limits, and next decision

Adafactor appears sufficiently reliable for a **longer controlled training pilot**, not yet for an unattended full run. It passed 12 full-parameter updates, left observed allocator headroom, produced finite gradients/loss, and resumed exactly. However, this benchmark used only three short clips from one configuration and reciter; `nvidia-smi` is sampled; the test does not cover longer utterances, other reciters, sustained thermals, or concurrent GPU workloads. A pilot should be bounded, periodically checkpointed, and continue monitoring OOM and non-finite behavior.

Before that pilot, implement or explicitly choose the training scheduler: the config says linear with 5% warm-up, but no training-loop scheduler is currently wired. Keep the optimizer's configured `lr=1e-5` explicit; decide whether the intended schedule should modify it. The eligible corpus row count also remains unknown because no full-dataset validity pass was run. No model architecture, vocabulary, labels, or dataset records were changed, and no full epoch or Part B run was started.

## Verification

The revised benchmark result is in `reports/memory_optimization_benchmark.json`. Focused verification passed **28 tests** (`tests/test_model.py`, `tests/test_training_collation.py`, and `tests/test_training_checkpointing.py`); the complete project suite passed **95 tests**. `git diff --check` also passed.
