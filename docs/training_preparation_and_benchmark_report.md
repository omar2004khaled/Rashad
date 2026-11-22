# Stage 7 — Part A: Training preparation and GPU benchmark

**Scope:** verify the original weights, implement batching/configuration/checkpoint resume, and run one controlled training step on a real sample. No epoch or full-dataset training was performed; only one audio sample was decoded for the benchmark.

## Existing work reviewed

- `src/quran_model/configuration.py` and `src/quran_model/modeling.py`: Stage 6 Wav2Vec2-BERT architecture, eleven heads, CTC IDs/reduction, per-level loss.
- `src/quran_training_data/preparation.py`, `encoding.py`, `batching.py`, and `splitting.py`: lazy one-record preparation, immutable Stage 5 examples, exact target IDs, and reciter split assignment.
- `configs/training_data.json`, `configs/reciter_split_manifest.json`, `configs/audio_preprocessing.json`, and Stage 1–6 reports: current inclusion policy, 228,941 source train rows, model audio expectations, and known limitations.
- Upstream `src/quran_muaalem/modeling/configuration_multi_level_ctc.py`, `modeling_multi_level_ctc.py`, `vocab.py`, `multi_level_tokenizer.py`, and `src/quran_muaalem/inference.py` in `../quran-muaalem-reference`.

Stage 5's 228,941 count is source rows assigned to train **before filtering**. No full eligible-row count is available. Determining it would require a corpus-wide pass through annotation/audio validity; this benchmark intentionally did not fetch the entire dataset.

## Pretrained checkpoint loading: verified

Checkpoint: `obadx/muaalem-model-v3_2`, revision `01a1ef9fbe40d144ef845101e89ff924aed3fef5`. It was already present in the Hugging Face local cache, so no checkpoint or dataset download was needed for verification. The verified checkpoint has **605,754,251 parameters** and 816 tensor entries.

`QuranMultiLevelCTC.from_pretrained_checkpoint` now reads the source safetensors one tensor at a time, maps `wav2vec2_bert.<suffix>` to `encoder.<suffix>`, preserves `level_to_lm_head.<level>.*`, and compares every key and shape before allocating/loading the full model. Missing, unexpected, or mismatched tensors raise `CheckpointCompatibilityError` with a `CheckpointLoadReport`; the API cannot return a partially/uninitialized model as a successful load.

Actual checkpoint load result:

| Check | Result |
|---|---:|
| Matched tensors | 816 / 816 |
| Loaded parameters | 605,754,251 |
| Missing keys | 0 |
| Unexpected keys | 0 |
| Shape mismatches | 0 |
| Output heads loaded | All eleven; source head keys already match |

Automated small-checkpoint tests exercise the prefix mapping and assert correct loaded tensors. Additional tests create missing, unexpected, and shape-mismatched source tensors and verify the loader reports and rejects each case. The real cached checkpoint load independently passed the complete audit above.

## Data collator and partial supervision

`src/quran_training_data/collation.py` adds `QuranTrainingCollator`. It accepts Stage 5 `TrainingExample` waveforms with a matching HF feature extractor, or precomputed `(frames,160)` feature sequences. It right-pads features with zeros, emits an attention mask and feature-frame `input_lengths`, right-pads each target level with `-100`, and provides target lengths. It preserves Stage 5 annotations and target IDs; a mono `(samples,1)` array is squeezed to the already-declared mono waveform shape, while multi-channel input is rejected.

Examples with different annotated levels are supported. The collator emits per-level boolean `label_level_mask` values and all-`-100` rows for an absent level. The model selects supervised rows before CTC loss, so absent supervision does not become an empty-transcript target. Blank remains ID 0, target IDs start at 1, and loss weights are not renormalized over partial levels. Automated tests cover variable feature/audio lengths, padding, lengths, masks, source immutability, invalid data, and a collator-to-model loss/backward pass.

The collator's `input_lengths` count feature frames. The model calculates CTC output lengths with the Wav2Vec2-BERT adapter's own length function; target lengths are independently calculated and checked against the padded IDs.

## Initial training configuration

`configs/training.json` is loaded by `src/quran_training/config.py`:

| Setting | Initial value | Rationale |
|---|---:|---|
| Mode | `full_finetune` | Explicitly identifies the intended full-encoder objective; benchmark also reports a separate frozen-encoder comparison. |
| Optimizer | AdamW | Standard adaptive optimizer; no optimizer substitution. |
| Learning rate | `1e-5` | Conservative starting point for fine-tuning a large pretrained encoder; requires validation tuning. |
| Weight decay | `0.01` | Conventional decoupled regularization starting value. |
| Scheduler | Linear | Common finite-run schedule; selected total steps determine its shape. |
| Warmup | 5% | Conservative short warmup for optimizer stability. |
| Micro-batch | 1 | Smallest practical batch for 6 GiB physical VRAM. |
| Gradient accumulation | 16 | Effective batch 16 without increasing activation batch size. |
| Gradient clipping | 1.0 | Bounds large gradient norms. |
| Precision | FP16 autocast | Tested on this RTX 3060; GradScaler used in the benchmark. |
| Gradient checkpointing | enabled | Recomputes encoder activations to reduce activation storage. |
| Epochs | 5 | A provisional planning value, not a validated stopping criterion. |
| Evaluation/save interval | 500 optimizer steps | Initial periodic evaluation and restart point. |
| Seed/workers | 42 / 0 | Reproducible seed; zero workers avoids Windows multiprocessing overhead in this environment. |

The config validates supported values and round-trips through JSON. It describes settings; this Part A does not implement or start the full training loop.

## Restartable training checkpoints

`src/quran_training/checkpointing.py` atomically saves model, optimizer, scheduler, optional mixed-precision scaler, epoch, global step, config/extra metadata, and Python/NumPy/PyTorch CPU/CUDA RNG states. Resume requires model, optimizer, and scheduler objects and raises if required state is absent, preventing a silent optimizer/scheduler reset. Tests perform an optimizer update, save, mutate state, resume, and compare model/optimizer/scheduler and RNG state.

## GPU benchmark

Reproduce with:

```powershell
$env:PYTHONPATH = "src"
python scripts/benchmark_training.py --scan-limit 5 --examples 3 --steps 3
```

The script searches at most five streaming rows from `moshaf_0.1`, prepares three eligible training examples through the existing Stage 2–5 code, and loads the **verified pretrained checkpoint**. It runs three batch-size-one microbatches/optimizer updates per mode, one on each distinct example; it does not loop an epoch. Machine: NVIDIA GeForce RTX 3060 Laptop GPU, PyTorch 2.11.0+cu128, CUDA 12.8; PyTorch reports 6,441,926,656 total device bytes (6 GiB).

Real benchmark examples: `moshaf_0.1:0.1:001.0000` (6.06 s), `001.0001` (6.02 s), and `001.0002` (6.28 s), all 16 kHz; feature lengths were 302, 300, and 313 frames. No preparation exclusions occurred. All are from one configuration/reciter and a narrow duration range, so this remains a small, nonrepresentative sample.

| Measurement | Full fine-tune | Frozen encoder comparison |
|---|---:|---:|
| Trainable parameters | 605,754,251 | 76,875 (eleven heads) |
| Micro-batch | 1 | 1 |
| Precision | FP16 autocast + GradScaler | FP16 autocast + GradScaler |
| Gradient checkpointing | yes | no (encoder is frozen) |
| Forward (mean) | 1.508 s | 0.0825 s |
| Backward (mean) | 2.317 s | 0.00914 s |
| Optimizer step (mean) | 3.976 s | 0.00245 s |
| Mean total step / throughput | 7.801 s / 0.1282 examples/s | 0.0941 s / 10.63 examples/s |
| PyTorch peak allocated / reserved | 11.30 / 11.47 GiB | 2.33 / 2.36 GiB |
| `nvidia-smi` sampled peak device memory | 5,977 MiB (89 samples) | 2,563 MiB (2 samples) |
| Finite loss, gradient and update | yes / yes / yes | yes / yes / yes |
| CUDA OOM | no | no |

The three-step full-model run across three clips completed with FP16, micro-batch one, and gradient checkpointing. PyTorch's allocator reported a peak of 11.30 GiB, which exceeds the device's reported 6 GiB; an independent `nvidia-smi` sampler saw device memory rise to 5,977 MiB of 6,144 MiB (89 samples). Those measures differ materially under this Windows/WDDM run, so the benchmark does not establish a reliable 6 GiB full-model fit. It completed with sampled device memory nearly saturated and PyTorch allocation above physical capacity; there is effectively no practical headroom for other GPU use or longer examples. Treat full fine-tuning as **not robustly feasible on this 6 GB card without another memory-saving method**, and benchmark that method before any full run. Mean throughput was 0.1282 examples/s. The first AdamW step allocates optimizer state and is included in the three-step mean. The benchmark catches and records `torch.cuda.OutOfMemoryError`; this run did not raise one. The `nvidia-smi` sampler is periodic and may miss brief peaks, especially for the short frozen comparison (two samples).

The frozen-encoder comparison fits at about 2.32 GiB peak allocated and successfully updates all heads, but it **changes the training objective**. It is a possible constrained baseline only if separately approved; it is not equivalent to full fine-tuning. Gradient accumulation cannot reduce the model, gradients, and optimizer-state memory of a single full-model microbatch. Smaller audio batches are already tested at batch size one. Further full fine-tuning would need a different memory strategy (for example a memory-efficient optimizer or offload) and a fresh benchmark before training.

Raw measurements are saved in `reports/gpu_benchmark.json`.

## Measured throughput projections and uncertainty

The actual eligible count is unknown; use `eligible_rows / measured_examples_per_second` for a projection. For scale only, applying the measured three-step mean rates to the **228,941 source train rows before filtering** gives:

| Mode | At source-row count, one epoch | At source-row count, 5 epochs |
|---|---:|---:|
| Full fine-tune, no-headroom benchmark | 496.1 hours (20.67 days) | 103.4 days |
| Frozen encoder, altered objective | 5.98 hours | 29.9 hours |

These are arithmetic extrapolations from three clips of 6.02–6.28 seconds, not stable throughput forecasts. They omit evaluation, checkpoint writes, data stalls, warmup, retries, and broader duration/reciter/configuration variation. Because the eligible count is below or equal to source count, multiply the applicable per-row time by the eventual eligible count instead. The frozen-mode estimate is only useful if the different objective is accepted; full fine-tune's estimate is not a recommendation because its memory readings leave no reliable headroom.

## Verification and open decisions

- Original cached checkpoint: exact mapping verified, 816 tensor matches, no missing/unexpected/mismatched keys.
- `tests/test_model.py`, `tests/test_training_collation.py`, and `tests/test_training_checkpointing.py`: automated coverage for mapping, model loss masks, variable lengths, config, and restart state.
- Final verification: `py -m pytest tests/test_model.py tests/test_training_collation.py tests/test_training_checkpointing.py -q` — **24 passed**; full `py -m pytest -q` — **91 passed**.
- One real training example and a one-step GPU benchmark were used; no epoch, filtering census, or full dataset download occurred.

Before Part B, decide whether to keep full fine-tuning and adopt a memory strategy that fits the 6 GiB limit, or explicitly approve frozen/partially frozen encoder training as a different objective. Also establish the eligible training-row count and tune settings using validation data. No Part B training was started.
