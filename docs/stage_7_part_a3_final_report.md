# Stage 7 Part A.3 — Final Verification Report

## Decision

**Part A.3 is incomplete. Stage 7 Part B was not started.** The required two-update experiment failed the approved GPU safety guard during the first backward pass, before any optimizer update. The experiment was stopped as required. No resource threshold, model architecture, or training configuration was changed.

## Phase 1 — Selection

The latest candidate scan in [`runs/candidate_availability_20261122T154412Z`](../runs/candidate_availability_20261122T154412Z/) found 196 selector-level training candidates across the scanned windows; 169 newly inspected rows also passed annotation validation and label encoding. The final bounded selection used the persisted candidate IDs from that scan, then reloaded the two chosen configurations through the Stage 2 loader to recover complete records and audio for the chosen items.

The selected set is in [`runs/pilot_final_selection_20261122T160812Z/selection_report.json`](../runs/pilot_final_selection_20261122T160812Z/selection_report.json), with the human-readable record at `selection_report.md`.

- **Training:** 32 unique examples, 16 each from `moshaf_11.0` / reciter 11 and `moshaf_27.0` / reciter 27. Both are manifest-training groups. The candidate scan had 22 and 20 annotation-valid candidates in those configurations, respectively.
- **Validation:** one example, `moshaf_13.0:13.0:002.0055`, from manifest-validation reciter 13 and moshaf `13.0`.
- **Candidate-pool requirement:** passed; 196 selector candidates and 169 new annotation-valid candidates exceed 32.
- **Identity and split requirements:** all 32 training IDs are unique; reciter and moshaf sets are disjoint between train and validation.
- **Sura/text preference:** not met. All disjoint held-out validation candidates were absent in the inspected 80-row validation windows. The selected validation example shares sura 2 and normalized text with training. This is explicitly recorded as a preference miss, not a reciter/moshaf split violation.
- The earlier final-selection attempts that stopped on dataset-server 502 responses or a code assertion are retained in their run directories; no prior run was overwritten.

The Stage 2 streaming loader returns encoded audio bytes even when decoding is disabled. To recover full records for selection, it streamed 80 rows from each of two training and three validation configurations (400 bounded rows). It decoded only the selected 32 training and one validation example; it did not download the complete dataset. The current cached dataset `main` pointer was `8e3fe50ec5efcc4beab054acdbd0178bc690c246`, matching the split manifest revision.

## Phase 2 — Selected audio and labels

The final selection report records metadata and decoded duration for each of the 33 selected examples. All 33 passed without rejection or replacement:

- Training decoded duration range: **3.17–6.07 seconds**.
- Validation decoded duration: **5.92 seconds**.
- Maximum decoded duration: **6.07 seconds**, below the strict 6.3-second cap.
- Sampling rate: **16,000 Hz** for every item; channel layout: **mono** for every item.
- All decoded samples were finite. Audio preparation returned no invalid-audio errors.
- `prepare_record`, the annotation validator, and `LabelCodec` accepted the labels. Phoneme/sifat alignment passed; all examples provide all eleven label levels.

Audio arrays and encoded labels are retained in `prepared_audio.npz` and `prepared_labels.json` within the selection run directory for reproducibility. The source annotations were not edited.

## Phase 3 — Controlled optimizer experiment

The experiment used the verified original checkpoint `obadx/muaalem-model-v3_2` at revision `01a1ef9fbe40d144ef845101e89ff924aed3fef5`. The checkpoint audit matched **all 816 tensors** and **605,754,251 parameters**, with no missing keys, unexpected keys, or shape mismatches. All parameters were configured for full fine-tuning.

The run used the existing configuration: Adafactor, fixed learning rate `1e-5`, weight decay `0.01`, FP16 autocast and GradScaler, gradient checkpointing, microbatch 1, accumulation 16, linear schedule, and one pilot epoch planned for two updates. Safety limits remained at 92% PyTorch reserved memory, 5,700 MiB NVIDIA device use, and 2 GiB minimum available system RAM.

The failure is recorded in [`runs/pilot_a3_updates_20261122T161535Z/first_half/failure_report.json`](../runs/pilot_a3_updates_20261122T161535Z/first_half/failure_report.json), and the top-level result is `pilot_result.json` in that run directory.

- Failure type: `ResourceLimitReached`, raised by the configured guard after the first microbatch backward, at example `moshaf_11.0:11.0:002.0038` (3.89 seconds).
- Device: NVIDIA GeForce RTX 3060 Laptop GPU, device 0, **6,441,926,656 bytes** total capacity.
- Reserved memory: **6,205,472,768 bytes (96.3%)**, above the unchanged 92% threshold. Peak allocated memory was **6,114,705,920 bytes**.
- NVIDIA monitoring sampled 49 times; the measured peak was **5,981 MiB**, above its 5,700 MiB limit. The reserved-memory check raised first. Minimum available system RAM was about **7.92 GiB**, above its 2 GiB floor.
- This was a safety-guard stop, **not a CUDA OOM**.
- Global step remained **0**. `metrics.jsonl` is empty. The initial checkpoint exists and was inspected using `FakeTensorMode`: global step 0, Adafactor state entries 0, scheduler total steps 2 / warmup 1 / current step 0, and GradScaler scale 65,536. No optimizer step, parameter update, scheduler advancement, optimizer-state initialization, or resume occurred.
- Validation evaluation ran before training reached the backward pass. The engine’s parameter-version check is part of evaluation; however, the exception prevented the validation loss from being returned and persisted in the run result. No post-update validation exists.
- Encoder and supervised-head parameter changes are therefore unverified and must be treated as not demonstrated.

The saved initial checkpoint is `runs/pilot_a3_updates_20261122T161535Z/first_half/checkpoints/initial.pt`. It is a step-zero checkpoint only and does not demonstrate resumability after an optimizer update.

## Acceptance gate

| Requirement | Result |
|---|---|
| Exactly 32 training examples and 1–3 validation examples | Pass |
| Candidate pool greater than 32 | Pass |
| Unique IDs, at least two training reciters/configurations | Pass |
| Held-out validation reciter and moshaf | Pass |
| Strict decoded duration, rate, mono, finite audio, valid labels | Pass |
| Two successful optimizer updates | **Fail — zero updates** |
| Finite training losses/gradients and nonzero parameter changes | **Not demonstrated** |
| Scheduler advancement and learning-rate trace | **Fail — scheduler stayed at step 0** |
| Post-update held-out validation without parameter mutation | **Not demonstrated** |
| Save/reload/resume after an actual optimizer update | **Not demonstrated** |
| GPU memory within approved limits | **Fail — guard stopped at 96.3% reserved** |
| Focused and full test suites for this final state | **Not run** |

The prior preflight report documents earlier runs of 14 focused and 109 full tests, but those are not a passing test result for this final pilot state. This investigation stopped at the failed GPU gate; no test suite was launched afterward.

## Exact blocker and next decision

The 6 GiB GPU cannot complete even the first accumulated update within the existing 92% reserved-memory and 5,700 MiB device-use limits for this full-fine-tuning setup. The run stopped before the first optimizer step, so the Stage 7 Part A.3 completion gate is not met. The user’s instruction prohibits lowering limits, freezing layers, reducing model size, or silently changing hyperparameters; no such change was made.

Consequently, **do not proceed to Part B**. A later attempt needs an approved resource/configuration change or a machine with sufficient GPU memory, followed by a fresh bounded two-update experiment and all remaining acceptance checks. No full-dataset training, full-dataset download, validation/test training, or test-based model selection occurred.
