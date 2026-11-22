# Stage 6 — Multi-Level CTC model architecture

## Scope and evidence

This stage implements the published model shape without training or downloading model weights. Reference implementation inspected in the read-only upstream checkout `../quran-muaalem-reference`:

- `src/quran_muaalem/modeling/configuration_multi_level_ctc.py` defines the custom config, head vocab sizes, loss weights, blank ID, and Wav2Vec2-BERT parameters.
- `src/quran_muaalem/modeling/modeling_multi_level_ctc.py` instantiates the encoder, dropout, one linear head per level, feature-frame length calculation, and weighted CTC loss.
- `src/quran_muaalem/modeling/vocab.py` defines the phoneme and categorical vocab construction with `[PAD]` at ID 0.
- `src/quran_muaalem/modeling/multi_level_tokenizer.py` builds separate token streams for phonemes and each `SifaOutput` property.
- `src/quran_muaalem/inference.py` passes 16 kHz waves through `AutoFeatureExtractor`, then into the model as `input_features`.
- `src/quran_muaalem/decode.py` contains inference-time greedy CTC decoding; this stage does not replace or reimplement that decoder.
- `docs/training_dataset_report.md` records Stage 5's label IDs, per-group sifat alignment, and `-100` target padding convention.

The publicly accessible `obadx/muaalem-model-v3_2` checkpoint `config.json` and `vocab.json` were inspected at revision `01a1ef9fbe40d144ef845101e89ff924aed3fef5` via Hugging Face metadata/resolve endpoints. A range request read only the 107,000-byte safetensors header of `model.safetensors` (2,423,124,012 bytes total); the weight tensors were not downloaded. Its header lists 816 tensor keys, with encoder keys under `wav2vec2_bert.*` and head keys under `level_to_lm_head.*`.

## Architecture

```text
Stage 5 waveform: float32, mono, 16 kHz
       │
       ▼
Wav2Vec2-BERT AutoFeatureExtractor (160-dimensional features)
       │ input_features: (batch, feature_frames, 160)
       ▼
Wav2Vec2BertModel (24 Conformer layers; hidden size 1024)
       │
       ├── checkpoint adapter (one layer, kernel 3, stride 2)
       ▼
shared hidden frames (batch, output_frames, 1024)
       ├── linear CTC head: phonemes (43)
       ├── linear CTC head: hams_or_jahr (3)
       ├── linear CTC head: shidda_or_rakhawa (4)
       ├── linear CTC head: tafkheem_or_taqeeq (4)
       ├── linear CTC head: itbaq (3)
       ├── linear CTC head: safeer (3)
       ├── linear CTC head: qalqla (3)
       ├── linear CTC head: tikraar (3)
       ├── linear CTC head: tafashie (3)
       ├── linear CTC head: istitala (3)
       └── linear CTC head: ghonna (3)
```

Each head independently maps the same hidden frame sequence to logits `(batch, output_frames, head_classes)`. Phoneme labels are phonetic symbols (42 nonblank symbols); each of the ten categorical sifat labels corresponds to a phoneme group, not an individual Unicode character. Separate streams are decoded with CTC at inference and aligned by the original inference/decode path.

## Encoder and head configuration

The original checkpoint config reports a Wav2Vec2-BERT encoder with hidden size 1024, 24 hidden layers, 16 attention heads, intermediate size 4096, feature input dimension 160, relative-key position embeddings, depthwise convolution kernel 31, and one adapter layer (kernel 3, stride 2, output hidden size 1024). It sets hidden/attention/activation/feature-projection dropout to 0, layerdrop to 0, conformer convolution dropout to 0.1, and final head dropout to 0.1. Spec augment is enabled in the checkpoint config, with `mask_time_prob=0` and `mask_feature_prob=0`. A direct comparison against the checkpoint config found all 42 encoder settings included in the implementation defaults identical; the eleven head sizes and all loss weights also match exactly.

Output classes include blank ID 0:

| Level | Classes | Meaning of nonblank IDs |
|---|---:|---|
| `phonemes` | 43 | 42 phonetic symbols |
| `hams_or_jahr` | 3 | two categorical values |
| `shidda_or_rakhawa` | 4 | three categorical values |
| `tafkheem_or_taqeeq` | 4 | three categorical values |
| `itbaq` | 3 | two categorical values |
| `safeer` | 3 | two categorical values |
| `qalqla` | 3 | two categorical values |
| `tikraar` | 3 | two categorical values |
| `tafashie` | 3 | two categorical values |
| `istitala` | 3 | two categorical values |
| `ghonna` | 3 | two categorical values |

The actual ordered ID-to-symbol/category mapping remains the checkpoint vocabulary established by Stage 5; this implementation does not rebuild or alter it. Stage 5's `src/quran_training_data/encoding.py` (`LabelCodec`) derives those mappings from the pinned `quran-transcript==0.6.1` alphabet and `SifaOutput` literals, and Stage 5 verified its IDs against the saved `vocab.json`. The new model test asserts that `LabelCodec.level_vocab_sizes` exactly equals all eleven output sizes. Safetensors header tensor shapes independently confirm the head matrices are `[43,1024]` for phonemes and `[3|4,1024]` for each sifat level, with matching output biases.

## Lengths, masks, CTC blank and target padding

The model receives Wav2Vec2-BERT feature frames, not raw waveform samples. With no feature attention mask, each batch item's input length is the feature tensor's frame count. With a mask, its valid feature length is the sum of that row. The encoder's `_get_feat_extract_output_lengths` computes the output length, including adapter convolution downsampling. Logit tensors are padded to the longest batch output; `output_lengths` gives valid per-item CTC lengths.

All levels use `pad_token_id=0` as the CTC blank. The unpadded label domain is IDs `1..vocab_size-1`. Stage 5 pads batch targets with `-100`; the loss masks those locations out before flattening targets. `-100` is never treated as the blank and is never sent to `CTCLoss`. The model's optional loss applies per-level weighted CTC loss with the checkpoint weights and `ctc_loss_reduction="mean"`, `ctc_zero_infinity=false`. `mean` is the default both in the checkpoint and in `QuranMultiLevelCTCConfig`; explicitly configured supported PyTorch reductions such as `sum` are retained and passed to `CTCLoss`. Config serialization/reloading preserves the default `mean` setting.

### Partial-level supervision

When labels are supplied, the original `src/quran_muaalem/modeling/modeling_multi_level_ctc.py` iterates the provided label dictionary and calculates loss only for those levels; it does not require all eleven label tensors. The new `QuranMultiLevelCTC.forward` preserves this intentional partial-level behavior: every head still returns logits, but only supplied levels contribute to loss. Their configured weights are used unchanged and are not renormalized across the subset. This supports batches where only a subset of annotations is available while keeping the reference loss convention. Unknown level names remain errors.

Checkpoint loss weights: phonemes `0.4`; `shidda_or_rakhawa` and `tafkheem_or_taqeeq` `0.0605` each; the remaining eight properties `0.059875` each.

## Input contract

The original `Muaalem.__call__` in `src/quran_muaalem/inference.py` checks sampling rate 16,000, runs waveforms through the checkpoint `AutoFeatureExtractor`, then calls the model using that feature mapping. The model itself consumes `input_features=(B,T,160)`. New `QuranMultiLevelCTC.forward` preserves that architecture-level contract. `forward_waveforms` is a convenience adapter for Stage 5 float waveform batches and a caller-supplied matching HF feature extractor. This keeps feature extraction explicit and avoids silently changing waveform/audio behavior.

## Implementation and checkpoint compatibility

New project files:

- `src/quran_model/configuration.py`: separately stores the complete Wav2Vec2-BERT config, head dimensions, loss weights, and pad conventions.
- `src/quran_model/modeling.py`: Wav2Vec2-BERT encoder, dropout, eleven independent linear heads, output lengths, optional weighted CTC loss, and a feature-extractor waveform helper.
- `tests/test_model.py`: tiny synthetic Wav2Vec2-BERT tests for outputs, masks/lengths, target padding and blank, gradients, waveform adapter, and serialization.
- `notebooks/06_model_architecture.ipynb`: a small, random-weight forward demonstration.

Exact matches to the checkpoint: encoder family and reported default settings, feature dimension, adapter setup, all eleven head names and class dimensions, blank ID, target padding value, and per-level CTC loss weights/reduction. Head parameter dimensions match; head initialization remains random unless loaded from trained weights.

Serialization intentionally nests encoder settings under `encoder_config`, while the original custom checkpoint stores Wav2Vec2-BERT settings at the top level under custom `model_type="multi_level_ctc"`. Therefore config JSON files are not byte-for-byte interchangeable. Original model state keys use `wav2vec2_bert.*`; this implementation uses `encoder.*`. Head keys retain `level_to_lm_head.<level>.*`. `from_pretrained_encoder` requests only `Wav2Vec2BertModel` weights from a supplied checkpoint and keeps new heads initialized, but loading the full original model into this class is not yet verified because it would require checkpoint tensor loading and compatibility mapping. The checkpoint's public tensor header confirms key namespaces and head tensors without loading tensors.

Remaining compatibility check before training: perform a controlled encoder-only weight load from the original checkpoint (or map `wav2vec2_bert.*` to `encoder.*`) and compare missing/unexpected keys and tensor shapes. This was deliberately not done here to comply with the no-large-weight-download restriction. A full checkpoint load also requires the original `vocab.json` mappings and appropriate HF feature extractor; head sizes alone do not define label IDs.

## Verification

`tests/test_model.py` uses a compact Wav2Vec2-BERT config (one layer, hidden size 16, with adapter behavior tested separately) and synthetic inputs. It verifies eleven output tensors and exact class sizes, feature masks and valid lengths, CTC blank/padding validation, finite loss and gradients through the encoder and heads, partial-level loss, default and explicit reduction configuration, prepared waveform adapter behavior with a fake extractor, and `save_pretrained`/`from_pretrained` config and weight round-trip. The notebook uses the same tiny encoder and prints every head shape. The production default config retains the original adapter.

Latest verification after the reduction and label-completeness review: `py -m pytest tests/test_model.py -q` — **11 passed**; `py -m pytest -q` — **passed**; notebook execution with `py -m jupyter nbconvert --to notebook --execute notebooks/06_model_architecture.ipynb --inplace` — **completed successfully**. The notebook kernel emitted a Windows Proactor/ZMQ runtime warning, but all cells executed and outputs were saved.

Notebook execution from the project root requires the project dependencies and `nbclient`/`nbformat`; run with:

```powershell
$env:PYTHONPATH = "src"
python -m jupyter nbconvert --to notebook --execute notebooks/06_model_architecture.ipynb --inplace
```

No model was trained, and no full checkpoint weights were downloaded.
