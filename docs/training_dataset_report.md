# Stage 5 — Training dataset preparation report

**Scope:** target encoding, record policies, reproducible split manifest, bounded statistics, and a small loader-to-audio-to-label smoke run. No model was trained, no annotations were edited, and the complete audio corpus was not downloaded or materialized.

## Evidence reviewed

Project evidence: `docs/dataset_exploration_report.md`, `docs/dataset_loading_report.md`, `docs/audio_preprocessing_report.md`, `docs/annotation_validation_report.md`; `configs/dataset.json`, `configs/audio_preprocessing.json`, `configs/training_data.json`; `src/quran_dataset/loader.py`, `src/quran_audio/preprocessing.py`, and `src/quran_annotations/validation.py`.

Read-only upstream checkout: `../quran-muaalem-reference`.

- `src/quran_muaalem/modeling/vocab.py` creates phoneme IDs by putting `[PAD]` at 0 and enumerating `quran_transcript.alphabet.phonetics` from ID 1. For each `SifaOutput` `Literal` field, it enumerates the literal values from ID 1 after mapping them to bracketed Arabic tokens.
- `src/quran_muaalem/modeling/multi_level_tokenizer.py` tokenizes the raw phonetic string and each sifat property as a separate level. Sifat properties are assembled one label per `SifaOutput` entry. The project has separate tokenizers/output vocabularies for the phoneme level and each of the ten categorical sifat fields.
- `src/quran_muaalem/modeling/modeling_multi_level_ctc.py` uses `config.pad_token_id` as the CTC blank and masks target labels below zero before calculating loss. Its comments specify padded label positions use `-100`.
- `src/quran_muaalem/modeling/configuration_multi_level_ctc.py` documents separate output sizes per level.
- `src/quran_muaalem/decode.py` uses ID 0 as the CTC blank and removes it during greedy decoding. `src/quran_muaalem/inference.py` generates references with `quran_phonetizer(..., remove_spaces=True)` and chunks phonemes to align sifat groups.
- `docs/en/quran-transcript/usage.md` and `docs/en/muaalem/python-api.md` show the `remove_spaces=True` expectation. The installed `quran_transcript.phonetics.phonetizer.quran_phonetizer` source replaces whitespace with the reference script's space marker and removes that marker when `remove_spaces=True`.
- The upstream checkout has no training loop, dataset collator, or script that prepared the published dataset. Its `uv.lock` pins `quran-transcript==0.6.1`; the environment had 0.6.4. The phoneme inventory and `SifaOutput` field/value definitions were compared across those versions.

## Phoneme inventory and 42/43 discrepancy

There is no missing 43rd ordinary phoneme in the installed reference package:

1. The original lockfile pins `quran-transcript` 0.6.1. Its wheel's `alphabet.phonetics` object has **42** symbol values. Installed 0.6.4 also has **42**, with the same ordered values and `SifaOutput` fields.
2. The original model checkpoint [`obadx/muaalem-model-v3_2` config](https://huggingface.co/obadx/muaalem-model-v3_2/blob/01a1ef9fbe40d144ef845101e89ff924aed3fefb/config.json), revision `01a1ef9fbe40d144ef845101e89ff924aed3fefb`, has `level_to_vocab_size.phonemes = 43`. Its [`vocab.json`](https://huggingface.co/obadx/muaalem-model-v3_2/blob/01a1ef9fbe40d144ef845101e89ff924aed3fefb/vocab.json) phoneme mapping has 43 entries total: `[PAD]` at ID 0 plus the same 42 symbols at IDs 1–42. The ID mapping exactly matches the original `vocab.py` construction order.
3. The upstream `docs/en/quran-transcript/phonetics.md` calls its table “Phonemes (43)”. The table heading is therefore consistent with the model's total class count only if it counts `[PAD]`; it is not evidence of a 43rd ordinary symbol. The additional tokenizer metadata tokens `<s>`, `</s>`, and `<unk>` are not output classes in the saved phoneme head.
4. All symbols in the bounded real annotation sample were within the reference alphabet. No new symbol was added.

The project now pins `quran-transcript==0.6.1`, matching the upstream lockfile. The exact ordered inventory is generated from that package rather than copied into code.

### Space handling is necessary for the CTC head

Published `phonemes` strings contain ordinary spaces between words. The original checkpoint tokenizer maps a space to its word-delimiter token (`|`, ID 46), while the phoneme output head has only IDs 0–42. The upstream inference path asks `quran_phonetizer` for `remove_spaces=True`; on a sampled dataset row, removing the spaces produced IDs exactly equal to `Wav2Vec2CTCTokenizer` for the published checkpoint. Passing the row's spaces directly would add an ID outside that output head.

Prepared examples therefore preserve two separate fields:

- `phoneme_text`: the original annotation verbatim;
- `phoneme_text_for_encoding`: a copy with ASCII inter-word spaces removed, matching the upstream phonetizer option.

Only the copy is encoded. Other unsupported whitespace/symbols are rejected. An unsupported symbol tested against the original tokenizer produced a `None` ID and caused tokenizer encoding to fail; the saved model head has no unknown-token output class. `LabelCodec` raises `UnknownLabelError` instead of emitting a fabricated or out-of-range label.

## Target IDs, padding, and sifat groups

`src/quran_training_data/encoding.py` reproduces the ID assignment rules:

- Phonemes are encoded as individual Unicode symbols with IDs 1–42; `[PAD]` is 0.
- The model's CTC blank is the same ID as pad: 0. Unpadded target sequences never contain 0.
- Batch positions outside a target sequence are `-100`, matching the model's loss mask. This is distinct from the CTC blank class.
- Every sifat property is a separate target level. Each `Literal` value is assigned its upstream ordinal ID starting at 1, with 0 reserved for pad/blank. The source mapping in `vocab.py` and published checkpoint category vocabularies were checked to agree.
- A sifat entry describes one group from `chunck_phonemes`, not one Unicode character. Phoneme labels have one ID per symbol; each sifat level has one ID per group. `TrainingExample` retains raw group strings and their symbol IDs, and preparation rejects any group/count/order mismatch.

Published model head sizes, including ID 0:

| Level | Classes |
|---|---:|
| `phonemes` | 43 |
| `hams_or_jahr` | 3 |
| `shidda_or_rakhawa` | 4 |
| `tafkheem_or_taqeeq` | 4 |
| `itbaq` | 3 |
| `safeer` | 3 |
| `qalqla` | 3 |
| `tikraar` | 3 |
| `tafashie` | 3 |
| `istitala` | 3 |
| `ghonna` | 3 |

Round-trip decoders reject pad/blank and ignore IDs inside a target sequence. Phoneme and sifat labels are round-tripped in tests and in the real-data smoke check.

## Preparation module and example trace

`src/quran_training_data/` contains:

- `preparation.py`: lazy `TrainingDataBuilder` over the Stage 2 `DatasetLoader`; one-record `prepare_record`; source tracing; policy evaluation; explicit `ExcludedRecord` reasons; optional JSONL exclusion ledger.
- `encoding.py`: strict reference-compatible phoneme and sifat encoders/decoders and level vocabulary sizes.
- `batching.py`: per-level right padding and target lengths.
- `splitting.py`: reproducible reciter-group assignment, row-to-split lookup, leakage assertion, and expected random-segment group overlap calculation.
- `statistics.py`: incremental duration, label-frequency, inclusion/exclusion, and split counters. It retains summary counters only, not waveforms.

An example preserves a compound source key (`configuration:moshaf_id:segment_index`), reciter/moshaf IDs, recording/index metadata, timestamps, spans, flags, match ratio, and original text fields. Processed audio uses the Stage 3 `AudioProcessingConfig`; the immutable decoded source snapshot remains in `ProcessedAudio.original`. The tested defaults convert to float32, target 16 kHz, do not downmix mono data unnecessarily, normalize, denoise, or trim. Stage 3 records no-op operations, so the smoke output lists configured operations together with their `applied` status.

## Inclusion policy

`configs/training_data.json` records the defaults:

- Include non-Quran records when their annotations and audio are valid. This is explicit and configurable. The original repository has no training implementation establishing whether such records were included; the published records do carry validated phoneme/sifat targets, and the upstream phonetic package has explicit non-Quran phrases. The default preserves these labeled records instead of silently discarding them. Decide whether they fit the intended training objective before training.
- Null spans are allowed and retained. They occur on non-Quran rows; span meaning is still unresolved.
- No minimum match-ratio threshold is set because the score's formula is undocumented.
- No short/long duration cutoff is set because corpus-wide duration distribution and a model input limit were not established.
- Require mono 16 kHz model input by default, matching the original inference contract. Stage 3 resamples to 16 kHz by default. Stereo is not silently downmixed: enable the Stage 3 mono conversion explicitly or the row is excluded with `audio:not_mono`; a residual wrong rate is excluded with `audio:wrong_sample_rate`.
- Missing phonemes/sifat and invalid annotations are excluded with explicit reason codes by default. Policies can instead be set to `error` to stop at the first such row. Invalid target symbols, malformed audio, and failed group alignment also receive reasons.
- Optional match-ratio and duration thresholds are configurable. Every `PreparedRecord` is either an example or an exclusion; `TrainingDataBuilder` can append exclusions to a JSONL ledger.

Because no original training loop exists, the non-Quran default is a documented data-preservation choice, not a recovered historical training policy.

## Leakage-aware split manifest

Only a `train` split is published. `configs/reciter_split_manifest.json` is a custom, deterministic manifest pinned to dataset revision `8e3fe50ec5efcc4beab054acdbd0178bc690c246` (Hugging Face repo metadata), built from the 22-row reciter metadata and the per-configuration row counts. It maps every reciter and each of the 27 moshaf set IDs to a split; rows resolve by `reciter_id` and are cross-checked against `moshaf_id`. It does not enumerate or download each audio row.

The selected group strategy holds out whole reciters, keeping all moshaf sets/recordings of a reciter together. A seeded exhaustive search over three validation and three test reciter groups minimizes squared deviation from 80/10/10 source-row proportions. It assigns:

| Split | Source rows before quality filtering | Share | Reciters | Moshaf sets |
|---|---:|---:|---:|---:|
| Train | 228,941 | 79.899% | 16 | 21 |
| Validation | 29,125 | 10.164% | 3 | 3 |
| Test | 28,471 | 9.936% | 3 | 3 |
| Total | 286,537 | 100% | 22 | 27 |

The manifest script is `scripts/build_reciter_split_manifest.py`; rerun with `python scripts/build_reciter_split_manifest.py`. It fetches only dataset revision, metadata rows, and config row counts, not audio.

### Random segment split comparison

The 22 reciter groups range from 7,922 to 42,472 source rows. Under independent 80/10/10 per-segment random assignment, the probability that a group with even the minimum 7,922 rows is absent from validation is `0.9^7922` (about `10^-362`); absence from test has the same probability. The expected overlap calculation in `expected_random_segment_group_leakage` therefore gives all **22 reciters in each split** and all 22 reciters represented in multiple splits. This is the expected leakage under the stated random-assignment model, not a sampled random split. The reciter-group split prevents that speaker/recording overlap.

The sample confirms repeated segment identifiers/content across reciters/configurations in the opening rows. Reciter holdout does not prevent the same Quran text/ayah from appearing across different reciters; a text-exclusive split would require a different evaluation objective and likely many connected/combined groups. No corpus-wide duplicate audio hash scan was performed.

## Statistics

### Corpus metadata and split distribution

The datasets-server size metadata reports **286,537 recitation rows** across 27 configurations. All 22 reciters and 27 moshaf sets are covered by the split manifest. These are source counts before validating every annotation or downloading each audio asset. Corpus-wide usable/excluded counts are therefore **not yet known**.

### Bounded annotation sample

A separate bounded sample inspected the first five annotation rows from `moshaf_0.1`, `moshaf_1.0`, and `moshaf_26.1` (15 records total). It contained 12 `has_quran=true` and 3 `has_quran=false` rows, 3 unique reciters and 3 unique moshaf sets. All 15 passed Stage 4 annotation validation; 0 had missing phonemes/sifat/text; 3 had both spans null (all three were non-Quran opening invocations); none had only one span null. Observed `match_ratio` values ran from about 0.8696 to 1.0; no cutoff was applied. The real loader smoke included 3/3 records with no exclusions. These are sample results, not corpus-level rates.

| Sample measure | Min | Median | Max |
|---|---:|---:|---:|
| Audio duration metadata (seconds) | 3.70 | 5.075 | 6.28 |
| Space-free phoneme target length (symbols) | 19 | 32 | 41 |
| Sifat groups per row | 10 | 15 | 19 |

Stage 3's limited decoded samples were finite, mono, 16 kHz, and below its clipping threshold. The Stage 5 loader smoke also decoded 3 short WAVs (4.71–6.06 s), with expected float32/mono/16 kHz properties and no enabled denoising, normalization, or trimming.

Phoneme character counts across the 15 annotation rows (spaces excluded):

| Symbol | Count | Symbol | Count | Symbol | Count |
|---|---:|---|---:|---|---:|
| `ء` | 9 | `ا` | 48 | `ب` | 12 |
| `ج` | 3 | `ح` | 15 | `د` | 9 |
| `ذ` | 3 | `ر` | 33 | `س` | 3 |
| `ش` | 6 | `ط` | 3 | `ع` | 6 |
| `ك` | 3 | `ل` | 33 | `م` | 33 |
| `ن` | 18 | `ه` | 9 | `و` | 3 |
| `ي` | 6 | `َ` | 66 | `ُ` | 9 |
| `ِ` | 60 | `ۥ` | 6 | `ۦ` | 50 |

The following 18 reference symbols had count 0 in this small sample: `ت`, `ث`, `خ`, `ز`, `ص`, `ض`, `ظ`, `غ`, `ف`, `ق`, `۪`, `ـ`, `ٲ`, `ڇ`, `ں`, `۾`, `ۜ`, `ؙ`. Do not treat sample absence as evidence they are unused in the full corpus.

Sifat category counts over the sample's **216 groups**:

| Property | Observed category counts |
|---|---|
| `ghonna` | `maghnoon` 51; `not_maghnoon` 165 |
| `hams_or_jahr` | `hams` 33; `jahr` 183 |
| `shidda_or_rakhawa` | `shadeed` 33; `between` 99; `rikhw` 84 |
| `tafkheem_or_taqeeq` | `mofakham` 24; `moraqaq` 192; `low_mofakham` 0 |
| `itbaq` | `monfateh` 213; `motbaq` 3 |
| `safeer` | `safeer` 3; `no_safeer` 213 |
| `qalqla` | `moqalqal` 0; `not_moqalqal` 216 |
| `tikraar` | `mokarar` 18; `not_mokarar` 198 |
| `tafashie` | `motafashie` 3; `not_motafashie` 213 |
| `istitala` | `mostateel` 0; `not_mostateel` 216 |

The beginning-of-recording sample overrepresents the opening invocations and Al-Fatiha excerpts. Its durations (3.70–6.28 seconds), target lengths (19–41 symbols), and group counts (10–19) are not corpus outlier limits; no min/max filter was applied. These frequencies cannot be used to prune or weight classes. `DatasetStatistics` can produce full streaming distributions while each eligible record is processed; no full pass was run here.

## Tests and verification

`tests/test_training_data.py` covers codec round-trips, unsupported labels, pad/blank/ignore IDs, output level sizes, group alignment, mono/rate-preserving audio preparation, source immutability, inclusion/exclusion reasons and ledger output, deterministic split generation, traceability, split leakage, and statistics.

Commands:

```powershell
$env:PYTHONPATH = "src"
python -m pytest
python scripts/real_training_data_smoke.py --configurations moshaf_0.1 moshaf_1.0 moshaf_26.1 --limit 1
python scripts/build_reciter_split_manifest.py
python -m nbconvert --to notebook --execute notebooks/05_training_dataset_preparation.ipynb --inplace
```

Project tests: **67 passed**. The real loader smoke prepared **3/3** samples, including non-Quran opening rows, with no exclusions. All phoneme target IDs round-tripped to their space-free strings, sifat group counts aligned, source annotation objects remained unchanged, and every example retained source IDs and the manifest split. The notebook executed **8/8 code cells** without errors and prepared **6/6** examples (three Quran, three non-Quran), with no exclusions. It uses a six-row bound (two per configuration); no corpus sweep is performed.

## Limitations and decisions before training

1. The original repo has no training pipeline, so the exact historical treatment of non-Quran rows, quality filtering, and collator batching cannot be recovered. Review the explicit policy before applying it to all rows.
2. Full-corpus annotation validation, usable/excluded counts, label frequencies, and duration outliers remain unknown. Current frequencies are a biased 15-row opening sample.
3. The group manifest is computed from metadata and published config row counts before quality filtering; exact post-filter split sizes require a full streaming preparation/audit pass.
4. The split prevents reciter/moshaf overlap but not Quran text/ayah overlap between different reciters. Confirm that speaker-independent evaluation is the intended metric.
5. The meanings/threshold of `match_ratio`, timestamp semantics, and span boundaries remain unresolved from Stage 4. Therefore no default score cutoff or span filtering is applied.
6. The tokenizer's model output and special-token metadata are not identical concepts. Target encoding rejects unsupported chars, and only the phoneme head's IDs 1–42 are permitted; `[PAD]`/blank 0 and `-100` batch padding remain distinct roles.

No model was built or trained. No full dataset/audio download or annotation edit was performed.
