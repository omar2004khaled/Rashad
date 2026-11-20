# Quran Muaalem dataset exploration report

**Exploration date:** 2026-11-20  
**Dataset:** [`obadx/muaalem-annotated-v3`](https://huggingface.co/datasets/obadx/muaalem-annotated-v3)  
**Original project:** [`obadx/quran-muaalem`](https://github.com/obadx/quran-muaalem)  
**Scope:** read-only source review and limited API/sample inspection. No preprocessing or training was performed. The 97.3 GB dataset payload was not downloaded.

## Executive findings

- The current datasets-server metadata reports **29 configurations**: 27 recitation configurations, `moshaf_metadata`, and `reciters_metadata`. Each exposes only a `train` split. Reported total is **286,586 rows**, including 27,000 metadata rows; therefore recitation configurations account for **286,537 rows**.
- Each recitation configuration reports 24 columns. The accessible first-row feature declarations for 21 of 27 recitation configurations match exactly. Six configurations return a server post-processing error: `moshaf_0.0`, `moshaf_5.0`, `moshaf_11.0`, `moshaf_19.0`, `moshaf_26.0`, and `moshaf_30.0`. Their counts are available. A small `/rows` sample could be retrieved for `moshaf_0.0`, but its declared schema was unavailable. For the other five sets, first-row inspection failed; they were not sampled through `/rows` in this run.
- Inspectable rows contain short WAV clips; HF declares `audio` as an Audio feature with 16,000 Hz. Two actual asset WAVs decoded at 16 kHz, mono, with durations of 6.06 s and 4.90 s, matching the row metadata.
- Rows distinguish Tarteel transcript, Imlaey text, Uthmani text, a phonetic script (`phonemes`), and per-unit phonetic-property records (`sifat`). Quran content is not guaranteed in every row: the inspected opening record is isti`aatha (`has_quran=false`); next records contain Quran text and populated `start_span`/`end_span`.
- This is already an annotated dataset, not just raw audio plus plain transcripts. The labels include a phonetic representation and sifat attributes, though their derivation and quality need broader validation before training.

## 1. Original project inspection

The provided working directory initially contained only `.git` and no project source checkout. A shallow, read-only clone of the upstream public repository was inspected at `../quran-muaalem-reference` (not modified). Main relevant paths:

| Original repository path | Purpose and finding |
| --- | --- |
| `README_EN.md` (also `README.md`) | Describes the intended pipeline: collect recitations, split by pauses (the README says not by ayah), obtain text with the Tarteel Quran Whisper model, correct it using tasmeea, convert Imlaey to Uthmani, phonetize with `quran-transcript`, then train Wav2Vec2-BERT. Also gives inference audio input expectation of 16 kHz. |
| `docs/en/training/data-sources.md` | Names external collection (`prepare-quran-dataset`), segmenter (`recitations-segmenter`), Quran-tuned Whisper, and `quran-transcript`; explains flow from raw audio to phoneme and sifat labels. |
| `docs/en/training/pipeline.md` | Conceptual collection ? segmentation ? transcription ? correction ? script conversion ? phonetization ? training flow. It recommends artifacts and train/valid/test splits, but does not implement this pipeline. |
| `docs/en/training/index.md`, `docs/en/training/evaluation.md` | Overview and suggested metrics/splits. Evaluation doc recommends PER, sifat F1, alignment metrics, etc.; it is guidance rather than runnable training/evaluation code. |
| `docs/en/pipeline/moshaf-attributes.md`, `docs/pipeline/moshaf-attributes.md` | Explain that `MoshafAttributes` options affect generated phonetic targets (madd, takbeer, sakt, etc.). |
| `src/quran_muaalem/modeling/multi_level_tokenizer.py` | `MultiLevelTokenizer` uses `quran_transcript.SifaOutput` and `quran_phonetizer`; tokenizes phonemes plus literal-valued sifat fields using saved HF tokenizers. This is inference-side tokenization support, not dataset preparation. |
| `src/quran_muaalem/modeling/vocab.py` | Builds vocabulary definitions from `quran_transcript` phonetics and sifat values. |
| `src/quran_muaalem/modeling/modeling_multi_level_ctc.py`, `configuration_multi_level_ctc.py` | Multi-head CTC model and model configuration. |
| `src/quran_muaalem/inference.py`, `decode.py` | Consumes 16 kHz audio with references and decodes/aligned phoneme and sifat predictions. |
| `src/quran_muaalem/muaalem_typing.py` | Types for model outputs/phonetic labels. |
| `pyproject.toml`, `uv.lock` | Package/build/dependency configuration. No `datasets` training dependency or training entrypoint is defined. |
| `tests/` | Inference/model/helper tests; no end-to-end dataset loading or training tests found. |
| `assets/` | Small demo/test audio and model cache; not the training corpus. |

**Included:** inference/model architecture, phoneme/sifat vocab/tokenizer support, user documentation describing an intended upstream data pipeline. **Not included:** a dataset loader for `muaalem-annotated-v3`, dataset transformation scripts, label-generation job, dataset split manifests, a training loop, optimizer/scheduler configuration, or reproducible training configuration. The docs? pipeline is conceptual and references external projects. Thus the current upstream code does not show a definitive training-time input contract for this published dataset.

## 2. Configurations, splits, counts

Counts below come from `https://datasets-server.huggingface.co/size?dataset=obadx/muaalem-annotated-v3`; split names come from the corresponding `/splits` endpoint. All are `train` only.

| Configuration | Rows | Columns |
| --- | ---: | ---: |
| `moshaf_0.0` | 9,133 | 24 |
| `moshaf_0.1` | 10,764 | 24 |
| `moshaf_0.2` | 9,971 | 24 |
| `moshaf_0.3` | 12,604 | 24 |
| `moshaf_1.0` | 10,939 | 24 |
| `moshaf_11.0` | 10,363 | 24 |
| `moshaf_12.0` | 9,880 | 24 |
| `moshaf_13.0` | 9,377 | 24 |
| `moshaf_19.0` | 11,278 | 24 |
| `moshaf_2.0` | 9,942 | 24 |
| `moshaf_2.1` | 10,394 | 24 |
| `moshaf_22.0` | 10,332 | 24 |
| `moshaf_24.0` | 9,868 | 24 |
| `moshaf_25.0` | 7,922 | 24 |
| `moshaf_26.0` | 11,565 | 24 |
| `moshaf_26.1` | 11,850 | 24 |
| `moshaf_27.0` | 11,213 | 24 |
| `moshaf_28.0` | 10,535 | 24 |
| `moshaf_29.0` | 11,061 | 24 |
| `moshaf_3.0` | 10,444 | 24 |
| `moshaf_30.0` | 11,312 | 24 |
| `moshaf_4.0` | 10,994 | 24 |
| `moshaf_5.0` | 11,482 | 24 |
| `moshaf_6.0` | 12,435 | 24 |
| `moshaf_7.0` | 9,907 | 24 |
| `moshaf_8.0` | 10,330 | 24 |
| `moshaf_9.0` | 10,642 | 24 |
| `moshaf_metadata` | 27 | 57 |
| `reciters_metadata` | 22 | 5 |

The 27 audio configurations sum to 286,537 rows. The repo metadata has 27 mushaf-set records and 22 reciter records. **No validation or test split is exposed by the datasets-server API.** This does not imply that one cannot create splits; it means there are no published split names in the inspected metadata.

## 3. Schema and columns

The first-row API returned identical feature declarations for all 21 successfully inspected recitation configurations. Six failed (listed above), so ?common schema? is verified for 21, not every config. Each config?s count metadata still reports 24 columns.

### Recitation configurations (24 declared fields)

| Field | Declared type | What the inspected data supports |
| --- | --- | --- |
| `audio` | Audio, sampling rate 16,000 | HF audio feature. Sample asset MIME/type is WAV; two actual WAVs decoded successfully. |
| `segment_index` | string | Example values `001.0000`, `001.0001`; appears to identify a segment within its recording set. It repeats across different sets, so use a compound key such as `(moshaf_id, segment_index)`. |
| `tarteel_transcript` | list[string] | Transcript output as a list of strings; inspected entries contain Arabic text. |
| `moshaf_id` | string | Recording-set identifier, e.g. `0.1`, `1.0`. |
| `moshaf_name` | string | Arabic recording-set name. |
| `reciter_id` | int32 | Numeric reciter identifier. |
| `reciter_arabic_name` | string | Arabic reciter name. |
| `reciter_english_name` | string | English reciter name. |
| `sura_or_aya_index` | string | Example `001`. Interpretation depends on `index_type`. |
| `index_type` | string | Example `sura`; likely describes recording indexing convention. Do not confuse it with per-row Quran span. |
| `sample_rate` | int32 | Sample rate metadata, 16000 in inspected rows. |
| `duration_seconds` | float32 | Segment duration metadata. |
| `timestamp_seconds` | list[float32] | Inspected first sample contains `[0.0, duration]`; likely boundaries in original audio, but broader semantics not verified. |
| `imlaey` | string | Arabic Imlaey-script text. |
| `uthmani` | string | Arabic Uthmani-script text, visibly differs in orthographic symbols (e.g. `?????????` vs `?????????`). |
| `has_quran` | bool | Whether row contains Quran text; explicitly false on sampled isti`aatha segment. |
| `has_istiaatha` | bool | Whether sampled row includes isti`aatha. |
| `has_bismillah` | bool | Flag for bismillah; false in first isti`aatha sample. |
| `has_sadaka` | bool | Dataset flag; exact intended meaning (?sadaqallah?/closing utterance) not verified from card in this inspection. |
| `start_span` | struct `{sura_idx:int32, aya_idx:int32, imlaey:int32, uthmani:int32}` or null | Quran span starting coordinates; null for sampled non-Quran isti`aatha. Quran sample starts at sura 1, aya 1, offsets 0. |
| `end_span` | same struct or null | Quran span ending coordinates; sampled values include character/word offset-like counts. Exact offset convention needs source documentation. |
| `match_ratio` | float32 | Present as an alignment/match score (0.9565 on sample isti`aatha; 1.0 on several Quran samples). Its formula and thresholds are unknown. |
| `phonemes` | string | Phonetic script, not ordinary Arabic transcript. Example includes doubled letters and Quranic combining signs such as `?`. |
| `sifat` | list of structs | Per-unit entries. The observed struct has 11 string fields: `phonemes`, `ghonna`, `hams_or_jahr`, `istitala`, `itbaq`, `qalqla`, `safeer`, `shidda_or_rakhawa`, `tafashie`, `tafkheem_or_taqeeq`, `tikraar`. `phonemes` associates the unit; other fields encode properties. |

### `moshaf_metadata` (27 rows, 57 fields)

The declared types are mostly `string`, with numeric `int32`/`float32`, booleans, and two nested list/struct fields. Column names and observed example support these groups:

- **Identity/source:** `id`, `path`, `name`, `publisher`, `sources`, `specific_sources`, `comments`.
- **Reciter/rewaya:** `reciter_id`, `reciter_arabic_name`, `reciter_english_name`, `rewaya`.
- **Coverage/status/summary:** `is_downloaded`, `is_complete`, `is_annotated`, `missing_recitations`, `num_recitations`, `total_size_mb`, `total_duraion_minutes` (spelling as published), `alias_start_sec`, `segmented_by`.
- **Recitation settings/rule choices:** `recitation_speed`, `takbeer`, `madd_monfasel_len`, `madd_mottasel_len`, `madd_mottasel_waqf`, `madd_aared_len`, `madd_alleen_len`, `madd_yaa_alayn_alharfy`, `saken_before_hamz`, `ghonna_lam_and_raa`, `meem_aal_imran`, `meem_mokhfah`, `sakt_iwaja`, `sakt_marqdena`, `sakt_man_raq`, `sakt_bal_ran`, `sakt_maleeyah`, `between_anfal_and_tawba`, `noon_and_yaseen`, `yaa_ataan`, `yabsut`, `bastah`, `bimusaytir`, `almusaytirun`, `start_with_ism`, `irkab_maana`, `idgham_nakhluqkum`, `noon_tamnna`, `yalhath_dhalik`, `tasheel_or_madd`, `harakat_daaf`, `raa_yasr`, `raa_misr`, `raa_nudhur`, `raa_firq`, `raa_alqitr`, `alif_salasila`.

The names above cover the 57 declared fields; the notebook displays the exact API type declaration and nesting for every field. The dataset card explains many Hafs settings and values. They describe selected recitation/mushaf conventions; they are not per-segment error labels.

### `reciters_metadata` (22 rows, 5 fields)

- `arabic_name`: string
- `moshaf_set_ids`: list[string]
- `english_name`: string
- `country_code`: string
- `id`: int32

Inspected example: `id=0`, Mahmoud Khalil Al-Husari / ????? ???? ??????, country code `EG`, linked to `0.0`, `0.1`, `0.2`, `0.3`.

## 4. Real sample examples

The following values came from the first rows API (`/rows`, small windows), not fabricated examples:

| Set / record | Reciter | Duration, SR | Annotation example | Notes |
| --- | --- | --- | --- | --- |
| `moshaf_0.1`, `001.0000` | Mahmoud Khalil Al-Husari | 6.06 s, 16 kHz | Imlaey: `??????? ????????? ???? ???????????? ??????????`; Uthmani: `??????? ????????? ???? ???????????? ??????????`; `has_quran=false`; `has_istiaatha=true`; no spans; `match_ratio?0.9565`. | Non-Quran invocation at opening; has phonemes and 19 sifat entries. |
| `moshaf_0.1`, `001.0001` | same | 6.02 s, 16 kHz | `?????? ??????? ???????????? ??????????`; `has_quran=true`; start/end `{sura_idx:1, aya_idx:1, imlaey:0/4, uthmani:0/4}`; `match_ratio=1.0`. | Quran row; 15 sifat entries. |
| `moshaf_0.1`, `001.0002` | same | 6.28 s, 16 kHz | `????????? ??????? ????? ?????????????`; span sura 1 aya 2, offsets 0?4; phonetic string `???????? ????????? ????? ??????????????`. | Demonstrates distinct phonetic target. |
| `moshaf_1.0`, `001.0000` | Mohammed Sidiq Al-Minshawi | 4.90 s, 16 kHz | Same isti`aatha Imlaey text; `has_quran=false`; `match_ratio?0.8696`; 19 sifat entries. | Different recording, same set of opening material. |

`moshaf_0.1` and `moshaf_1.0` rows repeat segment indexes because the index is set-local. Do not treat `segment_index` alone as globally unique.

## 5. Audio characteristics and verification

- API-declared Audio feature sampling rate: **16,000 Hz** for all 21 recitation schemas accessible via first-row API.
- Audio asset type on sampled records: **WAV** (`audio/wav`).
- Independently downloaded only two sample assets, then decoded from in-memory bytes with `soundfile`:
  - `moshaf_0.1 / 001.0000`: 16,000 Hz, 1 channel, 96,960 samples, 6.06 s, finite values, peak absolute amplitude ?0.591.
  - `moshaf_1.0 / 001.0000`: 16,000 Hz, 1 channel, 78,400 samples, 4.90 s, finite values, peak absolute amplitude ?0.475.
- Stored row `sample_rate` and `duration_seconds` agree with decoded sample values for these two records.
- These are tiny samples only. Other rates, channel counts, corrupt audio, duration outliers, or clipping may exist elsewhere; no full-corpus claim is made.
- The notebook exposes audio widgets so a person running it can listen. This exploration verified that the audio decodes, not subjective audio quality or correctness of the recitation.

## 6. Annotation meaning and likely training implications

### What labels are present?

1. **Tarteel transcript**: model transcript output stored as a list of strings. It may differ from the cleaned/aligned reference text; for the sampled initial invocation, it omits the leading hamza/alif seen in `imlaey`.
2. **Imlaey**: conventional Arabic orthographic reference in the inspected sample.
3. **Uthmani**: Uthmani Quran orthography; visible hamza/alif notation differs from Imlaey in the sample.
4. **Phonemes**: a Quran-specific phonetic script generated for pronunciation. It encodes audible/recitation features such as repeated letters or madd-like symbols. It is not a standard IPA transcription and is not the plain Quran text.
5. **Sifat**: per-phoneme/group labels for attributes such as ghonna, hams/jahr, itbaq, qalqla, safeer, tafashie, tafkheem/taqeeq, tikraar, istitala, and shidda/rakhawa.
6. **Moshaf choices**: recording-level metadata encodes selected Quranic reading/madd/sakt alternatives and related choices. These settings contextualize how a phonetic reference is produced.

The original README (`README_EN.md`, ?development steps?) says the intended pipeline converts Imlaey to Uthmani, then converts Uthmani to the Quranic phonetic script ?describing all tajweed rules except ishmam? using `quran-transcript`. It also says this dataset is annotated. The examples support that distinction: the phonetic script and sifat are explicit; **there are no columns named as per-segment tajweed-rule event tags** in the 24-column schema. So Tajweed is represented indirectly in phonetic target conventions and sifat, not as a separate list of named rule occurrences in the inspected schema. The exact mapping of every sign/property to tajweed concepts requires the `quran-transcript` implementation/version and dataset generation code.

### What is the unit?

Observed audio records are short individual segments. Quran samples include a start/end span with sura, aya, and Imlaey/Uthmani offsets. The first sampled row is a separate isti`aatha segment with no Quran span. `segment_index` increments locally; `index_type="sura"` and recording metadata `segmented_by="sura"` describe set-level indexing/segmentation fields, while the project README separately describes pause-based segmentation. The exact universal segmentation rule cannot be established from a handful of records. Data are therefore neither safely described as complete-surah audio per row nor assumed to be one-ayah-per-row throughout.

### Likely target transformations (decisions for Stage 2, not performed here)

- Select target field(s) explicitly: phoneme sequence and/or per-unit `sifat` labels if replicating the described model; do not tokenize `imlaey`/`uthmani` as if they were the original model's primary target.
- Preserve alignment between `phonemes` and `sifat` list entries; verify all sampled/full records where `len(sifat)` and phonetic units may differ.
- Decide how to include/exclude non-Quran segments using flags, not by assuming all transcript text is Quran.
- Join moshaf/reciter metadata by IDs and carry the moshaf-specific attributes into any label reconstruction.
- Define treatment for null spans, low `match_ratio`, missing/malformed annotations, and inconsistent records after auditing more data.
- Create leakage-aware train/validation/test splits by reciter or mushaf set; only a single `train` split exists upstream, and nearly identical same-surah recitations across sets could otherwise leak across splits.

## 7. Data quality checks (sample-limited)

Observed checks:

- First five rows were inspected in `moshaf_0.0`, `moshaf_0.1`, `moshaf_1.0`, and `moshaf_26.1`. The `/rows` endpoint returned records for `moshaf_0.0`, although its `/first-rows` schema endpoint returns HTTP 500.
- In the inspected successful rows, Imlaey, Uthmani, and phoneme strings were non-empty; sample metadata `sample_rate` was consistently 16,000; duration values were positive and plausible (3.02?6.28 s across the first five records in the four sampled sets).
- The two downloaded audio samples decoded, were mono and finite, with decoded rate/duration consistent with metadata.
- Sample `segment_index` values recur across sets by design/context; uniqueness must include `moshaf_id`. No duplicate `(moshaf_id, segment_index)` was observed in the sampled rows.
- Sampled Quran rows had `match_ratio=1.0`; sampled isti`aatha rows had lower values. This is not evidence that the rest of the corpus is correct or incorrect.
- Metadata counts are internally consistent at the top level: 27 moshaf metadata rows and 22 reciter metadata rows; examples join matching names and IDs.

Not checked corpus-wide: duplicate clips/content hashes, invalid audio, empty/missing labels, rate/duration distributions, `sifat`/phoneme alignment, duplicate identifiers, match-ratio distribution, repeated recitations between sets, and annotation correctness. These require a broader streamed audit and/or reading more data, which is beyond the limited exploration scope.

## 8. Unknowns and limitations

- HF first-row/data viewer post-processing fails for six audio configs. For `moshaf_0.0`, `/rows` returned a small sample but the schema declaration remained unavailable. The other five sets were not sampled through `/rows`. Reported counts/column counts still come from metadata.
- Full rows are in large Parquet/audio shards. Their total reported size is about **97.34 billion bytes** (roughly 97.3 GB decimal); no full shard was downloaded.
- No repository revision hash was pinned for the data-server API results; this is a snapshot of live metadata as of the exploration date.
- This environment has no original project checkout in the user workspace. The upstream `obadx/quran-muaalem` repository was cloned adjacent to the workspace read-only. The report references exact upstream file paths relative to its repository root.
- Data generation scripts in linked repositories (`prepare-quran-dataset`, `recitations-segmenter`, `quran-transcript`) were not audited deeply here, so annotation provenance, exact offsets, and `match_ratio` formula remain unknown.
- Listening was made available through notebook audio widgets; no listener-level pronunciation judgement was recorded.

## 9. Implications and decisions before Stage 2

Before implementing preprocessing/training, decide:

1. Which prediction target(s) to reproduce: phoneme string only, phoneme plus all `sifat` heads, or another target; the upstream model uses multi-level CTC heads (`src/quran_muaalem/modeling/modeling_multi_level_ctc.py`) and its tokenizer builds phoneme/sifat vocabularies (`src/quran_muaalem/modeling/multi_level_tokenizer.py`).
2. Whether to trust published labels directly or regenerate them using a pinned `quran-transcript` version and set-specific `MoshafAttributes`.
3. Which audio segments are eligible (`has_quran`, invocation/bismillah/sadaka flags, span availability, minimum match score).
4. How to define segment identity and splits without leakage across recording sets or reciters.
5. How much of the six currently inaccessible configurations to include; recheck them later or access underlying files through an alternative path before relying on them.
6. How to validate phoneme-to-sifat alignment and confirm unknown fields/offset conventions.

## Reproducibility

Run `notebooks/01_dataset_exploration.ipynb` from the workspace root. It fetches the current `/size`, `/splits`, and limited `/rows`/`first-rows` API responses, and downloads only two short audio assets to memory for decoding/playback. The notebook is the executable record of the inspection. Because it contacts a live service, rerunning later may produce changed counts or availability.
