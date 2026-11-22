# Quran Muaalem annotation validation report

**Scope:** read-only Stage 4 annotation investigation; no annotations were changed, no full-corpus tokenization or download was performed, and no model was trained. Real checks used small rows-API windows from `moshaf_0.1`, `moshaf_1.0`, and `moshaf_26.1`.

## Evidence and source paths

The Stage 1 schema/count and Stage 2 loader context are recorded in `docs/dataset_exploration_report.md`, `docs/dataset_loading_report.md`, `configs/dataset.json`, and `src/quran_dataset/loader.py`. The Stage 2 loader deliberately yields source rows without annotation changes.

The upstream reference checkout is the read-only sibling directory `../quran-muaalem-reference` (the local paths below are relative to that checkout):

- `src/quran_muaalem/modeling/vocab.py` builds phoneme symbols from `quran_transcript.alphabet.phonetics`; categorical sifat vocab values are taken from the `Literal` types on `SifaOutput`, mapped to Arabic text in brackets. `[PAD]` uses id 0.
- `src/quran_muaalem/modeling/multi_level_tokenizer.py` uses the raw phoneme script as the phoneme-level input. It turns each sifat property into its bracketed Arabic category text and tokenizes that per level. It constructs `SifaOutput` from dict entries, enforcing the reference model schema.
- `src/quran_muaalem/inference.py` uses `chunck_phonemes` on predicted phonetic text and compares phoneme groups with `sifat` reference groups. Its API requires 16 kHz audio and reference phonetic scripts.
- `docs/en/quran-transcript/phonetics.md` describes phonetic strings and `SifaOutput` groups. The doc file is mojibake in this Windows checkout, so the details below are grounded in the installed `quran-transcript` 0.6.4 package and actual row data, not a translated reading of its table.
- The upstream repository documents the intended annotation-generation stage in `docs/en/training/pipeline.md` and `docs/en/quran-transcript/usage.md`, but has no script that generated this published dataset's rows. No available source defines `start_span`/`end_span` conventions or `match_ratio` calculation.

## Structure and fields

The Stage 1 report describes 27 recitation configurations, each with a `train` split and 24 declared columns. The two metadata configurations are separate and are not the subject of this annotation validator. This Stage 4 run inspected three recitation configurations; it did not re-count the full dataset.

| Field | Observed type / nullability | Supported interpretation |
|---|---|---|
| `phonemes` | non-empty string in checked rows; feature declared string | Quran-specific phonetic script, not plain Imlaey/Uthmani text. Symbols are drawn from the reference package alphabet. Repeated letters and combining marks are visible in records. No spaces were present in the examples inspected. |
| `sifat` | list of objects; 11 string fields per entry in inspected rows | Each object has `phonemes` plus the ten categorical `SifaOutput` properties below. |
| `uthmani` | non-empty string in checked rows | Uthmani-script text annotation. |
| `imlaey` | non-empty string in checked rows | Imlaey-script text annotation. |
| `has_quran` | boolean in checked rows | Dataset flag. Example invocations are false; Quran ayah excerpts are true. |
| `has_istiaatha` | boolean in checked rows | Dataset segment flag; true for the sampled opening invocation. Its precise generation rule is not documented in the source checkout. |
| `has_bismillah` | boolean in checked rows | Dataset segment flag; a real row `002.0000` in `moshaf_0.1` is true while `has_quran` is false and spans are null. This establishes an observed use, not every case the flag may cover. |
| `has_sadaka` | boolean in checked rows | Dataset field. No positive row was found in the first 1,000 rows of either `moshaf_0.1` or `moshaf_1.0`; semantics and positive example remain unverified. |
| `start_span`, `end_span` | null or object with integer keys `sura_idx`, `aya_idx`, `imlaey`, `uthmani` | Quran-coordinate-like objects. Quran examples have both endpoints; non-Quran invocation and bismillah examples have both null. Units and endpoint convention are unresolved. |
| `match_ratio` | finite float in inspected rows | Observed values include `1.0`, `0.9565…`, `0.8695…`. Formula, target, and acceptance threshold are unknown. |
| `timestamp_seconds` | two floats in inspected rows | Values are increasing; first row starts at 0. Later rows have nonzero starts. They resemble recording-time boundaries, but semantics are not documented in available generation code. |
| `duration_seconds` | positive float in inspected rows | Segment duration metadata; Stage 1 decoded samples and found duration agreement on two audio assets. |

All of the above fields were present with expected non-null types on the 15 records in the Stage 4 smoke run. This is a sample observation, not a claim of corpus-wide completeness. The configured schemas in the Stage 1 report are available for more detailed typing; this stage does not silently generalize to uninspected rows.

## Actual records

Values below are from the Hugging Face datasets-server `/rows` API, not hand-created examples. Long signed audio URLs are omitted.

### Non-Quran invocation (`moshaf_0.1`, `001.0000`)

- `imlaey`: `أَعُوذُ بِاللَّهِ مِنَ الشَّيْطَانِ الرَّجِيمِ`
- `uthmani`: `أَعُوذُ بِٱللَّهِ مِنَ ٱلشَّيْطَانِ ٱلرَّجِيمِ`
- `phonemes`: `ءَعُۥۥذُ بِللَااهِ مِنَ ششَيطَاانِ ررَجِۦۦۦۦم`
- flags: `has_quran=false`, `has_istiaatha=true`, `has_bismillah=false`, `has_sadaka=false`
- spans: both null; `match_ratio=0.95652174949646`; duration `6.059999942779541` s
- 19 sifat entries; first entry's group is `ءَ`, and its observed categories include `hams_or_jahr=jahr`, `shidda_or_rakhawa=shadeed`, `tafkheem_or_taqeeq=moraqaq`.

### Quran text (`moshaf_0.1`, `001.0001`)

- `imlaey`: `بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ`
- `uthmani`: `بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ`
- `phonemes`: `بِسمِ للَااهِ ررَحمَاانِ ررَحِۦۦۦۦم`
- flags: `has_quran=true`, the other three flags false
- spans: start `{sura_idx: 1, aya_idx: 1, imlaey: 0, uthmani: 0}`, end `{sura_idx: 1, aya_idx: 1, imlaey: 4, uthmani: 4}`
- 15 sifat groups. The reference `chunck_phonemes(phonemes)` sequence exactly equals the 15 `sifat[*].phonemes` entries.

### Explicit bismillah flag (`moshaf_0.1`, `002.0000`)

- Text is `بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ` (Imlaey) and `بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ` (Uthmani).
- `has_quran=false`, `has_istiaatha=false`, `has_bismillah=true`, `has_sadaka=false`; both spans null; 15 sifat entries.
- The row has phoneme and sifat labels despite `has_quran=false`; therefore these labels are not limited to Quran-flagged rows.

### Cross-configuration examples

For `moshaf_1.0` and `moshaf_26.1`, the first five segment indices matched the same content pattern: invocation at `001.0000`, then Bismillah/ayah text and following excerpts. The phoneme string and grouping sometimes differ in repeated madd-like glyph counts across recordings/configurations (e.g. `ررَحِۦۦۦۦم` versus `ررَحِۦۦم`); this report does not infer whether that reflects recitation length, source generation, or a data issue. The current annotation validator checks each row against the reference grouping but does not require cross-record strings to be identical.

## Phoneme representation and model compatibility

The installed `quran-transcript` version inspected was **0.6.4**. `alphabet.phonetics` contains **42** reference symbol values. The upstream `docs/en/quran-transcript/phonetics.md` labels its scheme table “Phonemes (43)”, so that documentation count and the installed reference object disagree; the validator reads the installed object instead of hard-coding the count. The scheme includes Arabic consonant symbols, vowel/diacritic and madd signs, and specialized symbols. The package's source uses these symbols in a phonetic transcription engine. Available documentation does not provide a reliable character-by-character phonetic realization guide for every entry; such definitions should not be invented.

For the real text row `001.0001`, `chunck_phonemes` returned 15 groups, exactly matching `sifat[*].phonemes` in order. The same equality held for the sampled invocation in `moshaf_0.1`, and all 15 smoke rows passed this alignment rule. This means one sifat record corresponds to one **reference phoneme group**, not necessarily one individual Unicode character: groups can contain repeated base letters and marks (for example `للَ`, `اا`, or `ۦۦۦۦ`).

The original model implementation consumes raw phonetic strings at the phoneme level and tokenizes each categorical sifat level separately. It obtains phoneme vocabulary from the same reference package, while sifat category names are encoded as bracketed Arabic strings in the model vocab. The dataset's sampled representation is therefore structurally compatible with the original model's input contract. This is not a claim that the dataset labels were generated with the exact same package version, model checkpoint vocabulary, or every same configuration; no dataset generation manifest or full-dataset audit was available.

The ten categorical properties and allowed values are derived directly from `SifaOutput.model_fields` in the installed reference package:

| Property | Allowed values |
|---|---|
| `hams_or_jahr` | `hams`, `jahr` |
| `shidda_or_rakhawa` | `shadeed`, `between`, `rikhw` |
| `tafkheem_or_taqeeq` | `mofakham`, `moraqaq`, `low_mofakham` |
| `itbaq` | `monfateh`, `motbaq` |
| `safeer` | `safeer`, `no_safeer` |
| `qalqla` | `moqalqal`, `not_moqalqal` |
| `tikraar` | `mokarar`, `not_mokarar` |
| `tafashie` | `motafashie`, `not_motafashie` |
| `istitala` | `mostateel`, `not_mostateel` |
| `ghonna` | `maghnoon`, `not_maghnoon` |

All ten are required `Literal` values in `SifaOutput` 0.6.4. `phonemes` is required string data. No property is nullable in that reference model; no null property value occurred in the inspected rows.

## Spans, flags, and unit of a row

Rows are short audio segments. The Quran examples inspected carry an ayah coordinate plus `imlaey` and `uthmani` integer coordinates. A sampled Quran row's span values ran from 0 to 4 for a four-word excerpt, which is consistent with a word-position interpretation, but does **not** establish the offset unit or whether the ending is inclusive/exclusive. The upstream project checkout contains no data-generation script explaining the span fields. Do not use these values as character offsets or slice indices until the dataset producer's convention is verified.

The sampled opening `has_quran=false` invocation and `has_bismillah=true` segment both have phonetic labels and null spans. Thus rows are not restricted to complete Quran ayahs; at least some non-Quran segment types are included. A `has_sadaka=true` example was not found in bounded searches (first 1,000 rows each of two configs); its semantics and availability remain unknown. `has_quran` and category flags are observed dataset annotations; this investigation did not prove all logical combinations or determine how their source generator assigns them.

## Validation rules implemented

`src/quran_annotations/validation.py` is read-only and reports errors separately from warnings. It checks:

- non-empty phoneme string; symbols against the reference alphabet; chunker success;
- non-empty sifat list, required reference fields, categorical values, and exact ordered group alignment;
- non-empty `imlaey`/`uthmani` strings;
- flag field presence/type as a warning (since nullability is not documented by source generation);
- null-or-object span shape, exact keys, and non-negative integer values. It does not assert Quran bounds or interpret offsets;
- `match_ratio` numeric range and basic timestamp/duration shape/order when those values are present;
- likely flag/span contradictions as warnings, because their source rules have not been established.

It does not mutate source records, normalize text, repair missing annotations, or declare warnings fatal. The allowed phoneme inventory, chunking, and sifat categories come from `quran-transcript`, declared as a project dependency to keep this reference explicit.

## Data quality results and limitations

- **15 real rows**, five each from `moshaf_0.1`, `moshaf_1.0`, `moshaf_26.1`: no validation errors or warnings. All had non-empty phonemes/text/sifat; the sifat group sequence matched the reference chunker; observed categorical values were accepted.
- Both non-Quran and Quran rows were checked. A separate API row confirmed a `has_bismillah=true`, non-Quran row also has annotations and null spans.
- No duplicate `(configuration, segment_index)` identifier appeared in the smoke sample; this tiny check does not establish global uniqueness.
- A bounded scan of the first 1,000 rows from each of `moshaf_0.1` and `moshaf_1.0` found a bismillah row but no `has_sadaka=true` row.
- This stage did not download/decode audio; audio characteristics are in `docs/audio_preprocessing_report.md`. It did not test corpus-wide schema variation, empty values, corrupt assets, timestamp semantics, or annotation correctness against an independent Quran text source.
- Exact meanings of `match_ratio`, span offsets/end convention, `timestamp_seconds`, `has_sadaka`, and the complete flag-generation rules remain unresolved. No data generation script was present in the original project checkout.

## Tests and execution

Unit tests: `python -m pytest tests/test_annotations.py -q` — **9 passed**.

Real data smoke test:

```powershell
$env:PYTHONPATH = "src"
python scripts/annotation_validation_smoke.py --configurations moshaf_0.1 moshaf_1.0 moshaf_26.1 --limit 5
```

Result: 15/15 rows passed with no warnings. The script limits each configuration to at most 100 rows (default 4) and fetches only selected rows from the datasets-server API.

Notebook:

```powershell
python -m pip install -e ".[notebook]"
jupyter nbconvert --to notebook --execute notebooks/04_annotation_validation.ipynb --inplace
```

The notebook uses bounded API windows and is intended to be run from the repository root. Audio URLs are displayed only as descriptors; no assets or full shards are fetched.

## Decisions needed before Stage 5

1. Obtain dataset-generation documentation or producer confirmation for span coordinate units, boundaries, and null behavior.
2. Confirm `match_ratio` definition and whether values have any quality-control threshold.
3. Confirm precise definitions/assignment rules for all segment flags, especially `has_sadaka`.
4. Decide whether non-Quran segments should remain eligible targets (the evidence shows they have phoneme and sifat labels; this stage does not filter them).
5. Confirm the training-time reference package version and model checkpoint vocab compatibility before constructing model labels.
6. Run a larger, stratified annotation audit before training; this bounded sample is not a corpus-wide quality estimate.
