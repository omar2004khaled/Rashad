# Audio preprocessing report

**Stage:** 3 ? audio inspection and conservative preprocessing  
**Dataset:** `obadx/muaalem-annotated-v3`  
**Scope:** streamed samples from three recording-set configurations; no full dataset download, annotation changes, training, or automatic corpus processing.

## Summary

The 15 real audio records inspected from `moshaf_0.1`, `moshaf_1.0`, and `moshaf_26.1` were all finite, mono, 16 kHz audio. Durations ranged from 3.70 to 6.28 seconds. No sample reached the configured full-scale clipping threshold (`abs(sample) >= 0.99999`). For those samples, the model-compatible default pipeline did not change any sample: they were already float32, mono, and 16 kHz.

No denoising or trimming is justified by this small sample. One opening clip contains about 0.453 seconds of quiet audio at its beginning; it was retained. Optional normalization was demonstrated on a quiet final-word tail proxy, preserving the tail and duration without clipping. The final 200 ms is not a word-aligned window because dataset annotations contain no word-level timestamps.

## Original inference expectation

The inspected upstream project is the adjacent read-only clone `../quran-muaalem-reference`.

- `docs/en/muaalem/python-api.md` specifies a 16 kHz input and demonstrates `librosa.load(..., sr=16000, mono=True)`.
- `src/quran_muaalem/inference.py` accepts the supplied rate and raises when it is not 16 kHz.
- `README_EN.md` repeats the 16 kHz requirement and mono loading example.

The upstream example resamples/downmixes during inference loading. This stage offers those operations as explicit functions, but does not apply them unless configured. Dataset records inspected here already met the rate and channel expectations.

## Implemented module

`src/quran_audio/preprocessing.py` contains:

- `load_audio_sample`: obtains one record through the Stage 2 `DatasetLoader`, decodes it, and keeps an immutable source snapshot.
- `analyze_audio`: sample rate, channels, duration, min/max, peak and RMS amplitudes, dBFS levels, clipping counts, finite-value status, and measured quiet leading/trailing runs.
- `detect_silence`: returns threshold-based leading/trailing durations. It never crops or changes the audio.
- `to_float32`: explicit numeric conversion; signed/unsigned integer PCM is scaled into unit range, while floating values are cast only.
- `convert_to_mono`: explicit arithmetic-mean downmix. It is off in the default configuration because a downmix changes channel information and can alter phase interactions.
- `resample_audio`: `scipy.signal.resample_poly`, called only if the current rate differs from the requested rate.
- `normalize_peak`: optional uniform gain with target, maximum positive gain, and low-level safeguards.
- `reduce_noise`: optional stationary noise reduction, off by default, and requires a caller-supplied noise profile. It uses the `noisereduce` optional extra unless a reducer function is injected.
- `preprocess_audio`: applies only configured operations and stores a before/after record for each attempted operation, including no-ops.
- `compare_audio`: reports measurements and a samplewise difference only when rate, duration/sample count, and channel grid match.

Each `ProcessedAudio` retains `original` samples separately from the current samples. Transformations make new arrays and do not edit the source. `configs/audio_preprocessing.json` controls defaults.

## Real sample measurements

The smoke check read the first five records of each of three configurations (15 rows total). Silence uses an absolute threshold of ?60 dBFS; clipping uses `abs(sample) >= 0.99999`. Tail RMS is measured over the last 200 ms.

| Configuration | Duration range (s) | Peak dBFS range | RMS dBFS range | Leading quiet range (s) | Trailing quiet range (s) | Tail RMS range (dBFS) | Clipped samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `moshaf_0.1` | 5.075?6.28 | ?4.56 to ?1.15 | ?22.03 to ?19.84 | 0?0.453 | 0 | ?52.49 to ?50.64 | 0 |
| `moshaf_1.0` | 3.70?5.18 | ?6.47 to ?5.51 | ?23.22 to ?17.63 | 0.001?0.038 | 0?0.003 | ?64.76 to ?62.90 | 0 |
| `moshaf_26.1` | 4.50?5.66 | ?7.97 to ?2.82 | ?22.88 to ?17.11 | 0?0.374 | 0 | ?47.10 to ?41.79 | 0 |

Across all 15 clips, the observed amplitude range was approximately ?0.876 to +0.783. All decoded sample values were finite; every file decoded as mono at 16 kHz. The quiet leading runs are measured intervals at the selected threshold, not proof that they should be removed. Peak and RMS differences between recording sets are measurable, but do not by themselves indicate an audio defect.

No broad audio-quality diagnosis (noise, room acoustics, recitation correctness, or perceptual distortion) can be made from these amplitude statistics. The notebook provides audio controls so a person can listen; this run verified decoding and measurements but did not record a human listening judgment.

## Quiet word-ending check

The selected example is `moshaf_1.0 / 001.0004`, whose Uthmani text ends with `????????`. The final 200 ms tail RMS was ?63.49 dBFS, and the trailing run below ?60 dBFS was 0 seconds; the last sample was nonzero. This is a low-level clip tail proxy, not a word/phoneme-aligned measurement because there are no word-level timestamps.

Optional peak normalization was run individually with target ?1 dBFS and maximum gain +6 dB:

| Measurement | Before | After |
| --- | ---: | ---: |
| Peak level | ?5.73 dBFS | ?1.00 dBFS (`0.8913` amplitude) |
| Tail RMS, final 200 ms | ?63.49 dBFS | ?58.76 dBFS |
| Duration | 3.70 s | 3.70 s |
| Clipped samples | 0 | 0 |
| Last sample | Nonzero | Nonzero |

The measured gain was +4.73 dB and the final 200 ms was multiplied by that same gain; its relative waveform shape was retained. The original source array remained sample-for-sample equal to its pre-transform snapshot. This demonstrates the configured peak normalization on one sample; normalization stays disabled by default.

## Configuration choices

`configs/audio_preprocessing.json` sets:

- **Target rate: 16,000 Hz.** Matches the original inference API. Resampling is a no-op for the inspected clips and only runs for a different source rate.
- **Mono conversion: off.** All 15 samples were already mono. If a future record is multichannel, downmixing needs an explicit choice because it discards channel separation.
- **float32 conversion: on.** The model example consumes floating waveforms; the inspected `soundfile` arrays were already float32, so this step was a no-op. Integer conversion is explicit and logged.
- **Normalization: off.** Current sample peaks are below full scale and default inference does not require peak normalization. When enabled, target is ?1 dBFS, maximum positive gain +6 dB, and clips peaking below ?50 dBFS are left unchanged to avoid boosting very quiet material/noise.
- **Noise reduction: off.** No noise diagnosis was made and no noise profile is supplied. Enabling it without an explicit profile is rejected. The optional stationary reducer is conservative but can still alter recitation detail, so listen and compare before using it.
- **Silence measurement: ?60 dBFS.** Used for an explicit, reproducible measurement only; it never deletes audio.
- **Clipping flag: 0.99999 full scale.** A measurement threshold for float audio, not a limiter or repair operation.

No pitch or playback-speed change is implemented. Resampling changes the sample grid/rate but preserves playback timing; normalizing uniformly changes level only. Every operation logs whether it ran, its parameters, and before/after measurements.

## Tests and verification

- `python -m pytest`: **38 passed** (21 Stage 2 tests plus 17 preprocessing tests). Synthetic coverage includes mono/stereo, resampling and duration, float conversion, normalization safeguards, silence, clipping, invalid/empty/non-finite audio, original-array preservation, and noise reduction disabled by default.
- `python scripts/audio_preprocessing_smoke.py`: accessed 15 real records from three configurations. The default pipeline produced exact sample equality for all 15; source arrays remained unchanged. The quiet ending normalization check retained its final samples and duration with zero clipped samples.
- `notebooks/03_audio_preprocessing.ipynb`: executed from beginning to end with no cell errors. It includes measurements, waveform plots, original/processed comparisons, and playable original/normalized audio controls.
- The real sample check is limited to 15 clips; synthetic tests cover transformations such as actual stereo downmix and 48 kHz to 16 kHz resampling.

## Limitations and unresolved questions

- Dataset `timestamp_seconds` is segment-level; the inspected schema offers no word/phoneme timestamps. The selected last-200-ms window cannot establish exact consonant preservation at a named word boundary.
- Only three configurations and five initial rows per configuration were inspected. This is not a corpus-wide rate, clipping, silence, noise, or channel audit.
- All inspected audio was mono/16 kHz WAV. Stereo and non-16-kHz behavior is validated synthetically, not with real dataset examples.
- No human listening review was recorded. Use the notebook?s playback controls for an expert recitation-quality judgment before selecting optional normalization or denoising policies.
- Noise reduction is optional and not exercised on real audio. Its dependency is installable with `pip install -e ".[audio-denoise]"`; it requires an explicit noise profile and is disabled in configuration.
- The Stage 2 smoke runner may emit a Windows/Hugging Face stream cleanup warning after a successful run; the Stage 3 smoke completed and returned its measurements.
