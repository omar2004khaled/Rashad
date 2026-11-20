from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from quran_dataset import AudioData
from quran_audio import (
    AudioProcessingConfig,
    AudioProcessingError,
    ProcessedAudio,
    analyze_audio,
    compare_audio,
    convert_to_mono,
    detect_silence,
    load_audio_sample,
    normalize_peak,
    preprocess_audio,
    reduce_noise,
    resample_audio,
    to_float32,
)


def make_signal(samples, rate=16000):
    samples = np.asarray(samples)
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    duration = samples.shape[0] / rate
    return ProcessedAudio.from_audio_data(AudioData(samples, rate, channels, duration))


def test_config_loads_safe_defaults():
    config = AudioProcessingConfig.from_file(Path("configs/audio_preprocessing.json"))
    assert config.target_sample_rate == 16000
    assert config.convert_to_mono is False
    assert config.normalize is False
    assert config.noise_reduction is False


def test_load_audio_sample_uses_stage2_loader_and_keeps_source():
    samples = np.array([[0.1], [-0.1]], dtype=np.float32)
    record = {"audio": {"array": samples, "sampling_rate": 16000}, "sample_rate": 16000}

    class FakeLoader:
        def get_sample(self, configuration, index, split=None, streaming=True):
            assert (configuration, index, split, streaming) == ("moshaf_0.1", 0, None, True)
            return record

    loaded_record, signal = load_audio_sample(FakeLoader(), "moshaf_0.1")
    assert loaded_record is record
    np.testing.assert_array_equal(signal.samples, samples)
    np.testing.assert_array_equal(signal.original.samples, samples)
    assert not signal.original.samples.flags.writeable


def test_mono_input_is_not_rewritten_and_stereo_downmix_is_explicit():
    mono = make_signal(np.array([[0.1], [-0.3]], dtype=np.float32))
    mono_before = mono.original.samples.copy()
    unchanged = convert_to_mono(mono)
    assert unchanged.operations[-1].applied is False
    np.testing.assert_array_equal(unchanged.samples, mono.samples)
    np.testing.assert_array_equal(mono.original.samples, mono_before)

    stereo_samples = np.array([[0.2, 0.4], [-0.2, 0.0], [0.6, 0.2]], dtype=np.float32)
    stereo = make_signal(stereo_samples)
    original_copy = stereo.original.samples.copy()
    downmixed = convert_to_mono(stereo)
    expected = stereo_samples.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(downmixed.samples, expected, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(stereo.original.samples, original_copy)
    assert downmixed.channels == 1
    assert downmixed.operations[-1].applied is True
    assert downmixed.operations[-1].parameters["source_channels"] == 2


def test_resampling_only_occurs_for_a_different_target_rate_and_preserves_duration():
    rate = 48000
    samples = np.sin(2 * np.pi * 440 * np.arange(rate) / rate).astype(np.float32)
    signal = make_signal(samples, rate)
    original_copy = signal.original.samples.copy()
    same_rate = resample_audio(signal, rate)
    assert same_rate.operations[-1].applied is False
    np.testing.assert_array_equal(same_rate.samples, signal.samples)

    converted = resample_audio(signal, 16000)
    assert converted.sampling_rate == 16000
    assert converted.operations[-1].applied is True
    assert abs(len(converted.samples) / converted.sampling_rate - 1.0) <= 1 / 16000
    np.testing.assert_array_equal(signal.original.samples, original_copy)
    assert converted.original.sampling_rate == 48000


def test_float32_conversion_and_integer_scaling_are_explicit():
    signal = make_signal(np.array([0.25, -0.5], dtype=np.float64))
    converted = to_float32(signal)
    assert converted.samples.dtype == np.float32
    np.testing.assert_allclose(converted.samples, [0.25, -0.5])
    np.testing.assert_array_equal(signal.original.samples, np.array([0.25, -0.5]))

    pcm = make_signal(np.array([0, 32767, -32768], dtype=np.int16))
    scaled = to_float32(pcm)
    assert scaled.samples.dtype == np.float32
    assert scaled.samples[1] < 1.0
    assert scaled.samples[2] == -1.0
    assert scaled.operations[-1].parameters["integer_pcm_scaled_to_unit_range"] is True


def test_normalization_is_opt_in_guarded_and_does_not_mutate_original():
    samples = np.array([[0.10], [-0.05], [0.02]], dtype=np.float32)
    signal = make_signal(samples)
    original_copy = signal.original.samples.copy()
    normalized = normalize_peak(signal, target_dbfs=-1, max_gain_db=24)
    assert normalized.operations[-1].applied is True
    assert np.max(np.abs(normalized.samples)) == pytest.approx(10 ** (-1 / 20), rel=1e-6)
    assert analyze_audio(normalized).clipped_sample_count == 0
    assert len(normalized.samples) == len(signal.samples)
    np.testing.assert_array_equal(signal.original.samples, original_copy)

    config = AudioProcessingConfig(target_sample_rate=None, normalize=False, noise_reduction=False)
    unchanged = preprocess_audio(signal, config)
    assert not any(op.name == "normalize_peak" and op.applied for op in unchanged.operations)


def test_normalization_safeguards_quiet_and_excessive_gain_samples():
    quiet = make_signal(np.array([1e-4, -5e-5], dtype=np.float32))
    skipped_quiet = normalize_peak(quiet)
    assert skipped_quiet.operations[-1].applied is False
    assert "safeguard" in skipped_quiet.operations[-1].reason

    moderate = make_signal(np.array([0.1, -0.1], dtype=np.float32))
    skipped_gain = normalize_peak(moderate, max_gain_db=3)
    assert skipped_gain.operations[-1].applied is False
    assert "exceeds" in skipped_gain.operations[-1].reason


def test_silence_detection_measures_without_trimming():
    samples = np.concatenate([np.zeros(1600), np.full(8000, 0.1), np.zeros(3200)]).astype(np.float32)
    signal = make_signal(samples)
    original_copy = signal.original.samples.copy()
    silence = detect_silence(signal, threshold_dbfs=-60)
    assert silence.leading_seconds == pytest.approx(0.1)
    assert silence.trailing_seconds == pytest.approx(0.2)
    assert silence.measured_only is True
    assert len(signal.samples) == len(samples)
    np.testing.assert_array_equal(signal.original.samples, original_copy)


def test_clipping_indicator_reports_full_scale_samples():
    signal = make_signal(np.array([0.0, 0.2, -0.4, 1.0, -1.0], dtype=np.float32))
    properties = analyze_audio(signal, clipping_threshold=0.99999)
    assert properties.minimum_amplitude == -1.0
    assert properties.maximum_amplitude == 1.0
    assert properties.peak_amplitude == 1.0
    assert properties.clipped_sample_count == 2
    assert properties.clipped_sample_fraction == pytest.approx(2 / 5)


@pytest.mark.parametrize(
    "samples,rate",
    [
        (np.array([], dtype=np.float32), 16000),
        (np.array([0.0, np.nan], dtype=np.float32), 16000),
        (np.zeros(10, dtype=np.float32), 0),
        (np.zeros((2, 2, 2), dtype=np.float32), 16000),
    ],
)
def test_invalid_audio_is_rejected(samples, rate):
    with pytest.raises(AudioProcessingError):
        analyze_audio(samples, rate)


def test_noise_reduction_disabled_by_default_and_requires_explicit_profile():
    signal = make_signal(np.array([0.1, -0.1, 0.05], dtype=np.float32))
    original_copy = signal.original.samples.copy()

    def must_not_run(**kwargs):
        raise AssertionError("reducer should not run when disabled")

    untouched = reduce_noise(signal, reducer=must_not_run)
    assert untouched.operations[-1].applied is False
    assert untouched.operations[-1].reason == "disabled"
    np.testing.assert_array_equal(untouched.samples, signal.samples)
    np.testing.assert_array_equal(signal.original.samples, original_copy)

    with pytest.raises(AudioProcessingError, match="noise_profile"):
        reduce_noise(signal, enabled=True)


def test_opt_in_noise_reduction_uses_profile_and_preserves_duration():
    signal = make_signal(np.linspace(-0.3, 0.3, 100, dtype=np.float32))
    noise = np.zeros(20, dtype=np.float32)
    received = {}

    def reducer(y, y_noise, sr, stationary, prop_decrease):
        received.update(sr=sr, stationary=stationary, prop_decrease=prop_decrease, noise_len=len(y_noise))
        return y * 0.9

    output = reduce_noise(signal, enabled=True, noise_profile=noise, strength=0.2, reducer=reducer)
    assert output.operations[-1].applied is True
    assert received == {"sr": 16000, "stationary": True, "prop_decrease": 0.2, "noise_len": 20}
    assert len(output.samples) == len(signal.samples)
    assert output.sampling_rate == signal.sampling_rate
    np.testing.assert_array_equal(signal.original.samples, signal.samples)


def test_preprocess_config_records_each_operation_and_retains_original():
    signal = make_signal(np.array([[0.1, 0.2], [0.0, 0.1], [-0.2, -0.1]], dtype=np.float64), 48000)
    source_copy = signal.original.samples.copy()
    config = AudioProcessingConfig(
        target_sample_rate=16000,
        convert_to_mono=True,
        convert_to_float32=True,
        normalize=False,
        noise_reduction=False,
    )
    output = preprocess_audio(signal, config)
    assert [op.name for op in output.operations] == ["to_float32", "convert_to_mono", "resample", "noise_reduction"]
    assert all(isinstance(op.applied, bool) for op in output.operations)
    assert output.sampling_rate == 16000
    assert output.channels == 1
    assert output.samples.dtype == np.float32
    np.testing.assert_array_equal(signal.original.samples, source_copy)
    assert output.original.sampling_rate == 48000


def test_compare_audio_reports_sample_delta_only_for_matching_grids():
    signal = make_signal(np.array([0.1, -0.1, 0.05], dtype=np.float32))
    normalized = normalize_peak(signal, max_gain_db=24)
    comparison = compare_audio(signal, normalized)
    assert comparison["samplewise_comparable"] is True
    assert comparison["difference"]["max_absolute"] > 0

    resampled = resample_audio(make_signal(np.zeros(480, dtype=np.float32), 48000), 16000)
    resampled_comparison = compare_audio(resampled.original, resampled)
    assert resampled_comparison["samplewise_comparable"] is False
    assert resampled_comparison["difference"] is None
