"""Conservative audio measurements and explicitly opt-in transformations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import gcd, log10
from typing import Any, Callable, Mapping
import importlib
import json

import numpy as np
from scipy.signal import resample_poly

from quran_dataset import AudioData, DatasetLoader, decode_audio


class AudioProcessingError(ValueError):
    """Invalid audio or unsafe transformation parameters."""


@dataclass(frozen=True)
class TransformationRecord:
    name: str
    applied: bool
    parameters: Mapping[str, Any]
    before: Mapping[str, Any]
    after: Mapping[str, Any] | None
    reason: str | None = None


@dataclass(frozen=True)
class ProcessedAudio:
    """Current waveform plus an immutable snapshot of the decoded source."""

    original: AudioData
    samples: np.ndarray
    sampling_rate: int
    operations: tuple[TransformationRecord, ...] = ()

    @property
    def channels(self) -> int:
        return 1 if self.samples.ndim == 1 else int(self.samples.shape[1])

    @classmethod
    def from_audio_data(cls, audio: AudioData) -> "ProcessedAudio":
        _validate_samples(audio.samples, audio.sampling_rate)
        original_samples = np.array(audio.samples, copy=True)
        original_samples.setflags(write=False)
        original = AudioData(
            samples=original_samples,
            sampling_rate=int(audio.sampling_rate),
            channels=int(audio.channels),
            duration_seconds=float(audio.duration_seconds),
        )
        return cls(original, original_samples, int(audio.sampling_rate))


@dataclass(frozen=True)
class AudioProperties:
    sampling_rate: int
    channels: int
    duration_seconds: float
    samples: int
    minimum_amplitude: float
    maximum_amplitude: float
    peak_amplitude: float
    peak_dbfs: float
    rms_amplitude: float
    rms_dbfs: float
    clipping_threshold: float
    clipped_sample_count: int
    clipped_sample_fraction: float
    finite: bool
    leading_silence_seconds: float
    trailing_silence_seconds: float
    silence_threshold_dbfs: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AudioSilence:
    leading_seconds: float
    trailing_seconds: float
    threshold_dbfs: float
    threshold_amplitude: float
    measured_only: bool = True


@dataclass(frozen=True)
class AudioProcessingConfig:
    """Explicit switches; only the model's target rate defaults to 16 kHz."""

    target_sample_rate: int | None = 16000
    convert_to_mono: bool = False
    convert_to_float32: bool = True
    normalize: bool = False
    normalize_target_dbfs: float = -1.0
    normalize_max_gain_db: float = 6.0
    normalize_minimum_peak_dbfs: float = -50.0
    noise_reduction: bool = False
    noise_reduction_strength: float = 0.35
    silence_threshold_dbfs: float = -60.0
    clipping_threshold: float = 0.99999

    @classmethod
    def from_file(cls, path: str) -> "AudioProcessingConfig":
        with open(path, encoding="utf-8") as file:
            return cls(**json.load(file))


def _validate_samples(samples: np.ndarray, sampling_rate: int) -> None:
    if not isinstance(samples, np.ndarray):
        raise AudioProcessingError("Audio samples must be a NumPy array")
    if samples.size == 0:
        raise AudioProcessingError("Audio samples are empty")
    if samples.ndim not in (1, 2):
        raise AudioProcessingError(f"Expected 1D or 2D samples, got shape {samples.shape}")
    if not isinstance(sampling_rate, (int, np.integer)) or sampling_rate <= 0:
        raise AudioProcessingError("Sampling rate must be a positive integer")
    if not np.isfinite(samples).all():
        raise AudioProcessingError("Audio contains non-finite samples")


def load_audio_sample(
    loader: DatasetLoader,
    configuration: str,
    index: int = 0,
    *,
    split: str | None = None,
) -> tuple[Mapping[str, Any], ProcessedAudio]:
    """Load a record via Stage 2 and decode a protected source snapshot."""
    record = loader.get_sample(configuration, index, split=split, streaming=True)
    audio = decode_audio(record)
    return record, ProcessedAudio.from_audio_data(audio)


def _as_signal(value: ProcessedAudio | AudioData | np.ndarray, sampling_rate: int | None = None) -> ProcessedAudio:
    if isinstance(value, ProcessedAudio):
        return value
    if isinstance(value, AudioData):
        return ProcessedAudio.from_audio_data(value)
    if sampling_rate is None:
        raise AudioProcessingError("sampling_rate is required for a raw NumPy array")
    _validate_samples(value, sampling_rate)
    if value.ndim == 1:
        channels = 1
        duration = value.shape[0] / sampling_rate
    else:
        channels = value.shape[1]
        duration = value.shape[0] / sampling_rate
    return ProcessedAudio.from_audio_data(AudioData(value, sampling_rate, channels, duration))


def _rms_dbfs(rms: float) -> float:
    return float("-inf") if rms <= 0 else float(20.0 * log10(rms))


def detect_silence(
    audio: ProcessedAudio | AudioData | np.ndarray,
    sampling_rate: int | None = None,
    *,
    threshold_dbfs: float = -60.0,
) -> AudioSilence:
    """Measure quiet leading/trailing runs; never crop or alter the waveform."""
    signal = _as_signal(audio, sampling_rate)
    if not np.isfinite(threshold_dbfs) or threshold_dbfs >= 0:
        raise AudioProcessingError("threshold_dbfs must be a finite negative dBFS value")
    threshold = 10.0 ** (threshold_dbfs / 20.0)
    matrix = signal.samples[:, None] if signal.samples.ndim == 1 else signal.samples
    per_frame_peak = np.max(np.abs(matrix), axis=1)
    active = np.flatnonzero(per_frame_peak > threshold)
    if active.size == 0:
        leading = trailing = len(per_frame_peak) / signal.sampling_rate
    else:
        leading = int(active[0]) / signal.sampling_rate
        trailing = (len(per_frame_peak) - int(active[-1]) - 1) / signal.sampling_rate
    return AudioSilence(float(leading), float(trailing), float(threshold_dbfs), float(threshold))


def analyze_audio(
    audio: ProcessedAudio | AudioData | np.ndarray,
    sampling_rate: int | None = None,
    *,
    silence_threshold_dbfs: float = -60.0,
    clipping_threshold: float = 0.99999,
) -> AudioProperties:
    signal = _as_signal(audio, sampling_rate)
    _validate_samples(signal.samples, signal.sampling_rate)
    if not 0 < clipping_threshold <= 1.0:
        raise AudioProcessingError("clipping_threshold must be in (0, 1]")
    values = signal.samples.astype(np.float64, copy=False)
    peak = float(np.max(np.abs(values)))
    rms = float(np.sqrt(np.mean(np.square(values))))
    clipped = int(np.count_nonzero(np.abs(values) >= clipping_threshold))
    silence = detect_silence(signal, threshold_dbfs=silence_threshold_dbfs)
    frames = values.shape[0]
    return AudioProperties(
        sampling_rate=signal.sampling_rate,
        channels=signal.channels,
        duration_seconds=float(frames / signal.sampling_rate),
        samples=int(frames),
        minimum_amplitude=float(np.min(values)),
        maximum_amplitude=float(np.max(values)),
        peak_amplitude=peak,
        peak_dbfs=_rms_dbfs(peak),
        rms_amplitude=rms,
        rms_dbfs=_rms_dbfs(rms),
        clipping_threshold=float(clipping_threshold),
        clipped_sample_count=clipped,
        clipped_sample_fraction=float(clipped / values.size),
        finite=bool(np.isfinite(values).all()),
        leading_silence_seconds=silence.leading_seconds,
        trailing_silence_seconds=silence.trailing_seconds,
        silence_threshold_dbfs=float(silence_threshold_dbfs),
    )


def _operation(
    signal: ProcessedAudio,
    name: str,
    samples: np.ndarray,
    sample_rate: int,
    *,
    applied: bool,
    parameters: Mapping[str, Any],
    reason: str | None = None,
) -> ProcessedAudio:
    before = analyze_audio(signal).as_dict()
    if applied:
        _validate_samples(samples, sample_rate)
        output = replace(signal, samples=samples, sampling_rate=int(sample_rate))
        after = analyze_audio(output).as_dict()
    else:
        output = signal
        after = before
    entry = TransformationRecord(name, applied, dict(parameters), before, after, reason)
    return replace(output, operations=signal.operations + (entry,))


def convert_to_mono(audio: ProcessedAudio | AudioData | np.ndarray, sampling_rate: int | None = None) -> ProcessedAudio:
    signal = _as_signal(audio, sampling_rate)
    if signal.channels == 1:
        return _operation(signal, "convert_to_mono", signal.samples, signal.sampling_rate,
                          applied=False, parameters={"method": "arithmetic_mean"}, reason="already mono")
    mono = np.mean(signal.samples, axis=1, keepdims=True)
    return _operation(signal, "convert_to_mono", mono, signal.sampling_rate,
                      applied=True, parameters={"method": "arithmetic_mean", "source_channels": signal.channels})


def resample_audio(
    audio: ProcessedAudio | AudioData | np.ndarray,
    target_sample_rate: int,
    sampling_rate: int | None = None,
) -> ProcessedAudio:
    signal = _as_signal(audio, sampling_rate)
    if not isinstance(target_sample_rate, int) or target_sample_rate <= 0:
        raise AudioProcessingError("target_sample_rate must be a positive integer")
    if target_sample_rate == signal.sampling_rate:
        return _operation(signal, "resample", signal.samples, signal.sampling_rate,
                          applied=False, parameters={"target_sample_rate": target_sample_rate}, reason="already at target rate")
    if not np.issubdtype(signal.samples.dtype, np.floating):
        signal = to_float32(signal)
    divisor = gcd(signal.sampling_rate, target_sample_rate)
    up = target_sample_rate // divisor
    down = signal.sampling_rate // divisor
    changed = resample_poly(signal.samples, up, down, axis=0).astype(signal.samples.dtype, copy=False)
    return _operation(signal, "resample", changed, target_sample_rate, applied=True,
                      parameters={"source_sample_rate": signal.sampling_rate,
                                  "target_sample_rate": target_sample_rate,
                                  "algorithm": "scipy.signal.resample_poly", "up": up, "down": down})


def to_float32(audio: ProcessedAudio | AudioData | np.ndarray, sampling_rate: int | None = None) -> ProcessedAudio:
    signal = _as_signal(audio, sampling_rate)
    if signal.samples.dtype == np.float32:
        return _operation(signal, "to_float32", signal.samples, signal.sampling_rate,
                          applied=False, parameters={"dtype": "float32"}, reason="already float32")
    samples = signal.samples
    if np.issubdtype(samples.dtype, np.signedinteger):
        info = np.iinfo(samples.dtype)
        converted = samples.astype(np.float64) / float(max(abs(info.min), info.max + 1))
    elif np.issubdtype(samples.dtype, np.unsignedinteger):
        info = np.iinfo(samples.dtype)
        midpoint = (info.max + 1) / 2.0
        converted = (samples.astype(np.float64) - midpoint) / midpoint
    elif np.issubdtype(samples.dtype, np.floating):
        converted = samples
    else:
        raise AudioProcessingError(f"Unsupported sample dtype {samples.dtype}")
    return _operation(signal, "to_float32", np.asarray(converted, dtype=np.float32), signal.sampling_rate,
                      applied=True, parameters={"source_dtype": str(samples.dtype), "target_dtype": "float32",
                                                "integer_pcm_scaled_to_unit_range": bool(np.issubdtype(samples.dtype, np.integer))})


def normalize_peak(
    audio: ProcessedAudio | AudioData | np.ndarray,
    sampling_rate: int | None = None,
    *,
    target_dbfs: float = -1.0,
    max_gain_db: float = 6.0,
    minimum_peak_dbfs: float = -50.0,
) -> ProcessedAudio:
    signal = _as_signal(audio, sampling_rate)
    if not np.isfinite(target_dbfs) or target_dbfs >= 0:
        raise AudioProcessingError("target_dbfs must be finite and below 0 dBFS")
    if not np.isfinite(max_gain_db) or max_gain_db < 0:
        raise AudioProcessingError("max_gain_db must be finite and non-negative")
    if not np.isfinite(minimum_peak_dbfs) or minimum_peak_dbfs >= 0:
        raise AudioProcessingError("minimum_peak_dbfs must be finite and below 0 dBFS")
    peak = float(np.max(np.abs(signal.samples)))
    params = {"target_dbfs": target_dbfs, "max_gain_db": max_gain_db,
              "minimum_peak_dbfs": minimum_peak_dbfs}
    if peak == 0:
        return _operation(signal, "normalize_peak", signal.samples, signal.sampling_rate,
                          applied=False, parameters=params, reason="zero-valued audio")
    peak_dbfs = _rms_dbfs(peak)
    if peak_dbfs < minimum_peak_dbfs:
        return _operation(signal, "normalize_peak", signal.samples, signal.sampling_rate,
                          applied=False, parameters=params, reason="peak below amplification safeguard")
    gain_db = target_dbfs - peak_dbfs
    if gain_db > max_gain_db:
        return _operation(signal, "normalize_peak", signal.samples, signal.sampling_rate,
                          applied=False, parameters={**params, "requested_gain_db": gain_db},
                          reason="requested gain exceeds safeguard")
    gain = 10.0 ** (gain_db / 20.0)
    output = (signal.samples.astype(np.float64) * gain).astype(np.float32)
    if float(np.max(np.abs(output))) > 1.0:
        raise AudioProcessingError("Normalization would clip; output was rejected")
    return _operation(signal, "normalize_peak", output, signal.sampling_rate, applied=True,
                      parameters={**params, "gain_db": gain_db, "linear_gain": gain})


def reduce_noise(
    audio: ProcessedAudio | AudioData | np.ndarray,
    sampling_rate: int | None = None,
    *,
    enabled: bool = False,
    noise_profile: np.ndarray | None = None,
    strength: float = 0.35,
    reducer: Callable[..., np.ndarray] | None = None,
) -> ProcessedAudio:
    """Optional stationary spectral reduction requiring caller-supplied noise audio."""
    signal = _as_signal(audio, sampling_rate)
    params = {"enabled": enabled, "strength": strength, "requires_noise_profile": True}
    if not enabled:
        return _operation(signal, "noise_reduction", signal.samples, signal.sampling_rate,
                          applied=False, parameters=params, reason="disabled")
    if not 0.0 <= strength <= 1.0:
        raise AudioProcessingError("strength must be between 0 and 1")
    if noise_profile is None or np.size(noise_profile) == 0:
        raise AudioProcessingError("Enabled noise reduction requires an explicit noise_profile")
    if signal.channels != 1:
        raise AudioProcessingError("Convert to mono explicitly before noise reduction")
    noise = np.asarray(noise_profile, dtype=np.float32)
    if not np.isfinite(noise).all():
        raise AudioProcessingError("Noise profile contains non-finite samples")
    if reducer is None:
        try:
            reducer = importlib.import_module("noisereduce").reduce_noise
        except ImportError as exc:
            raise AudioProcessingError(
                "Optional noise reduction requires the 'audio-denoise' extra"
            ) from exc
    waveform = signal.samples[:, 0] if signal.samples.ndim == 2 else signal.samples
    output = np.asarray(reducer(y=waveform, y_noise=noise, sr=signal.sampling_rate,
                                stationary=True, prop_decrease=strength), dtype=np.float32)
    if output.shape != waveform.shape:
        raise AudioProcessingError("Noise reducer changed the waveform length")
    if output.ndim == 1 and signal.samples.ndim == 2:
        output = output[:, None]
    return _operation(signal, "noise_reduction", output, signal.sampling_rate, applied=True,
                      parameters={**params, "stationary": True, "noise_profile_samples": int(noise.size),
                                  "implementation": getattr(reducer, "__module__", "custom reducer")})


def preprocess_audio(
    audio: ProcessedAudio | AudioData | np.ndarray,
    config: AudioProcessingConfig = AudioProcessingConfig(),
    *,
    sampling_rate: int | None = None,
    noise_profile: np.ndarray | None = None,
) -> ProcessedAudio:
    """Apply only explicitly enabled operations, always retaining the source copy."""
    signal = _as_signal(audio, sampling_rate)
    if config.convert_to_float32:
        signal = to_float32(signal)
    if config.convert_to_mono:
        signal = convert_to_mono(signal)
    if config.target_sample_rate is not None:
        signal = resample_audio(signal, config.target_sample_rate)
    if config.normalize:
        signal = normalize_peak(
            signal,
            target_dbfs=config.normalize_target_dbfs,
            max_gain_db=config.normalize_max_gain_db,
            minimum_peak_dbfs=config.normalize_minimum_peak_dbfs,
        )
    signal = reduce_noise(
        signal,
        enabled=config.noise_reduction,
        noise_profile=noise_profile,
        strength=config.noise_reduction_strength,
    )
    return signal


def compare_audio(original: ProcessedAudio | AudioData, processed: ProcessedAudio) -> dict[str, Any]:
    """Compare measurements; samplewise differences only when grids match."""
    original_properties = analyze_audio(original)
    processed_properties = analyze_audio(processed)
    same_grid = (
        original_properties.sampling_rate == processed_properties.sampling_rate
        and original_properties.samples == processed_properties.samples
        and original_properties.channels == processed_properties.channels
    )
    result: dict[str, Any] = {
        "original": original_properties.as_dict(),
        "processed": processed_properties.as_dict(),
        "duration_delta_seconds": processed_properties.duration_seconds - original_properties.duration_seconds,
        "samplewise_comparable": same_grid,
        "operations": [asdict(operation) for operation in processed.operations],
    }
    if same_grid:
        before = original.samples
        after = processed.samples
        diff = after.astype(np.float64) - before.astype(np.float64)
        result["difference"] = {
            "max_absolute": float(np.max(np.abs(diff))),
            "rms": float(np.sqrt(np.mean(np.square(diff)))),
        }
    else:
        result["difference"] = None
    return result
