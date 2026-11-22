"""Bounded, resumable fine-tuning loop for prepared Quran Muaalem examples."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import subprocess
import threading
import time
from typing import Any, Callable, Sequence

import psutil
import torch

from .checkpointing import load_training_checkpoint, save_training_checkpoint
from .config import TrainingConfig


@dataclass(frozen=True)
class TrainingRunConfig:
    """Bounds and cadence for one selected-example training run."""

    epochs: int = 1
    max_optimizer_steps: int = 1
    max_train_examples: int = 1
    checkpoint_every_steps: int = 1
    log_every_steps: int = 1
    seed: int = 42
    max_gpu_reserved_fraction: float = 0.92
    max_nvidia_smi_used_mib: int = 5700
    min_system_available_gib: float = 2.0
    require_all_parameter_gradients: bool = True
    scheduler_warmup_ratio: float | None = None

    def __post_init__(self):
        if min(self.epochs, self.max_optimizer_steps, self.max_train_examples,
               self.checkpoint_every_steps, self.log_every_steps) < 1:
            raise ValueError("epochs, limits, and logging/checkpoint intervals must be positive")
        if not 0 < self.max_gpu_reserved_fraction <= 1:
            raise ValueError("max_gpu_reserved_fraction must be in (0, 1]")
        if self.max_nvidia_smi_used_mib < 1 or self.min_system_available_gib < 0:
            raise ValueError("resource limits must be nonnegative")
        if self.scheduler_warmup_ratio is not None and not 0 <= self.scheduler_warmup_ratio < 1:
            raise ValueError("scheduler_warmup_ratio must be None or in [0, 1)")


class ResourceLimitReached(RuntimeError):
    """Training stopped at a configured memory safety threshold."""


class MemoryMonitor:
    """Sample physical GPU and host memory while training is active."""

    def __init__(self, *, interval_seconds: float = 0.1, device_index: int | None = 0):
        self.interval_seconds = interval_seconds
        self.device_index = device_index
        self._stop = threading.Event()
        self._process = psutil.Process()
        self._phase = "setup"
        self._context: dict[str, Any] = {"global_step": None, "example_id": None, "duration_seconds": None}
        self._lock = threading.Lock()
        self.samples: list[dict[str, Any]] = []
        self.cuda_memory_snapshots: list[dict[str, Any]] = []
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def start(self):
        self._thread.start()

    def set_phase(self, phase: str):
        with self._lock:
            self._phase = phase

    def set_context(self, *, global_step: int, example_id: str | None = None,
                    duration_seconds: float | None = None):
        with self._lock:
            self._context = {
                "global_step": global_step,
                "example_id": example_id,
                "duration_seconds": duration_seconds,
            }

    def record_cuda_memory(self, *, phase: str, event: str, device: torch.device) -> dict[str, Any] | None:
        """Capture synchronized current and peak PyTorch memory at a phase boundary."""
        if device.type != "cuda":
            return None
        torch.cuda.synchronize(device)
        snapshot = {
            "timestamp": time.time(),
            "phase": phase,
            "event": event,
            "device": str(device),
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        self.cuda_memory_snapshots.append(snapshot)
        return snapshot

    def _sample(self):
        while not self._stop.is_set():
            sample = {
                "timestamp": time.time(),
                "process_rss_bytes": self._process.memory_info().rss,
                "system_available_bytes": psutil.virtual_memory().available,
                "nvidia_smi_used_mib": None,
            }
            try:
                if self.device_index is None:
                    raise FileNotFoundError("GPU sampling is disabled for a CPU run")
                result = subprocess.run(
                    ["nvidia-smi", f"--id={self.device_index}", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    check=True, capture_output=True, text=True, timeout=5,
                )
                sample["nvidia_smi_used_mib"] = int(result.stdout.strip().splitlines()[0])
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            with self._lock:
                sample["phase"] = self._phase
                sample.update(self._context)
                self.samples.append(sample)
            self._stop.wait(self.interval_seconds)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=6)

    def summary(self, *, include_samples: bool = False) -> dict[str, Any]:
        samples = list(self.samples)
        gpu = [row["nvidia_smi_used_mib"] for row in samples if row["nvidia_smi_used_mib"] is not None]
        result = {
            "poll_wait_interval_seconds": self.interval_seconds,
            "device_index": self.device_index,
            "sample_count": len(samples),
            "nvidia_smi_sample_count": len(gpu),
            "nvidia_smi_peak_used_mib": max(gpu) if gpu else None,
            "nvidia_smi_samples_by_phase": {
                phase: len([row for row in samples if row["phase"] == phase and row["nvidia_smi_used_mib"] is not None])
                for phase in sorted({row["phase"] for row in samples})
            },
            "process_rss_peak_bytes": max((row["process_rss_bytes"] for row in samples), default=None),
            "system_available_min_bytes": min((row["system_available_bytes"] for row in samples), default=None),
        }
        if include_samples:
            result["samples"] = samples
        result["cuda_memory_snapshots"] = list(self.cuda_memory_snapshots)
        return result


def _accumulate_gradients_on_cpu(named_parameters, accumulated: dict[str, torch.Tensor]) -> None:
    """Move each microbatch's scaled gradients off-device and sum them in-place."""
    for name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        gradient_cpu = gradient.detach().to(device="cpu")
        if name in accumulated:
            accumulated[name].add_(gradient_cpu)
        else:
            accumulated[name] = gradient_cpu
        parameter.grad = None
        # On CUDA, the detached host copy is now owned by `accumulated`; drop
        # this local reference and the old device grad before copying the next
        # parameter. This bounds transient copies to one parameter at a time.
        del gradient_cpu, gradient


def _restore_gradients_to_parameters(named_parameters, accumulated, device) -> None:
    """Restore gradients while releasing their CPU buffers one parameter at a time."""
    for name, parameter in named_parameters:
        gradient = accumulated.pop(name, None)
        if gradient is not None:
            parameter.grad = gradient.to(device=device)
            del gradient


def build_linear_warmup_scheduler(optimizer, *, total_steps: int, warmup_ratio: float):
    """Linear warmup followed by linear decay, with explicit integer step counts."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    warmup_steps = math.ceil(total_steps * warmup_ratio) if warmup_ratio else 0

    def factor(current_step: int) -> float:
        if warmup_steps and current_step < warmup_steps:
            return float(current_step) / float(warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        return max(0.0, float(total_steps - current_step) / float(decay_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    scheduler.total_steps = total_steps
    scheduler.warmup_steps = warmup_steps
    return scheduler


def total_optimizer_steps(example_count: int, accumulation_steps: int, epochs: int) -> int:
    if min(example_count, accumulation_steps, epochs) < 1:
        raise ValueError("example_count, accumulation_steps, and epochs must be positive")
    return math.ceil(example_count / accumulation_steps) * epochs


def optimizer_update_succeeded(*, scaler_enabled: bool, scale_before: float, scale_after: float) -> bool:
    """GradScaler lowers its scale when it skipped an optimizer update for overflow."""
    return not scaler_enabled or scale_after >= scale_before


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            moved[key] = {name: item.to(device) if torch.is_tensor(item) else item for name, item in value.items()}
        else:
            moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def _example_field(example: Any, key: str, default=None):
    return example.get(key, default) if isinstance(example, dict) else getattr(example, key, default)


def _example_duration_seconds(example: Any) -> float | None:
    direct_duration = _example_field(example, "duration_seconds")
    if direct_duration is not None:
        return float(direct_duration)
    audio = _example_field(example, "audio")
    duration = _example_field(audio, "duration_seconds") if audio is not None else None
    if duration is not None:
        return float(duration)
    samples = _example_field(audio, "samples") if audio is not None else None
    rate = _example_field(audio, "sampling_rate") if audio is not None else None
    if samples is not None and rate:
        return float(len(samples) / rate)
    return None


def _autocast_context(device: torch.device, precision: str):
    if precision == "no" or (device.type == "cpu" and precision == "fp16"):
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def _validate_split(examples: Sequence[Any], expected: str, max_examples: int):
    selected = list(examples[:max_examples])
    if not selected:
        raise ValueError(f"No {expected} examples were provided")
    keys = set()
    for example in selected:
        split = _example_field(example, "split")
        source_key = _example_field(example, "source_key")
        if split != expected:
            raise ValueError(f"Expected {expected} example, got split={split!r} for {source_key!r}")
        if source_key in keys:
            raise ValueError(f"Duplicate {expected} example key: {source_key}")
        keys.add(source_key)
    return selected


class QuranTrainer:
    """Training loop over already prepared, immutable Stage 5 examples."""

    def __init__(
        self,
        *,
        model,
        collator,
        training_config: TrainingConfig,
        run_config: TrainingRunConfig,
        output_dir: str | Path,
        device: str | torch.device | None = None,
    ):
        self.model = model
        self.collator = collator
        self.training_config = training_config
        self.run_config = run_config
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def evaluate(self, examples: Sequence[Any], *, max_examples: int) -> dict[str, Any]:
        selected = _validate_split(examples, "validation", max_examples)
        was_training = self.model.training
        parameter_versions = [parameter._version for parameter in self.model.parameters()]
        losses = []
        monitor = getattr(self, "_active_monitor", None)
        if monitor is not None:
            monitor.set_phase("validation")
        self.model.eval()
        try:
            with torch.inference_mode():
                for example in selected:
                    if monitor is not None:
                        monitor.set_context(
                            global_step=getattr(self, "_active_global_step", 0),
                            example_id=_example_field(example, "source_key"),
                            duration_seconds=_example_duration_seconds(example),
                        )
                    batch = _move_batch(self.collator([example]), self.device)
                    with _autocast_context(self.device, self.training_config.mixed_precision):
                        output = self.model(**batch)
                    loss = float(output.loss.detach().float().cpu())
                    if not math.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite validation loss for {_example_field(example, 'source_key')}"
                        )
                    losses.append(loss)
        finally:
            self.model.train(was_training)
        after_versions = [parameter._version for parameter in self.model.parameters()]
        if parameter_versions != after_versions:
            raise RuntimeError("Validation mutated one or more model parameters")
        return {
            "split": "validation",
            "examples": len(losses),
            "mean_ctc_loss": sum(losses) / len(losses),
            "per_head_losses": None,
            "per_head_loss_note": "The current model output exposes aggregate CTC loss only.",
            "parameters_unchanged": True,
        }

    def _save(self, path: Path, *, optimizer, scheduler, scaler, epoch: int,
              global_step: int, cursor: int, examples_seen: int, extra: dict[str, Any]):
        save_training_checkpoint(
            path,
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            config={
                "training_config": self.training_config.__dict__,
                "run_config": asdict(self.run_config),
            },
            extra={
                **extra,
                "next_epoch": epoch,
                "next_example_offset": cursor,
                "examples_seen": examples_seen,
            },
        )

    def _check_resources(self, monitor: MemoryMonitor, *, peak_reserved_bytes: int | None = None):
        if self.device.type != "cuda":
            return
        properties = torch.cuda.get_device_properties(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        reserved_for_check = max(reserved, peak_reserved_bytes or 0)
        fraction = reserved_for_check / properties.total_memory
        if fraction >= self.run_config.max_gpu_reserved_fraction:
            raise ResourceLimitReached(
                f"CUDA reserved memory reached {fraction:.1%} of device capacity "
                f"({reserved_for_check}/{properties.total_memory} bytes; current={reserved})"
            )
        sampled_gpu = [row["nvidia_smi_used_mib"] for row in monitor.samples
                       if row["nvidia_smi_used_mib"] is not None]
        if sampled_gpu and max(sampled_gpu) >= self.run_config.max_nvidia_smi_used_mib:
            raise ResourceLimitReached(
                f"nvidia-smi sampled {max(sampled_gpu)} MiB, threshold is "
                f"{self.run_config.max_nvidia_smi_used_mib} MiB"
            )
        available = psutil.virtual_memory().available / (1024 ** 3)
        if available <= self.run_config.min_system_available_gib:
            raise ResourceLimitReached(
                f"System available RAM is {available:.2f} GiB, threshold is "
                f"{self.run_config.min_system_available_gib:.2f} GiB"
            )

    def _write_failure_report(self, *, exception: BaseException, monitor: MemoryMonitor,
                              global_step: int, example: Any | None):
        device_index = self.device.index
        if self.device.type == "cuda" and device_index is None:
            device_index = torch.cuda.current_device()
        memory = {
            "current_allocated_bytes": None,
            "current_reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
        gpu = None
        if self.device.type == "cuda":
            try:
                memory.update({
                    "current_allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
                    "current_reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
                })
                properties = torch.cuda.get_device_properties(self.device)
                gpu = {
                    "name": properties.name,
                    "device_index": device_index,
                    "capacity_bytes": int(properties.total_memory),
                }
            except Exception as snapshot_error:
                memory["snapshot_error"] = f"{type(snapshot_error).__name__}: {snapshot_error}"
        sample_context = dict(monitor._context)
        current = {
            "example_id": sample_context.get("example_id") or (
                _example_field(example, "source_key") if example is not None else None
            ),
            "duration_seconds": sample_context.get("duration_seconds")
            if sample_context.get("duration_seconds") is not None
            else (_example_duration_seconds(example) if example is not None else None),
        }
        report = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "training_phase": monitor._phase,
            "global_step": global_step,
            "current_example": current,
            "device": str(self.device),
            "gpu": gpu,
            "configured_thresholds": {
                "max_gpu_reserved_fraction": self.run_config.max_gpu_reserved_fraction,
                "max_nvidia_smi_used_mib": self.run_config.max_nvidia_smi_used_mib,
                "min_system_available_gib": self.run_config.min_system_available_gib,
            },
            "model_runtime_diagnostics": getattr(self, "_model_runtime_diagnostics", None),
            "pytorch_memory": memory,
            "pytorch_phase_memory_snapshots": list(monitor.cuda_memory_snapshots),
            "current_process_rss_bytes": psutil.Process().memory_info().rss,
            "current_system_available_bytes": psutil.virtual_memory().available,
            "monitor": monitor.summary(include_samples=True),
        }
        destination = self.output_dir / "failure_report.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        temporary.replace(destination)

    def fit(
        self,
        train_examples: Sequence[Any],
        validation_examples: Sequence[Any],
        *,
        max_validation_examples: int,
        resume_from: str | Path | None = None,
    ) -> dict[str, Any]:
        train = _validate_split(train_examples, "train", len(train_examples))
        validation = _validate_split(validation_examples, "validation", max_validation_examples)
        accumulation = self.training_config.gradient_accumulation_steps
        if self.training_config.scheduler != "linear":
            raise ValueError(
                "QuranTrainer currently implements the configured 'linear' scheduler only"
            )
        planned_steps = total_optimizer_steps(len(train), accumulation, self.run_config.epochs)
        if self.run_config.max_optimizer_steps > planned_steps:
            raise ValueError(
                f"max_optimizer_steps={self.run_config.max_optimizer_steps} exceeds "
                f"the planned {planned_steps} optimizer steps"
            )

        self.model.to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        if self.training_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.model.train()
        encoder = getattr(self.model, "encoder", None)
        encoder_layers = list(getattr(getattr(encoder, "encoder", None), "layers", ()))
        checkpointed_layers = sum(bool(getattr(layer, "gradient_checkpointing", False))
                                  for layer in encoder_layers)
        attention_status = (
            self.model.attention_implementation_status()
            if hasattr(self.model, "attention_implementation_status") else None
        )
        self._model_runtime_diagnostics = {
            "gradient_checkpointing_configured": self.training_config.gradient_checkpointing,
            "gradient_checkpointing_layers_enabled": checkpointed_layers,
            "gradient_checkpointing_layers_total": len(encoder_layers),
            "attention": attention_status,
            "gradient_accumulation_device": self.training_config.gradient_accumulation_device,
        }
        if (self.training_config.gradient_checkpointing and encoder_layers
                and checkpointed_layers != len(encoder_layers)):
            raise RuntimeError(
                f"Gradient checkpointing active on {checkpointed_layers}/{len(encoder_layers)} encoder layers"
            )
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        trainable_named = [(name, parameter) for name, parameter in self.model.named_parameters()
                           if parameter.requires_grad]
        if not trainable or sum(parameter.numel() for parameter in trainable) != sum(
            parameter.numel() for parameter in self.model.parameters()
        ):
            raise RuntimeError("Full fine-tuning requires every model parameter to be trainable")

        optimizer = self.training_config.build_optimizer(trainable)
        effective_warmup_ratio = (
            self.run_config.scheduler_warmup_ratio
            if self.run_config.scheduler_warmup_ratio is not None
            else self.training_config.warmup_ratio
        )
        scheduler = build_linear_warmup_scheduler(
            optimizer, total_steps=planned_steps, warmup_ratio=effective_warmup_ratio
        )
        amp_enabled = self.device.type == "cuda" and self.training_config.mixed_precision == "fp16"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        global_step = 0
        examples_seen = 0
        epoch_start = 0
        offset_start = 0
        resume_start = None
        if resume_from is not None:
            restored = load_training_checkpoint(
                resume_from,
                model=self.model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                map_location="cpu",
                restore_rng=True,
            )
            stored_training_config = restored["config"].get("training_config")
            if stored_training_config is not None and stored_training_config != self.training_config.__dict__:
                raise ValueError("Resume checkpoint training configuration does not match this run")
            global_step = restored["global_step"]
            epoch_start = int(restored["extra"].get("next_epoch", restored["epoch"]))
            offset_start = int(restored["extra"].get("next_example_offset", 0))
            examples_seen = int(restored["extra"].get("examples_seen", 0))
            resume_start = {
                "global_step": global_step,
                "epoch": epoch_start,
                "next_example_offset": offset_start,
                "examples_seen": examples_seen,
                "scheduler_last_epoch": scheduler.last_epoch,
                "optimizer_state_entries": len(optimizer.state),
                "grad_scaler_scale": float(scaler.get_scale()),
            }
        else:
            self._save(
                self.checkpoint_dir / "initial.pt", optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, epoch=0, global_step=0, cursor=0, examples_seen=0,
                extra={"status": "initial"},
            )

        metrics_path = self.output_dir / "metrics.jsonl"
        monitor_device_index = None
        if self.device.type == "cuda":
            monitor_device_index = self.device.index
            if monitor_device_index is None:
                monitor_device_index = torch.cuda.current_device()
        monitor = MemoryMonitor(device_index=monitor_device_index)
        self._active_monitor = monitor
        self._active_global_step = global_step
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        monitor.start()
        step_rows = []
        stop_reason = "completed_epoch_budget_or_limits"
        last_checkpoint = Path(resume_from) if resume_from is not None else None
        started = time.perf_counter()
        current_example = None
        try:
            monitor.set_phase("initial_validation")
            initial_validation = self.evaluate(
                validation, max_examples=max_validation_examples
            )
            with metrics_path.open("a", encoding="utf-8") as metrics_file:
                for epoch in range(epoch_start, self.run_config.epochs):
                    indices = list(range(len(train)))
                    random.Random(self.run_config.seed + epoch).shuffle(indices)
                    position = offset_start if epoch == epoch_start else 0
                    offset_start = 0
                    while position < len(indices):
                        if global_step >= self.run_config.max_optimizer_steps:
                            stop_reason = "max_optimizer_steps"
                            break
                        if examples_seen >= self.run_config.max_train_examples:
                            stop_reason = "max_train_examples"
                            break
                        budget = self.run_config.max_train_examples - examples_seen
                        group_indices = indices[position:position + min(accumulation, budget)]
                        if not group_indices:
                            break
                        optimizer.zero_grad(set_to_none=True)
                        accumulated_cpu_gradients = (
                            {} if self.training_config.gradient_accumulation_device == "cpu" else None
                        )
                        group_losses = []
                        group_start = time.perf_counter()
                        for index in group_indices:
                            example = train[index]
                            current_example = example
                            self._active_global_step = global_step
                            monitor.set_context(
                                global_step=global_step,
                                example_id=_example_field(example, "source_key"),
                                duration_seconds=_example_duration_seconds(example),
                            )
                            monitor.set_phase("train_forward")
                            batch = _move_batch(self.collator([example]), self.device)
                            monitor.record_cuda_memory(
                                phase="train_forward", event="before_forward", device=self.device
                            )
                            with _autocast_context(self.device, self.training_config.mixed_precision):
                                output = self.model(**batch)
                            monitor.record_cuda_memory(
                                phase="train_forward", event="after_forward", device=self.device
                            )
                            loss = output.loss
                            if loss is None or not torch.isfinite(loss):
                                raise FloatingPointError(
                                    f"Non-finite/missing training loss for {_example_field(example, 'source_key')}"
                                )
                            group_losses.append(float(loss.detach().float().cpu()))
                            monitor.set_phase("train_backward")
                            scaler.scale(loss / len(group_indices)).backward()
                            backward_snapshot = monitor.record_cuda_memory(
                                phase="train_backward", event="after_backward_before_offload",
                                device=self.device,
                            )
                            self._check_resources(
                                monitor,
                                peak_reserved_bytes=(backward_snapshot["peak_reserved_bytes"]
                                                     if backward_snapshot else None),
                            )
                            if accumulated_cpu_gradients is not None:
                                monitor.set_phase("gradient_offload")
                                _accumulate_gradients_on_cpu(trainable_named, accumulated_cpu_gradients)
                            del output, batch, loss
                            if accumulated_cpu_gradients is not None and self.device.type == "cuda":
                                # The gradient tensors now have CPU owners. Return their freed
                                # allocator blocks before the next microbatch's activation pass.
                                torch.cuda.empty_cache()
                            monitor.record_cuda_memory(
                                phase="train_backward", event="after_gradient_offload",
                                device=self.device,
                            )
                            self._check_resources(monitor)

                        if accumulated_cpu_gradients is not None:
                            monitor.set_phase("gradient_restore")
                            _restore_gradients_to_parameters(
                                trainable_named, accumulated_cpu_gradients, self.device
                            )
                            del accumulated_cpu_gradients
                            monitor.record_cuda_memory(
                                phase="train_backward", event="gradients_restored_for_optimizer",
                                device=self.device,
                            )
                        scaler.unscale_(optimizer)
                        if self.run_config.require_all_parameter_gradients:
                            missing = [name for name, parameter in self.model.named_parameters()
                                       if parameter.requires_grad and parameter.grad is None]
                            if missing:
                                raise RuntimeError(f"Trainable parameters without gradients: {missing[:10]}")
                        gradient_norm = torch.nn.utils.clip_grad_norm_(
                            trainable, self.training_config.max_grad_norm, error_if_nonfinite=True
                        )
                        if not torch.isfinite(gradient_norm):
                            raise FloatingPointError("Gradient norm is non-finite")
                        lr_used = [group["lr"] for group in optimizer.param_groups]
                        old_scale = scaler.get_scale()
                        monitor.set_phase("train_optimizer")
                        monitor.record_cuda_memory(
                            phase="train_optimizer", event="before_optimizer", device=self.device
                        )
                        scaler.step(optimizer)
                        scaler.update()
                        monitor.record_cuda_memory(
                            phase="train_optimizer", event="after_optimizer", device=self.device
                        )
                        new_scale = scaler.get_scale()
                        update_succeeded = optimizer_update_succeeded(
                            scaler_enabled=scaler.is_enabled(),
                            scale_before=old_scale,
                            scale_after=new_scale,
                        )
                        if update_succeeded:
                            scheduler.step()
                            global_step += 1
                            self._active_global_step = global_step
                        position += len(group_indices)
                        examples_seen += len(group_indices)
                        elapsed = time.perf_counter() - group_start
                        row = {
                            "event": "train_step",
                            "epoch": epoch,
                            "global_step": global_step,
                            "examples_seen": examples_seen,
                            "microbatches_accumulated": len(group_indices),
                            "optimizer_update_succeeded": update_succeeded,
                            "mean_ctc_loss": sum(group_losses) / len(group_losses),
                            "gradient_norm_before_clip": float(gradient_norm.detach().float().cpu()),
                            "learning_rate_used": lr_used,
                            "learning_rate_next": [group["lr"] for group in optimizer.param_groups],
                            "grad_scaler_scale": float(scaler.get_scale()),
                            "step_seconds": elapsed,
                            "examples_per_second": len(group_indices) / max(elapsed, 1e-9),
                            "finite_loss": True,
                            "finite_gradients": True,
                        }
                        step_rows.append(row)
                        if global_step % self.run_config.log_every_steps == 0:
                            metrics_file.write(json.dumps(row) + "\n")
                            metrics_file.flush()
                        self._check_resources(monitor)

                        if update_succeeded and global_step % self.run_config.checkpoint_every_steps == 0:
                            monitor.set_phase("checkpoint_save")
                            last_checkpoint = self.checkpoint_dir / "latest.pt"
                            self._save(
                                last_checkpoint, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                                epoch=epoch, global_step=global_step, cursor=position,
                                examples_seen=examples_seen, extra={"status": "periodic"},
                            )
                    if global_step >= self.run_config.max_optimizer_steps or examples_seen >= self.run_config.max_train_examples:
                        break

            final_checkpoint = self.checkpoint_dir / "final.pt"
            self._save(
                final_checkpoint, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                epoch=epoch if "epoch" in locals() else 0, global_step=global_step,
                cursor=position if "position" in locals() else 0,
                examples_seen=examples_seen, extra={"status": stop_reason},
            )
            last_checkpoint = final_checkpoint
            final_validation = self.evaluate(
                validation, max_examples=max_validation_examples
            )
        except BaseException as exc:
            stop_reason = (
                "cuda_oom" if isinstance(exc, torch.cuda.OutOfMemoryError)
                else "resource_limit" if isinstance(exc, ResourceLimitReached)
                else "training_exception"
            )
            try:
                monitor.stop()
                self._write_failure_report(
                    exception=exc, monitor=monitor, global_step=global_step,
                    example=current_example,
                )
            except BaseException as telemetry_error:
                try:
                    exc.add_note(
                        f"Failure telemetry could not be persisted: "
                        f"{type(telemetry_error).__name__}: {telemetry_error}"
                    )
                except Exception:
                    pass
            raise
        finally:
            monitor.stop()
            self._active_monitor = None
            self._active_global_step = None

        duration = time.perf_counter() - started
        memory = monitor.summary()
        gpu = torch.cuda.get_device_properties(self.device) if self.device.type == "cuda" else None
        memory.update({
            "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)) if gpu else None,
            "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)) if gpu else None,
            "gpu_total_bytes": int(gpu.total_memory) if gpu else None,
        })
        return {
            "status": stop_reason,
            "device": str(self.device),
            "trainable_parameters": sum(parameter.numel() for parameter in trainable),
            "train_examples_available": len(train),
            "validation_examples": len(validation),
            "max_train_examples": self.run_config.max_train_examples,
            "examples_seen": examples_seen,
            "optimizer_steps_planned": planned_steps,
            "optimizer_steps_max": self.run_config.max_optimizer_steps,
            "optimizer_steps_completed": global_step,
            "gradient_accumulation_steps": accumulation,
            "epochs_planned": self.run_config.epochs,
            "warmup_steps": scheduler.warmup_steps,
            "effective_warmup_ratio": effective_warmup_ratio,
            "scheduler_total_steps": scheduler.total_steps,
            "initial_validation": initial_validation,
            "final_validation": final_validation,
            "step_metrics": step_rows,
            "elapsed_seconds": duration,
            "examples_per_second_overall": examples_seen / max(duration, 1e-9),
            "memory": memory,
            "model_runtime_diagnostics": self._model_runtime_diagnostics,
            "last_checkpoint": str(last_checkpoint) if last_checkpoint else None,
            "metrics_jsonl": str(metrics_path),
            "resume": {"enabled": resume_from is not None, "restored_state": resume_start},
        }


__all__ = [
    "MemoryMonitor", "QuranTrainer", "ResourceLimitReached", "TrainingRunConfig",
    "build_linear_warmup_scheduler", "optimizer_update_succeeded", "total_optimizer_steps",
]
