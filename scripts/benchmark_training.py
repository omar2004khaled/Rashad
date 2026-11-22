"""Short RTX benchmark: one real batch, verified checkpoint, no epoch/training run."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time

import psutil
import torch
from transformers import AutoFeatureExtractor

from quran_audio import AudioProcessingConfig
from quran_dataset import DatasetConfig, DatasetLoader
from quran_model import QuranMultiLevelCTC
from quran_training import TrainingConfig
from quran_training import load_training_checkpoint, save_training_checkpoint
from quran_training_data import InclusionPolicy, QuranTrainingCollator, TrainingDataBuilder


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = "obadx/muaalem-model-v3_2"
REVISION = "01a1ef9fbe40d144ef845101e89ff924aed3fef5"


def _find_real_training_examples(configurations, scan_limit, desired_examples):
    dataset_config = DatasetConfig.from_file(ROOT / "configs/dataset.json")
    loader = DatasetLoader(dataset_config)
    audio_config = AudioProcessingConfig.from_file(str(ROOT / "configs/audio_preprocessing.json"))
    manifest = json.loads((ROOT / "configs/reciter_split_manifest.json").read_text(encoding="utf-8"))
    builder = TrainingDataBuilder(
        loader,
        audio_config=audio_config,
        policy=InclusionPolicy(),
        split_manifest=manifest,
    )
    exclusions = []
    scanned = 0
    examples = []
    for outcome in builder.iter_prepared(configurations, limit_per_configuration=scan_limit):
        scanned += 1
        if outcome.example is not None and outcome.example.split == "train":
            examples.append(outcome.example)
            if len(examples) >= desired_examples:
                return examples, {
                    "examples": [{
                        "configuration": example.configuration,
                        "source_key": example.source_key,
                        "duration_seconds": example.audio.samples.shape[0] / example.audio.sampling_rate,
                        "sampling_rate": example.audio.sampling_rate,
                    } for example in examples],
                    "scanned_rows": scanned, "excluded_rows": exclusions,
                }
        if outcome.excluded is not None:
            exclusions.append({"key": outcome.excluded.source_key, "reasons": list(outcome.excluded.reasons)})
    raise RuntimeError(f"No eligible train example found after scanning {scanned} rows: {exclusions}")


def _to_device(batch, device):
    return {key: ({name: value.to(device) for name, value in val.items()}
                  if isinstance(val, dict) else val.to(device) if torch.is_tensor(val) else val)
            for key, val in batch.items()}


class NvidiaSmiSampler:
    """Sample physical device memory use independently of PyTorch's allocator."""

    def __init__(self, interval_seconds=0.1):
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.values_mib = []
        self.phase = "setup"
        self.phase_samples = {}
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--id=0", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    check=True, capture_output=True, text=True, timeout=5,
                )
                value = int(result.stdout.strip().splitlines()[0])
                self.values_mib.append(value)
                self.phase_samples.setdefault(self.phase, []).append(value)
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self.stop_event.wait(self.interval_seconds)

    def start(self):
        self.thread.start()

    def set_phase(self, phase):
        self.phase = phase

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=6)


class SystemMemorySampler:
    """Poll this process RSS and system available RAM during the short run."""

    def __init__(self, interval_seconds=0.1):
        self.interval_seconds = interval_seconds
        self.process = psutil.Process()
        self.stop_event = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            memory = psutil.virtual_memory()
            self.samples.append({
                "process_rss_bytes": self.process.memory_info().rss,
                "system_available_bytes": memory.available,
                "system_used_bytes": memory.used,
            })
            self.stop_event.wait(self.interval_seconds)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)


def _tensor_memory(tensors):
    totals = {}
    for tensor in tensors:
        if torch.is_tensor(tensor):
            key = f"{tensor.device.type}:{tensor.dtype}"
            totals[key] = totals.get(key, 0) + tensor.numel() * tensor.element_size()
    return totals


def _model_memory(model):
    params = list(model.parameters())
    grads = [parameter.grad for parameter in params if parameter.grad is not None]
    return {
        "parameter_bytes_by_dtype": _tensor_memory(params),
        "gradient_bytes_by_dtype": _tensor_memory(grads),
        "trainable_parameter_count": sum(p.numel() for p in params if p.requires_grad),
        "parameters_with_grad_count": sum(p.numel() for p in params if p.requires_grad and p.grad is not None),
        "parameter_tensor_count": len(params),
        "gradient_tensor_count": len(grads),
    }


def _optimizer_memory(optimizer):
    tensors = [tensor for state in optimizer.state.values() for tensor in state.values() if torch.is_tensor(tensor)]
    return {
        "optimizer_state_bytes_by_dtype": _tensor_memory(tensors),
        "optimizer_state_tensor_count": len(tensors),
    }


def _tensor_state_digest(state):
    """Hash all tensor values in a nested state dict one tensor at a time."""
    digest = hashlib.sha256()

    def visit(value, prefix=""):
        if torch.is_tensor(value):
            cpu_tensor = value.detach().contiguous().cpu()
            digest.update(prefix.encode("utf-8"))
            digest.update(str(cpu_tensor.dtype).encode("ascii"))
            digest.update(str(tuple(cpu_tensor.shape)).encode("ascii"))
            digest.update(memoryview(cpu_tensor.numpy()).cast("B"))
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                visit(value[key], f"{prefix}/{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{prefix}/{index}")
        elif value is None or isinstance(value, (str, int, float, bool)):
            digest.update(prefix.encode("utf-8"))
            digest.update(repr(value).encode("utf-8"))
    visit(state)
    return digest.hexdigest()


def _benchmark_mode(model, batches, training_config, *, use_amp=True, steps=10, warmup_steps=2,
                    checkpoint_after_updates=7):
    device = torch.device("cuda")
    model.to(device)
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if training_config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    gradient_checkpointing_active = bool(getattr(model.encoder.encoder, "gradient_checkpointing", False))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = training_config.build_optimizer(trainable)
    # There is no training-loop scheduler implementation yet. Use a stateful constant
    # scheduler here so resume state can be verified without altering the fixed optimizer LR.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    device_batches = [_to_device(batch, device) for batch in batches]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    result = {
        "mode": "full_finetune",
        "optimizer": training_config.optimizer,
        "optimizer_foreach": training_config.optimizer_foreach,
        "gradient_checkpointing_configured": training_config.gradient_checkpointing,
        "gradient_checkpointing_active": gradient_checkpointing_active,
        "mixed_precision": "fp16 autocast" if use_amp else "disabled",
        "status": "not_started",
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "micro_batch_size": int(device_batches[0]["input_features"].shape[0]),
        "feature_frames_by_example": [int(batch["input_features"].shape[1]) for batch in device_batches],
        "warmup_steps": int(warmup_steps),
        "measured_steps": int(steps),
        "total_optimizer_updates": int(warmup_steps + steps),
        "parameter_dtype_before_autocast": str(next(model.parameters()).dtype),
        "scaler_enabled": use_amp,
    }
    parameter = model.level_to_lm_head["phonemes"].weight
    before = parameter.detach()[0, 0].float().item()
    encoder_parameter = model.encoder.feature_projection.projection.weight
    encoder_before = encoder_parameter.detach()[0, 0].float().item()
    changed = False
    gradient_present = False
    sampler = NvidiaSmiSampler()
    system_sampler = SystemMemorySampler()
    sampler.start()
    system_sampler.start()
    phase_measurements = {name: [] for name in ("forward", "backward", "optimizer", "checkpoint")}
    overall_peaks = {"allocated": 0, "reserved": 0}
    measured_peaks = {"allocated": 0, "reserved": 0}
    measured_times = {"forward": [], "backward": [], "optimizer": []}
    warmup_times = {"forward": [], "backward": [], "optimizer": []}
    resume_checkpoint_path = None
    resume_expected = None

    def begin_phase(name, measured):
        sampler.set_phase(("measured_" if measured else "warmup_") + name)
        torch.cuda.synchronize(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        return baseline_allocated, baseline_reserved

    def end_phase(name, baseline, measured):
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        record = {
            "baseline_allocated_bytes": int(baseline[0]),
            "baseline_reserved_bytes": int(baseline[1]),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "allocated_transient_delta_bytes": max(0, peak_allocated - int(baseline[0])),
            "reserved_transient_delta_bytes": max(0, peak_reserved - int(baseline[1])),
        }
        phase_measurements[name].append(record)
        overall_peaks["allocated"] = max(overall_peaks["allocated"], peak_allocated)
        overall_peaks["reserved"] = max(overall_peaks["reserved"], peak_reserved)
        if measured:
            measured_peaks["allocated"] = max(measured_peaks["allocated"], peak_allocated)
            measured_peaks["reserved"] = max(measured_peaks["reserved"], peak_reserved)
        return record

    try:
        total_updates = warmup_steps + steps
        for update_index in range(total_updates):
            is_measured = update_index >= warmup_steps
            timing_bucket = measured_times if is_measured else warmup_times
            device_batch = device_batches[update_index % len(device_batches)]
            optimizer.zero_grad(set_to_none=True)
            forward_base = begin_phase("forward", is_measured)
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                output = model(**device_batch)
            torch.cuda.synchronize(device)
            forward_done = time.perf_counter()
            end_phase("forward", forward_base, is_measured)
            result["loss"] = float(output.loss.detach().float().cpu())
            result["finite_loss"] = bool(torch.isfinite(output.loss).item())
            if not result["finite_loss"]:
                raise FloatingPointError("Non-finite CTC loss")
            backward_base = begin_phase("backward", is_measured)
            scaler.scale(output.loss).backward()
            torch.cuda.synchronize(device)
            backward_done = time.perf_counter()
            end_phase("backward", backward_base, is_measured)
            gradient_present = gradient_present or parameter.grad is not None
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                trainable, training_config.max_grad_norm, error_if_nonfinite=True
            )
            optimizer_base = begin_phase("optimizer", is_measured)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            torch.cuda.synchronize(device)
            step_done = time.perf_counter()
            end_phase("optimizer", optimizer_base, is_measured)
            timing_bucket["forward"].append(forward_done - start)
            timing_bucket["backward"].append(backward_done - forward_done)
            timing_bucket["optimizer"].append(step_done - backward_done)
            changed = changed or parameter.detach()[0, 0].float().item() != before
            completed_updates = update_index + 1
            if completed_updates == checkpoint_after_updates:
                sampler.set_phase("checkpoint_io")
                torch.cuda.synchronize(device)
                checkpoint_baseline = (
                    torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)
                )
                torch.cuda.reset_peak_memory_stats(device)
                handle = tempfile.NamedTemporaryFile(
                    prefix="quran-adafactor-resume-", suffix=".pt", delete=False
                )
                resume_checkpoint_path = Path(handle.name)
                handle.close()
                save_training_checkpoint(
                    resume_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=0,
                    global_step=completed_updates,
                    config=training_config.__dict__,
                )
                resume_expected = {
                    "model_digest": _tensor_state_digest(model.state_dict()),
                    "optimizer_digest": _tensor_state_digest(optimizer.state_dict()),
                    "scheduler_state": scheduler.state_dict(),
                    "global_step": completed_updates,
                }
                torch.cuda.synchronize(device)
                checkpoint_peak_allocated = int(torch.cuda.max_memory_allocated(device))
                checkpoint_peak_reserved = int(torch.cuda.max_memory_reserved(device))
                phase_measurements["checkpoint"].append({
                    "baseline_allocated_bytes": int(checkpoint_baseline[0]),
                    "baseline_reserved_bytes": int(checkpoint_baseline[1]),
                    "peak_allocated_bytes": checkpoint_peak_allocated,
                    "peak_reserved_bytes": checkpoint_peak_reserved,
                    "allocated_transient_delta_bytes": max(0, checkpoint_peak_allocated - int(checkpoint_baseline[0])),
                    "reserved_transient_delta_bytes": max(0, checkpoint_peak_reserved - int(checkpoint_baseline[1])),
                })
                overall_peaks["allocated"] = max(overall_peaks["allocated"], checkpoint_peak_allocated)
                overall_peaks["reserved"] = max(overall_peaks["reserved"], checkpoint_peak_reserved)
        total_time = sum(sum(values) for values in measured_times.values())
        measured_phase_transients = {
            phase: max((item["allocated_transient_delta_bytes"] for item in entries[warmup_steps:]), default=0)
            for phase, entries in phase_measurements.items()
        }
        all_phase_transients = {
            phase: max((item["allocated_transient_delta_bytes"] for item in entries), default=0)
            for phase, entries in phase_measurements.items()
        }
        if phase_measurements["checkpoint"]:
            measured_phase_transients["checkpoint"] = phase_measurements["checkpoint"][-1]["allocated_transient_delta_bytes"]
            all_phase_transients["checkpoint"] = phase_measurements["checkpoint"][-1]["allocated_transient_delta_bytes"]
        result.update({
            "status": "completed",
            "warmup_steps_excluded_from_timing": warmup_steps,
            "forward_seconds_mean": sum(measured_times["forward"]) / steps,
            "backward_seconds_mean": sum(measured_times["backward"]) / steps,
            "optimizer_step_seconds_mean": sum(measured_times["optimizer"]) / steps,
            "step_seconds_mean": total_time / steps,
            "examples_per_second": result["micro_batch_size"] * steps / total_time,
            "parameter_update_verified": changed,
            "gradient_present": gradient_present,
            "finite_gradients_verified": True,
            "scaler_scale": float(scaler.get_scale()),
            "optimizer_state_initialized": bool(optimizer.state),
            "optimizer_defaults": {
                key: value for key, value in optimizer.defaults.items()
                if isinstance(value, (str, int, float, bool, type(None), tuple))
            },
            "adafactor_learning_rate_semantics": (
                {
                    "implementation": "torch.optim.Adafactor",
                    "configured_lr": training_config.learning_rate,
                    "relative_step_option": "not exposed by PyTorch Adafactor; rho_t=min(lr,1/sqrt(step))",
                    "scale_parameter_option": "not exposed; PyTorch always scales by max(eps[1], RMS(parameter))",
                    "warmup_init_option": "not exposed",
                    "weight_decay": training_config.weight_decay,
                    "weight_decay_semantics": "PyTorch Adafactor applies decoupled weight decay using lr as coefficient",
                }
                if training_config.optimizer == "adafactor" else None
            ),
            "effective_optimizer_learning_rate_by_group": [
                group["lr"] for group in optimizer.param_groups
            ],
            "scheduler_during_benchmark": "constant LambdaLR (checkpoint-resume state verification only)",
            "checkpointing_encoder_gradient_present": model.encoder.feature_projection.projection.weight.grad is not None,
            "head_gradient_presence": {
                name: head.weight.grad is not None for name, head in model.level_to_lm_head.items()
            },
            "encoder_parameter_update_verified": model.encoder.feature_projection.projection.weight.detach()[0, 0].float().item() != encoder_before,
            "head_parameter_update_verified": parameter.detach()[0, 0].float().item() != before,
            "phase_memory_measurements": phase_measurements,
            "phase_allocated_transient_max_bytes_all_updates": all_phase_transients,
            "phase_allocated_transient_max_bytes_measured_updates": measured_phase_transients,
            "memory_measurement_definition": (
                "Forward, backward, optimizer, and checkpoint phases reset CUDA peak stats after recording their live allocated/reserved baseline; "
                "phase peak is absolute total at that phase, transient delta is peak minus that phase baseline. "
                "Phase deltas overlap live tensors and must not be summed. Run peaks are maxima of absolute phase peaks."
            ),
        })
        result.update(_model_memory(model))
        result.update(_optimizer_memory(optimizer))
    except torch.cuda.OutOfMemoryError as exc:
        result.update({"status": "cuda_oom", "error": str(exc).splitlines()[0]})
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    except Exception as exc:
        result.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        sampler.stop()
        system_sampler.stop()
        result["peak_allocated_bytes"] = int(overall_peaks["allocated"])
        result["peak_reserved_bytes"] = int(overall_peaks["reserved"])
        result["measured_updates_peak_allocated_bytes"] = int(measured_peaks["allocated"])
        result["measured_updates_peak_reserved_bytes"] = int(measured_peaks["reserved"])
        result["current_allocated_bytes"] = int(torch.cuda.memory_allocated(device))
        result["gpu_total_bytes"] = int(torch.cuda.get_device_properties(device).total_memory)
        result["nvidia_smi_peak_used_mib"] = max(sampler.values_mib) if sampler.values_mib else None
        result["nvidia_smi_samples"] = len(sampler.values_mib)
        result["nvidia_smi_poll_wait_interval_seconds"] = sampler.interval_seconds
        result["nvidia_smi_peak_by_phase_mib"] = {
            name: max(values) for name, values in sampler.phase_samples.items() if values
        }
        result["nvidia_smi_sample_count_by_phase"] = {
            name: len(values) for name, values in sampler.phase_samples.items()
        }
        result["system_memory_samples"] = len(system_sampler.samples)
        if system_sampler.samples:
            result["process_rss_peak_bytes"] = max(x["process_rss_bytes"] for x in system_sampler.samples)
            result["system_used_peak_bytes"] = max(x["system_used_bytes"] for x in system_sampler.samples)
            result["system_available_min_bytes"] = min(x["system_available_bytes"] for x in system_sampler.samples)
        result["gradient_checkpointing_active"] = gradient_checkpointing_active
        result["resume_checkpoint_path"] = str(resume_checkpoint_path) if resume_checkpoint_path else None
        result["resume_expected"] = resume_expected
    return result


def _verify_real_checkpoint_resume(checkpoint_path, training_config, batches, expected):
    """Reload a saved full-model update into fresh objects and run one more update."""
    device = torch.device("cuda")
    model, load_report = QuranMultiLevelCTC.from_pretrained_checkpoint(
        CHECKPOINT, revision=REVISION, device="cpu"
    )
    if not load_report.is_complete:
        raise RuntimeError("Fresh checkpoint model failed its source load audit")
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if training_config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    optimizer = training_config.build_optimizer(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=training_config.mixed_precision == "fp16")
    restored = load_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        map_location="cpu",
        restore_rng=False,
    )
    model_digest = _tensor_state_digest(model.state_dict())
    optimizer_digest = _tensor_state_digest(optimizer.state_dict())
    scheduler_state = scheduler.state_dict()
    verification = {
        "source_checkpoint_complete": load_report.is_complete,
        "model_state_restored_exactly": model_digest == expected["model_digest"],
        "optimizer_state_restored_exactly": optimizer_digest == expected["optimizer_digest"],
        "scheduler_state_restored_exactly": scheduler_state == expected["scheduler_state"],
        "global_step_restored": restored["global_step"] == expected["global_step"],
        "global_step": restored["global_step"],
        "restored_optimizer_state_tensor_count": _optimizer_memory(optimizer)["optimizer_state_tensor_count"],
        "learning_rate_after_restore": [group["lr"] for group in optimizer.param_groups],
    }
    if not all(verification[key] for key in (
        "source_checkpoint_complete", "model_state_restored_exactly",
        "optimizer_state_restored_exactly", "scheduler_state_restored_exactly",
        "global_step_restored",
    )):
        raise RuntimeError(f"Adafactor resume verification failed: {verification}")

    optimizer.zero_grad(set_to_none=True)
    batch = _to_device(batches[0], device)
    with torch.autocast("cuda", dtype=torch.float16,
                        enabled=training_config.mixed_precision == "fp16"):
        output = model(**batch)
    if not torch.isfinite(output.loss):
        raise FloatingPointError("Resume update produced non-finite loss")
    scaler.scale(output.loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), training_config.max_grad_norm, error_if_nonfinite=True
    )
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    torch.cuda.synchronize(device)
    verification["subsequent_update_succeeded"] = True
    verification["subsequent_loss_finite"] = bool(torch.isfinite(output.loss).item())
    verification["subsequent_global_step"] = restored["global_step"] + 1
    return verification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configurations", nargs="+", default=["moshaf_0.1"])
    parser.add_argument("--scan-limit", type=int, default=8)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--checkpoint-after-updates", type=int, default=7)
    parser.add_argument("--optimizers", nargs="+", default=["adamw", "adamw_no_foreach", "adamw8bit", "adafactor"])
    parser.add_argument("--output", default="reports/gpu_benchmark.json")
    args = parser.parse_args()
    if args.steps < 1 or args.warmup_steps < 0:
        parser.error("--steps must be at least one")
    if args.examples < 1 or args.examples > args.scan_limit:
        parser.error("--examples must be between 1 and --scan-limit")
    if not 1 <= args.checkpoint_after_updates <= args.steps + args.warmup_steps:
        parser.error("--checkpoint-after-updates must be within the total warmup plus measured updates")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; benchmark requires the requested GPU")
    device_name = torch.cuda.get_device_name(0)
    feature_extractor = AutoFeatureExtractor.from_pretrained(CHECKPOINT, revision=REVISION)
    training_config = TrainingConfig.from_file(ROOT / "configs/training.json")
    examples, sample_info = _find_real_training_examples(args.configurations, args.scan_limit, args.examples)
    collator = QuranTrainingCollator(feature_extractor)
    batches = [collator([example]) for example in examples]

    benchmark_results = {}
    load_results = {}
    for optimizer_name in args.optimizers:
        config_optimizer = "adamw" if optimizer_name == "adamw_no_foreach" else optimizer_name
        optimizer_config = TrainingConfig(**{
            **training_config.__dict__, "optimizer": config_optimizer,
            "optimizer_foreach": False if optimizer_name == "adamw_no_foreach" else training_config.optimizer_foreach,
        })
        model, load_report = QuranMultiLevelCTC.from_pretrained_checkpoint(
            CHECKPOINT, revision=REVISION, device="cpu"
        )
        if not load_report.is_complete:
            raise RuntimeError(f"Incomplete checkpoint mapping for {optimizer_name}: {load_report}")
        benchmark_results[optimizer_name] = _benchmark_mode(
            model, batches, optimizer_config,
            use_amp=optimizer_config.mixed_precision == "fp16", steps=args.steps,
            warmup_steps=args.warmup_steps,
            checkpoint_after_updates=args.checkpoint_after_updates,
        )
        if optimizer_name == "adafactor":
            checkpoint_path = benchmark_results[optimizer_name].pop("resume_checkpoint_path")
            resume_expected = benchmark_results[optimizer_name].pop("resume_expected")
            del model
            gc.collect()
            torch.cuda.empty_cache()
            try:
                benchmark_results[optimizer_name]["checkpoint_resume_verification"] = (
                    _verify_real_checkpoint_resume(
                        checkpoint_path, optimizer_config, batches, resume_expected
                    )
                )
            finally:
                Path(checkpoint_path).unlink(missing_ok=True)
        load_results[optimizer_name] = {
            "complete": load_report.is_complete,
            "matched_tensors": len(load_report.matched),
            "loaded_parameter_count": load_report.loaded_parameter_count,
            "missing": list(load_report.missing),
            "unexpected": list(load_report.unexpected),
            "shape_mismatches": list(load_report.shape_mismatches),
        }
        if optimizer_name != "adafactor":
            del model
        gc.collect()
        torch.cuda.empty_cache()

    data = {
        "gpu": device_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "training_config": training_config.__dict__,
        "verified_checkpoint": CHECKPOINT,
        "checkpoint_revision": REVISION,
        "checkpoint_load_by_optimizer": load_results,
        "real_sample": sample_info,
        "experiments": benchmark_results,
        "note": f"{args.steps} microbatches from {len(examples)} real examples for each full-finetune optimizer; no epoch or corpus training was run.",
    }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    previous_results = None
    if output_path.exists():
        try:
            loaded_results = json.loads(output_path.read_text(encoding="utf-8"))
            previous_results = loaded_results.get("previous_results_before_audit", loaded_results)
        except (OSError, json.JSONDecodeError):
            previous_results = None
    if previous_results is not None:
        data["previous_results_before_audit"] = previous_results
    data["measurement_method"] = {
        "phase_baseline": "Live allocated and reserved bytes captured separately immediately before each phase.",
        "phase_peak": "Absolute CUDA phase maximum after resetting peak stats at that phase baseline.",
        "phase_transient": "Absolute phase peak minus that same phase's live baseline; phase deltas overlap and are not additive.",
        "run_peak": "Maximum absolute phase peak across all warmup and measured phases.",
        "measured_run_peak": "Maximum absolute phase peak excluding warmup updates.",
        "memory_sampling": "nvidia-smi and psutil poll loops use a 0.1-second wait interval; actual cadence also includes query latency.",
    }
    output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    for name, item in benchmark_results.items():
        print(
            f"{name}: {item['status']}; trainable={item['trainable_parameters']:,}; "
            f"peak_allocated={item.get('peak_allocated_bytes')}; "
            f"peak_reserved={item.get('peak_reserved_bytes')}; "
            f"measured_step_s={item.get('step_seconds_mean')}"
        )
    print(f"Saved benchmark results to {output_path}")


if __name__ == "__main__":
    main()
