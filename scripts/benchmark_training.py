"""Short RTX benchmark: one real batch, verified checkpoint, no epoch/training run."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import threading
import time

import torch
from transformers import AutoFeatureExtractor

from quran_audio import AudioProcessingConfig
from quran_dataset import DatasetConfig, DatasetLoader
from quran_model import QuranMultiLevelCTC
from quran_training import TrainingConfig
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

    def __init__(self, interval_seconds=0.2):
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.values_mib = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--id=0", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    check=True, capture_output=True, text=True, timeout=5,
                )
                self.values_mib.append(int(result.stdout.strip().splitlines()[0]))
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self.stop_event.wait(self.interval_seconds)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=6)


def _benchmark_mode(model, batches, mode, training_config, *, gradient_checkpointing, use_amp=True, steps=3):
    device = torch.device("cuda")
    model.to(device)
    model.train()
    if mode == "frozen_encoder":
        model.encoder.requires_grad_(False)
        model.encoder.eval()
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    device_batches = [_to_device(batch, device) for batch in batches]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    result = {
        "mode": mode,
        "gradient_checkpointing": bool(gradient_checkpointing and mode == "full_finetune"),
        "mixed_precision": "fp16 autocast" if use_amp else "disabled",
        "status": "not_started",
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "micro_batch_size": int(device_batches[0]["input_features"].shape[0]),
        "feature_frames_by_example": [int(batch["input_features"].shape[1]) for batch in device_batches],
        "steps": int(steps),
    }
    parameter = model.level_to_lm_head["phonemes"].weight
    before = parameter.detach()[0, 0].float().item()
    forward_times = []
    backward_times = []
    optimizer_times = []
    changed = False
    gradient_present = False
    sampler = NvidiaSmiSampler()
    sampler.start()
    try:
        for _ in range(steps):
            device_batch = device_batches[_ % len(device_batches)]
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                output = model(**device_batch)
            torch.cuda.synchronize(device)
            forward_done = time.perf_counter()
            result["loss"] = float(output.loss.detach().float().cpu())
            result["finite_loss"] = bool(torch.isfinite(output.loss).item())
            if not result["finite_loss"]:
                raise FloatingPointError("Non-finite CTC loss")
            scaler.scale(output.loss).backward()
            torch.cuda.synchronize(device)
            backward_done = time.perf_counter()
            gradient_present = gradient_present or parameter.grad is not None
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, training_config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            step_done = time.perf_counter()
            forward_times.append(forward_done - start)
            backward_times.append(backward_done - forward_done)
            optimizer_times.append(step_done - backward_done)
            changed = changed or parameter.detach()[0, 0].float().item() != before
        total_time = sum(forward_times) + sum(backward_times) + sum(optimizer_times)
        result.update({
            "status": "completed",
            "forward_seconds_mean": sum(forward_times) / steps,
            "backward_seconds_mean": sum(backward_times) / steps,
            "optimizer_step_seconds_mean": sum(optimizer_times) / steps,
            "step_seconds_mean": total_time / steps,
            "examples_per_second": result["micro_batch_size"] * steps / total_time,
            "parameter_update_verified": changed,
            "gradient_present": gradient_present,
            "scaler_scale": float(scaler.get_scale()),
        })
    except torch.cuda.OutOfMemoryError as exc:
        result.update({"status": "cuda_oom", "error": str(exc).splitlines()[0]})
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    except Exception as exc:
        result.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        sampler.stop()
        result["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        result["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        result["current_allocated_bytes"] = int(torch.cuda.memory_allocated(device))
        result["gpu_total_bytes"] = int(torch.cuda.get_device_properties(device).total_memory)
        result["nvidia_smi_peak_used_mib"] = max(sampler.values_mib) if sampler.values_mib else None
        result["nvidia_smi_samples"] = len(sampler.values_mib)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configurations", nargs="+", default=["moshaf_0.1"])
    parser.add_argument("--scan-limit", type=int, default=8)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", default="reports/gpu_benchmark.json")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least one")
    if args.examples < 1 or args.examples > args.scan_limit:
        parser.error("--examples must be between 1 and --scan-limit")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; benchmark requires the requested GPU")
    device_name = torch.cuda.get_device_name(0)
    feature_extractor = AutoFeatureExtractor.from_pretrained(CHECKPOINT, revision=REVISION)
    training_config = TrainingConfig.from_file(ROOT / "configs/training.json")
    examples, sample_info = _find_real_training_examples(args.configurations, args.scan_limit, args.examples)
    collator = QuranTrainingCollator(feature_extractor)
    batches = [collator([example]) for example in examples]

    model, load_report = QuranMultiLevelCTC.from_pretrained_checkpoint(
        CHECKPOINT, revision=REVISION, device="cpu"
    )
    if not load_report.is_complete:
        raise RuntimeError(f"Incomplete checkpoint mapping: {load_report}")
    full_result = _benchmark_mode(
        model, batches, "full_finetune", training_config,
        gradient_checkpointing=training_config.gradient_checkpointing,
        use_amp=training_config.mixed_precision == "fp16", steps=args.steps,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # A frozen-encoder run is a separate objective and is reported as such.
    frozen_model, frozen_load = QuranMultiLevelCTC.from_pretrained_checkpoint(
        CHECKPOINT, revision=REVISION, device="cpu"
    )
    if not frozen_load.is_complete:
        raise RuntimeError(f"Incomplete checkpoint mapping for frozen trial: {frozen_load}")
    frozen_result = _benchmark_mode(
        frozen_model, batches, "frozen_encoder", training_config,
        gradient_checkpointing=False,
        use_amp=training_config.mixed_precision == "fp16", steps=args.steps,
    )
    del frozen_model
    gc.collect()
    torch.cuda.empty_cache()

    data = {
        "gpu": device_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "training_config": training_config.__dict__,
        "verified_checkpoint": CHECKPOINT,
        "checkpoint_revision": REVISION,
        "checkpoint_load": {
            "complete": load_report.is_complete,
            "matched_tensors": len(load_report.matched),
            "loaded_parameter_count": load_report.loaded_parameter_count,
            "missing": list(load_report.missing),
            "unexpected": list(load_report.unexpected),
            "shape_mismatches": list(load_report.shape_mismatches),
        },
        "real_sample": sample_info,
        "full_finetune": full_result,
        "frozen_encoder_comparison": frozen_result,
        "note": f"{args.steps} microbatches from {len(examples)} real examples per mode; no epoch or corpus training was run.",
    }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data, indent=2))
    print(f"Saved benchmark results to {output_path}")


if __name__ == "__main__":
    main()
