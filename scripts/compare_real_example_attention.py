"""Compare eager and SDPA on one already-decoded, held-out-safe A.3 sample."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
import torch
from transformers import AutoFeatureExtractor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quran_model import QuranMultiLevelCTC  # noqa: E402
from quran_training_data import QuranTrainingCollator  # noqa: E402

SELECTION_RUN = ROOT / "runs/pilot_final_selection_20261122T160812Z"
MODEL_ID = "obadx/muaalem-model-v3_2"
MODEL_REVISION = "01a1ef9fbe40d144ef845101e89ff924aed3fef5"
EXAMPLE_KEY = "moshaf_11.0:11.0:002.0038"
MAX_RESERVED_FRACTION = 0.92
MAX_NVIDIA_MIB = 5700


def _nvidia_used_mib():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return int(result.stdout.strip().splitlines()[0])
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _snapshot(device, phase):
    torch.cuda.synchronize(device)
    return {
        "phase": phase,
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "nvidia_smi_used_mib": _nvidia_used_mib(),
    }


def main():
    run_id = datetime.now(timezone.utc).strftime("attention_compare_%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    result = {"run_id": run_id, "status": "RUNNING", "example_key": EXAMPLE_KEY,
              "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "memory_limits": {"reserved_fraction": MAX_RESERVED_FRACTION,
                                "nvidia_mib": MAX_NVIDIA_MIB}}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; attention comparison requires the RTX GPU")
        device = torch.device("cuda:0")
        baseline = _nvidia_used_mib()
        result["gpu_idle_baseline_nvidia_mib"] = baseline
        if isinstance(baseline, int) and baseline >= MAX_NVIDIA_MIB:
            raise RuntimeError(f"GPU already at or above the approved {MAX_NVIDIA_MIB} MiB threshold")

        selection = json.loads((SELECTION_RUN / "selection_report.json").read_text(encoding="utf-8"))
        labels_blob = json.loads((SELECTION_RUN / "prepared_labels.json").read_text(encoding="utf-8"))
        selected_index = next(i for i, row in enumerate(selection["decoded_training_examples"])
                              if row["source_key"] == EXAMPLE_KEY)
        audio_arrays = np.load(SELECTION_RUN / "prepared_audio.npz", allow_pickle=False)
        waveform = np.asarray(audio_arrays[f"train_{selected_index:02d}"]).squeeze()
        label_record = labels_blob["training"][selected_index]
        extractor = AutoFeatureExtractor.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, local_files_only=True
        )
        collator = QuranTrainingCollator(extractor)
        example = {
            "source_key": EXAMPLE_KEY,
            "split": "train",
            "audio_samples": waveform,
            "sampling_rate": 16000,
            "labels": {key: tuple(value) for key, value in label_record["labels"].items()},
        }
        batch_cpu = collator([example])
        result["example"] = {
            "waveform_samples": int(waveform.size),
            "decoded_duration_seconds": float(waveform.size / 16000),
            "input_features_shape": list(batch_cpu["input_features"].shape),
            "attention_mask_shape": list(batch_cpu["attention_mask"].shape),
            "attention_mask_valid_frames": int(batch_cpu["attention_mask"].sum()),
            "input_lengths": batch_cpu["input_lengths"].tolist(),
            "target_lengths": {k: v.tolist() for k, v in batch_cpu["target_lengths"].items()},
        }
        model, load_report = QuranMultiLevelCTC.from_pretrained_checkpoint(
            MODEL_ID, revision=MODEL_REVISION, device="cpu", local_files_only=True
        )
        if not load_report.is_complete or load_report.loaded_parameter_count != 605_754_251:
            raise RuntimeError(f"Checkpoint audit failed: {load_report}")
        model.to(device)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        # Make backend comparison deterministic: eager and SDPA can consume
        # different RNG streams for attention dropout. Dropout is disabled only
        # for this equivalence diagnostic; the actual pilot keeps model.train()
        # dropout probabilities unchanged.
        dropout_modules = [module for module in model.modules() if isinstance(module, torch.nn.Dropout)]
        original_dropout_probabilities = [module.p for module in dropout_modules]
        for module in dropout_modules:
            module.p = 0.0
        layers = list(model.encoder.encoder.layers)
        result["gradient_checkpointing"] = {
            "enabled_layers": sum(bool(layer.gradient_checkpointing) for layer in layers),
            "total_layers": len(layers),
            "wrapper_class": type(layers[0]).__mro__[1].__name__,
        }
        if result["gradient_checkpointing"]["enabled_layers"] != len(layers):
            raise RuntimeError("Gradient checkpointing is not active on every encoder layer")
        result["checkpoint_tensor_count"] = len(load_report.matched)
        batch = {k: (v.to(device) if torch.is_tensor(v) else
                     {n: x.to(device) for n, x in v.items()} if isinstance(v, dict) else v)
                 for k, v in batch_cpu.items()}
        precision_scaler = torch.amp.GradScaler("cuda", enabled=True)
        outputs = {}
        gradients = {}
        measurements = {}
        for backend in ("eager", "sdpa"):
            torch.manual_seed(1729)
            torch.cuda.manual_seed_all(1729)
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            model.set_attention_implementation(backend)
            phase = [_snapshot(device, "before_forward")]
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(**batch)
            if not torch.isfinite(output.loss):
                raise FloatingPointError(f"{backend} returned a non-finite loss")
            phase.append(_snapshot(device, "after_forward"))
            outputs[backend] = {
                "loss": float(output.loss.detach().float().cpu()),
                "logits": {name: logits.detach().float().cpu() for name, logits in output.logits.items()},
                "attention_status": model.attention_implementation_status(),
            }
            precision_scaler.scale(output.loss).backward()
            phase.append(_snapshot(device, "after_backward"))
            gradient_finite = all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
            if not gradient_finite:
                raise FloatingPointError(f"{backend} produced non-finite gradients")
            gradients[backend] = {
                "all_present_gradients_finite": gradient_finite,
                "gradient_tensor_count": sum(parameter.grad is not None for parameter in model.parameters()),
                "encoder_feature_projection": model.encoder.feature_projection.projection.weight.grad.detach().float().cpu(),
                "phoneme_head": model.level_to_lm_head["phonemes"].weight.grad.detach().float().cpu(),
            }
            peak_reserved = max(item["peak_reserved_bytes"] for item in phase)
            cap = torch.cuda.get_device_properties(device).total_memory
            peak_nvidia = max((x["nvidia_smi_used_mib"] for x in phase
                               if isinstance(x["nvidia_smi_used_mib"], int)), default=None)
            measurements[backend] = {
                "phases": phase,
                "peak_reserved_fraction": peak_reserved / cap,
                "peak_nvidia_smi_used_mib": peak_nvidia,
            }
            if peak_reserved >= int(MAX_RESERVED_FRACTION * cap):
                raise RuntimeError(
                    f"{backend} single-example diagnostic reached reserved peak "
                    f"{peak_reserved}/{cap} bytes, exceeding the unchanged safety threshold"
                )
            if peak_nvidia is not None and peak_nvidia >= MAX_NVIDIA_MIB:
                raise RuntimeError(f"{backend} diagnostic exceeded {MAX_NVIDIA_MIB} MiB NVIDIA threshold")
            del output
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        result["attention_comparison"] = {}
        for name in outputs["eager"]["logits"]:
            difference = outputs["sdpa"]["logits"][name] - outputs["eager"]["logits"][name]
            eager_logits = outputs["eager"]["logits"][name]
            sdpa_logits = outputs["sdpa"]["logits"][name]
            result["attention_comparison"][name] = {
                "max_absolute_logit_difference": float(difference.abs().max()),
                "mean_absolute_logit_difference": float(difference.abs().mean()),
                "mean_eager_absolute_logit": float(eager_logits.abs().mean()),
                "relative_mean_absolute_difference": float(
                    difference.abs().mean() / eager_logits.abs().mean().clamp_min(1e-12)
                ),
                "argmax_token_agreement_fraction": float(
                    (sdpa_logits.argmax(dim=-1) == eager_logits.argmax(dim=-1)).float().mean()
                ),
                "cosine_similarity": float(torch.nn.functional.cosine_similarity(
                    sdpa_logits.reshape(1, -1), eager_logits.reshape(1, -1)
                ).item()),
            }
        result["loss_absolute_difference"] = abs(outputs["sdpa"]["loss"] - outputs["eager"]["loss"])
        result["losses_finite"] = all(np.isfinite(outputs[b]["loss"]) for b in outputs)
        result["gradients_finite"] = all(gradients[b]["all_present_gradients_finite"] for b in gradients)
        result["gradient_comparison"] = {
            "encoder_feature_projection_max_absolute_difference": float(
                (gradients["sdpa"]["encoder_feature_projection"] -
                 gradients["eager"]["encoder_feature_projection"]).abs().max()
            ),
            "phoneme_head_max_absolute_difference": float(
                (gradients["sdpa"]["phoneme_head"] - gradients["eager"]["phoneme_head"]).abs().max()
            ),
        }
        result["measurements"] = measurements
        result["output_compatibility"] = all(
            row["relative_mean_absolute_difference"] < 1e-3
            and row["argmax_token_agreement_fraction"] >= 0.999
            and row["cosine_similarity"] >= 0.99999
            for row in result["attention_comparison"].values()
        )
        for module, probability in zip(dropout_modules, original_dropout_probabilities):
            module.p = probability
        result["status"] = "PASS" if (
            result["output_compatibility"] and result["losses_finite"] and result["gradients_finite"]
        ) else "FAIL_COMPATIBILITY"
    except BaseException as exc:
        result["status"] = "FAILED"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        (run_dir / "attention_comparison.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
        )
        print(json.dumps({key: value for key, value in result.items()
                          if key not in {"traceback", "attention_comparison"}}, indent=2, default=str))
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
