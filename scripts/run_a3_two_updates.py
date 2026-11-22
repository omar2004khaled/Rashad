"""Run exactly two full-fine-tuning pilot updates with checkpoint continuation."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoFeatureExtractor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quran_audio import ProcessedAudio  # noqa: E402
from quran_dataset import AudioData  # noqa: E402
from quran_model import QuranMultiLevelCTC  # noqa: E402
from quran_training import QuranTrainer, MemoryMonitor, TrainingConfig, TrainingRunConfig  # noqa: E402
from quran_training_data import QuranTrainingCollator  # noqa: E402


SELECTION_RUN = ROOT / "runs/pilot_final_selection_20261122T160812Z"
MODEL_ID = "obadx/muaalem-model-v3_2"
MODEL_REVISION = "01a1ef9fbe40d144ef845101e89ff924aed3fef5"


def load_examples():
    report = json.loads((SELECTION_RUN / "selection_report.json").read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or not report.get("audio_validation_pass"):
        raise RuntimeError("Final audio selection report did not pass")
    labels = json.loads((SELECTION_RUN / "prepared_labels.json").read_text(encoding="utf-8"))
    result = {"train": [], "validation": []}
    with np.load(SELECTION_RUN / "prepared_audio.npz", allow_pickle=False) as archive:
        for split, records in (("train", report["decoded_training_examples"]),
                               ("validation", report["decoded_validation_examples"])):
            for index, record in enumerate(records):
                prefix = "train" if split == "train" else "validation"
                samples = np.asarray(archive[f"{prefix}_{index:02d}"], dtype=np.float32)
                sample_rate = int(record["sample_rate"])
                audio = AudioData(samples=samples, sampling_rate=sample_rate, channels=1,
                                  duration_seconds=len(samples) / sample_rate)
                processed = ProcessedAudio.from_audio_data(audio)
                row_labels = labels["training" if split == "train" else "validation"][index]["labels"]
                result[split].append(SimpleNamespace(
                    source_key=record["source_key"], split=split, audio=processed,
                    labels={k: tuple(int(x) for x in v) for k, v in row_labels.items()},
                    metadata={"reciter_id": record["reciter_id"], "moshaf_id": record["moshaf_id"]},
                    duration_seconds=record["duration_seconds_decoded"],
                ))
    return result, report


def main() -> int:
    run_id = datetime.now(timezone.utc).strftime("pilot_a3_updates_%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    result = {"run_id": run_id, "selection_run": str(SELECTION_RUN.relative_to(ROOT)),
              "status": "RUNNING", "training_started": False,
              "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
              "optimizer": "adafactor", "micro_batch_size": 1,
              "gradient_accumulation_steps": 16, "requested_successful_updates": 2}
    host_monitor = MemoryMonitor(interval_seconds=0.2, device_index=0)
    host_monitor.set_phase("pilot_setup")
    host_monitor.start()
    try:
        host_monitor.set_phase("selected_examples_load")
        splits, selection_report = load_examples()
        training_config = TrainingConfig.from_file(ROOT / "configs/training.json")
        if (training_config.optimizer != "adafactor" or training_config.learning_rate != 1e-5
                or training_config.gradient_accumulation_steps != 16
                or training_config.micro_batch_size != 1
                or training_config.mixed_precision != "fp16"
                or not training_config.gradient_checkpointing):
            raise RuntimeError(f"Training configuration violates approved pilot settings: {training_config}")
        training_config.to_file(run_dir / "training_config_snapshot.json")
        result["scheduler_policy"] = {
            "production_learning_rate": training_config.learning_rate,
            "production_warmup_ratio": training_config.warmup_ratio,
            "pilot_warmup_ratio_override": 0.0,
            "reason": "A two-update smoke pilot must exercise two nonzero-LR parameter updates; production config remains unchanged.",
        }
        host_monitor.set_phase("feature_extractor_load")
        extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        collator = QuranTrainingCollator(extractor)

        host_monitor.set_phase("first_checkpoint_model_load")
        first_model, first_load_report = QuranMultiLevelCTC.from_pretrained_checkpoint(
            MODEL_ID, revision=MODEL_REVISION, device="cuda", local_files_only=True
        )
        if (not first_load_report.is_complete or first_load_report.loaded_parameter_count != 605_754_251
                or len(first_load_report.matched) != 816):
            raise RuntimeError(f"Original checkpoint tensor audit failed: {first_load_report}")
        parameter_keys = ["encoder.feature_projection.projection.weight"] + [
            f"level_to_lm_head.{name}.weight" for name in first_model.level_to_lm_head
        ]
        initial_parameters = {
            key: first_model.state_dict()[key].detach().cpu().clone() for key in parameter_keys
        }
        result["checkpoint_load"] = {
            "complete": first_load_report.is_complete,
            "loaded_parameter_count": first_load_report.loaded_parameter_count,
            "matched_tensor_count": len(first_load_report.matched),
            "missing_count": len(first_load_report.missing),
            "unexpected_count": len(first_load_report.unexpected),
            "shape_mismatch_count": len(first_load_report.shape_mismatches),
        }

        first_run = TrainingRunConfig(epochs=1, max_optimizer_steps=1, max_train_examples=16,
                                      checkpoint_every_steps=1, log_every_steps=1, seed=42,
                                      scheduler_warmup_ratio=0.0)
        first_run_json = asdict(first_run)
        (run_dir / "first_run_config.json").write_text(json.dumps(first_run_json, indent=2) + "\n",
                                                       encoding="utf-8")
        first_trainer = QuranTrainer(
            model=first_model, collator=collator, training_config=training_config,
            run_config=first_run, output_dir=run_dir / "first_half", device="cuda",
        )
        host_monitor.set_phase("first_update_and_validation")
        result["training_started"] = True
        result["first_half"] = first_trainer.fit(
            splits["train"], splits["validation"], max_validation_examples=1
        )
        if result["first_half"]["optimizer_steps_completed"] != 1:
            raise RuntimeError("First half did not complete exactly one optimizer update")
        first_lr = result["first_half"]["step_metrics"][0]["learning_rate_used"][0]
        if first_lr <= 0:
            raise RuntimeError(f"Pilot first update used a non-positive learning rate: {first_lr}")
        resume_checkpoint = result["first_half"]["last_checkpoint"]
        host_monitor.set_phase("release_first_model_and_optimizer")
        del first_trainer, first_model
        gc.collect()
        torch.cuda.empty_cache()

        host_monitor.set_phase("resume_model_checkpoint_load")
        resumed_model, resumed_load_report = QuranMultiLevelCTC.from_pretrained_checkpoint(
            MODEL_ID, revision=MODEL_REVISION, device="cuda", local_files_only=True
        )
        if not resumed_load_report.is_complete or resumed_load_report.loaded_parameter_count != 605_754_251:
            raise RuntimeError(f"Fresh model checkpoint audit failed before resume: {resumed_load_report}")
        resume_run = TrainingRunConfig(epochs=1, max_optimizer_steps=2, max_train_examples=32,
                                       checkpoint_every_steps=1, log_every_steps=1, seed=42,
                                       scheduler_warmup_ratio=0.0)
        (run_dir / "resume_run_config.json").write_text(
            json.dumps(asdict(resume_run), indent=2) + "\n", encoding="utf-8"
        )
        resume_trainer = QuranTrainer(
            model=resumed_model, collator=collator, training_config=training_config,
            run_config=resume_run, output_dir=run_dir / "resumed", device="cuda",
        )
        host_monitor.set_phase("resumed_update_and_validation")
        result["resumed"] = resume_trainer.fit(
            splits["train"], splits["validation"], max_validation_examples=1,
            resume_from=resume_checkpoint,
        )
        resume_state = result["resumed"]["resume"]["restored_state"]
        if not resume_state or resume_state["global_step"] != 1:
            raise RuntimeError(f"Resume did not restore step 1: {resume_state}")
        if resume_state["optimizer_state_entries"] <= 0:
            raise RuntimeError(f"Resume had no Adafactor optimizer state: {resume_state}")
        if result["resumed"]["optimizer_steps_completed"] != 2:
            raise RuntimeError("Resumed run did not finish at optimizer step 2")
        if len(result["resumed"]["step_metrics"]) != 1 or not result["resumed"]["step_metrics"][0]["optimizer_update_succeeded"]:
            raise RuntimeError("The post-resume optimizer update did not succeed")
        if result["resumed"]["step_metrics"][0]["learning_rate_used"][0] <= 0:
            raise RuntimeError("The resumed pilot update used a non-positive learning rate")

        del resume_trainer
        final_state = resumed_model.state_dict()
        parameter_deltas = {}
        for key, before in initial_parameters.items():
            after = final_state[key].detach().cpu()
            parameter_deltas[key] = float(torch.linalg.vector_norm(after - before).item())
        result["parameter_delta_l2"] = parameter_deltas
        result["encoder_parameters_changed"] = parameter_deltas[parameter_keys[0]] > 0
        result["all_supervised_heads_changed"] = all(parameter_deltas[k] > 0 for k in parameter_keys[1:])

        update_rows = (result["first_half"]["step_metrics"] + result["resumed"]["step_metrics"])
        result["learning_rate_trace"] = [
            {"global_step": row["global_step"], "learning_rate_used": row["learning_rate_used"],
             "learning_rate_next": row["learning_rate_next"], "gradient_norm": row["gradient_norm_before_clip"]}
            for row in update_rows
        ]
        result["scheduler"] = {
            "total_steps": result["resumed"]["scheduler_total_steps"],
            "warmup_steps": result["resumed"]["warmup_steps"],
            "effective_warmup_ratio": result["resumed"]["effective_warmup_ratio"],
            "restored_last_epoch": resume_state["scheduler_last_epoch"],
        }
        result["validation"] = {
            "initial": result["first_half"]["initial_validation"],
            "after_first_update": result["resumed"]["initial_validation"],
            "after_second_update": result["resumed"]["final_validation"],
        }
        result["memory"] = {
            "first_half": result["first_half"]["memory"],
            "resumed": result["resumed"]["memory"],
        }
        result["status"] = "PASS" if (
            result["encoder_parameters_changed"] and result["all_supervised_heads_changed"]
            and result["resumed"]["status"] == "max_optimizer_steps"
            and all(v["parameters_unchanged"] for v in result["validation"].values())
        ) else "FAIL_ACCEPTANCE_CHECK"
    except BaseException as exc:
        result["status"] = "FAILED"
        result["exception_type"] = type(exc).__name__
        result["exception_message"] = str(exc)
        result["traceback"] = traceback.format_exc()
        if (run_dir / "first_half/failure_report.json").exists():
            result["first_half_failure_report"] = "first_half/failure_report.json"
        if (run_dir / "resumed/failure_report.json").exists():
            result["resume_failure_report"] = "resumed/failure_report.json"
    finally:
        host_monitor.stop()
        result["host_memory_telemetry"] = host_monitor.summary(include_samples=True)
        (run_dir / "host_memory_telemetry.json").write_text(
            json.dumps(result["host_memory_telemetry"], indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        (run_dir / "pilot_result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
        )
        summary = {k: result.get(k) for k in ("status", "exception_type", "exception_message",
                                              "checkpoint_load", "encoder_parameters_changed",
                                              "all_supervised_heads_changed", "scheduler", "learning_rate_trace",
                                              "memory", "first_half_failure_report", "resume_failure_report")}
        (run_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n",
                                                    encoding="utf-8")
        print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2, default=str))
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
