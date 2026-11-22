import pytest
import torch
from transformers import Wav2Vec2BertConfig

from quran_model import QuranMultiLevelCTC, QuranMultiLevelCTCConfig
from quran_training import (
    QuranTrainer,
    TrainingConfig,
    TrainingRunConfig,
    build_linear_warmup_scheduler,
    optimizer_update_succeeded,
    total_optimizer_steps,
)
from quran_training.engine import _accumulate_gradients_on_cpu, _restore_gradients_to_parameters
from quran_training_data import QuranTrainingCollator


def tiny_model():
    encoder = Wav2Vec2BertConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        feature_projection_input_dim=160,
        conv_depthwise_kernel_size=3,
        conformer_conv_dropout=0.0,
        apply_spec_augment=False,
        layerdrop=0.0,
    )
    return QuranMultiLevelCTC(
        QuranMultiLevelCTCConfig(encoder_config=encoder, final_dropout=0.0)
    )


def examples(count, split):
    return [
        {
            "source_key": f"{split}:{index}",
            "split": split,
            "input_features": torch.randn(8 + index % 2, 160),
            "labels": {"phonemes": (1, 2, 3)},
        }
        for index in range(count)
    ]


def training_config():
    return TrainingConfig(
        optimizer="adafactor",
        learning_rate=1e-3,
        weight_decay=0.0,
        scheduler="linear",
        warmup_ratio=0.0,
        gradient_accumulation_steps=2,
        gradient_accumulation_device="cpu",
        mixed_precision="no",
        gradient_checkpointing=False,
        max_grad_norm=1.0,
    )


def run_config(*, max_steps, max_examples):
    return TrainingRunConfig(
        epochs=2,
        max_optimizer_steps=max_steps,
        max_train_examples=max_examples,
        checkpoint_every_steps=1,
        require_all_parameter_gradients=False,
        min_system_available_gib=0.0,
    )


def test_linear_scheduler_counts_steps_and_warmup():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = build_linear_warmup_scheduler(optimizer, total_steps=10, warmup_ratio=0.2)
    assert scheduler.total_steps == 10
    assert scheduler.warmup_steps == 2
    assert optimizer.param_groups[0]["lr"] == 0.0
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.5
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 1.0
    for _ in range(8):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.0


def test_two_update_pilot_scheduler_learning_rate_sequence():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=1e-5)
    assert total_optimizer_steps(32, 16, 1) == 2
    scheduler = build_linear_warmup_scheduler(optimizer, total_steps=2, warmup_ratio=0.05)
    assert scheduler.warmup_steps == 1
    used_lrs = []
    for _ in range(2):
        used_lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert used_lrs == [0.0, 1e-5]
    assert optimizer.param_groups[0]["lr"] == 0.0


def test_pilot_warmup_override_keeps_both_bounded_update_lrs_nonzero(tmp_path):
    torch.manual_seed(31)
    config = training_config()
    config = TrainingConfig(**{**config.__dict__, "warmup_ratio": 0.05})
    run = run_config(max_steps=2, max_examples=4)
    run = TrainingRunConfig(**{**run.__dict__, "epochs": 1, "scheduler_warmup_ratio": 0.0})
    model = tiny_model()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    trainer = QuranTrainer(
        model=model, collator=QuranTrainingCollator(feature_dim=160),
        training_config=config, run_config=run, output_dir=tmp_path, device="cpu",
    )
    result = trainer.fit(examples(4, "train"), examples(1, "validation"), max_validation_examples=1)
    assert config.warmup_ratio == 0.05
    assert result["effective_warmup_ratio"] == 0.0
    assert result["warmup_steps"] == 0
    assert [row["learning_rate_used"][0] for row in result["step_metrics"]] == [1e-3, 5e-4]
    assert all(row["optimizer_update_succeeded"] for row in result["step_metrics"])
    assert any(not torch.equal(before[name], p) for name, p in model.named_parameters())


def test_total_optimizer_steps_rounds_partial_accumulation_up():
    assert total_optimizer_steps(17, 16, 3) == 6


def test_run_config_scheduler_override_validation():
    assert run_config(max_steps=1, max_examples=2).scheduler_warmup_ratio is None
    with pytest.raises(ValueError, match="scheduler_warmup_ratio"):
        TrainingRunConfig(scheduler_warmup_ratio=1.0)


def test_cpu_gradient_accumulation_matches_device_resident_sum():
    torch.manual_seed(33)
    expected = torch.nn.Linear(4, 3)
    offloaded = torch.nn.Linear(4, 3)
    offloaded.load_state_dict(expected.state_dict())
    batches = [(torch.randn(2, 4), torch.randn(2, 3)) for _ in range(2)]

    for x, target in batches:
        (torch.nn.functional.mse_loss(expected(x), target) / 2).backward()

    accumulated = {}
    named_parameters = list(offloaded.named_parameters())
    for x, target in batches:
        (torch.nn.functional.mse_loss(offloaded(x), target) / 2).backward()
        assert all(parameter.grad is not None for _, parameter in named_parameters)
        _accumulate_gradients_on_cpu(named_parameters, accumulated)
        assert all(parameter.grad is None for _, parameter in named_parameters)
        assert all(value.device.type == "cpu" for value in accumulated.values())
    _restore_gradients_to_parameters(named_parameters, accumulated, torch.device("cpu"))
    assert accumulated == {}
    for (_, expected_parameter), (_, offloaded_parameter) in zip(
        expected.named_parameters(), offloaded.named_parameters()
    ):
        torch.testing.assert_close(offloaded_parameter.grad, expected_parameter.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA transfer check requires a GPU")
def test_cpu_gradient_accumulation_cuda_transfer_and_reset():
    torch.manual_seed(34)
    expected = torch.nn.Linear(4, 3).cuda()
    offloaded = torch.nn.Linear(4, 3).cuda()
    offloaded.load_state_dict(expected.state_dict())
    batches = [(torch.randn(2, 4, device="cuda"), torch.randn(2, 3, device="cuda"))
               for _ in range(2)]
    accumulated = {}
    named_parameters = list(offloaded.named_parameters())
    for x, target in batches:
        (torch.nn.functional.mse_loss(expected(x), target) / 2).backward()
        (torch.nn.functional.mse_loss(offloaded(x), target) / 2).backward()
        _accumulate_gradients_on_cpu(named_parameters, accumulated)
        assert all(parameter.grad is None for _, parameter in named_parameters)
        assert all(value.device.type == "cpu" for value in accumulated.values())
    _restore_gradients_to_parameters(named_parameters, accumulated, torch.device("cuda"))
    assert not accumulated
    for (_, expected_parameter), (_, actual_parameter) in zip(
        expected.named_parameters(), offloaded.named_parameters()
    ):
        assert actual_parameter.grad.device.type == "cuda"
        torch.testing.assert_close(actual_parameter.grad.cpu(), expected_parameter.grad.cpu())
    optimizer = torch.optim.SGD(offloaded.parameters(), lr=1e-3)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert all(parameter.grad is None for parameter in offloaded.parameters())


def test_grad_scaler_skip_detection_controls_scheduler_stepping():
    assert optimizer_update_succeeded(scaler_enabled=True, scale_before=65536, scale_after=32768) is False
    assert optimizer_update_succeeded(scaler_enabled=True, scale_before=65536, scale_after=65536) is True
    assert optimizer_update_succeeded(scaler_enabled=True, scale_before=65536, scale_after=131072) is True
    assert optimizer_update_succeeded(scaler_enabled=False, scale_before=1, scale_after=1) is True


def test_training_accumulates_gradients_and_validation_preserves_parameters(tmp_path):
    torch.manual_seed(4)
    model = tiny_model()
    collator = QuranTrainingCollator(feature_dim=160)
    trainer = QuranTrainer(
        model=model,
        collator=collator,
        training_config=training_config(),
        run_config=run_config(max_steps=1, max_examples=2),
        output_dir=tmp_path / "first",
        device="cpu",
    )
    result = trainer.fit(examples(4, "train"), examples(2, "validation"), max_validation_examples=2)
    assert result["optimizer_steps_completed"] == 1
    assert result["examples_seen"] == 2
    assert result["step_metrics"][0]["microbatches_accumulated"] == 2
    assert result["step_metrics"][0]["finite_gradients"] is True
    assert result["initial_validation"]["parameters_unchanged"] is True
    assert result["final_validation"]["parameters_unchanged"] is True
    assert result["initial_validation"]["split"] == "validation"


def test_training_resumes_from_checkpoint_and_continues_cursor(tmp_path):
    train = examples(4, "train")
    validation = examples(1, "validation")
    collator = QuranTrainingCollator(feature_dim=160)
    first_model = tiny_model()
    first_trainer = QuranTrainer(
        model=first_model,
        collator=collator,
        training_config=training_config(),
        run_config=run_config(max_steps=1, max_examples=2),
        output_dir=tmp_path / "first",
        device="cpu",
    )
    first = first_trainer.fit(train, validation, max_validation_examples=1)

    resumed_trainer = QuranTrainer(
        model=tiny_model(),
        collator=collator,
        training_config=training_config(),
        run_config=run_config(max_steps=2, max_examples=4),
        output_dir=tmp_path / "resumed",
        device="cpu",
    )
    resumed = resumed_trainer.fit(
        train,
        validation,
        max_validation_examples=1,
        resume_from=first["last_checkpoint"],
    )
    assert resumed["resume"]["enabled"] is True
    assert resumed["optimizer_steps_completed"] == 2
    assert resumed["examples_seen"] == 4
    assert resumed["step_metrics"][-1]["microbatches_accumulated"] == 2


def test_validation_rejects_training_split_examples(tmp_path):
    trainer = QuranTrainer(
        model=tiny_model(),
        collator=QuranTrainingCollator(feature_dim=160),
        training_config=training_config(),
        run_config=run_config(max_steps=1, max_examples=2),
        output_dir=tmp_path,
        device="cpu",
    )
    try:
        trainer.evaluate(examples(1, "train"), max_examples=1)
    except ValueError as exc:
        assert "Expected validation example" in str(exc)
    else:
        raise AssertionError("Training split was incorrectly accepted for validation")


def test_resource_failure_persists_telemetry_and_original_exception(tmp_path):
    model = tiny_model()
    trainer = QuranTrainer(
        model=model,
        collator=QuranTrainingCollator(feature_dim=160),
        training_config=training_config(),
        run_config=run_config(max_steps=1, max_examples=2),
        output_dir=tmp_path,
        device="cpu",
    )
    def fail_after_backward(monitor, **kwargs):
        raise __import__("quran_training").ResourceLimitReached("synthetic resource stop")
    trainer._check_resources = fail_after_backward
    train = examples(2, "train")
    for row in train:
        row["duration_seconds"] = 4.25
    try:
        trainer.fit(train, examples(1, "validation"), max_validation_examples=1)
    except Exception as exc:
        assert type(exc).__name__ == "ResourceLimitReached"
        assert str(exc) == "synthetic resource stop"
    else:
        raise AssertionError("Resource guard exception was not propagated")
    import json
    report = json.loads((tmp_path / "failure_report.json").read_text(encoding="utf-8"))
    assert report["exception_type"] == "ResourceLimitReached"
    assert report["global_step"] == 0
    assert report["current_example"]["example_id"].startswith("train:")
    assert report["current_example"]["duration_seconds"] == 4.25
    assert "samples" in report["monitor"]


def test_telemetry_write_failure_does_not_hide_training_exception(tmp_path):
    from quran_training import ResourceLimitReached

    trainer = QuranTrainer(
        model=tiny_model(),
        collator=QuranTrainingCollator(feature_dim=160),
        training_config=training_config(),
        run_config=run_config(max_steps=1, max_examples=2),
        output_dir=tmp_path,
        device="cpu",
    )
    def fail_guard(monitor, **kwargs):
        raise ResourceLimitReached("original resource stop")
    def fail_telemetry(**kwargs):
        raise OSError("disk unavailable")
    trainer._check_resources = fail_guard
    trainer._write_failure_report = fail_telemetry
    try:
        trainer.fit(examples(2, "train"), examples(1, "validation"), max_validation_examples=1)
    except ResourceLimitReached as exc:
        assert str(exc) == "original resource stop"
        assert any("disk unavailable" in note for note in getattr(exc, "__notes__", []))
    else:
        raise AssertionError("Original resource exception was hidden")
