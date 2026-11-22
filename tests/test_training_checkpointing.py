import random

import numpy as np
import pytest
import torch

from quran_model import QuranMultiLevelCTC
from quran_model import QuranMultiLevelCTCConfig
from quran_training import TrainingConfig, load_training_checkpoint, save_training_checkpoint
from transformers import Wav2Vec2BertConfig


def tiny_model():
    encoder = Wav2Vec2BertConfig(
        hidden_size=16, num_hidden_layers=1, num_attention_heads=4,
        intermediate_size=32, feature_projection_input_dim=160,
        conv_depthwise_kernel_size=3, conformer_conv_dropout=0.0,
        apply_spec_augment=False, layerdrop=0.0,
    )
    return QuranMultiLevelCTC(QuranMultiLevelCTCConfig(encoder_config=encoder, final_dropout=0.0))


def test_training_config_file_roundtrip(tmp_path):
    path = tmp_path / "training.json"
    config = TrainingConfig()
    config.to_file(path)
    assert TrainingConfig.from_file(path) == config
    assert config.optimizer == "adafactor"
    assert config.micro_batch_size == 1
    assert config.mode == "full_finetune"
    assert config.gradient_accumulation_device == "cpu"


def test_memory_efficient_optimizer_config_choices():
    assert TrainingConfig(optimizer="adafactor").optimizer == "adafactor"
    assert TrainingConfig(optimizer="adamw8bit").optimizer == "adamw8bit"
    assert TrainingConfig(optimizer="adamw", optimizer_foreach=False).build_optimizer(
        [torch.nn.Parameter(torch.ones(1))]
    ).defaults["foreach"] is False
    with pytest.raises(ValueError, match="optimizer must be"):
        TrainingConfig(optimizer="unknown")
    with pytest.raises(ValueError, match="gradient_accumulation_device"):
        TrainingConfig(gradient_accumulation_device="cuda")


def test_adafactor_uses_configured_fixed_learning_rate_and_weight_decay():
    parameter = torch.nn.Parameter(torch.ones(2))
    config = TrainingConfig(optimizer="adafactor", learning_rate=1e-5, weight_decay=0.01)
    optimizer = config.build_optimizer([parameter])
    assert optimizer.defaults["lr"] == 1e-5
    assert optimizer.defaults["weight_decay"] == 0.01
    assert "relative_step" not in optimizer.defaults
    assert "scale_parameter" not in optimizer.defaults
    assert "warmup_init" not in optimizer.defaults


def test_8bit_optimizer_state_can_be_checkpointed_and_restored(tmp_path):
    bnb = pytest.importorskip("bitsandbytes")
    parameter = torch.nn.Parameter(torch.randn(5000))
    model = torch.nn.Linear(1, 1)  # its state dict is intentionally separate from the tested tensor
    optimizer = bnb.optim.AdamW8bit([parameter], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    scheduler.step()
    path = tmp_path / "8bit.pt"
    save_training_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler,
        epoch=1, global_step=1,
    )
    expected = optimizer.state_dict()
    with torch.no_grad():
        parameter.add_(3)
    info = load_training_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, restore_rng=False,
    )
    actual = optimizer.state_dict()
    assert info["global_step"] == 1
    for p_id, state in expected["state"].items():
        for key, value in state.items():
            if torch.is_tensor(value):
                assert torch.equal(value, actual["state"][p_id][key])


def test_adafactor_optimizer_state_can_be_checkpointed_and_restored(tmp_path):
    model = tiny_model()
    optimizer = torch.optim.Adafactor(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    loss = model(
        torch.randn(1, 8, 160), labels={"phonemes": torch.tensor([[1]])}
    ).loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    path = tmp_path / "adafactor.pt"
    save_training_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler,
        epoch=1, global_step=1,
    )
    expected_state = optimizer.state_dict()
    expected_weights = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    info = load_training_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, restore_rng=False,
    )
    assert info["epoch"] == 1 and info["global_step"] == 1
    assert all(torch.equal(expected_weights[k], model.state_dict()[k]) for k in expected_weights)
    restored_state = optimizer.state_dict()
    assert restored_state["param_groups"] == expected_state["param_groups"]
    assert restored_state["state"].keys() == expected_state["state"].keys()
    for parameter_id, expected in expected_state["state"].items():
        actual = restored_state["state"][parameter_id]
        for key, value in expected.items():
            if torch.is_tensor(value):
                assert torch.equal(value, actual[key])
            else:
                assert value == actual[key]
    before_next_update = {name: value.detach().clone() for name, value in model.named_parameters()}
    next_loss = model(
        torch.randn(1, 8, 160), labels={"phonemes": torch.tensor([[1]])}
    ).loss
    assert torch.isfinite(next_loss)
    next_loss.backward()
    optimizer.step()
    scheduler.step()
    assert any(not torch.equal(before_next_update[name], value)
               for name, value in model.named_parameters())


def test_checkpoint_resume_restores_model_optimizer_scheduler_and_rng(tmp_path):
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    (model(torch.randn(1, 8, 160), labels={"phonemes": torch.tensor([[1]])}).loss).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    path = tmp_path / "resume.pt"
    expected_model = {name: param.detach().clone() for name, param in model.state_dict().items()}
    expected_optimizer = optimizer.state_dict()
    expected_scheduler = scheduler.state_dict()
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    save_training_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler,
        epoch=2, global_step=19, config={"mode": "full_finetune"},
    )
    expected_rng = (random.random(), float(np.random.random()), torch.rand(1))

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    optimizer.param_groups[0]["lr"] = 0.2
    scheduler.step()
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    info = load_training_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, restore_rng=True,
    )
    assert info["epoch"] == 2 and info["global_step"] == 19
    assert info["config"] == {"mode": "full_finetune"}
    assert optimizer.state_dict()["param_groups"][0]["lr"] == expected_optimizer["param_groups"][0]["lr"]
    assert scheduler.state_dict() == expected_scheduler
    assert all(torch.equal(expected_model[k], model.state_dict()[k]) for k in expected_model)
    assert random.random() == expected_rng[0]
    assert np.random.random() == expected_rng[1]
    assert torch.equal(torch.rand(1), expected_rng[2])


def test_resume_requires_optimizer_and_scheduler(tmp_path):
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="optimizer and scheduler are required"):
        load_training_checkpoint(tmp_path / "missing.pt", model=model, optimizer=optimizer, scheduler=None)
