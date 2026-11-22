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
    assert config.micro_batch_size == 1
    assert config.mode == "full_finetune"


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
