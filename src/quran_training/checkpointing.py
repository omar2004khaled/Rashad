"""Atomic save/restore of complete training state, including random generators."""

from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_training_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    scaler=None,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a restartable checkpoint atomically on the destination volume."""
    if optimizer is None:
        raise ValueError("optimizer is required for a resumable training checkpoint")
    if scheduler is None:
        raise ValueError("scheduler is required for a resumable training checkpoint")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "rng": _rng_state(),
        "config": config or {},
        "extra": extra or {},
    }
    try:
        torch.save(state, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_training_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer,
    scheduler,
    scaler=None,
    map_location="cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore weights and optimizer/scheduler state; missing state is an error."""
    if optimizer is None or scheduler is None:
        raise ValueError("optimizer and scheduler are required to resume training")
    # mmap prevents eagerly materializing every model/optimizer storage in the
    # host working set while the destination model and optimizer already exist.
    checkpoint = torch.load(path, map_location=map_location, weights_only=False, mmap=True)
    for required in ("model", "optimizer", "scheduler", "epoch", "global_step"):
        if required not in checkpoint:
            raise ValueError(f"Training checkpoint is missing required state: {required}")
    if checkpoint.get("scaler") is not None and scaler is None:
        raise ValueError("Checkpoint contains mixed-precision scaler state; provide a scaler to resume")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None:
        if checkpoint.get("scaler") is None:
            raise ValueError("A scaler was supplied but checkpoint has no scaler state")
        scaler.load_state_dict(checkpoint["scaler"])
    if restore_rng:
        if "rng" not in checkpoint:
            raise ValueError("Training checkpoint is missing RNG states")
        _restore_rng_state(checkpoint["rng"])
    restored = {
        "epoch": int(checkpoint["epoch"]),
        "global_step": int(checkpoint["global_step"]),
        "config": checkpoint.get("config", {}),
        "extra": checkpoint.get("extra", {}),
    }
    # Ensure mapped checkpoint storages are not held through the next forward.
    del checkpoint
    return restored
