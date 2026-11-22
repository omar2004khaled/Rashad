"""Explicit, serializable initial settings for controlled Quran model training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    mode: str = "full_finetune"
    optimizer: str = "adamw"
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.01
    scheduler: str = "linear"
    warmup_ratio: float = 0.05
    gradient_accumulation_steps: int = 16
    max_grad_norm: float = 1.0
    mixed_precision: str = "fp16"
    gradient_checkpointing: bool = True
    micro_batch_size: int = 1
    epochs: int = 5
    evaluation_steps: int = 500
    checkpoint_steps: int = 500
    seed: int = 42
    num_workers: int = 0

    def __post_init__(self):
        if self.mode not in {"full_finetune", "frozen_encoder"}:
            raise ValueError("mode must be full_finetune or frozen_encoder")
        if self.optimizer != "adamw":
            raise ValueError("Only AdamW is currently implemented")
        if self.scheduler not in {"linear", "cosine", "constant"}:
            raise ValueError("scheduler must be linear, cosine, or constant")
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be no, fp16, or bf16")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if min(self.gradient_accumulation_steps, self.micro_batch_size, self.epochs,
               self.evaluation_steps, self.checkpoint_steps) < 1:
            raise ValueError("batch sizes, epochs, and step frequencies must be positive")

    def to_file(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_file(cls, path: str | Path) -> "TrainingConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
