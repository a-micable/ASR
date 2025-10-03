"""Learning rate schedulers with warmup support."""

from __future__ import annotations

import math
from typing import Literal

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

SchedulerType = Literal["cosine", "linear", "cosine_with_warmup", "linear_with_warmup"]


def _warmup_factor(current_step: int, warmup_steps: int) -> float:
    """Linear warmup multiplier."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, float(current_step) / float(max(1, warmup_steps)))


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Cosine decay schedule with linear warmup.

    Args:
        optimizer: PyTorch optimizer.
        num_warmup_steps: Warmup step count.
        num_training_steps: Total training steps.
        num_cycles: Cosine cycle fraction.
        last_epoch: Last epoch index for resume.

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return _warmup_factor(current_step, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Linear decay schedule with linear warmup.

    Args:
        optimizer: PyTorch optimizer.
        num_warmup_steps: Warmup step count.
        num_training_steps: Total training steps.
        last_epoch: Last epoch index for resume.

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return _warmup_factor(current_step, num_warmup_steps)
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_scheduler(
    optimizer: Optimizer,
    scheduler_type: SchedulerType,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """
    Factory for LR schedulers.

    Args:
        optimizer: PyTorch optimizer.
        scheduler_type: One of cosine, linear, cosine_with_warmup, linear_with_warmup.
        num_warmup_steps: Warmup steps.
        num_training_steps: Total training steps.

    Returns:
        Configured LambdaLR scheduler.
    """
    if scheduler_type in {"cosine", "cosine_with_warmup"}:
        warmup = num_warmup_steps if "warmup" in scheduler_type else 0
        return get_cosine_schedule_with_warmup(
            optimizer, warmup, num_training_steps
        )
    if scheduler_type in {"linear", "linear_with_warmup"}:
        warmup = num_warmup_steps if "warmup" in scheduler_type else 0
        return get_linear_schedule_with_warmup(
            optimizer, warmup, num_training_steps
        )
    raise ValueError(f"Unknown scheduler type: {scheduler_type}")
# num_cycles parameter controls cosine period; default 0.5 for half cycle
