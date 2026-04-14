"""Mixed-precision training utilities and gradient scaling helpers."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


def get_grad_scaler(enabled: bool = True) -> GradScaler:
    """
    Create a GradScaler for AMP training.

    Args:
        enabled: Whether to actually scale gradients. Should be False for CPU.

    Returns:
        Configured GradScaler.
    """
    return GradScaler(enabled=enabled)


def detect_precision(fp16: bool, bf16: bool) -> str:
    """
    Determine the effective training precision.

    Args:
        fp16: Whether FP16 is requested.
        bf16: Whether BF16 is requested.

    Returns:
        One of 'fp16', 'bf16', or 'fp32'.
    """
    if not torch.cuda.is_available():
        logger.info("No CUDA device; using fp32")
        return "fp32"
    if fp16 and bf16:
        logger.warning("Both fp16 and bf16 requested; defaulting to fp16")
        return "fp16"
    if fp16:
        return "fp16"
    if bf16:
        # BF16 requires Ampere (A100) or newer
        if torch.cuda.is_bf16_supported():
            return "bf16"
        else:
            logger.warning("BF16 requested but not supported on this GPU; falling back to fp16")
            return "fp16"
    return "fp32"


def clip_grad_norm(
    parameters: Any,
    max_norm: float,
    scaler: GradScaler | None = None,
    optimizer: Optimizer | None = None,
) -> float:
    """
    Clip gradient norm, unscaling first if a GradScaler is used.

    Args:
        parameters: Model parameters.
        max_norm: Maximum gradient norm.
        scaler: Optional GradScaler (used with AMP).
        optimizer: Required if scaler is not None.

    Returns:
        Total gradient norm before clipping.
    """
    if scaler is not None and optimizer is not None:
        scaler.unscale_(optimizer)
    total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    return float(total_norm)


class PrecisionContext:
    """
    Context manager for mixed-precision forward passes.

    Wraps `torch.autocast` for a cleaner training loop interface.
    """

    def __init__(self, precision: str = "fp32") -> None:
        """
        Initialize precision context.

        Args:
            precision: One of 'fp16', 'bf16', 'fp32'.
        """
        self.precision = precision
        self._ctx = None

    def __enter__(self) -> "PrecisionContext":
        if self.precision in {"fp16", "bf16"} and torch.cuda.is_available():
            dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
            self._ctx = torch.autocast(device_type="cuda", dtype=dtype)
            self._ctx.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(*args)
            self._ctx = None
# scaler.unscale_(optimizer) called before clip_grad_norm_ for correct scaling
# bf16 requires Ampere GPU (A100+); graceful fallback to fp16 on older cards
