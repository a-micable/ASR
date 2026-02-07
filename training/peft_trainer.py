"""LoRA/PEFT fine-tuning for Whisper using parameter-efficient adapters."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from transformers import (
    Seq2SeqTrainer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from training.config import PipelineConfig
from training.trainer import WhisperDataCollator, WhisperTrainer

logger = logging.getLogger(__name__)


def _check_peft_available() -> bool:
    try:
        import peft  # noqa: F401
        return True
    except ImportError:
        return False


class PEFTWhisperTrainer(WhisperTrainer):
    """
    Whisper trainer with LoRA (Low-Rank Adaptation) support via PEFT.

    Enables memory-efficient fine-tuning by training only a small number
    of adapter parameters injected into attention layers.

    Requires: pip install peft
    """

    def __init__(
        self,
        config: PipelineConfig,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        target_modules: list[str] | None = None,
    ) -> None:
        """
        Initialize PEFT trainer.

        Args:
            config: Pipeline configuration.
            lora_r: LoRA rank (dimensionality of adapter matrices).
            lora_alpha: LoRA scaling factor.
            lora_dropout: Dropout probability for adapter layers.
            target_modules: Attention module names to apply LoRA to.
        """
        if not _check_peft_available():
            raise ImportError(
                "PEFT is required for LoRA training. "
                "Install with: pip install peft>=0.7.0"
            )
        super().__init__(config)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or [
            "q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"
        ]

    def load_model_and_processor(self) -> None:
        """Load Whisper model and apply LoRA adapters."""
        from peft import LoraConfig, TaskType, get_peft_model

        super().load_model_and_processor()

        assert self.model is not None
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        trainable, total = self._count_parameters()
        logger.info(
            "LoRA applied | trainable: %d (%.2f%%) | total: %d",
            trainable,
            100 * trainable / total,
            total,
        )

    def _count_parameters(self) -> tuple[int, int]:
        """Return (trainable, total) parameter counts."""
        assert self.model is not None
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total

    def save_checkpoint(self, path: str | Path) -> None:
        """Save merged LoRA adapter checkpoint."""
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call train() first.")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save adapter weights only
        self.trainer.model.save_pretrained(str(path))
        if self.processor:
            self.processor.save_pretrained(str(path))
        logger.info("PEFT checkpoint saved to %s", path)

    def merge_and_save(self, path: str | Path) -> None:
        """
        Merge LoRA weights into base model and save full model.

        This produces a standalone model that doesn't require PEFT at inference.

        Args:
            path: Output directory for merged model.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        try:
            merged = self.model.merge_and_unload()
        except AttributeError:
            logger.warning("merge_and_unload not available; saving adapter-only weights")
            self.model.save_pretrained(str(path))
        else:
            merged.save_pretrained(str(path))
            logger.info("Merged LoRA model saved to %s", path)

        if self.processor:
            self.processor.save_pretrained(str(path))
# Module: LoRA/PEFT fine-tuning; reduces trainable params by ~99%
