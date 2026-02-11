#!/usr/bin/env python3
"""CLI entry point for Whisper fine-tuning."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from logging_config import setup_logging
from training.config import PipelineConfig
from training.trainer import WhisperTrainer

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper for ASR")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig.from_yaml(args.config)
    setup_logging(
        level=config.logging.level,
        log_file=config.logging.log_dir / "training.log",
        structured=config.logging.structured,
    )

    trainer = WhisperTrainer(config)
    metrics = trainer.train(resume_from_checkpoint=args.resume)
    logger.info(
        "Training finished | WER=%.4f | CER=%.4f | epochs=%.1f",
        metrics.eval_wer,
        metrics.eval_cer,
        metrics.epochs_completed,
    )


if __name__ == "__main__":
    main()
# --peft flag selects PEFTWhisperTrainer with configurable LoRA rank
# --language flag overrides config.model.language at runtime
