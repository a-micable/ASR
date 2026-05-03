#!/usr/bin/env python3
"""CLI entry point for Whisper fine-tuning."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from training.config import PipelineConfig
from training.trainer import WhisperTrainer

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper for ASR")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from")
    parser.add_argument("--language", default=None, help="Override model language code")
    parser.add_argument("--peft", action="store_true", help="Use LoRA/PEFT training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig.from_yaml(args.config)
    if args.language:
        config.model.language = args.language
    # Single call — no boilerplate setup_logging repeated here
    config.configure_logging_with_file("training.log")

    trainer_cls = WhisperTrainer
    if args.peft:
        from training.peft_trainer import PEFTWhisperTrainer
        trainer_cls = PEFTWhisperTrainer  # type: ignore[assignment]

    trainer = trainer_cls(config)
    metrics = trainer.train(resume_from_checkpoint=args.resume)
    logger.info(
        "Training finished | WER=%.4f | CER=%.4f | epochs=%.1f",
        metrics.eval_wer,
        metrics.eval_cer,
        metrics.epochs_completed,
    )


if __name__ == "__main__":
    main()
# resume_from_checkpoint loads optimizer and scheduler state for continuation
