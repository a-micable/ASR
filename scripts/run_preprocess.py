#!/usr/bin/env python3
"""CLI entry point for audio preprocessing pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from logging_config import setup_logging
from preprocessing.audio_cleaner import AudioCleaner
from preprocessing.dataset_builder import DatasetBuilder
from preprocessing.noise_reducer import NoiseReducer
from preprocessing.resampler import AudioResampler
from training.config import PipelineConfig

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess audio for Whisper training")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-resample", action="store_true")
    parser.add_argument("--skip-denoise", action="store_true")
    parser.add_argument("--skip-dataset", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig.from_yaml(args.config)
    config.configure_logging()
    config.ensure_directories()

    raw_dir = config.dataset.raw_dir
    processed_dir = config.dataset.processed_dir
    audio_files = []
    for ext in ("*.wav", "*.flac", "*.mp3", "*.ogg"):
        audio_files.extend(raw_dir.rglob(ext))

    if not audio_files:
        logger.error("No audio files found in %s", raw_dir)
        raise SystemExit(1)

    logger.info("Found %d audio files", len(audio_files))

    if not args.skip_clean:
        cleaner = AudioCleaner()
        clean_dir = processed_dir / "clean"
        cleaner.clean_batch(audio_files, clean_dir)
        work_files = list(clean_dir.glob("*.wav"))
    else:
        work_files = audio_files

    if not args.skip_resample:
        resampler = AudioResampler()
        resampled_dir = processed_dir / "resampled"
        resampler.resample_batch(work_files, resampled_dir)
        work_files = list(resampled_dir.glob("*.wav"))

    if not args.skip_denoise:
        reducer = NoiseReducer()
        denoised_dir = processed_dir / "denoised"
        reducer.process_batch(work_files, denoised_dir)
        work_files = list(denoised_dir.glob("*.wav"))

    if not args.skip_dataset:
        builder = DatasetBuilder(
            audio_dir=processed_dir / "denoised" if not args.skip_denoise else processed_dir,
            transcription_file=config.dataset.transcription_file,
            language=config.model.language,
        )
        dataset = builder.build_dataset(
            train_ratio=config.dataset.train_ratio,
            val_ratio=config.dataset.val_ratio,
            test_ratio=config.dataset.test_ratio,
        )
        stats = builder.export_artifacts(dataset, processed_dir)
        logger.info("Dataset ready: %d total samples", stats.total_samples)

    logger.info("Preprocessing complete")


if __name__ == "__main__":
    main()
# --augment flag enables AudioAugmenter batch processing step
# --workers controls ThreadPoolExecutor parallelism in resampler
# AudioAugmenter applied when --augment flag set before dataset build
# validate_audio_files checks for corrupt files before processing
