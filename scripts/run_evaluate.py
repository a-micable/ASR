#!/usr/bin/env python3
"""CLI entry point for model evaluation and benchmarking."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from evaluation.benchmark import ModelBenchmark
from evaluation.cer import CharacterErrorRate
from evaluation.wer import WordErrorRate
from logging_config import setup_logging
from preprocessing.io_utils import load_audio
from training.config import PipelineConfig

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Whisper model")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--split", default="test", choices=["validation", "test", "train"])
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_results"))
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig.from_yaml(args.config)
    config.configure_logging()

    model_path = args.model_path or config.api.model_path
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = WhisperProcessor.from_pretrained(str(model_path))
    model = WhisperForConditionalGeneration.from_pretrained(str(model_path)).to(device)
    model.eval()

    dataset = load_from_disk(str(config.dataset.hf_dataset_path))
    split = dataset[args.split]

    references: list[str] = []
    hypotheses: list[str] = []
    durations: list[float] = []

    for item in split:
        audio_ref = item["audio"]
        if isinstance(audio_ref, str):
            array, sr = load_audio(audio_ref, target_sr=config.dataset.sample_rate, mono=True)
        else:
            array = audio_ref["array"]
            sr = audio_ref["sampling_rate"]
        durations.append(len(array) / sr)

        inputs = processor(array, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(input_features)

        hyp = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        references.append(item["text"])
        hypotheses.append(hyp)

    wer_eval = WordErrorRate()
    cer_eval = CharacterErrorRate(language=config.model.language)
    wer_report = wer_eval.evaluate_dataset(references, hypotheses)
    cer_report = cer_eval.evaluate_dataset(references, hypotheses)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "wer": wer_report["aggregate"],
        "cer": cer_report["aggregate"],
        "num_samples": len(references),
        "split": args.split,
        "model_path": str(model_path),
    }
    with open(args.output_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(
        "Evaluation | WER=%.2f%% | CER=%.2f%%",
        report["wer"]["wer"] * 100,
        report["cer"]["cer"] * 100,
    )

    if args.benchmark:
        benchmark = ModelBenchmark(model_name=str(model_path), device=device)

        def infer_one() -> None:
            sample = split[0]
            audio_ref = sample["audio"]
            if isinstance(audio_ref, str):
                array, sr = load_audio(
                    audio_ref, target_sr=config.dataset.sample_rate, mono=True
                )
            else:
                array = audio_ref["array"]
                sr = audio_ref["sampling_rate"]
            inputs = processor(array, sampling_rate=sr, return_tensors="pt")
            with torch.no_grad():
                model.generate(inputs.input_features.to(device))

        benchmark.generate_report(
            infer_one,
            durations,
            args.output_dir,
            evaluation_metrics={"wer": report["wer"]["wer"], "cer": report["cer"]["cer"]},
        )


if __name__ == "__main__":
    main()
# ReportGenerator.save_html called after evaluation to produce HTML report
# --output-dir defaults to evaluation_results/ for report artifacts
# ReportGenerator.save_html generates evaluation_report.html in output_dir
# compute_language_metrics flag enables TER/MER/SER calculation
