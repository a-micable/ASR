"""Whisper fine-tuning trainer using Hugging Face Trainer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

from evaluation.cer import CharacterErrorRate
from evaluation.wer import WordErrorRate
from logging_config import setup_logging
from training.callbacks import build_callbacks
from training.config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for final training metrics."""

    train_loss: float
    eval_wer: float
    eval_cer: float
    global_step: int
    epochs_completed: float


class WhisperDataCollator:
    """Collate audio features and tokenized labels for Whisper training."""

    def __init__(self, processor: WhisperProcessor) -> None:
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


class WhisperTrainer:
    """
    End-to-end Whisper fine-tuning orchestrator.

    Handles model loading, dataset preparation, training, validation,
    checkpointing, mixed precision, and multi-GPU via HF Trainer.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """
        Initialize trainer with pipeline configuration.

        Args:
            config: Full pipeline configuration.
        """
        self.config = config
        setup_logging(
            level=config.logging.level,
            log_file=config.logging.log_dir / "training.log",
            structured=config.logging.structured,
        )
        config.ensure_directories()

        self.processor: WhisperProcessor | None = None
        self.model: WhisperForConditionalGeneration | None = None
        self.trainer: Seq2SeqTrainer | None = None
        self.wer_evaluator = WordErrorRate()
        self.cer_evaluator = CharacterErrorRate(language=config.model.language)

    def load_model_and_processor(self) -> None:
        """Load Whisper model, tokenizer, and feature extractor."""
        model_cfg = self.config.model
        logger.info("Loading model: %s", model_cfg.name)

        self.processor = WhisperProcessor.from_pretrained(
            model_cfg.name,
            language=model_cfg.language,
            task=model_cfg.task,
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(model_cfg.name)

        if model_cfg.freeze_encoder:
            for param in self.model.model.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder frozen")

        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []

        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        logger.info(
            "Model loaded | params: %.1fM | device: %s",
            sum(p.numel() for p in self.model.parameters()) / 1e6,
            "cuda" if torch.cuda.is_available() else "cpu",
        )

    def load_dataset(self) -> DatasetDict:
        """
        Load HuggingFace dataset from disk.

        Returns:
            DatasetDict with train/validation/test splits.
        """
        path = self.config.dataset.hf_dataset_path
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {path}. Run dataset_builder first."
            )
        dataset = load_from_disk(str(path))
        if not isinstance(dataset, DatasetDict):
            dataset = DatasetDict({"train": dataset})
        logger.info("Loaded dataset from %s", path)
        return dataset

    def _prepare_dataset(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Map function: extract features and tokenize labels."""
        assert self.processor is not None
        audio = batch["audio"]

        if isinstance(audio, str):
            array, sampling_rate = librosa.load(
                audio,
                sr=self.config.dataset.sample_rate,
                mono=True,
            )
        elif isinstance(audio, dict):
            array = audio["array"]
            sampling_rate = audio["sampling_rate"]
        else:
            array = np.asarray(audio, dtype=np.float32)
            sampling_rate = self.config.dataset.sample_rate

        inputs = self.processor.feature_extractor(
            array,
            sampling_rate=sampling_rate,
            return_tensors="np",
        )
        batch["input_features"] = inputs.input_features[0]

        batch["labels"] = self.processor.tokenizer(batch["text"]).input_ids
        return batch

    def prepare_datasets(self, dataset_dict: DatasetDict) -> DatasetDict:
        """
        Preprocess datasets for training.

        Args:
            dataset_dict: Raw HF dataset.

        Returns:
            Processed DatasetDict.
        """
        assert self.processor is not None

        processed = dataset_dict.map(
            self._prepare_dataset,
            remove_columns=dataset_dict["train"].column_names,
            num_proc=min(4, self.config.training.dataloader_num_workers),
        )
        return processed

    def _compute_metrics(self, pred: Any) -> dict[str, float]:
        """Compute WER during evaluation."""
        assert self.processor is not None
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        label_ids = np.where(label_ids != -100, label_ids, self.processor.tokenizer.pad_token_id)

        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = self.wer_evaluator.compute_batch(pred_str, label_str)
        cer = self.cer_evaluator.compute_batch(pred_str, label_str)
        return {"wer": wer["wer"], "cer": cer["cer"]}

    def build_training_arguments(self) -> Seq2SeqTrainingArguments:
        """Create HuggingFace Seq2SeqTrainingArguments from config."""
        tc = self.config.training
        sc = self.config.scheduler

        use_fp16 = tc.fp16 and torch.cuda.is_available()
        use_bf16 = tc.bf16 and torch.cuda.is_available()

        return Seq2SeqTrainingArguments(
            output_dir=str(tc.output_dir),
            num_train_epochs=tc.num_train_epochs,
            per_device_train_batch_size=tc.per_device_train_batch_size,
            per_device_eval_batch_size=tc.per_device_eval_batch_size,
            gradient_accumulation_steps=tc.gradient_accumulation_steps,
            learning_rate=tc.learning_rate,
            weight_decay=tc.weight_decay,
            warmup_steps=sc.num_warmup_steps or tc.warmup_steps,
            max_grad_norm=tc.max_grad_norm,
            fp16=use_fp16,
            bf16=use_bf16 and not use_fp16,
            logging_steps=tc.logging_steps,
            eval_steps=tc.eval_steps,
            save_steps=tc.save_steps,
            save_total_limit=tc.save_total_limit,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=tc.load_best_model_at_end,
            metric_for_best_model=tc.metric_for_best_model,
            greater_is_better=tc.greater_is_better,
            predict_with_generate=True,
            generation_max_length=225,
            dataloader_num_workers=tc.dataloader_num_workers,
            seed=tc.seed,
            report_to=tc.report_to,
            logging_dir=str(self.config.logging.tensorboard_dir),
            remove_unused_columns=False,
            label_names=["labels"],
            ddp_find_unused_parameters=False,
        )

    def initialize_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
    ) -> Seq2SeqTrainer:
        """
        Build Seq2SeqTrainer instance.

        Args:
            train_dataset: Training split.
            eval_dataset: Validation split.

        Returns:
            Configured Seq2SeqTrainer.
        """
        if self.model is None or self.processor is None:
            self.load_model_and_processor()

        assert self.model is not None and self.processor is not None

        training_args = self.build_training_arguments()
        data_collator = WhisperDataCollator(self.processor)
        callbacks = build_callbacks(self.config, self.config.logging.log_dir)

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            processing_class=self.processor,
            callbacks=callbacks,
        )
        return self.trainer

    def train(self, resume_from_checkpoint: str | Path | None = None) -> TrainingMetrics:
        """
        Run full training pipeline.

        Args:
            resume_from_checkpoint: Optional checkpoint path to resume from.

        Returns:
            Final training metrics.
        """
        dataset_dict = self.load_dataset()
        if self.model is None:
            self.load_model_and_processor()

        processed = self.prepare_datasets(dataset_dict)
        train_ds = processed["train"]
        eval_ds = processed.get("validation", processed.get("val", train_ds))

        trainer = self.initialize_trainer(train_ds, eval_ds)

        checkpoint = resume_from_checkpoint or self.config.training.resume_from_checkpoint
        logger.info("Starting training%s", f" from {checkpoint}" if checkpoint else "")

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        eval_metrics = trainer.evaluate()
        best_dir = Path(self.config.training.output_dir) / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(best_dir))
        self.processor.save_pretrained(str(best_dir))

        metrics = TrainingMetrics(
            train_loss=train_result.training_loss,
            eval_wer=eval_metrics.get("eval_wer", eval_metrics.get("wer", 0.0)),
            eval_cer=eval_metrics.get("eval_cer", eval_metrics.get("cer", 0.0)),
            global_step=train_result.global_step,
            epochs_completed=train_result.metrics.get("epoch", 0.0),
        )
        logger.info(
            "Training complete | WER=%.4f | loss=%.4f",
            metrics.eval_wer,
            metrics.train_loss,
        )
        return metrics

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model checkpoint to path."""
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call train() first.")
        self.trainer.save_model(str(path))
        if self.processor:
            self.processor.save_pretrained(str(path))
        logger.info("Checkpoint saved to %s", path)
# fallback: eval_ds = processed.get('validation', processed.get('val', train_ds))
# epoch extracted via train_result.metrics.get('epoch', 0.0)
# WhisperDataCollator handles padding for both input_features and labels
# train_result.metrics.get('epoch', 0.0) provides safe fallback for epoch count
# str(checkpoint) ensures Path objects accepted by HF Trainer.train()
# TrainingMetrics(train_loss, eval_wer, eval_cer, global_step, epochs_completed)
