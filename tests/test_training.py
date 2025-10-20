"""Tests for training module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.optim import Adam

from training.callbacks import (
    CheckpointCallback,
    MetricLoggingCallback,
    TrainingVisualizationCallback,
    build_callbacks,
)
from training.config import PipelineConfig, TrainingConfig
from training.scheduler import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_scheduler,
)
from training.trainer import WhisperDataCollator, WhisperTrainer


class TestPipelineConfig:
    """Tests for configuration loading."""

    def test_default_config(self) -> None:
        config = PipelineConfig()
        assert config.model.name == "openai/whisper-small"
        assert config.model.language == "am"
        assert config.training.learning_rate == 1e-5

    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(
            "model:\n  name: openai/whisper-base\n  language: om\n"
            "training:\n  num_train_epochs: 5\n  learning_rate: 2e-5\n"
        )
        config = PipelineConfig.from_yaml(yaml_path)
        assert config.model.name == "openai/whisper-base"
        assert config.model.language == "om"
        assert config.training.num_train_epochs == 5

    def test_to_yaml(self, tmp_path: Path) -> None:
        config = PipelineConfig()
        out = tmp_path / "out.yaml"
        config.to_yaml(out)
        assert out.exists()

    def test_ensure_directories(self, tmp_path: Path) -> None:
        config = PipelineConfig(
            dataset={"processed_dir": tmp_path / "processed"},
            training={"output_dir": tmp_path / "checkpoints"},
            logging={"log_dir": tmp_path / "logs"},
        )
        config.ensure_directories()
        assert (tmp_path / "processed").exists()
        assert (tmp_path / "checkpoints").exists()


class TestSchedulers:
    """Tests for LR schedulers."""

    def test_cosine_with_warmup(self) -> None:
        optimizer = Adam([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 10, 100)
        lrs = []
        for _ in range(100):
            optimizer.step()
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])
        assert lrs[0] < lrs[9]
        assert lrs[-1] < lrs[50]
        assert lrs[-1] < lrs[50]

    def test_linear_with_warmup(self) -> None:
        optimizer = Adam([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
        scheduler = get_linear_schedule_with_warmup(optimizer, 10, 100)
        for _ in range(50):
            scheduler.step()
        assert scheduler.get_last_lr()[0] > 0

    def test_get_scheduler_factory(self) -> None:
        optimizer = Adam([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
        for sched_type in ["cosine", "linear", "cosine_with_warmup", "linear_with_warmup"]:
            sched = get_scheduler(optimizer, sched_type, 10, 100)  # type: ignore[arg-type]
            assert sched is not None

    def test_unknown_scheduler_raises(self) -> None:
        optimizer = Adam([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
        with pytest.raises(ValueError):
            get_scheduler(optimizer, "unknown", 10, 100)  # type: ignore[arg-type]


class TestCallbacks:
    """Tests for training callbacks."""

    def test_metric_logging_callback(self, tmp_path: Path) -> None:
        cb = MetricLoggingCallback(tmp_path / "metrics.jsonl")
        from transformers import TrainerControl, TrainerState, TrainingArguments

        args = TrainingArguments(output_dir=str(tmp_path))
        state = TrainerState()
        state.global_step = 10
        state.epoch = 0.5
        control = TrainerControl()
        cb.on_log(args, state, control, logs={"loss": 1.5})
        assert (tmp_path / "metrics.jsonl").exists()

    def test_checkpoint_callback(self, tmp_path: Path) -> None:
        cb = CheckpointCallback(tmp_path, metric_name="eval_wer")
        from transformers import TrainerControl, TrainerState, TrainingArguments

        args = TrainingArguments(output_dir=str(tmp_path))
        state = TrainerState()
        state.global_step = 100
        state.epoch = 1.0
        control = TrainerControl()
        cb.on_evaluate(args, state, control, metrics={"eval_wer": 0.25})
        cb.on_evaluate(args, state, control, metrics={"eval_wer": 0.20})
        info = json.loads((tmp_path / "best_checkpoint.json").read_text())
        assert info["best_metric"] == 0.20

    def test_visualization_callback(self, tmp_path: Path) -> None:
        cb = TrainingVisualizationCallback(tmp_path)
        from transformers import TrainerControl, TrainerState, TrainingArguments

        args = TrainingArguments(output_dir=str(tmp_path))
        state = TrainerState()
        state.global_step = 50
        control = TrainerControl()
        cb.on_log(args, state, control, logs={"loss": 0.8})
        cb.on_evaluate(args, state, control, metrics={"eval_wer": 0.3})
        cb.on_train_end(args, state, control)
        assert (tmp_path / "training_curves.json").exists()

    def test_build_callbacks(self, tmp_path: Path) -> None:
        config = PipelineConfig(training={"output_dir": tmp_path / "ckpt"})
        callbacks = build_callbacks(config, tmp_path / "logs")
        assert len(callbacks) >= 4


class TestWhisperTrainer:
    """Tests for WhisperTrainer (mocked model loading)."""

    def test_trainer_initialization(self, tmp_path: Path) -> None:
        config = PipelineConfig(
            training={"output_dir": tmp_path / "checkpoints"},
            logging={"log_dir": tmp_path / "logs"},
        )
        trainer = WhisperTrainer(config)
        assert trainer.config.model.language == "am"
        assert trainer.processor is None

    def test_load_dataset_not_found(self, tmp_path: Path) -> None:
        config = PipelineConfig(
            dataset={"hf_dataset_path": tmp_path / "missing"},
            training={"output_dir": tmp_path / "checkpoints"},
            logging={"log_dir": tmp_path / "logs"},
        )
        trainer = WhisperTrainer(config)
        with pytest.raises(FileNotFoundError):
            trainer.load_dataset()

    @patch("training.trainer.WhisperProcessor")
    @patch("training.trainer.WhisperForConditionalGeneration")
    def test_load_model_and_processor(
        self,
        mock_model_cls: MagicMock,
        mock_processor_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_processor_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock(
            parameters=lambda: iter([torch.nn.Parameter(torch.zeros(1))])
        )
        config = PipelineConfig(
            training={"output_dir": tmp_path / "checkpoints"},
            logging={"log_dir": tmp_path / "logs"},
        )
        trainer = WhisperTrainer(config)
        trainer.load_model_and_processor()
        assert trainer.processor is not None
        assert trainer.model is not None

    def test_build_training_arguments(self, tmp_path: Path) -> None:
        config = PipelineConfig(
            training={
                "output_dir": tmp_path / "checkpoints",
                "num_train_epochs": 3,
                "fp16": False,
            },
            logging={"log_dir": tmp_path / "logs", "tensorboard_dir": tmp_path / "tb"},
        )
        trainer = WhisperTrainer(config)
        args = trainer.build_training_arguments()
        assert args.num_train_epochs == 3
        assert args.output_dir == str(tmp_path / "checkpoints")

    def test_data_collator(self) -> None:
        processor = MagicMock()
        processor.feature_extractor.pad.return_value = {"input_features": torch.zeros(2, 80, 3000)}
        processor.tokenizer.pad.return_value = MagicMock(
            input_ids=torch.ones(2, 10, dtype=torch.long),
            attention_mask=torch.ones(2, 10, dtype=torch.long),
        )
        collator = WhisperDataCollator(processor)
        features = [
            {"input_features": torch.zeros(80, 100), "labels": [1, 2, 3]},
            {"input_features": torch.zeros(80, 200), "labels": [4, 5]},
        ]
        batch = collator(features)
        assert "input_features" in batch
        assert "labels" in batch

    def test_save_checkpoint_without_trainer(self, tmp_path: Path) -> None:
        config = PipelineConfig(
            training={"output_dir": tmp_path / "checkpoints"},
            logging={"log_dir": tmp_path / "logs"},
        )
        trainer = WhisperTrainer(config)
        with pytest.raises(RuntimeError):
            trainer.save_checkpoint(tmp_path / "ckpt")

    def test_compute_metrics(self, tmp_path: Path) -> None:
        import numpy as np

        config = PipelineConfig(
            training={"output_dir": tmp_path / "checkpoints"},
            logging={"log_dir": tmp_path / "logs"},
        )
        trainer = WhisperTrainer(config)
        trainer.processor = MagicMock()
        trainer.processor.tokenizer.pad_token_id = 0
        trainer.processor.tokenizer.batch_decode.return_value = ["hello", "hello"]

        pred = MagicMock()
        pred.predictions = np.array([[1, 2], [3, 4]])
        pred.label_ids = np.array([[1, 2], [3, 4]])

        metrics = trainer._compute_metrics(pred)
        assert "wer" in metrics
        assert "cer" in metrics
        assert metrics["wer"] == 0.0
# TestWhisperTrainer.test_data_collator tests padding of unequal label lengths
# get_scheduler tested with cosine, linear, cosine_with_warmup, linear_with_warmup
