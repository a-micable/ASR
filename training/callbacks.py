"""Training callbacks for Whisper fine-tuning."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import (
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


class MetricLoggingCallback(TrainerCallback):
    """Log training metrics to JSON file for external monitoring."""

    def __init__(self, log_path: str | Path) -> None:
        """
        Initialize callback.

        Args:
            log_path: Path to JSON lines metrics log.
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> TrainerControl:
        """Append metrics on each log event."""
        if logs is None:
            return control

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": state.global_step,
            "epoch": state.epoch,
            **logs,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        logger.info(
            "Step %d | epoch %.2f | %s",
            state.global_step,
            state.epoch or 0,
            " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in logs.items()),
        )
        return control


class CheckpointCallback(TrainerCallback):
    """Enhanced checkpoint saving with best-model tracking."""

    def __init__(self, checkpoint_dir: str | Path, metric_name: str = "eval_wer") -> None:
        """
        Initialize checkpoint callback.

        Args:
            checkpoint_dir: Directory for best model symlink/info.
            metric_name: Metric to track for best checkpoint.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metric_name = metric_name
        self.best_metric: float | None = None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> TrainerControl:
        """Track best evaluation metric."""
        if metrics is None or self.metric_name not in metrics:
            return control

        current = metrics[self.metric_name]
        if self.best_metric is None or current < self.best_metric:
            self.best_metric = current
            info = {
                "best_metric": current,
                "metric_name": self.metric_name,
                "global_step": state.global_step,
                "epoch": state.epoch,
                "checkpoint": f"checkpoint-{state.global_step}",
            }
            info_path = self.checkpoint_dir / "best_checkpoint.json"
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2)
            logger.info("New best %s: %.4f at step %d", self.metric_name, current, state.global_step)

        return control


class TrainingVisualizationCallback(TrainerCallback):
    """Hook for training visualization (loss curves)."""

    def __init__(self, output_dir: str | Path) -> None:
        """
        Initialize visualization callback.

        Args:
            output_dir: Directory to save plot artifacts.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_losses: list[tuple[int, float]] = []
        self.eval_metrics: list[tuple[int, dict[str, float]]] = []

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> TrainerControl:
        """Collect loss values."""
        if logs and "loss" in logs:
            self.train_losses.append((state.global_step, logs["loss"]))
        return control

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> TrainerControl:
        """Collect evaluation metrics."""
        if metrics:
            self.eval_metrics.append((state.global_step, metrics.copy()))
        return control

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        """Save training curves as JSON for notebook rendering."""
        artifact = {
            "train_losses": self.train_losses,
            "eval_metrics": [
                {"step": step, **metrics} for step, metrics in self.eval_metrics
            ],
        }
        out_path = self.output_dir / "training_curves.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2)
        logger.info("Saved training curves to %s", out_path)
        return control


def build_callbacks(
    config: Any,
    log_dir: str | Path,
) -> list[TrainerCallback]:
    """
    Build standard callback list from pipeline config.

    Args:
        config: PipelineConfig or TrainingConfig-like object.
        log_dir: Directory for logs and artifacts.

    Returns:
        List of TrainerCallback instances.
    """
    training = getattr(config, "training", config)
    log_dir = Path(log_dir)

    callbacks: list[TrainerCallback] = [
        MetricLoggingCallback(log_dir / "metrics.jsonl"),
        CheckpointCallback(training.output_dir, metric_name="eval_wer"),
        TrainingVisualizationCallback(log_dir / "visualizations"),
        EarlyStoppingCallback(early_stopping_patience=training.early_stopping_patience),
    ]
    return callbacks
# build_callbacks returns [MetricLogging, Checkpoint, Visualization, EarlyStopping]
# best_metric updated only when current < best; tracked in on_evaluate
