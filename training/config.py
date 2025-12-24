"""Centralized pipeline configuration using Pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    """Whisper model configuration."""

    model_config = SettingsConfigDict(env_prefix="WHISPER_MODEL_")

    name: str = Field(default="openai/whisper-small", description="HF model ID")
    language: str = Field(default="am", description="ISO 639-1 language code")
    task: Literal["transcribe", "translate"] = "transcribe"
    use_peft: bool = False
    freeze_encoder: bool = False


class DatasetConfig(BaseSettings):
    """Dataset paths and split configuration."""

    model_config = SettingsConfigDict(env_prefix="DATASET_")

    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    augmented_dir: Path = Path("data/augmented")
    hf_dataset_path: Path = Path("data/processed/hf_dataset")
    transcription_file: Path | None = None
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    max_input_length: float = 30.0
    sample_rate: int = 16000


class TrainingConfig(BaseSettings):
    """Training hyperparameters and runtime settings."""

    model_config = SettingsConfigDict(env_prefix="TRAINING_")

    output_dir: Path = Path("checkpoints")
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    fp16: bool = True
    bf16: bool = False
    dataloader_num_workers: int = 4
    seed: int = 42
    logging_steps: int = 25
    eval_steps: int = 250
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    early_stopping_patience: int = 3
    resume_from_checkpoint: str | None = None
    report_to: list[str] = Field(default_factory=lambda: ["tensorboard"])

    @field_validator("learning_rate", mode="before")
    @classmethod
    def parse_lr(cls, v: float | str) -> float:
        """Parse learning rate from string or float."""
        return float(v)


class SchedulerConfig(BaseSettings):
    """Learning rate scheduler configuration."""

    model_config = SettingsConfigDict(env_prefix="SCHEDULER_")

    scheduler_type: Literal["cosine", "linear", "cosine_with_warmup", "linear_with_warmup"] = (
        "cosine_with_warmup"
    )
    num_warmup_steps: int = 500
    num_training_steps: int | None = None


class LoggingConfig(BaseSettings):
    """Logging and monitoring configuration."""

    model_config = SettingsConfigDict(env_prefix="LOG_")

    level: str = "INFO"
    log_dir: Path = Path("logs")
    structured: bool = True
    tensorboard_dir: Path = Path("logs/tensorboard")


class APIConfig(BaseSettings):
    """Inference API configuration."""

    model_config = SettingsConfigDict(env_prefix="API_")

    host: str = "0.0.0.0"
    port: int = 8000
    model_path: Path = Path("checkpoints/best")
    max_upload_size_mb: int = 50
    rate_limit: str = "60/minute"
    default_language: str = "am"
    workers: int = 1


class PipelineConfig(BaseSettings):
    """Top-level pipeline configuration aggregating all sub-configs."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    project_root: Path = Path(".")

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """
        Load configuration from a YAML file.

        Args:
            path: Path to YAML config file.

        Returns:
            PipelineConfig instance.
        """
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def ensure_directories(self) -> None:
        """Create required output directories."""
        for directory in [
            self.dataset.raw_dir,
            self.dataset.processed_dir,
            self.dataset.augmented_dir,
            self.training.output_dir,
            self.logging.log_dir,
            self.logging.tensorboard_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
# dataloader_num_workers defaults to 4; set to 0 on CPU via env override
# gradient_accumulation_steps=2 doubles effective batch size without extra memory
# project_root used for absolute path resolution in ensure_directories()
# env_prefix set per sub-config: WHISPER_MODEL_, DATASET_, TRAINING_, API_
# beam_size=5 default gives good WER/speed tradeoff for Whisper
# max_upload_size_mb: 50 limits uploads to 50MB to prevent abuse
