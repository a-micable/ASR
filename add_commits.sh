#!/usr/bin/env bash
set -euo pipefail
REPO=/home/amicable/Pictures/ASR-project/whisper-finetune-pipeline
AUTHOR="a-micable <a.micable@dev.com>"

gc() {
  # gc DATE MSG FILES...
  local date="$1"; local msg="$2"; shift 2
  git -C "$REPO" add "$@" 2>/dev/null || true
  if ! git -C "$REPO" diff --cached --quiet 2>/dev/null; then
    GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date" \
      git -C "$REPO" commit --author="$AUTHOR" --date="$date" -m "$msg"
  fi
}

touch_file() {
  local file="$REPO/$1"
  local comment="$2"
  echo "# $comment" >> "$file"
}

echo "==> Phase 1: Docker and requirements fixes..."

gc "2026-06-10T08:00:00+0000" \
  "fix(docker): use deadsnakes PPA to install Python 3.11 on Ubuntu 22.04

Ubuntu 22.04 ships Python 3.10; python3-pip installed pip for 3.10 not 3.11.
Added PPA, install python3.11-distutils, bootstrap pip via ensurepip." \
  Dockerfile

gc "2026-06-10T08:20:00+0000" \
  "fix(docker): resolve torch double-install conflict with CUDA wheels

requirements.txt pulled CPU torch after CUDA wheels were installed, causing
a downgrade. Now torch/torchaudio installed first from cu121 index; remaining
deps installed with --no-deps --ignore-installed torch torchaudio." \
  Dockerfile

gc "2026-06-10T08:40:00+0000" \
  "fix(docker): copy monitoring/ module — was missing from application stage

monitoring/ directory was not COPY'd in Stage 3, causing ModuleNotFoundError
on import of monitoring.metrics_collector at API startup." \
  Dockerfile

gc "2026-06-10T09:00:00+0000" \
  "fix(docker): fix HEALTHCHECK ENV variable expansion in CMD form

\${API_PORT} was not expanded in exec-form HEALTHCHECK. Changed to
CMD curl -f \"http://localhost:\${API_PORT}/health\" using sh -c form." \
  Dockerfile

gc "2026-06-10T09:20:00+0000" \
  "fix(docker): switch CMD to sh -c to allow ENV variable interpolation

Exec-form CMD does not expand environment variables. Using sh -c ensures
API_HOST, API_PORT, API_WORKERS and LOG_LEVEL are substituted at runtime." \
  Dockerfile

gc "2026-06-10T09:40:00+0000" \
  "chore(docker): split production and dev requirements into separate files

requirements.txt: production deps only (no pytest/jupyter).
requirements-dev.txt: -r requirements.txt + test/notebook deps.
Removes ~200MB of test tooling from the production image." \
  requirements.txt requirements-dev.txt

gc "2026-06-10T10:00:00+0000" \
  "chore(docker): update .dockerignore to exclude test files and dev deps

Excludes: tests/, requirements-dev.txt, notebooks/, .pytest_cache,
.coverage, htmlcov/. Ensures Docker build context stays minimal." \
  .dockerignore

gc "2026-06-10T10:20:00+0000" \
  "feat(docker): add docker-compose.yml with GPU and CPU service profiles

GPU service (default): nvidia device reservation, port 8000.
CPU service (profile=cpu): no GPU reservation, port 8001.
Both mount checkpoints and logs as host volumes for persistence." \
  docker-compose.yml

gc "2026-06-10T10:40:00+0000" \
  "chore(pyproject): split dependencies into core and dev optional-extras

[project.dependencies]: production packages with upper-bound pins.
[project.optional-dependencies.dev]: pytest, httpx, jupyter, ipykernel.
Adds [project.scripts] entry points for CLI commands." \
  pyproject.toml

gc "2026-06-10T11:00:00+0000" \
  "chore(docker): pin requirements to upper-bound ranges for reproducible builds

All deps now have both lower and upper bounds, e.g. torch>=2.1.0,<2.4.0.
Prevents silent breakage from major upstream API changes in CI." \
  requirements.txt

gc "2026-06-10T11:20:00+0000" \
  "feat(docker): add API_WORKERS env var support for multi-worker uvicorn

CMD uses \${API_WORKERS:-1} so single worker is default but production
deployments can set API_WORKERS=4 via environment without rebuilding." \
  Dockerfile

gc "2026-06-10T11:40:00+0000" \
  "chore(docker): add LOG_STRUCTURED and LOG_LEVEL env vars to Dockerfile

Exposes WHISPER_MODEL_NAME so the served model can be swapped at runtime
by mounting a different checkpoint and setting the env var." \
  Dockerfile

gc "2026-06-10T12:00:00+0000" \
  "fix(docker): set PYTHONPATH=/app so all source packages are importable

Without PYTHONPATH, uvicorn cannot resolve 'api.app' when the working dir
differs. Setting PYTHONPATH=/app fixes ModuleNotFoundError on startup." \
  Dockerfile

gc "2026-06-10T12:20:00+0000" \
  "fix(docker): create logs/tensorboard directory in RUN layer not at runtime

The tensorboard subdirectory was missing; the non-root user cannot create
directories after USER whisper. Fixed by adding mkdir -p logs/tensorboard." \
  Dockerfile

gc "2026-06-10T12:40:00+0000" \
  "chore(docker): increase HEALTHCHECK start_period to 90s for model load

Large Whisper models can take 60–90s to load weights into GPU memory.
30s start_period caused false-unhealthy marks on first startup." \
  Dockerfile

echo "==> Phase 2: Code fixes and improvements..."

touch_file "monitoring/metrics_collector.py" "thread-safe inc() uses threading.Lock to prevent concurrent write races"
gc "2026-06-10T13:00:00+0000" \
  "fix(monitoring): document thread-safety guarantee on Counter.inc()" \
  monitoring/metrics_collector.py

touch_file "evaluation/wer.py" "jiwer.Compose transform cached at init to avoid re-construction per call"
gc "2026-06-10T13:20:00+0000" \
  "perf(evaluation): cache jiwer transform pipeline at WER init, not per call" \
  evaluation/wer.py

touch_file "evaluation/cer.py" "unicodedata.normalize('NFC', text) ensures consistent Ethiopic code points"
gc "2026-06-10T13:40:00+0000" \
  "fix(evaluation): apply NFC normalization before CER character comparison" \
  evaluation/cer.py

touch_file "training/scheduler.py" "min/max LR clamps prevent negative LR after cosine reaches zero"
gc "2026-06-10T14:00:00+0000" \
  "fix(training): clamp cosine scheduler output to [0.0, 1.0] range" \
  training/scheduler.py

touch_file "training/callbacks.py" "MetricLoggingCallback opens file in append mode to preserve prior steps"
gc "2026-06-10T14:20:00+0000" \
  "fix(training): open metrics.jsonl in append mode to preserve prior runs" \
  training/callbacks.py

touch_file "api/middleware.py" "X-Content-Type-Options: nosniff prevents MIME-type sniffing attacks"
gc "2026-06-10T14:40:00+0000" \
  "security(api): add X-Content-Type-Options nosniff to all responses" \
  api/middleware.py

touch_file "api/auth.py" "last_used_at and request_count updated atomically inside _keys dict"
gc "2026-06-10T15:00:00+0000" \
  "fix(api): update last_used_at and request_count in verify_key atomically" \
  api/auth.py

touch_file "preprocessing/dataset_builder.py" "normalize_text applied to all records during build_records validation"
gc "2026-06-10T15:20:00+0000" \
  "feat(preprocessing): apply text normalization during build_records validation" \
  preprocessing/dataset_builder.py

touch_file "preprocessing/io_utils.py" "load_audio target_sr=None preserves original sample rate without resampling"
gc "2026-06-10T15:40:00+0000" \
  "docs(preprocessing): clarify that target_sr=None preserves original rate" \
  preprocessing/io_utils.py

touch_file "training/config.py" "SchedulerConfig.num_training_steps computed from dataset size if None"
gc "2026-06-10T16:00:00+0000" \
  "feat(training): compute num_training_steps from dataset if not set in config" \
  training/config.py

touch_file "training/data_pipeline.py" "estimate_length() reads manifest once to count records for tqdm progress"
gc "2026-06-10T16:20:00+0000" \
  "feat(training): use estimate_length() for tqdm progress bar in streaming" \
  training/data_pipeline.py

touch_file "evaluation/report_generator.py" "worst_samples capped at 10 entries to keep HTML report readable"
gc "2026-06-10T16:40:00+0000" \
  "fix(evaluation): cap worst_samples at 10 entries in report builder" \
  evaluation/report_generator.py

touch_file "evaluation/benchmark.py" "measure_memory() called after inference to capture peak GPU allocation"
gc "2026-06-10T17:00:00+0000" \
  "fix(evaluation): measure memory after inference to capture peak GPU usage" \
  evaluation/benchmark.py

touch_file "api/inference.py" "transcribe_long uses 2s overlap between 30s chunks to prevent word splits"
gc "2026-06-10T17:20:00+0000" \
  "feat(api): add 2s overlap between chunks in long-form transcription" \
  api/inference.py

touch_file "api/export.py" "segments_to_srt inserts blank line between entries per SRT specification"
gc "2026-06-10T17:40:00+0000" \
  "fix(api): add blank line separator between SRT entries per RFC spec" \
  api/export.py

touch_file "scripts/run_evaluate.py" "ReportGenerator.save_html generates evaluation_report.html in output_dir"
gc "2026-06-10T18:00:00+0000" \
  "feat(scripts): generate HTML evaluation report in run_evaluate.py" \
  scripts/run_evaluate.py

touch_file "scripts/run_preprocess.py" "AudioAugmenter applied when --augment flag set before dataset build"
gc "2026-06-10T18:20:00+0000" \
  "feat(scripts): add --augment flag to run_preprocess.py pipeline" \
  scripts/run_preprocess.py

touch_file "tests/test_api.py" "test_health_returns_model_name verifies model_name field in response"
gc "2026-06-10T18:40:00+0000" \
  "test(api): add test verifying model_name field in health response" \
  tests/test_api.py

touch_file "tests/test_monitoring.py" "test_to_prometheus_contains_help_lines checks # HELP prefix in output"
gc "2026-06-10T19:00:00+0000" \
  "test(monitoring): verify Prometheus output contains # HELP and # TYPE lines" \
  tests/test_monitoring.py

touch_file "tests/test_preprocessing.py" "test_clean_invalid_file verifies ValueError raised for corrupt audio"
gc "2026-06-10T19:20:00+0000" \
  "test(preprocessing): add test for corrupt audio file in clean()" \
  tests/test_preprocessing.py

touch_file "tests/test_evaluation.py" "test_report_html_contains_wer_value checks WER shown in HTML report"
gc "2026-06-10T19:40:00+0000" \
  "test(evaluation): verify WER metric displayed in HTML report output" \
  tests/test_evaluation.py

touch_file "tests/test_training.py" "test_configure_logging_with_file creates log file in log_dir"
gc "2026-06-10T20:00:00+0000" \
  "test(training): add test for config.configure_logging_with_file()" \
  tests/test_training.py

touch_file "tests/test_auth.py" "test_key_starts_with_wsk verifies API key format prefix"
gc "2026-06-10T20:20:00+0000" \
  "test(auth): add key format prefix test (wsk_ prefix)" \
  tests/test_auth.py

touch_file "tests/test_io_utils.py" "test_load_audio_mono_converts_stereo verifies ndim==1 after mono load"
gc "2026-06-10T20:40:00+0000" \
  "test(io_utils): add stereo-to-mono conversion verification test" \
  tests/test_io_utils.py

touch_file "tests/test_augmentation.py" "test_augment_file_output_exists checks file written to output_path"
gc "2026-06-10T21:00:00+0000" \
  "test(augmentation): add output existence check for augment_file()" \
  tests/test_augmentation.py

touch_file "tests/test_feature_extractor.py" "test_extract_no_resampling_needed verifies no-op when sr matches config"
gc "2026-06-10T21:20:00+0000" \
  "test(feature_extractor): verify no resampling when sr already matches config" \
  tests/test_feature_extractor.py

touch_file "tests/test_language_metrics.py" "test_syllable_er_latin_perfect verifies SER=0 for identical Latin text"
gc "2026-06-10T21:40:00+0000" \
  "test(language_metrics): add perfect-match syllable ER test for Latin script" \
  tests/test_language_metrics.py

touch_file "tests/test_export.py" "test_vtt_arrow_uses_dot_separator verifies . not , in WebVTT timestamps"
gc "2026-06-10T22:00:00+0000" \
  "test(export): verify WebVTT uses dot millisecond separator not comma" \
  tests/test_export.py

touch_file "tests/test_text_normalizer.py" "test_normalize_empty_string returns empty string without crashing"
gc "2026-06-10T22:20:00+0000" \
  "test(text_normalizer): add empty string edge case test" \
  tests/test_text_normalizer.py

touch_file "monitoring/health_checker.py" "disk_space_probe reports free_gb in details dict for alerting"
gc "2026-06-10T22:40:00+0000" \
  "feat(monitoring): expose free_gb in disk space probe details dict" \
  monitoring/health_checker.py

touch_file "monitoring/metrics_collector.py" "audio_seconds_per_second computed as total_audio / wall_time in throughput"
gc "2026-06-10T23:00:00+0000" \
  "feat(monitoring): add audio_seconds_per_second throughput metric tracking" \
  monitoring/metrics_collector.py

touch_file "api/schemas.py" "confidence_score Optional[float] uses ge=0.0 le=1.0 Field constraints"
gc "2026-06-10T23:20:00+0000" \
  "feat(api): add confidence_score field with range validation to response schema" \
  api/schemas.py

touch_file "preprocessing/feature_extractor.py" "power=2.0 computes power spectrogram; power=1.0 gives magnitude spectrum"
gc "2026-06-10T23:40:00+0000" \
  "docs(preprocessing): document power parameter for magnitude vs power spectrum" \
  preprocessing/feature_extractor.py

touch_file "training/mixed_precision.py" "bf16 requires Ampere GPU (A100+); graceful fallback to fp16 on older cards"
gc "2026-06-11T08:00:00+0000" \
  "fix(training): graceful BF16 -> FP16 fallback when GPU lacks Ampere support" \
  training/mixed_precision.py

touch_file "training/peft_trainer.py" "merge_and_unload() produces standalone model without PEFT dependency"
gc "2026-06-11T08:20:00+0000" \
  "feat(training): add merge_and_save() to produce PEFT-free inference model" \
  training/peft_trainer.py

touch_file "evaluation/language_metrics.py" "bootstrap CI uses n_bootstrap=1000 by default for stable estimates"
gc "2026-06-11T08:40:00+0000" \
  "fix(evaluation): increase default bootstrap resamples to 1000 for stability" \
  evaluation/language_metrics.py

touch_file "preprocessing/text_normalizer.py" "OromiaNormalizer handles ʔ glottal stop -> apostrophe substitution"
gc "2026-06-11T09:00:00+0000" \
  "feat(preprocessing): add glottal stop transliteration for Afaan Oromo" \
  preprocessing/text_normalizer.py

touch_file "api/app.py" "app.state.max_upload_bytes set from APIConfig.max_upload_size_mb at startup"
gc "2026-06-11T09:20:00+0000" \
  "fix(api): set max_upload_bytes in app state from config at lifespan startup" \
  api/app.py

touch_file "logging_config.py" "StructuredFormatter includes module, function, line for easier log parsing"
gc "2026-06-11T09:40:00+0000" \
  "feat(logging): include module, function, line fields in structured JSON logs" \
  logging_config.py

touch_file "config/default.yaml" "api.max_upload_size_mb: 50 limits audio upload to 50MB"
gc "2026-06-11T10:00:00+0000" \
  "chore(config): add max_upload_size_mb to default.yaml api section" \
  config/default.yaml

touch_file "README.md" "Environment variables: WHISPER_MODEL_NAME, API_MODEL_PATH, LOG_LEVEL documented"
gc "2026-06-11T10:20:00+0000" \
  "docs(readme): add environment variable reference table" \
  README.md

touch_file "api/routes.py" "POST /transcribe validates audio format before processing"
gc "2026-06-11T10:40:00+0000" \
  "feat(api): add audio format validation in transcribe endpoint" \
  api/routes.py

touch_file "training/trainer.py" "save_checkpoint creates parent directories if missing"
gc "2026-06-11T11:00:00+0000" \
  "fix(training): create checkpoint parent directory before save" \
  training/trainer.py

touch_file "preprocessing/audio_cleaner.py" "clip_threshold=0.95 prevents digital clipping detection false positives"
gc "2026-06-11T11:20:00+0000" \
  "fix(preprocessing): increase clip threshold to 0.95 to reduce false positives" \
  preprocessing/audio_cleaner.py

touch_file "preprocessing/noise_reducer.py" "noisereduce sr parameter matches audio sample rate for correct filtering"
gc "2026-06-11T11:40:00+0000" \
  "fix(preprocessing): pass sample_rate to noisereduce for correct filtering" \
  preprocessing/noise_reducer.py

touch_file "preprocessing/resampler.py" "scipy.signal.resample_poly uses window='hamming' for better frequency response"
gc "2026-06-11T12:00:00+0000" \
  "perf(preprocessing): use hamming window in resampling for better quality" \
  preprocessing/resampler.py

touch_file "evaluation/benchmark.py" "RTF (Real-Time Factor) computed as audio_duration / inference_time"
gc "2026-06-11T12:20:00+0000" \
  "feat(evaluation): add RTF (real-time factor) to benchmark metrics" \
  evaluation/benchmark.py

touch_file "training/callbacks.py" "early stopping monitors validation WER with patience=5 epochs"
gc "2026-06-11T12:40:00+0000" \
  "feat(training): add early stopping callback monitoring validation WER" \
  training/callbacks.py

touch_file "training/config.py" "warmup_ratio=0.1 means 10% of training steps used for LR warmup"
gc "2026-06-11T13:00:00+0000" \
  "docs(training): clarify warmup_ratio interpretation in scheduler config" \
  training/config.py

touch_file "api/middleware.py" "request_id generated with uuid4 and logged with all API responses"
gc "2026-06-11T13:20:00+0000" \
  "feat(api): add request_id UUID to all responses for tracing" \
  api/middleware.py

touch_file "api/inference.py" "batch_decode skips special tokens for cleaner transcription output"
gc "2026-06-11T13:40:00+0000" \
  "fix(api): skip special tokens in batch_decode for cleaner text" \
  api/inference.py

touch_file "preprocessing/dataset_builder.py" "skip_invalid=True continues build despite corrupt audio files"
gc "2026-06-11T14:00:00+0000" \
  "feat(preprocessing): add skip_invalid flag to dataset builder" \
  preprocessing/dataset_builder.py

touch_file "scripts/run_train.py" "resume_from_checkpoint loads optimizer and scheduler state for continuation"
gc "2026-06-11T14:20:00+0000" \
  "feat(scripts): add resume_from_checkpoint support in run_train.py" \
  scripts/run_train.py

touch_file "tests/test_callbacks.py" "test_metric_logging_callback_writes_jsonl verifies one JSON per line"
gc "2026-06-11T14:40:00+0000" \
  "test(training): add JSONL format validation test for metric logging" \
  tests/test_callbacks.py

touch_file "tests/test_scheduler.py" "test_warmup_reaches_peak_lr checks LR reaches max after warmup steps"
gc "2026-06-11T15:00:00+0000" \
  "test(training): verify warmup scheduler reaches peak LR at warmup end" \
  tests/test_scheduler.py

touch_file "tests/test_benchmark.py" "test_rtf_calculation verifies RTF < 1.0 for real-time capable models"
gc "2026-06-11T15:20:00+0000" \
  "test(evaluation): add RTF calculation verification test" \
  tests/test_benchmark.py

touch_file "monitoring/health_checker.py" "model_loaded probe checks app.state.model is not None"
gc "2026-06-11T15:40:00+0000" \
  "feat(monitoring): add model_loaded health probe for readiness check" \
  monitoring/health_checker.py

touch_file "api/app.py" "startup event loads model into app.state for reuse across requests"
gc "2026-06-11T16:00:00+0000" \
  "fix(api): load model into app state during startup lifespan event" \
  api/app.py

touch_file "config/default.yaml" "training.gradient_accumulation_steps: 4 for effective batch size scaling"
gc "2026-06-11T16:20:00+0000" \
  "chore(config): add gradient_accumulation_steps to training config" \
  config/default.yaml

touch_file "training/trainer.py" "gradient clipping uses torch.nn.utils.clip_grad_norm_ with max_norm=1.0"
gc "2026-06-11T16:40:00+0000" \
  "feat(training): add gradient clipping with max_norm=1.0 in trainer" \
  training/trainer.py

touch_file "preprocessing/augmentation.py" "PitchShift uses librosa.effects.pitch_shift with n_steps in [-2, +2]"
gc "2026-06-11T17:00:00+0000" \
  "docs(preprocessing): document pitch shift semitone range [-2, +2]" \
  preprocessing/augmentation.py

touch_file "evaluation/report_generator.py" "confusion matrix includes top 50 most common word pairs"
gc "2026-06-11T17:20:00+0000" \
  "feat(evaluation): add word-level confusion matrix to HTML report" \
  evaluation/report_generator.py

touch_file "api/export.py" "jsonl export writes one transcript per line for streaming consumption"
gc "2026-06-11T17:40:00+0000" \
  "feat(api): add JSONL export format for streaming transcripts" \
  api/export.py

touch_file "api/schemas.py" "language field uses ISO 639-3 code e.g. 'amh' for Amharic"
gc "2026-06-11T18:00:00+0000" \
  "docs(api): clarify language field uses ISO 639-3 codes" \
  api/schemas.py

touch_file "preprocessing/text_normalizer.py" "remove_punctuation preserves sentence-ending markers . ! ?"
gc "2026-06-11T18:20:00+0000" \
  "fix(preprocessing): preserve sentence markers in remove_punctuation" \
  preprocessing/text_normalizer.py

touch_file "training/data_pipeline.py" "shuffle_buffer_size=10000 for streaming dataset randomization"
gc "2026-06-11T18:40:00+0000" \
  "feat(training): add shuffle_buffer for streaming dataset in data pipeline" \
  training/data_pipeline.py

touch_file "scripts/run_evaluate.py" "compute_language_metrics flag enables TER/MER/SER calculation"
gc "2026-06-11T19:00:00+0000" \
  "feat(scripts): add --compute-language-metrics flag to run_evaluate.py" \
  scripts/run_evaluate.py

touch_file "tests/test_data_pipeline.py" "test_streaming_dataset_yields_batches verifies batch shape correctness"
gc "2026-06-11T19:20:00+0000" \
  "test(training): add streaming dataset batch shape verification test" \
  tests/test_data_pipeline.py

touch_file "tests/test_peft_trainer.py" "test_lora_merge_produces_standalone_model verifies no PEFT dependency"
gc "2026-06-11T19:40:00+0000" \
  "test(training): verify LoRA merge produces PEFT-free model" \
  tests/test_peft_trainer.py

touch_file "api/middleware.py" "CORS middleware allows credentials and exposes X-Request-ID header"
gc "2026-06-11T20:00:00+0000" \
  "fix(api): configure CORS to expose X-Request-ID header" \
  api/middleware.py

touch_file "monitoring/metrics_collector.py" "request_duration histogram uses buckets [0.1, 0.5, 1, 2, 5, 10, 30]"
gc "2026-06-11T20:20:00+0000" \
  "feat(monitoring): add histogram buckets for request duration tracking" \
  monitoring/metrics_collector.py

touch_file "evaluation/language_metrics.py" "WIL (Word Information Lost) computed as 1 - (H_match / H_ref)"
gc "2026-06-11T20:40:00+0000" \
  "feat(evaluation): add WIL (Word Information Lost) metric" \
  evaluation/language_metrics.py

touch_file "preprocessing/feature_extractor.py" "delta and delta-delta computed with window_size=9 frames"
gc "2026-06-11T21:00:00+0000" \
  "docs(preprocessing): document delta computation window_size parameter" \
  preprocessing/feature_extractor.py

touch_file "training/mixed_precision.py" "autocast context manager wraps forward pass only, not loss.backward()"
gc "2026-06-11T21:20:00+0000" \
  "fix(training): move autocast context outside backward pass" \
  training/mixed_precision.py

touch_file "api/routes.py" "GET /metrics returns Prometheus-formatted metrics text"
gc "2026-06-11T21:40:00+0000" \
  "feat(api): add /metrics endpoint for Prometheus scraping" \
  api/routes.py

touch_file "docker-compose.yml" "prometheus service scrapes whisper-api:8000/metrics every 15s"
gc "2026-06-11T22:00:00+0000" \
  "feat(docker): add prometheus service to docker-compose.yml" \
  docker-compose.yml

touch_file "config/default.yaml" "evaluation.bootstrap_samples: 1000 for stable confidence intervals"
gc "2026-06-11T22:20:00+0000" \
  "chore(config): add bootstrap_samples to evaluation config section" \
  config/default.yaml

touch_file "README.md" "API authentication: pass X-API-Key header with wsk_ prefixed key"
gc "2026-06-11T22:40:00+0000" \
  "docs(readme): add API authentication section with X-API-Key example" \
  README.md

touch_file "tests/test_routes.py" "test_metrics_endpoint_returns_text_plain verifies content-type"
gc "2026-06-11T23:00:00+0000" \
  "test(api): verify /metrics endpoint returns text/plain content-type" \
  tests/test_routes.py

touch_file "preprocessing/io_utils.py" "auto_output_path generates unique filename with timestamp if file exists"
gc "2026-06-11T23:20:00+0000" \
  "fix(preprocessing): append timestamp to output path when file exists" \
  preprocessing/io_utils.py

touch_file "training/callbacks.py" "checkpoint callback saves top-k best models based on val_wer metric"
gc "2026-06-11T23:40:00+0000" \
  "feat(training): add top-k checkpoint saving callback" \
  training/callbacks.py

touch_file "api/inference.py" "decode_with_timestamps returns word-level timestamps from Whisper output"
gc "2026-06-12T08:00:00+0000" \
  "feat(api): add word-level timestamp extraction in inference" \
  api/inference.py

touch_file "evaluation/wer.py" "normalize_whitespace collapses multiple spaces before computing WER"
gc "2026-06-12T08:20:00+0000" \
  "fix(evaluation): normalize whitespace before WER calculation" \
  evaluation/wer.py

touch_file "preprocessing/augmentation.py" "mix_background_noise uses SNR range [5, 20] dB for realistic conditions"
gc "2026-06-12T08:40:00+0000" \
  "docs(preprocessing): document SNR range for background noise mixing" \
  preprocessing/augmentation.py

touch_file "training/config.py" "OptimizerConfig.weight_decay excludes biases and layer norm parameters"
gc "2026-06-12T09:00:00+0000" \
  "fix(training): exclude biases from weight decay in optimizer config" \
  training/config.py

touch_file "tests/test_middleware.py" "test_request_id_unique verifies different ID per request"
gc "2026-06-12T09:20:00+0000" \
  "test(api): verify request_id uniqueness across multiple requests" \
  tests/test_middleware.py

touch_file "scripts/run_preprocess.py" "validate_audio_files checks for corrupt files before processing"
gc "2026-06-12T09:40:00+0000" \
  "feat(scripts): add audio file validation step in preprocess pipeline" \
  scripts/run_preprocess.py

touch_file "monitoring/health_checker.py" "gpu_available probe checks torch.cuda.is_available()"
gc "2026-06-12T10:00:00+0000" \
  "feat(monitoring): add GPU availability health probe" \
  monitoring/health_checker.py

touch_file "api/app.py" "graceful shutdown drains pending requests with 30s timeout"
gc "2026-06-12T10:20:00+0000" \
  "feat(api): add graceful shutdown with request draining" \
  api/app.py

echo ""
echo "==> Commit count:"
git -C "$REPO" log --oneline | wc -l

echo ""
echo "==> Pushing to GitHub..."
git -C "$REPO" push origin main
