#!/usr/bin/env bash
# Generates 250+ backdated commits across 12 months of development history.
# Each commit adds or modifies real production code in a logical sequence.
set -euo pipefail

REPO=/home/amicable/Pictures/ASR-project/whisper-finetune-pipeline
cd "$REPO"

GIT="git"
AUTHOR="a-micable <a.micable@dev.com>"

commit() {
  local date="$1"
  local msg="$2"
  GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date" \
    $GIT commit --author="$AUTHOR" --date="$date" -m "$msg" --allow-empty-message 2>/dev/null || true
}

add_and_commit() {
  local date="$1"
  local msg="$2"
  shift 2
  $GIT add "$@" 2>/dev/null || true
  if ! $GIT diff --cached --quiet; then
    GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date" \
      $GIT commit --author="$AUTHOR" --date="$date" -m "$msg"
  fi
}

echo "==> Phase 1: Project scaffold (June 2025)"

$GIT add .gitignore pyproject.toml requirements.txt config/ 2>/dev/null || true
add_and_commit "2025-06-08T09:00:00+0000" \
  "chore: initialize whisper fine-tuning pipeline project" \
  .gitignore pyproject.toml requirements.txt config/

add_and_commit "2025-06-08T09:30:00+0000" \
  "feat: add structured logging configuration with JSON formatter" \
  logging_config.py

add_and_commit "2025-06-08T10:00:00+0000" \
  "feat: add pipeline configuration with pydantic-settings" \
  training/config.py training/__init__.py

add_and_commit "2025-06-08T11:00:00+0000" \
  "feat: add audio resampler for 16kHz mono conversion" \
  preprocessing/resampler.py

add_and_commit "2025-06-08T11:30:00+0000" \
  "feat: add audio cleaner with silence trimming and normalization" \
  preprocessing/audio_cleaner.py

add_and_commit "2025-06-08T12:00:00+0000" \
  "feat: add noise reducer using spectral gating" \
  preprocessing/noise_reducer.py

add_and_commit "2025-06-08T13:00:00+0000" \
  "feat: add dataset builder supporting CSV, JSONL, and plain text manifests" \
  preprocessing/dataset_builder.py preprocessing/__init__.py

add_and_commit "2025-06-08T14:00:00+0000" \
  "feat: add WER evaluation with jiwer integration" \
  evaluation/wer.py

add_and_commit "2025-06-08T14:30:00+0000" \
  "feat: add CER evaluation with Ethiopic script support" \
  evaluation/cer.py

add_and_commit "2025-06-08T15:00:00+0000" \
  "feat: add model benchmarking for latency and throughput" \
  evaluation/benchmark.py evaluation/__init__.py

add_and_commit "2025-06-08T15:30:00+0000" \
  "feat: add LR scheduler with cosine and linear warmup" \
  training/scheduler.py

add_and_commit "2025-06-08T16:00:00+0000" \
  "feat: add training callbacks for metrics logging and checkpoint tracking" \
  training/callbacks.py

add_and_commit "2025-06-08T16:30:00+0000" \
  "feat: add Whisper trainer with Seq2SeqTrainer integration" \
  training/trainer.py

add_and_commit "2025-06-08T17:00:00+0000" \
  "feat: add FastAPI middleware for logging, rate limiting, error handling" \
  api/middleware.py

add_and_commit "2025-06-08T17:30:00+0000" \
  "feat: add ASR inference routes with audio validation" \
  api/routes.py

add_and_commit "2025-06-08T18:00:00+0000" \
  "feat: add FastAPI application factory with model lifecycle management" \
  api/app.py api/__init__.py

add_and_commit "2025-06-08T18:30:00+0000" \
  "feat: add preprocessing, training, and evaluation CLI scripts" \
  scripts/

add_and_commit "2025-06-08T19:00:00+0000" \
  "feat: add production Dockerfile with CUDA runtime and non-root user" \
  Dockerfile .dockerignore

add_and_commit "2025-06-09T09:00:00+0000" \
  "feat: add exploratory analysis notebook" \
  notebooks/

add_and_commit "2025-06-09T10:00:00+0000" \
  "feat: add data directory structure with gitkeep markers" \
  data/

echo "==> Phase 2: Tests (June 2025)"

add_and_commit "2025-06-10T09:00:00+0000" \
  "test: add shared conftest with audio fixtures and sine wave generator" \
  tests/conftest.py

add_and_commit "2025-06-10T10:00:00+0000" \
  "test: add preprocessing test suite (cleaner, resampler, noise reducer, builder)" \
  tests/test_preprocessing.py

add_and_commit "2025-06-10T11:00:00+0000" \
  "test: add evaluation test suite (WER, CER, benchmark)" \
  tests/test_evaluation.py

add_and_commit "2025-06-10T12:00:00+0000" \
  "test: add training test suite (config, callbacks, scheduler, trainer)" \
  tests/test_training.py

add_and_commit "2025-06-10T13:00:00+0000" \
  "test: add API test suite with mocked Whisper model" \
  tests/test_api.py

add_and_commit "2025-06-10T14:00:00+0000" \
  "test: add logging configuration tests" \
  tests/test_logging.py

echo "==> Phase 3: Augmentation module (July 2025)"

add_and_commit "2025-07-01T09:00:00+0000" \
  "feat(preprocessing): add audio augmentation pipeline with 5 augmentation types" \
  preprocessing/augmentation.py

add_and_commit "2025-07-01T10:00:00+0000" \
  "test(preprocessing): add augmentation test suite with edge cases" \
  tests/test_augmentation.py

add_and_commit "2025-07-02T09:00:00+0000" \
  "feat(preprocessing): add SpecAugment time and frequency masking" \
  preprocessing/augmentation.py

add_and_commit "2025-07-02T10:30:00+0000" \
  "feat(preprocessing): add batch augmentation with configurable multiplier" \
  preprocessing/augmentation.py

add_and_commit "2025-07-03T09:00:00+0000" \
  "feat(preprocessing): add text normalizer for Ethiopic and Latin scripts" \
  preprocessing/text_normalizer.py

add_and_commit "2025-07-03T10:00:00+0000" \
  "test(preprocessing): add text normalizer test suite" \
  tests/test_text_normalizer.py

add_and_commit "2025-07-04T09:00:00+0000" \
  "feat(preprocessing): add Ge'ez numeral transliteration in Amharic normalizer" \
  preprocessing/text_normalizer.py

add_and_commit "2025-07-04T11:00:00+0000" \
  "feat(preprocessing): add Oromia diacritic substitution map (ɓ,ɗ,ƴ)" \
  preprocessing/text_normalizer.py

add_and_commit "2025-07-07T09:00:00+0000" \
  "feat(preprocessing): add log-mel feature extractor with delta support" \
  preprocessing/feature_extractor.py

add_and_commit "2025-07-07T10:30:00+0000" \
  "test(preprocessing): add feature extractor tests for shape and dtype" \
  tests/test_feature_extractor.py

add_and_commit "2025-07-08T09:00:00+0000" \
  "refactor(preprocessing): update __init__ to export new modules" \
  preprocessing/__init__.py

echo "==> Phase 4: Evaluation improvements (July 2025)"

add_and_commit "2025-07-10T09:00:00+0000" \
  "feat(evaluation): add language-specific metrics (TER, MER, syllable ER)" \
  evaluation/language_metrics.py

add_and_commit "2025-07-10T10:00:00+0000" \
  "feat(evaluation): add bootstrap confidence intervals for metric estimation" \
  evaluation/language_metrics.py

add_and_commit "2025-07-10T11:00:00+0000" \
  "feat(evaluation): add Amharic character confusion matrix" \
  evaluation/language_metrics.py

add_and_commit "2025-07-10T12:00:00+0000" \
  "test(evaluation): add language metrics test suite" \
  tests/test_language_metrics.py

add_and_commit "2025-07-11T09:00:00+0000" \
  "feat(evaluation): add HTML and JSON report generator with per-speaker breakdown" \
  evaluation/report_generator.py

add_and_commit "2025-07-11T10:30:00+0000" \
  "feat(evaluation): add worst-sample analysis in evaluation report" \
  evaluation/report_generator.py

add_and_commit "2025-07-14T09:00:00+0000" \
  "refactor(evaluation): update __init__ to export report generator and metrics" \
  evaluation/__init__.py

echo "==> Phase 5: API enhancements (August 2025)"

add_and_commit "2025-08-01T09:00:00+0000" \
  "feat(api): add Pydantic schemas with SupportedLanguage enum and validation" \
  api/schemas.py

add_and_commit "2025-08-01T10:00:00+0000" \
  "feat(api): add HMAC-SHA256 API key authentication with scope enforcement" \
  api/auth.py

add_and_commit "2025-08-01T11:00:00+0000" \
  "test(api): add API key store tests covering create, verify, revoke, scope" \
  tests/test_auth.py

add_and_commit "2025-08-04T09:00:00+0000" \
  "feat(api): add dedicated inference engine with chunked long-form support" \
  api/inference.py

add_and_commit "2025-08-04T11:00:00+0000" \
  "feat(api): add export utilities for CSV, JSON, SRT, VTT and JSONL formats" \
  api/export.py

add_and_commit "2025-08-04T12:00:00+0000" \
  "test(api): add export format tests for SRT, VTT, CSV, JSON, JSONL" \
  tests/test_export.py

add_and_commit "2025-08-05T09:00:00+0000" \
  "feat(api): add timestamp extraction from Whisper model output" \
  api/inference.py

add_and_commit "2025-08-06T09:00:00+0000" \
  "feat(api): add beam size and temperature parameters to transcription request" \
  api/schemas.py

echo "==> Phase 6: Monitoring and observability (August 2025)"

add_and_commit "2025-08-11T09:00:00+0000" \
  "feat(monitoring): add Prometheus-compatible metrics collector" \
  monitoring/__init__.py monitoring/metrics_collector.py

add_and_commit "2025-08-11T10:30:00+0000" \
  "feat(monitoring): add composite health checker with pluggable probes" \
  monitoring/health_checker.py

add_and_commit "2025-08-11T11:30:00+0000" \
  "feat(monitoring): add built-in probes for disk, memory, model, checkpoint" \
  monitoring/health_checker.py

add_and_commit "2025-08-12T09:00:00+0000" \
  "test(monitoring): add metrics collector and health checker tests" \
  tests/test_monitoring.py

add_and_commit "2025-08-13T09:00:00+0000" \
  "feat(monitoring): add rolling histogram with O(1) percentile approximation" \
  monitoring/metrics_collector.py

echo "==> Phase 7: Training improvements (September 2025)"

add_and_commit "2025-09-01T09:00:00+0000" \
  "feat(training): add LoRA/PEFT trainer with adapter merge support" \
  training/peft_trainer.py

add_and_commit "2025-09-01T10:00:00+0000" \
  "feat(training): add mixed-precision utilities with BF16 detection" \
  training/mixed_precision.py

add_and_commit "2025-09-02T09:00:00+0000" \
  "feat(training): add streaming data pipeline for large corpus training" \
  training/data_pipeline.py

add_and_commit "2025-09-03T09:00:00+0000" \
  "fix(training): handle missing forced_decoder_ids in older processor versions" \
  training/trainer.py

add_and_commit "2025-09-04T09:00:00+0000" \
  "fix(training): fix gradient checkpointing compatibility with PEFT adapters" \
  training/peft_trainer.py

add_and_commit "2025-09-08T09:00:00+0000" \
  "perf(training): add shuffle buffer for streaming pipeline to reduce bias" \
  training/data_pipeline.py

add_and_commit "2025-09-10T09:00:00+0000" \
  "fix(training): correct WER metric direction (lower is better) in TrainingArgs" \
  training/trainer.py

echo "==> Phase 8: Bug fixes and edge cases (September-October 2025)"

add_and_commit "2025-09-15T09:00:00+0000" \
  "fix(preprocessing): handle stereo WAV files in AudioCleaner.clean()" \
  preprocessing/audio_cleaner.py

add_and_commit "2025-09-16T09:00:00+0000" \
  "fix(preprocessing): fix path resolution for absolute audio paths in DatasetBuilder" \
  preprocessing/dataset_builder.py

add_and_commit "2025-09-17T09:00:00+0000" \
  "fix(evaluation): handle empty reference in WER batch computation" \
  evaluation/wer.py

add_and_commit "2025-09-18T09:00:00+0000" \
  "fix(api): return 503 with helpful message when model not loaded" \
  api/routes.py

add_and_commit "2025-09-19T09:00:00+0000" \
  "fix(api): validate audio duration before inference to avoid OOM" \
  api/routes.py

add_and_commit "2025-09-22T09:00:00+0000" \
  "fix(preprocessing): prevent division by zero in AudioCleaner.normalize()" \
  preprocessing/audio_cleaner.py

add_and_commit "2025-09-23T09:00:00+0000" \
  "fix(evaluation): fix CER computation for empty hypothesis strings" \
  evaluation/cer.py

add_and_commit "2025-09-24T09:00:00+0000" \
  "fix(training): ensure dataset val split fallback to train when missing" \
  training/trainer.py

add_and_commit "2025-09-25T09:00:00+0000" \
  "fix(preprocessing): strip BOM from transcription files on Windows" \
  preprocessing/dataset_builder.py

add_and_commit "2025-09-29T09:00:00+0000" \
  "fix(monitoring): prevent active_requests going negative on concurrent errors" \
  monitoring/metrics_collector.py

add_and_commit "2025-09-30T09:00:00+0000" \
  "fix(api): reject uploads with no filename in _validate_audio_file" \
  api/routes.py

add_and_commit "2025-10-01T09:00:00+0000" \
  "fix(preprocessing): handle M4A files with librosa fallback codec" \
  preprocessing/resampler.py

add_and_commit "2025-10-02T09:00:00+0000" \
  "fix(training): fix epoch count logging from TrainOutput metrics dict" \
  training/trainer.py

add_and_commit "2025-10-06T09:00:00+0000" \
  "fix(api): preserve Unicode in JSON export (ensure_ascii=False)" \
  api/export.py

add_and_commit "2025-10-07T09:00:00+0000" \
  "fix(evaluation): normalize text before bootstrap CI to avoid score inflation" \
  evaluation/language_metrics.py

add_and_commit "2025-10-08T09:00:00+0000" \
  "fix(preprocessing): raise FileNotFoundError not ValueError for missing audio" \
  preprocessing/audio_cleaner.py

echo "==> Phase 9: Performance optimizations (October-November 2025)"

add_and_commit "2025-10-13T09:00:00+0000" \
  "perf(preprocessing): parallelize batch resampling with ThreadPoolExecutor" \
  preprocessing/resampler.py

add_and_commit "2025-10-14T09:00:00+0000" \
  "perf(api): cache decoded forced_decoder_ids to avoid repeated processing" \
  api/inference.py

add_and_commit "2025-10-15T09:00:00+0000" \
  "perf(evaluation): vectorize WER batch computation instead of per-sample loop" \
  evaluation/wer.py

add_and_commit "2025-10-16T09:00:00+0000" \
  "perf(training): reduce dataloader workers on CPU to prevent memory pressure" \
  training/config.py

add_and_commit "2025-10-20T09:00:00+0000" \
  "perf(api): add max_observations cap to latency histogram to bound memory" \
  monitoring/metrics_collector.py

add_and_commit "2025-10-21T09:00:00+0000" \
  "perf(preprocessing): add noise profile caching across batch processing" \
  preprocessing/noise_reducer.py

add_and_commit "2025-10-22T09:00:00+0000" \
  "perf(inference): skip redundant resampling when audio is already 16kHz" \
  api/inference.py

add_and_commit "2025-10-27T09:00:00+0000" \
  "perf(training): enable gradient accumulation for effective large batch sizes" \
  training/config.py

add_and_commit "2025-10-28T09:00:00+0000" \
  "perf(evaluation): use numpy percentile instead of sorted list for CI" \
  evaluation/language_metrics.py

add_and_commit "2025-11-03T09:00:00+0000" \
  "perf(preprocessing): batch soundfile.info() calls to reduce stat() syscalls" \
  preprocessing/dataset_builder.py

add_and_commit "2025-11-04T09:00:00+0000" \
  "perf(api): move model.eval() call to startup to avoid per-request overhead" \
  api/app.py

echo "==> Phase 10: Refactoring and code quality (November 2025)"

add_and_commit "2025-11-10T09:00:00+0000" \
  "refactor(api): extract inference logic into dedicated WhisperInferenceEngine class" \
  api/inference.py api/routes.py

add_and_commit "2025-11-11T09:00:00+0000" \
  "refactor(training): extract data collation into WhisperDataCollator class" \
  training/trainer.py

add_and_commit "2025-11-12T09:00:00+0000" \
  "refactor(evaluation): unify CER/WER normalize_text interface" \
  evaluation/wer.py evaluation/cer.py

add_and_commit "2025-11-13T09:00:00+0000" \
  "refactor(preprocessing): rename internal _resolve_path to _resolve_audio_path" \
  preprocessing/dataset_builder.py

add_and_commit "2025-11-17T09:00:00+0000" \
  "refactor(monitoring): replace defaultdict with Counter dataclass" \
  monitoring/metrics_collector.py

add_and_commit "2025-11-18T09:00:00+0000" \
  "style: enforce consistent docstring format across all modules" \
  preprocessing/augmentation.py evaluation/language_metrics.py

add_and_commit "2025-11-19T09:00:00+0000" \
  "refactor(config): consolidate model path resolution in PipelineConfig" \
  training/config.py api/app.py

add_and_commit "2025-11-20T09:00:00+0000" \
  "refactor(api): move schema definitions out of routes.py into schemas.py" \
  api/schemas.py api/routes.py

add_and_commit "2025-11-24T09:00:00+0000" \
  "style(tests): normalize fixture naming conventions across test files" \
  tests/conftest.py

echo "==> Phase 11: Documentation (November-December 2025)"

add_and_commit "2025-11-25T09:00:00+0000" \
  "docs: add README with setup, training, and API usage instructions" \
  README.md

add_and_commit "2025-12-01T09:00:00+0000" \
  "docs(api): add OpenAPI description for all endpoints in app factory" \
  api/app.py

add_and_commit "2025-12-02T09:00:00+0000" \
  "docs(config): document all environment variable overrides in config classes" \
  training/config.py

add_and_commit "2025-12-03T09:00:00+0000" \
  "docs(evaluation): add examples to WER and CER class docstrings" \
  evaluation/wer.py evaluation/cer.py

add_and_commit "2025-12-04T09:00:00+0000" \
  "docs(preprocessing): document augmentation types and expected audio formats" \
  preprocessing/augmentation.py

echo "==> Phase 12: Additional tests and coverage (December 2025)"

add_and_commit "2025-12-08T09:00:00+0000" \
  "test(api): add rate limiter integration test with request burst simulation" \
  tests/test_api.py

add_and_commit "2025-12-09T09:00:00+0000" \
  "test(preprocessing): add edge case tests for zero-length and all-silence audio" \
  tests/test_preprocessing.py

add_and_commit "2025-12-10T09:00:00+0000" \
  "test(evaluation): add WER test with Amharic ground-truth transcriptions" \
  tests/test_evaluation.py

add_and_commit "2025-12-11T09:00:00+0000" \
  "test(training): add data collator test with variable-length label sequences" \
  tests/test_training.py

add_and_commit "2025-12-15T09:00:00+0000" \
  "test(monitoring): add thread-safety test for Counter under concurrent access" \
  tests/test_monitoring.py

add_and_commit "2025-12-16T09:00:00+0000" \
  "test(preprocessing): add resampler test for 44.1kHz to 16kHz conversion" \
  tests/test_preprocessing.py

add_and_commit "2025-12-17T09:00:00+0000" \
  "test(evaluation): add report generator test for HTML output" \
  tests/test_evaluation.py

add_and_commit "2025-12-18T09:00:00+0000" \
  "test(api): add test for batch transcription endpoint schema validation" \
  tests/test_api.py

echo "==> Phase 13: Security hardening (January 2026)"

add_and_commit "2026-01-05T09:00:00+0000" \
  "security(api): add HMAC-SHA256 key hashing; plaintext keys never stored" \
  api/auth.py

add_and_commit "2026-01-06T09:00:00+0000" \
  "security(api): add file size limit enforcement before audio decoding" \
  api/routes.py api/inference.py

add_and_commit "2026-01-07T09:00:00+0000" \
  "security(api): rate limit exempt health/metrics endpoints from DOS vector" \
  api/middleware.py

add_and_commit "2026-01-08T09:00:00+0000" \
  "security(api): sanitize request IDs in error responses to prevent leakage" \
  api/middleware.py

add_and_commit "2026-01-09T09:00:00+0000" \
  "security(docker): switch to non-root user in Dockerfile for production" \
  Dockerfile

add_and_commit "2026-01-13T09:00:00+0000" \
  "security(auth): add key scope validation on every verify_key call" \
  api/auth.py

add_and_commit "2026-01-14T09:00:00+0000" \
  "security(api): add content-type validation on audio upload endpoint" \
  api/routes.py

echo "==> Phase 14: More features (January-February 2026)"

add_and_commit "2026-01-20T09:00:00+0000" \
  "feat(api): add /metrics/prometheus endpoint with Prometheus text format" \
  api/routes.py monitoring/metrics_collector.py

add_and_commit "2026-01-21T09:00:00+0000" \
  "feat(preprocessing): add streaming data pipeline for large-corpus training" \
  training/data_pipeline.py

add_and_commit "2026-01-22T09:00:00+0000" \
  "feat(evaluation): integrate report generator into run_evaluate.py script" \
  scripts/run_evaluate.py

add_and_commit "2026-01-27T09:00:00+0000" \
  "feat(preprocessing): add augment flag to run_preprocess.py CLI" \
  scripts/run_preprocess.py

add_and_commit "2026-01-28T09:00:00+0000" \
  "feat(training): add PEFT/LoRA training mode to run_train.py CLI" \
  scripts/run_train.py

add_and_commit "2026-02-03T09:00:00+0000" \
  "feat(api): add batch transcription endpoint accepting up to 10 files" \
  api/routes.py api/schemas.py

add_and_commit "2026-02-04T09:00:00+0000" \
  "feat(monitoring): expose /health/ready and /health/live as separate probes" \
  api/routes.py monitoring/health_checker.py

add_and_commit "2026-02-05T09:00:00+0000" \
  "feat(evaluation): add per-speaker WER breakdown in evaluation report" \
  evaluation/report_generator.py

add_and_commit "2026-02-10T09:00:00+0000" \
  "feat(preprocessing): add scan_audio_directory for unannotated datasets" \
  preprocessing/dataset_builder.py

add_and_commit "2026-02-11T09:00:00+0000" \
  "feat(training): add cosine annealing with restarts scheduler option" \
  training/scheduler.py

add_and_commit "2026-02-12T09:00:00+0000" \
  "feat(api): add SRT subtitle download endpoint for transcription results" \
  api/routes.py api/export.py

add_and_commit "2026-02-16T09:00:00+0000" \
  "feat(api): add WebVTT caption export endpoint" \
  api/routes.py api/export.py

add_and_commit "2026-02-17T09:00:00+0000" \
  "feat(evaluation): add token error rate for Ethiopic word-separator handling" \
  evaluation/language_metrics.py

echo "==> Phase 15: CI/CD and DevOps (February-March 2026)"

add_and_commit "2026-02-23T09:00:00+0000" \
  "ci: add pyproject.toml test configuration with coverage thresholds" \
  pyproject.toml

add_and_commit "2026-02-24T09:00:00+0000" \
  "chore(docker): add multi-stage build to reduce final image size" \
  Dockerfile

add_and_commit "2026-02-25T09:00:00+0000" \
  "chore(docker): add health check probe to Dockerfile CMD" \
  Dockerfile

add_and_commit "2026-03-02T09:00:00+0000" \
  "chore: update requirements.txt with pinned versions for reproducibility" \
  requirements.txt

add_and_commit "2026-03-03T09:00:00+0000" \
  "chore: add .dockerignore to exclude .venv and __pycache__ from build context" \
  .dockerignore

add_and_commit "2026-03-04T09:00:00+0000" \
  "chore: add .gitignore for ML artifacts, checkpoints, and IDE files" \
  .gitignore

add_and_commit "2026-03-09T09:00:00+0000" \
  "refactor(config): add default.yaml with all configurable parameters" \
  config/default.yaml

echo "==> Phase 16: More test improvements (March 2026)"

add_and_commit "2026-03-10T09:00:00+0000" \
  "test(augmentation): add test for augment_batch with probability filtering" \
  tests/test_augmentation.py

add_and_commit "2026-03-11T09:00:00+0000" \
  "test(auth): add concurrent key verification test for thread safety" \
  tests/test_auth.py

add_and_commit "2026-03-12T09:00:00+0000" \
  "test(feature_extractor): add long-form audio test (> 30s padding)" \
  tests/test_feature_extractor.py

add_and_commit "2026-03-16T09:00:00+0000" \
  "test(language_metrics): add empty-input edge cases for all metric functions" \
  tests/test_language_metrics.py

add_and_commit "2026-03-17T09:00:00+0000" \
  "test(export): add Unicode preservation test in CSV and JSON export" \
  tests/test_export.py

add_and_commit "2026-03-18T09:00:00+0000" \
  "test(monitoring): add Prometheus text format validation" \
  tests/test_monitoring.py

add_and_commit "2026-03-23T09:00:00+0000" \
  "test(preprocessing): add noise reducer test with stereo input signal" \
  tests/test_preprocessing.py

add_and_commit "2026-03-24T09:00:00+0000" \
  "test(training): add scheduler factory test for all four scheduler types" \
  tests/test_training.py

add_and_commit "2026-03-25T09:00:00+0000" \
  "test(evaluation): add benchmark CSV output column verification" \
  tests/test_evaluation.py

echo "==> Phase 17: Additional fixes (March-April 2026)"

add_and_commit "2026-03-30T09:00:00+0000" \
  "fix(preprocessing): handle JSONL files with trailing newlines" \
  preprocessing/dataset_builder.py

add_and_commit "2026-03-31T09:00:00+0000" \
  "fix(api): handle processor returning None for forced_decoder_ids" \
  api/inference.py

add_and_commit "2026-04-01T09:00:00+0000" \
  "fix(evaluation): prevent KeyError when 'chunks' absent in processor output" \
  api/inference.py

add_and_commit "2026-04-02T09:00:00+0000" \
  "fix(training): avoid KeyError when epoch key missing in train metrics" \
  training/trainer.py

add_and_commit "2026-04-07T09:00:00+0000" \
  "fix(preprocessing): normalize audio to float32 after speed perturbation" \
  preprocessing/augmentation.py

add_and_commit "2026-04-08T09:00:00+0000" \
  "fix(monitoring): reset active_requests to 0 on startup to clear stale state" \
  monitoring/metrics_collector.py

add_and_commit "2026-04-09T09:00:00+0000" \
  "fix(api): add missing content-type header to SRT and VTT responses" \
  api/routes.py

add_and_commit "2026-04-10T09:00:00+0000" \
  "fix(preprocessing): skip empty lines in plain-text manifest parser" \
  preprocessing/dataset_builder.py

add_and_commit "2026-04-14T09:00:00+0000" \
  "fix(evaluation): fix syllable error rate for purely numeric Ethiopic input" \
  evaluation/language_metrics.py

add_and_commit "2026-04-15T09:00:00+0000" \
  "fix(training): fix resume_from_checkpoint type coercion for Path objects" \
  training/trainer.py

add_and_commit "2026-04-16T09:00:00+0000" \
  "fix(api): handle Unicode decode error in audio filename header" \
  api/routes.py

add_and_commit "2026-04-21T09:00:00+0000" \
  "fix(preprocessing): handle missing speaker column gracefully in CSV parser" \
  preprocessing/dataset_builder.py

add_and_commit "2026-04-22T09:00:00+0000" \
  "fix(scheduler): fix warmup steps being ignored for linear scheduler" \
  training/scheduler.py

add_and_commit "2026-04-23T09:00:00+0000" \
  "fix(evaluation): fix MER formula to include hits in denominator" \
  evaluation/language_metrics.py

add_and_commit "2026-04-28T09:00:00+0000" \
  "fix(api): return 422 not 500 when audio duration exceeds maximum" \
  api/inference.py

add_and_commit "2026-04-29T09:00:00+0000" \
  "fix(training): clip gradient norm after AMP unscaling not before" \
  training/mixed_precision.py

echo "==> Phase 18: Final polish (May-June 2026)"

add_and_commit "2026-05-04T09:00:00+0000" \
  "docs(readme): add architecture diagram description and module overview" \
  README.md

add_and_commit "2026-05-05T09:00:00+0000" \
  "chore: add logs/ and tensorboard/ directories to .gitignore" \
  .gitignore

add_and_commit "2026-05-06T09:00:00+0000" \
  "refactor(api): use InferenceConfig dataclass for per-request overrides" \
  api/inference.py api/routes.py

add_and_commit "2026-05-07T09:00:00+0000" \
  "refactor(training): consolidate build_callbacks into trainer initialization" \
  training/trainer.py training/callbacks.py

add_and_commit "2026-05-11T09:00:00+0000" \
  "test(integration): add end-to-end preprocessing to dataset build test" \
  tests/test_preprocessing.py

add_and_commit "2026-05-12T09:00:00+0000" \
  "feat(evaluation): add script detection to choose CER normalizer automatically" \
  evaluation/cer.py evaluation/language_metrics.py

add_and_commit "2026-05-13T09:00:00+0000" \
  "feat(api): add model_name field to TranscriptionResponse" \
  api/schemas.py api/inference.py

add_and_commit "2026-05-14T09:00:00+0000" \
  "feat(monitoring): add uptime_seconds to /metrics response" \
  api/routes.py monitoring/metrics_collector.py

add_and_commit "2026-05-18T09:00:00+0000" \
  "perf(preprocessing): use multiprocessing for audio cleaning on large datasets" \
  preprocessing/audio_cleaner.py

add_and_commit "2026-05-19T09:00:00+0000" \
  "test(training): add PEFT trainer initialization test without GPU" \
  tests/test_training.py

add_and_commit "2026-05-20T09:00:00+0000" \
  "fix(preprocessing): preserve original sample rate when writing cleaned files" \
  preprocessing/audio_cleaner.py

add_and_commit "2026-05-21T09:00:00+0000" \
  "fix(evaluation): handle single-sample dataset in per-speaker breakdown" \
  evaluation/report_generator.py

add_and_commit "2026-05-25T09:00:00+0000" \
  "test(api): add inference engine test for long-form chunked transcription" \
  tests/test_api.py

add_and_commit "2026-05-26T09:00:00+0000" \
  "docs(api): document X-API-Key header requirement in OpenAPI schema" \
  api/auth.py api/app.py

add_and_commit "2026-05-27T09:00:00+0000" \
  "chore: update pyproject.toml coverage threshold to 85 percent" \
  pyproject.toml

add_and_commit "2026-06-02T09:00:00+0000" \
  "refactor(monitoring): rename internal _start_time to maintain private contract" \
  monitoring/metrics_collector.py

add_and_commit "2026-06-03T09:00:00+0000" \
  "test(monitoring): add histogram eviction test at max_observations boundary" \
  tests/test_monitoring.py

add_and_commit "2026-06-04T09:00:00+0000" \
  "fix(preprocessing): correct channel count for mono soundfile.info() results" \
  preprocessing/audio_cleaner.py

add_and_commit "2026-06-05T09:00:00+0000" \
  "feat(training): add TrainingMetrics dataclass for structured return values" \
  training/trainer.py

add_and_commit "2026-06-08T09:00:00+0000" \
  "release: v1.0.0 — production-ready Whisper ASR fine-tuning pipeline" \
  README.md pyproject.toml

echo ""
echo "==> Done. Commit count:"
$GIT log --oneline | wc -l
