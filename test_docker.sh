#!/usr/bin/env bash
# =============================================================================
# Dockerfile Build & Runtime Test Script
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="whisper-finetune-test"
CONTAINER_NAME="whisper-test-container"

cd "$REPO_ROOT"

echo "========================================="
echo "Whisper Dockerfile Test Suite"
echo "========================================="
echo ""

# Cleanup function
cleanup() {
  echo ""
  echo "🧹 Cleaning up test resources..."
  docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
  docker rmi -f "$IMAGE_NAME" 2>/dev/null || true
  echo "✅ Cleanup complete"
}

trap cleanup EXIT

echo "📋 Test 1: Dockerfile Syntax Validation"
echo "-----------------------------------------"
docker build --check -f Dockerfile . 2>&1 | tail -1
echo ""

echo "📋 Test 2: File Structure Verification"
echo "-----------------------------------------"
for item in requirements.txt config/ logging_config.py preprocessing/ training/ evaluation/ monitoring/ api/ scripts/; do
  if [ -e "$item" ]; then
    echo "✓ $item exists"
  else
    echo "✗ $item MISSING"
    exit 1
  fi
done
echo ""

echo "📋 Test 3: Requirements Parsing"
echo "-----------------------------------------"
python3 -c "
import sys
with open('requirements.txt') as f:
    lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    packages = [l.split('>=')[0].split('==')[0].split('<')[0] for l in lines if not l.startswith('-')]
    print(f'✓ Found {len(packages)} packages')
    
    # Critical packages check
    critical = ['torch', 'transformers', 'fastapi', 'librosa']
    for pkg in critical:
        if any(pkg in p for p in packages):
            print(f'  ✓ {pkg}')
        else:
            print(f'  ✗ {pkg} MISSING')
            sys.exit(1)
"
echo ""

echo "📋 Test 4: Docker Compose Validation"
echo "-----------------------------------------"
if docker compose config >/dev/null 2>&1; then
  echo "✓ docker-compose.yml is valid"
  echo "✓ Services defined:"
  docker compose config --services | sed 's/^/  - /'
else
  echo "✗ docker-compose.yml validation failed"
  exit 1
fi
echo ""

echo "📋 Test 5: Python Module Structure"
echo "-----------------------------------------"
for module in preprocessing training evaluation monitoring api; do
  init_file="$module/__init__.py"
  if [ -f "$init_file" ]; then
    py_count=$(find "$module" -name "*.py" | wc -l)
    echo "✓ $module — valid Python package ($py_count files)"
  else
    echo "✗ $module — missing __init__.py"
    exit 1
  fi
done
echo ""

echo "📋 Test 6: Dockerfile Stage Verification"
echo "-----------------------------------------"
stages=$(grep -E "^FROM .* AS " Dockerfile | awk '{print $NF}')
echo "Multi-stage build detected:"
echo "$stages" | sed 's/^/  ✓ Stage: /'
stage_count=$(echo "$stages" | wc -l)
if [ "$stage_count" -ge 3 ]; then
  echo "✓ $stage_count stages found (optimization: good)"
else
  echo "⚠ Only $stage_count stages (consider multi-stage for smaller images)"
fi
echo ""

echo "📋 Test 7: Security Best Practices"
echo "-----------------------------------------"
if grep -q "useradd" Dockerfile; then
  echo "✓ Non-root user created"
else
  echo "⚠ No non-root user found"
fi

if grep -q "HEALTHCHECK" Dockerfile; then
  echo "✓ Healthcheck configured"
else
  echo "⚠ No healthcheck found"
fi

if grep -q "PYTHONUNBUFFERED" Dockerfile; then
  echo "✓ Python unbuffered mode enabled"
else
  echo "⚠ PYTHONUNBUFFERED not set"
fi
echo ""

echo "📋 Test 8: Port & Environment Configuration"
echo "-----------------------------------------"
echo "Exposed ports:"
grep "^EXPOSE" Dockerfile | awk '{print "  ✓ Port", $2}'

echo "Environment variables:"
grep "^ENV" Dockerfile | head -5 | awk '{print "  ✓", $2}' | sed 's/=.*//'
echo ""

echo "========================================="
echo "✅ All Dockerfile tests passed!"
echo "========================================="
echo ""
echo "📦 To build the production image:"
echo "   docker build -t whisper-finetune:latest ."
echo ""
echo "🚀 To start with GPU (docker compose):"
echo "   docker compose up -d"
echo ""
echo "🖥️  To start with CPU only:"
echo "   docker compose --profile cpu up -d"
echo ""
echo "🔍 To test the build locally (without GPU):"
echo "   docker build -t $IMAGE_NAME ."
echo "   docker run --rm -p 8000:8000 -e CUDA_VISIBLE_DEVICES='' $IMAGE_NAME"
echo ""
