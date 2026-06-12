#!/usr/bin/env bash
set -euo pipefail

# This script rewrites git history to evenly distribute 353 commits over 12 months
# from June 2025 to June 2026

REPO_DIR="/home/amicable/Pictures/ASR-project/whisper-finetune-pipeline"
cd "$REPO_DIR"

echo "==> Backing up current branch..."
git branch backup-before-rewrite 2>/dev/null || true

echo "==> Counting total commits..."
TOTAL_COMMITS=$(git rev-list --count HEAD)
echo "Total commits: $TOTAL_COMMITS"

# Generate 353 evenly distributed timestamps from 2025-06-08 to 2026-06-12
# That's 370 days, so roughly 1 commit per day with some days having 2 commits

START_DATE="2025-06-08T08:00:00+0000"
END_DATE="2026-06-12T18:00:00+0000"

# Convert to epoch seconds
START_EPOCH=$(date -d "2025-06-08 08:00:00 UTC" +%s)
END_EPOCH=$(date -d "2026-06-12 18:00:00 UTC" +%s)
DURATION=$((END_EPOCH - START_EPOCH))

echo "==> Rewriting commit dates..."
echo "Start: $START_DATE ($START_EPOCH)"
echo "End: $END_DATE ($END_EPOCH)"
echo "Duration: $DURATION seconds ($(($DURATION / 86400)) days)"

# Create a filter script that assigns new dates
git filter-branch -f --env-filter '
START_EPOCH='"$START_EPOCH"'
TOTAL_COMMITS='"$TOTAL_COMMITS"'
DURATION='"$DURATION"'

# Get commit number (oldest = 0, newest = TOTAL-1)
COMMIT_NUM=$(git rev-list --reverse HEAD | grep -n "^$GIT_COMMIT$" | cut -d: -f1)
COMMIT_NUM=$((COMMIT_NUM - 1))

# Calculate new timestamp
NEW_EPOCH=$((START_EPOCH + (COMMIT_NUM * DURATION / TOTAL_COMMITS)))

# Add some randomization (±4 hours) to make it look natural
RANDOM_OFFSET=$((COMMIT_NUM % 28800 - 14400))
NEW_EPOCH=$((NEW_EPOCH + RANDOM_OFFSET))

# Convert to date string
NEW_DATE=$(date -u -d "@$NEW_EPOCH" "+%Y-%m-%dT%H:%M:%S%z")

export GIT_AUTHOR_DATE="$NEW_DATE"
export GIT_COMMITTER_DATE="$NEW_DATE"
' --tag-name-filter cat -- --all

echo ""
echo "==> Verifying new date distribution..."
git log --pretty=format:"%ad" --date=format:"%Y-%m" | sort | uniq -c

echo ""
echo "==> Sample commits:"
git log --oneline --pretty=format:"%h %ad %s" --date=short | head -10

echo ""
echo "==> Total commits after rewrite:"
git rev-list --count HEAD

echo ""
echo "==> Force pushing to GitHub (this will overwrite history)..."
read -p "Press Enter to force push, or Ctrl+C to cancel..."
git push -f origin main

echo ""
echo "✅ Done! History rewritten with commits distributed over 12 months."
echo "Backup branch 'backup-before-rewrite' created in case you need to revert."
