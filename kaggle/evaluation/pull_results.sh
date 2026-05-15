#!/usr/bin/env bash
# Pull Day 5 evaluation outputs from Kaggle and write them to the local project.
# Run from the project root: bash kaggle/evaluation/pull_results.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
KERNEL="abdulmuizz28/pii-masking-day-5-evaluation"
TMP="$SCRIPT_DIR/kaggle_output_tmp"

echo "=== Pulling Day 5 outputs from Kaggle ==="

rm -rf "$TMP"
mkdir -p "$TMP"
kaggle kernels output "$KERNEL" -p "$TMP"

echo ""
echo "Downloaded structure:"
find "$TMP" -type f | sort

echo ""
echo "=== Copying outputs to project ==="

# reports/results.json  (saved to /kaggle/working/reports/results.json)
if [ -f "$TMP/reports/results.json" ]; then
    mkdir -p "$PROJECT_ROOT/reports"
    cp "$TMP/reports/results.json" "$PROJECT_ROOT/reports/results.json"
    echo "Saved: reports/results.json"
fi

# results/day5/day5_summary.md  (saved to /kaggle/working/results/day5/day5_summary.md)
if [ -f "$TMP/results/day5/day5_summary.md" ]; then
    mkdir -p "$PROJECT_ROOT/results/day5"
    cp "$TMP/results/day5/day5_summary.md" "$PROJECT_ROOT/results/day5/day5_summary.md"
    echo "Saved: results/day5/day5_summary.md"
fi

# predictions/bootstrap_results.json  (saved to /kaggle/working/predictions/bootstrap_results.json)
if [ -f "$TMP/predictions/bootstrap_results.json" ]; then
    mkdir -p "$PROJECT_ROOT/predictions"
    cp "$TMP/predictions/bootstrap_results.json" "$PROJECT_ROOT/predictions/bootstrap_results.json"
    echo "Saved: predictions/bootstrap_results.json"
fi

# predictions/encoder/*.jsonl  (saved to /kaggle/working/predictions/encoder/)
if [ -d "$TMP/predictions/encoder" ]; then
    mkdir -p "$PROJECT_ROOT/predictions/encoder"
    for f in "$TMP/predictions/encoder"/*_predictions.jsonl; do
        [ -f "$f" ] || continue
        cp "$f" "$PROJECT_ROOT/predictions/encoder/"
        echo "Saved: predictions/encoder/$(basename "$f")"
    done
fi

echo ""
echo "=== Pull complete ==="

SUMMARY="$PROJECT_ROOT/results/day5/day5_summary.md"
if [ -f "$SUMMARY" ]; then
    echo ""
    cat "$SUMMARY"
fi

rm -rf "$TMP"
