#!/usr/bin/env bash
# Pull Day 7 independent evaluation outputs from Kaggle to the correct local paths.
# Run from the project root: bash kaggle/independent_eval/pull_results.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
KERNEL="abdulmuizz28/pii-masking-day-7-independent-evaluation"
TMP="$SCRIPT_DIR/kaggle_output_tmp"

echo "=== Pulling Day 7 outputs from Kaggle ==="
rm -rf "$TMP"
mkdir -p "$TMP"
kaggle kernels output "$KERNEL" -p "$TMP"

echo ""
echo "Downloaded structure:"
find "$TMP" -type f | sort

echo ""
echo "=== Copying outputs to project ==="

# predictions/wikiann/* (aggregate JSON + JSONL predictions)
WIKIANN_SRC="$TMP/predictions/wikiann"
WIKIANN_DST="$PROJECT_ROOT/predictions/wikiann"
if [ -d "$WIKIANN_SRC" ]; then
    mkdir -p "$WIKIANN_DST"
    for f in "$WIKIANN_SRC"/*; do
        [ -f "$f" ] || continue
        cp "$f" "$WIKIANN_DST/"
        echo "Saved: predictions/wikiann/$(basename "$f")"
    done
fi

# results/day7/bootstrap_results.json
BOOT_SRC="$TMP/results/day7/bootstrap_results.json"
if [ -f "$BOOT_SRC" ]; then
    mkdir -p "$PROJECT_ROOT/results/day7"
    cp "$BOOT_SRC" "$PROJECT_ROOT/results/day7/bootstrap_results.json"
    echo "Saved: results/day7/bootstrap_results.json"
fi

echo ""
echo "=== Pull complete ==="
rm -rf "$TMP"
