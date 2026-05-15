#!/usr/bin/env bash
# Push Day 5 evaluation dataset (code only, ~20 KB) + kernel to Kaggle.
# Model weights come from the Day 3 kernel source.
# LLM predictions come from the Day 4 kernel source.
# No large file upload needed.
#
# Run from the project root: bash kaggle/evaluation/push_to_kaggle.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATASET_DIR="$SCRIPT_DIR/dataset_staging"

echo "=== Day 5 Kaggle push ==="
echo "Project root : $PROJECT_ROOT"

# Stage code-only dataset (scripts + src, no model weights)
rm -rf "$DATASET_DIR"
mkdir -p "$DATASET_DIR"
cp "$SCRIPT_DIR/dataset-metadata.json" "$DATASET_DIR/"

mkdir -p "$DATASET_DIR/scripts"
cp "$PROJECT_ROOT/scripts/05_evaluate_all.py" "$DATASET_DIR/scripts/"

mkdir -p "$DATASET_DIR/src/pii_masking"
cp "$PROJECT_ROOT/src/pii_masking/day5_masking.py"  "$DATASET_DIR/src/pii_masking/"
cp "$PROJECT_ROOT/src/pii_masking/day4_evaluate.py" "$DATASET_DIR/src/pii_masking/"
[ -f "$PROJECT_ROOT/src/pii_masking/__init__.py" ] && \
    cp "$PROJECT_ROOT/src/pii_masking/__init__.py" "$DATASET_DIR/src/pii_masking/"

echo "Staged dataset size: $(du -sh "$DATASET_DIR" | cut -f1)  (code only - models come from Day 3 kernel)"

# Upload dataset
if kaggle datasets status abdulmuizz28/pii-masking-day5-eval &>/dev/null; then
    echo ""
    echo "=== Updating dataset to new version ==="
    kaggle datasets version -p "$DATASET_DIR" --dir-mode zip -m "Day 5 code-only eval assets"
else
    echo ""
    echo "=== Creating dataset (first time) ==="
    kaggle datasets create -p "$DATASET_DIR" --dir-mode zip
fi

# Push kernel
echo ""
echo "=== Pushing kernel ==="
cd "$SCRIPT_DIR"
kaggle kernels push

echo ""
echo "=== Done ==="
echo "Monitor : kaggle kernels status abdulmuizz28/pii-masking-day-5-evaluation"
echo "Logs    : kaggle kernels output abdulmuizz28/pii-masking-day-5-evaluation"
