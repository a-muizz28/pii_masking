#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Ensure kaggle is on PATH (WSL installs to ~/.local/bin)
export PATH="$PATH:$HOME/.local/bin"

echo "=== Day 3 Kaggle Push ==="
echo "Using kaggle user: $(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")"
echo ""

# Step 1: Upload/update the dataset
echo "[1/3] Uploading processed dataset to Kaggle..."
cd "$SCRIPT_DIR/dataset_upload"

if kaggle datasets list --user abdulmuizz28 2>/dev/null | grep -q "pii-masking-processed-dataset"; then
    echo "  Dataset exists — creating new version..."
    kaggle datasets version -m "Day 2 output $(date +%Y-%m-%d)" --dir-mode zip
else
    echo "  First upload — creating dataset..."
    kaggle datasets create --dir-mode zip
fi
cd "$PROJECT_ROOT"
echo "  Dataset upload queued. Waiting 30s for Kaggle to process..."
sleep 30
echo ""

# Step 2: Push the kernel
echo "[2/3] Pushing training kernel..."
cd "$SCRIPT_DIR"
kaggle kernels push
cd "$PROJECT_ROOT"
echo "  Kernel pushed. Training will start shortly on Kaggle's T4 GPU."
echo ""

# Step 3: Show status
echo "[3/3] Current kernel status:"
KAGGLE_USER=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
sleep 10
kaggle kernels status "$KAGGLE_USER/pii-masking-day3-training" || echo "  (Status may take a minute to appear)"

echo ""
echo "=== DONE ==="
echo "You can now CLOSE YOUR LAPTOP. Training runs on Kaggle's servers."
echo ""
echo "To check progress anytime:"
echo "  bash kaggle/check_status.sh"
echo ""
echo "When complete, pull results with:"
echo "  bash kaggle/pull_results.sh"
