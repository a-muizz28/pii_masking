#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PATH="$PATH:$HOME/.local/bin"

echo "=== Day 4 LLM Kaggle Push ==="
echo "Using kaggle user: $(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")"
echo ""

# Push the notebook only. The dataset source is already configured in kernel-metadata.json.
echo "[1/2] Pushing LLM inference kernel..."
cp "$PROJECT_ROOT/notebooks/04_llm_inference.ipynb" "$SCRIPT_DIR/04_llm_inference.ipynb"
cd "$SCRIPT_DIR"
kaggle kernels push
cd "$PROJECT_ROOT"
echo "  Kernel pushed. Inference will start shortly on Kaggle's T4 GPU."
echo ""

# Show status
echo "[2/2] Current kernel status:"
KAGGLE_USER=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
sleep 10
kaggle kernels status "$KAGGLE_USER/pii-masking-day-4-llm-inference" || echo "  (Status may take a minute to appear)"

echo ""
echo "=== DONE ==="
echo "Only the kernel was pushed. No dataset version was created."
echo ""
echo "To check progress anytime:"
echo "  bash kaggle/llm/check_status.sh"
echo ""
echo "When complete, pull results with:"
echo "  bash kaggle/llm/pull_llm_results.sh"
