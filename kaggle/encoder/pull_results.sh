#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PATH="$PATH:$HOME/.local/bin"
cd "$PROJECT_ROOT"

KAGGLE_USER=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
KERNEL_ID="$KAGGLE_USER/pii-masking-day-3-training"

echo "=== Pulling Day 3 Results ==="
echo "Kernel: $KERNEL_ID"
echo ""

STATUS=$(kaggle kernels status "$KERNEL_ID" 2>&1)
echo "Status: $STATUS"
echo ""

mkdir -p results/day3_encoder_training
echo "Downloading outputs to results/day3_encoder_training/ ..."
kaggle kernels output "$KERNEL_ID" -p results/day3_encoder_training/
echo ""

if [ -f results/day3_encoder_training/training_summary.json ]; then
    echo "=== TRAINING SUMMARY ==="
    python3 -c "
import json
with open('results/day3_encoder_training/training_summary.json') as f:
    s = json.load(f)
print(f'DeBERTa-v3-small Test F1: {s[\"deberta_test_f1_mean\"]:.4f} +/- {s[\"deberta_test_f1_std\"]:.4f}')
print(f'DeBERTa Token FPR (mean): {s[\"deberta_test_fpr_mean\"]:.4f}')
print(f'DeBERTa Token FNR (mean): {s[\"deberta_test_fnr_mean\"]:.4f}')
print()
print('Per-run breakdown:')
for r in s['runs']:
    tm = r['test_metrics']
    print(f'  {r[\"run_label\"]:25s}  test_f1={tm.get(\"eval_overall_f1\",0):.4f}'
          f'  per_f1={tm.get(\"eval_per_f1\",0):.4f}'
          f'  email_f1={tm.get(\"eval_email_f1\",0):.4f}'
          f'  fpr={tm.get(\"eval_token_fpr\",0):.4f}'
          f'  fnr={tm.get(\"eval_token_fnr\",0):.4f}')
"
else
    echo "training_summary.json not found in results."
    echo "Files downloaded:"
    ls results/day3_encoder_training/
fi
