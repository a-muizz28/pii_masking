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

# Copy per-model result JSONs back into models/
for label in deberta_seed42 deberta_seed0 deberta_seed7 distilbert_seed42; do
    src="results/day3_encoder_training/${label}_result.json"
    if [ -f "$src" ]; then
        cp "$src" "models/${label}/${label}_result.json"
        echo "  Copied $src -> models/${label}/"
    fi
done
echo ""

if [ -f results/day3_encoder_training/training_summary.json ]; then
    echo "=== TRAINING SUMMARY ==="
    python3 -c "
import json
with open('results/day3_encoder_training/training_summary.json') as f:
    s = json.load(f)
mean_key = 'deberta_val_f1_mean' if 'deberta_val_f1_mean' in s else 'deberta_test_f1_mean'
std_key  = 'deberta_val_f1_std'  if 'deberta_val_f1_std'  in s else 'deberta_test_f1_std'
fpr_key  = 'deberta_val_fpr_mean' if 'deberta_val_fpr_mean' in s else 'deberta_test_fpr_mean'
fnr_key  = 'deberta_val_fnr_mean' if 'deberta_val_fnr_mean' in s else 'deberta_test_fnr_mean'
split    = 'val' if 'deberta_val_f1_mean' in s else 'test'
print(f'DeBERTa-v3-small {split} F1: {s[mean_key]:.4f} +/- {s[std_key]:.4f}')
print(f'DeBERTa Token FPR (mean): {s[fpr_key]:.4f}')
print(f'DeBERTa Token FNR (mean): {s[fnr_key]:.4f}')
print()
print('Per-run breakdown:')
for r in s['runs']:
    m = r.get('val_metrics') or r.get('test_metrics', {})
    print(f'  {r[\"run_label\"]:25s}  f1={m.get(\"eval_overall_f1\",0):.4f}'
          f'  per_f1={m.get(\"eval_per_f1\",0):.4f}'
          f'  email_f1={m.get(\"eval_email_f1\",0):.4f}'
          f'  fpr={m.get(\"eval_token_fpr\",0):.4f}'
          f'  fnr={m.get(\"eval_token_fnr\",0):.4f}')
"
else
    echo "training_summary.json not found in results."
    echo "Files downloaded:"
    ls results/day3_encoder_training/
fi
