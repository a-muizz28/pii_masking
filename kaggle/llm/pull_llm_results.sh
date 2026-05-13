#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PATH="$PATH:$HOME/.local/bin"
cd "$PROJECT_ROOT"

KAGGLE_USER=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
KERNEL_ID="$KAGGLE_USER/pii-masking-day-4-llm-inference"

echo "=== Pulling Day 4 LLM Results ==="
echo "Kernel: $KERNEL_ID"
echo ""

STATUS=$(kaggle kernels status "$KERNEL_ID" 2>&1)
echo "Status: $STATUS"
echo ""

mkdir -p results/day4_llm_inference
mkdir -p predictions/llm

echo "Downloading outputs to results/day4_llm_inference/ ..."
kaggle kernels output "$KERNEL_ID" -p results/day4_llm_inference/
echo ""

# Move cache file to predictions/llm/ if present
if [ -f results/day4_llm_inference/raw_outputs.jsonl ]; then
    mv results/day4_llm_inference/raw_outputs.jsonl predictions/llm/raw_outputs.jsonl
    echo "Moved raw_outputs.jsonl -> predictions/llm/"
fi

if [ -f results/day4_llm_inference/llm_metrics.json ]; then
    echo "=== DAY 4 LLM METRICS ==="
    python3 -c "
import json
with open('results/day4_llm_inference/llm_metrics.json') as f:
    m = json.load(f)
print(f'Span F1 Overall : {m[\"span_f1_overall\"]:.4f}')
print(f'Span F1 PER     : {m[\"span_f1_per\"]:.4f}')
print(f'Span F1 EMAIL   : {m[\"span_f1_email\"]:.4f}')
print(f'Token FPR PER   : {m[\"token_fpr_per\"]*100:.2f}%')
print(f'Token FNR PER   : {m[\"token_fnr_per\"]*100:.2f}%')
print(f'Token FPR EMAIL : {m[\"token_fpr_email\"]*100:.2f}%')
print(f'Token FNR EMAIL : {m[\"token_fnr_email\"]*100:.2f}%')
print(f'Redaction Leak  : {m[\"redaction_leak_rate\"]*100:.2f}%')
print(f'Parse Failures  : {m[\"n_parse_failures\"]}/{m[\"n_sentences\"]} ({m[\"parse_failure_rate\"]*100:.1f}%)')
"
else
    echo "llm_metrics.json not found. Files downloaded:"
    ls results/day4_llm_inference/
fi

if [ -f results/day4_llm_inference/day4_summary.md ]; then
    echo ""
    echo "=== COMPARISON SUMMARY ==="
    cat results/day4_llm_inference/day4_summary.md
fi
