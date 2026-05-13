#!/bin/bash
export PATH="$PATH:$HOME/.local/bin"

KAGGLE_USER=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
echo "Checking status for: $KAGGLE_USER/pii-masking-day-4-llm-inference"
echo ""
kaggle kernels status "$KAGGLE_USER/pii-masking-day-4-llm-inference"
echo ""
echo "Status meanings: queued=waiting for GPU | running=inference now | complete=done | error=failed"
