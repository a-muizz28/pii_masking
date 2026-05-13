#!/bin/bash
export PATH="$PATH:$HOME/.local/bin"

KAGGLE_USER=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
echo "Checking status for: $KAGGLE_USER/pii-masking-day-3-training"
echo ""
kaggle kernels status "$KAGGLE_USER/pii-masking-day-3-training"
echo ""
echo "Status meanings: queued=waiting for GPU | running=training now | complete=done | error=failed"
