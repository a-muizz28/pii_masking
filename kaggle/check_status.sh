#!/bin/bash
VENV_PATH="$(dirname "$(dirname "$(readlink -f "$0")")")/.venv"
source "$VENV_PATH/bin/activate"

KAGGLE_USER=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
echo "Checking status for: $KAGGLE_USER/pii-masking-day3-training"
echo ""
kaggle kernels status "$KAGGLE_USER/pii-masking-day3-training"
echo ""
echo "Status meanings: queued=waiting for GPU | running=training now | complete=done | error=failed"
