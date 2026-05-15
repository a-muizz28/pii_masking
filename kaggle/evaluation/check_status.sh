#!/usr/bin/env bash
# Check the status of the Day 5 evaluation kernel.
# Run from anywhere: bash kaggle/evaluation/check_status.sh

set -euo pipefail

KERNEL="abdulmuizz28/pii-masking-day-5-evaluation"

echo "=== Kernel status ==="
kaggle kernels status "$KERNEL"

echo ""
echo "To stream logs (once running):"
echo "  kaggle kernels output $KERNEL"
