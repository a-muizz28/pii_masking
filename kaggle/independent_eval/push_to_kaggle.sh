#!/usr/bin/env bash
set -euo pipefail

# Push Day 7 independent evaluation notebook to Kaggle.
# Run from the project root after adding Day 3 and Day 4 outputs as kernel sources.

mkdir -p kaggle/independent_eval
cp notebooks/07_independent_eval.ipynb kaggle/independent_eval/07_independent_eval.ipynb
rm -rf kaggle/independent_eval/src
cp -R src kaggle/independent_eval/src
kaggle kernels push -p kaggle/independent_eval
