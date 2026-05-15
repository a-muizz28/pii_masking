#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="results/wikiann"
mkdir -p "$OUT_DIR"
kaggle kernels output abdulmuizz28/pii-masking-day-7-independent-evaluation -p "$OUT_DIR"
