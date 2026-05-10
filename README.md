# pii_masking

PII masking project scaffold for detecting and masking personally identifiable information, with a focus on names and email addresses using both an encoder-based model and an LLM-based pipeline.

## What

This repository is structured for:

- data validation and preprocessing
- synthetic email injection
- encoder training and evaluation
- LLM inference and masking
- metrics and error analysis

## How To Run

```bash
python scripts/01_validate_data.py
python scripts/02_inject_emails.py
python scripts/03_train_encoder.py
python scripts/04_run_llm.py
python scripts/05_evaluate.py
python scripts/06_error_analysis.py
```

## Key Results

Results will be written to:

- `predictions/`
- `errors/`
- `models/`
- `reports/figures/`
