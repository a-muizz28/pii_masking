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

## Locked Design Decisions

| Decision | Value |
|---|---|
| Email tokenization | 90% single-token, 10% multi-token |
| Label alignment | Strategy B (B→I propagation on all subwords) |
| Encoder (primary) | DeBERTa-v3-small (3 seeds: 42, 0, 7) |
| Encoder (baseline) | DistilBERT-base-cased (1 seed: 42) |
| LLM model | Llama-3.2-1B-Instruct (Q4_K_M GGUF via llama-cpp-python) |
| LLM prompt strategy | Template C only |
| Eval library | seqeval (primary) |
| Masking placeholders | [NAME], [EMAIL] |
| Injection rate | 0.6× PER-sentence count |
| Random seed | 42 (primary), 0 and 7 (multi-seed) |
| Train/val split | 85/15 stratified by entity-density |
| Statistical test | Paired bootstrap (1000 iterations, 95% CI) |

## Quickstart

```bash
python scripts/01_validate_data.py   # produces data/processed/*.parquet + report
python scripts/02_inject_emails.py   # produces train/val/test with emails
pytest tests/                        # must all pass before proceeding
```
