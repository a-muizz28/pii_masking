# pii_masking

PII masking project for detecting and masking personally identifiable information, focused on PERSON names and EMAIL addresses. The final report compares three approaches:

| Approach | Model | Role |
|---|---|---|
| A | DistilBERT-s42 | Lightweight encoder baseline and best WikiANN encoder in this run |
| B | DeBERTa-s7 | Best in-domain DeBERTa seed |
| C | LLaMA | Prompted local LLM baseline |

## What

This repository is structured for:

- data validation and preprocessing
- synthetic email injection
- encoder training and evaluation
- LLM inference and masking
- metrics and error analysis

## Numbering Convention

The numeric prefixes show the order in which the project stages were created and executed. Matching `src/pii_masking/day*_*.py` modules contain the reusable implementation used by the numbered scripts and notebooks.

| Stage | Runnable / notebook | Uses source module(s) | Main output |
|---|---|---|---|
| Day 1 | `notebooks/01_day1_overview.ipynb`, `scripts/01_validate_data.py` | `src/pii_masking/day1_data.py` | clean parquet splits, validation report, checksums, token-length histogram |
| Day 1 | `notebooks/01_day1_overview.ipynb`, `scripts/01_inject_emails.py` | `src/pii_masking/day1_data.py`, `src/pii_masking/day1_injection.py` | injected train/validation/test parquet files, injection report |
| Day 2 | `notebooks/02_eda.ipynb`, `scripts/02_preprocess.py` | `src/pii_masking/day2_preprocessing.py` | `data/processed/hf_dataset/` |
| Day 2 | `notebooks/02_eda.ipynb`, `scripts/02_smoke_test.py` | `src/pii_masking/day2_metrics.py` | `models/smoke_test_distilbert/smoke_test_results.json` |
| Day 3 | `notebooks/03_encoder_training.ipynb` | Kaggle notebook code + Day 2 dataset | encoder checkpoints, `results/day3_encoder_training/training_summary.json` |
| Day 4 | `scripts/04_llm_inference.py`, `notebooks/04_llm_inference.ipynb` | `src/pii_masking/day4_llm_inference.py`, `src/pii_masking/day4_evaluate.py` | LLaMA predictions/cache, `results/day4_llm_inference/llm_metrics.json`, Day 4 summary |
| Day 5 | `scripts/05_evaluate_all.py`, `src/pii_masking/day5_masking.py` | `src/pii_masking/day5_masking.py` | final comparison metrics, masking leak rate, bootstrap results |
| Day 6 | `notebooks/06_error_analysis.ipynb`, `scripts/06_error_analysis.py`, `src/pii_masking/day6_error_analysis.py` | `src/pii_masking/day6_error_analysis.py` | error bucket counts, annotation TSVs, error-distribution figure |
| Day 7 | `notebooks/07_independent_eval.ipynb`, `scripts/07_eval_independent.py`, `src/pii_masking/day7_eval_independent.py` | `src/pii_masking/day7_eval_independent.py` | WikiANN cross-domain metrics |

`scripts/` are command-line entrypoints. `src/pii_masking/` contains reusable implementation logic. `notebooks/` documents EDA artifacts and holds Kaggle/GPU experiments. `tests/` validates the reusable source modules.

## Configs

All tuneable values are centralised in `configs/`. No magic numbers are hardcoded in source files.

| File | Loaded by | Keys used |
|---|---|---|
| `configs/label_config.json` | `scripts/02_preprocess.py` | `label2id`, `max_length`, `model_name_smoke_test` (tokenizer) |
| `configs/encoder_distilbert.yaml` | `scripts/02_smoke_test.py` | `model_name`, `batch_size`, `learning_rate` (`epochs` overridden to 1 for the smoke run) |
| `configs/encoder_deberta_v3_small.yaml` | `notebooks/03_encoder_training.ipynb` (reference) | `model_name`, `batch_size`, `learning_rate`, `epochs` |
| `configs/llm_template_C.yaml` | `src/pii_masking/day4_llm_inference.py` | `system_prompt`, `temperature` |

To change the LLM prompt, edit `configs/llm_template_C.yaml`. To change encoder hyperparameters for the smoke test, edit `configs/encoder_distilbert.yaml`. Label schema and tokenisation settings live in `configs/label_config.json`.

## How To Run

```bash
python scripts/01_validate_data.py
python scripts/01_inject_emails.py
python scripts/02_preprocess.py
python scripts/02_smoke_test.py
python scripts/04_llm_inference.py
```

Encoder training is implemented in `notebooks/03_encoder_training.ipynb` for Kaggle/GPU execution. Day 5, Day 6, and Day 7 outputs are produced by `scripts/05_evaluate_all.py`, `scripts/06_error_analysis.py`, and `scripts/07_eval_independent.py`.

## Step-By-Step Flow

1. Run `scripts/01_validate_data.py` to validate raw WikiNeural JSON, repair BIO issues, split train/validation, and write clean parquet files.
2. Run `scripts/01_inject_emails.py` to inject synthetic email entities into the clean splits using `data/injection_config.json`.
3. Run `scripts/02_preprocess.py` to tokenize injected splits and align labels using Strategy B (reads `configs/label_config.json`).
4. Run `scripts/02_smoke_test.py` locally to confirm the encoder pipeline works on a small DistilBERT subset before expensive training (reads `configs/encoder_distilbert.yaml`).
5. Run `notebooks/03_encoder_training.ipynb` on Kaggle/GPU for the full DeBERTa/DistilBERT encoder experiments.
6. Run `scripts/04_llm_inference.py` locally or `notebooks/04_llm_inference.ipynb` on Kaggle/GPU for the LLaMA zero-shot pipeline and Day 4 comparison against encoder metrics (prompt and temperature read from `configs/llm_template_C.yaml`).
7. Run `scripts/05_evaluate_all.py` to produce `results/day5/results.json` and `results/day5/bootstrap_results.json`.
8. Use `notebooks/06_error_analysis.ipynb` or `scripts/06_error_analysis.py` to compare Approach A, B, and C error buckets.
9. Use `notebooks/07_independent_eval.ipynb` for the cached WikiANN JSON review, or `scripts/07_eval_independent.py` to regenerate WikiANN metrics.

## Key Results

Results will be written to:

- `predictions/`
- `errors/`
- `models/`
- `reports/figures/`
- `results/day3_encoder_training/training_summary.json`
- `results/day4_llm_inference/llm_metrics.json`
- `results/day5/results.json`
- `results/day5/bootstrap_results.json`
- `results/day6/error_buckets.json`
- `predictions/wikiann/*.json`

## Locked Design Decisions

| Decision | Value |
|---|---|
| Email tokenization | 90% single-token, 10% multi-token |
| Label alignment | Strategy B (B->I propagation on all subwords) |
| Approach A | DistilBERT-base-cased seed 42 |
| Approach B | DeBERTa-v3-small seed 7 |
| Approach C | Llama-3.2-1B-Instruct (Q4_K_M GGUF via llama-cpp-python) |
| Encoder training set | DeBERTa-v3-small seeds 0, 7, 42; DistilBERT seed 42 |
| LLM model | Llama-3.2-1B-Instruct (Q4_K_M GGUF via llama-cpp-python) |
| LLM prompt strategy | Template C only |
| Eval library | seqeval (primary) |
| Masking placeholders | [NAME], [EMAIL] |
| Injection rate | 0.6x PER-sentence count |
| Random seed | 42 (primary), 0 and 7 (multi-seed) |
| Train/val split | 85/15 stratified by entity-density |
| Statistical test | Paired bootstrap (1000 iterations, 95% CI) |

## Quickstart

```bash
python scripts/01_validate_data.py   # produces data/processed/*.parquet + report
python scripts/01_inject_emails.py   # produces train/val/test with emails
python scripts/02_preprocess.py      # produces data/processed/hf_dataset
pytest tests/                        # must all pass before proceeding
```
