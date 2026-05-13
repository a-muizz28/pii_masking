# Day 4 Results - LLaMA Zero-Shot vs DeBERTa Fine-Tuned

## Inference Setup
- Model: Llama-3.2-1B-Instruct (Q4_K_M GGUF, T4 GPU, n_gpu_layers=-1)
- Prompt: Template C (zero-shot, definition-heavy, no fine-tuning)
- Temperature: 0.0 (deterministic)
- Test sentences: 3650
- Elapsed inference time: 1515.3s (0.42s/sent)

## Model Comparison

| Metric                   | DeBERTa-v3-small (mean +/- std, 3 seeds) | DistilBERT-cased (1 seed) | LLaMA-3.2-1B zero-shot |
|--------------------------|------------------------------------------|----------------------------|------------------------|
| Span F1 - PER            | 0.922 +/- 0.005 | 0.978 | 0.419 |
| Span F1 - EMAIL          | 0.946 +/- 0.040 | 0.998 | 0.426 |
| Span F1 - Overall        | 0.929 +/- 0.015 | 0.984 | 0.423 |
| Token FPR                | 0.8% +/- 0.0% | 0.3% | N/A |
| Token FNR                | 1.7% +/- 0.4% | 0.5% | N/A |
| Token FPR (PER)          | N/A | N/A | 12.3% |
| Token FNR (PER)          | N/A | N/A | 10.0% |
| Token FPR (EMAIL)        | N/A | N/A | 4.6% |
| Token FNR (EMAIL)        | N/A | N/A | 36.7% |
| Redaction Leak Rate      | N/A | N/A | 9.1% |
| Parse Failure Rate       | N/A | N/A | 0.0% |

## Notes
- Encoder metrics are copied from `results/day3/training_summary.json` test metrics. Their token FPR/FNR are aggregate token-level rates, not split by PER vs EMAIL, so the class-specific encoder cells are marked N/A.
- LLaMA inference is far slower than the encoder baselines even with GPU offload.
- Parse failures: 0 out of 3650 sentences (see parse_failures.jsonl)
- No fine-tuning was applied to LLaMA; results reflect zero-shot generalization at 1B scale.
