# Day 5 Evaluation Summary

DeBERTa seeds evaluated: [0, 7, 42]  (token-level rates and leak rate reported for seed 7)

## Main Comparison Table

| Metric | DeBERTa-v3-small (mean+/-std) | DistilBERT-cased (1 seed) | LLaMA-3.2-1B-Instruct (Template C) |
|-------------------------|------------------------------|---------------------------|--------------------------------------|
| Span F1 - PER           | 0.979+/-0.002 | 0.982 | 0.419 |
| Span F1 - EMAIL         | 1.000+/-0.000 | 0.999 | 0.426 |
| Macro Span F1           | 0.989+/-0.001 | 0.990 | 0.423 |
| Token FPR (binary)      | 0.001 | 0.001 | 0.165 |
| Token FNR (binary)      | 0.007 | 0.007 | 0.076 |
| PER FNR                 | 0.009 | 0.009 | 0.100 |
| EMAIL FNR               | 0.000 | 0.000 | 0.367 |
| Redaction Leak Rate     | 0.022 | 0.023 | 0.130 |

## Statistical Comparison (DeBERTa-v3-small seed 7 vs LLaMA)

- Observed F1 difference: 0.417
- 95% CI: [0.406, 0.429]
- p-value (one-sided): 0.000
- Conclusion: significant at alpha=0.05

## Statistical Comparison (DistilBERT-base-cased seed 42 vs LLaMA)

- Observed F1 difference: 0.416
- 95% CI: [0.405, 0.429]
- p-value (one-sided): 0.000
- Conclusion: significant at alpha=0.05
