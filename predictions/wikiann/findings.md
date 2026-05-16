# WikiANN Cross-Domain Evaluation Findings

Source results: `predictions/wikiann/eval_summary.json`  
Dataset: WikiANN English test split (`wikiann/en`, `split="test"`)  
Sentences: 3,652  
Entity coverage: PER only; no EMAIL spans are present.

## Approach Mapping

| Approach | Model | Role |
|---|---|---|
| A | DistilBERT-s42 | Lightweight encoder baseline and best OOD encoder in this run |
| B | DeBERTa-s7 | Best in-domain DeBERTa seed |
| C | LLaMA | Prompted local LLM baseline |

## OOD Results

| Approach | PER F1 | Token FPR | Token FNR | Redaction Leak Rate |
|---|---:|---:|---:|---:|
| A: DistilBERT-s42 | 0.777 | 0.021 | 0.215 | 19.3% |
| B: DeBERTa-s7 | 0.718 | 0.016 | 0.274 | 28.0% |
| C: LLaMA | 0.628 | 0.144 | 0.126 | 13.7% |

## Main Finding

Approach A is the strongest OOD encoder by PER F1. Compared with Approach B, it improves PER F1 by 0.060 and reduces leak rate by 8.7 percentage points. Approach C has the lowest leak rate because it over-predicts PERSON spans, but its token FPR is much higher than either encoder.

## Domain Shift

| Model | In-domain PER F1 | WikiANN PER F1 | Change |
|---|---:|---:|---:|
| DistilBERT-s42 | 0.982 | 0.777 | -0.205 |
| DeBERTa-s7 | 0.981 | 0.718 | -0.263 |

Both encoders lose substantial PER F1 on WikiANN. The likely causes are name-origin shift, sentence-style shift, and denser entity contexts than the WikiNeural-derived training split.

## Reporting Note

WikiANN has no EMAIL annotations in this setup. The `overall_f1` value in `eval_summary.json` is therefore set to PER F1 rather than a macro average that would include a non-existent EMAIL class.
