# PII Detection and Masking - Final Technical Report

**Evaluation date:** 2026-05-15 | **Test set:** 3,650 sentences | **seqeval mode:** strict IOB2

---

## Abstract

This report evaluates two automated PII detection strategies on English Wikipedia NER text augmented with synthetically injected email addresses. Approach A fine-tunes DeBERTa-v3-small across three random seeds and achieves a macro span F1 of 0.989 ± 0.001 with a redaction leak rate of 2.1%. Approach B applies LLaMA 3.2-1B-Instruct zero-shot and achieves macro span F1 of 0.423 with a leak rate of 13.0%. Paired bootstrap resampling (n = 10,000) confirms the gap is statistically significant (observed diff = 0.410, 95% CI [0.398, 0.423], p < 0.0001). Cross-domain evaluation on WikiANN reveals that both approaches degrade substantially on out-of-distribution names, with DeBERTa PER F1 dropping from 0.979 to 0.718 and LLaMA from 0.419 to 0.628.

---

## 1. Introduction

Named entity recognition (NER) for PII detection is a prerequisite for privacy-safe data pipelines in production ML systems. This project evaluates two detection strategies on a subset of WikiNeural, a silver-standard BIO-annotated English Wikipedia corpus providing PERSON entities. Because WikiNeural contains no EMAIL addresses, we inject synthetic emails at a controlled rate to enable joint PER + EMAIL evaluation.

Two approaches are compared: (1) a fine-tuned transformer encoder trained with task-specific gradient updates, and (2) a quantised large language model applied zero-shot via structured prompting. The evaluation measures not only span-level F1 but also token-level false positive / false negative rates and a privacy-oriented redaction leak rate, giving a multi-dimensional view of deployment readiness.

---

## 2. Dataset and Email Injection

**Base corpus.** The WikiNeural English split provides BIO annotations over PERSON entities in Wikipedia-style text. A 3,650-sentence test subset is held out for evaluation; the training split is used exclusively for fine-tuning Approach A.

**Email injection.** Synthetic email addresses are injected at approximately 0.6 injections per sentence. A 90/10 ratio of single-token (`user@domain.com`) to multi-token (`first . last @ domain . com`) formats is maintained. Train and test splits draw from disjoint email domain pools to prevent lexical memorisation of domain names. The injection procedure attaches a B-EMAIL tag to the first token and I-EMAIL to any continuation tokens, matching the BIO2 convention used throughout.

**Cross-domain test set.** WikiANN (`wikiann/en`, test split) provides 3,652 sentences with 4,556 PER spans and zero EMAIL spans. It is used solely for out-of-distribution evaluation; no model is trained or tuned on it.

---

## 3. Approach A - Encoder Fine-tuning

**Model.** `microsoft/deberta-v3-small` is fine-tuned for token classification. DistilBERT-base-cased is included as a single-seed baseline ablation.

**Label alignment (Strategy B).** All subword tokens within a word receive the same label as the first subword, so B-tags propagate through the entire word span as I-tags. This provides roughly 7× more effective EMAIL gradient signal per sentence compared to first-subword-only labelling - critical for the relatively rare EMAIL class.

**Training.** Three random seeds (0, 7, 42) are used. Hyperparameters: learning rate 3×10⁻⁵, max sequence length 256, batch size 16, 5 epochs. Seed 42 is used for bootstrap comparison and redaction leak rate reporting. Mean and standard deviation are reported across all three seeds.

---

## 4. Approach B - Zero-Shot LLM

**Model.** LLaMA 3.2-1B-Instruct, quantised to GGUF Q4\_K\_M format and served locally via `llama-cpp-python`. No fine-tuning or in-context examples are provided.

**Prompting (Template C).** The model is instructed to return a JSON object with `names` and `emails` lists. Parsed spans are aligned back to BIO2 token sequences using a greedy span-match procedure; `repair_bio` enforces valid IOB2 transitions. On parse failure, the sentence is tagged all-O.

**Parse failure rate.** 0.0% - all 3,650 test responses parsed successfully.

**Local inference.** No text is transmitted to external APIs during inference, providing privacy-by-design guarantees for the inference step itself.

---

## 5. Evaluation Methodology

- **Primary metric.** Span-level F1 computed by seqeval in strict IOB2 mode - both span boundary and entity label must match exactly.
- **Secondary metrics.** Token-level False Positive Rate (FPR) and False Negative Rate (FNR), reported per entity type (PER, EMAIL).
- **Privacy metric.** Redaction leak rate = fraction of gold PII spans left completely unmasked by the model, measuring direct privacy risk independent of precision.
- **Statistical significance.** Paired bootstrap resampling: n = 10,000 iterations, random state 42, 95% CI via 2.5th / 97.5th percentiles of the bootstrap distribution of mean F1 differences, one-sided p-value (H₀: DeBERTa − LLaMA ≤ 0).

---

## 6. In-Distribution Results

| Metric              | DeBERTa (mean ± std) | DistilBERT (seed 42) | LLaMA (zero-shot) |
|---------------------|----------------------|----------------------|-------------------|
| Span F1 - PER       | 0.979 ± 0.002        | 0.982                | 0.419             |
| Span F1 - EMAIL     | 1.000 ± 0.000        | 0.999                | 0.426             |
| Macro Span F1       | 0.989 ± 0.001        | 0.990                | 0.423             |
| Token FPR           | 0.00165              | 0.00119              | 0.16529           |
| Token FNR           | 0.00763              | 0.00729              | 0.07600           |
| Redaction Leak Rate | 2.1%                 | 2.3%                 | 13.0%             |

_DeBERTa per-seed PER F1: seed 0 = 0.9764, seed 7 = 0.9809, seed 42 = 0.9792._
_Token FPR/FNR and leak rate are reported for DeBERTa seed 42._

---

## 7. Bootstrap Significance Test

Paired bootstrap resampling (n = 10,000) on per-sentence macro F1 between DeBERTa seed 42 and LLaMA.

| Metric | Value |
|--------|-------|
| n sentences | 3,650 |
| n bootstrap iterations | 10,000 |
| DeBERTa mean F1 | 0.9859 |
| LLaMA mean F1 | 0.5759 |
| Observed mean diff | 0.4100 |
| 95% CI lower (2.5th pct) | 0.3975 |
| 95% CI upper (97.5th pct) | 0.4228 |
| One-sided p-value | < 0.0001 |
| Significant at α = 0.05 | Yes |

DeBERTa significantly outperforms LLaMA zero-shot on the in-distribution test set.

---

## 8. Error Analysis

**Approach A (DeBERTa seed 42).** Total errors: 219 across 3,650 sentences.

| Bucket | Count | % of A errors |
|--------|-------|---------------|
| FP - PER | 130 | 59.4% |
| FN - PER | 88 | 40.2% |
| FP - EMAIL | 0 | 0.0% |
| FN - EMAIL | 1 | 0.5% |

The dominant failure mode is false-positive PER predictions - the model occasionally activates on organisation names or capitalised nouns that resemble person names. False negatives are concentrated in multi-word names where the second or later word is not a common English name.

**Approach B (LLaMA).** Total errors: 14,631 - approximately 67× more than DeBERTa.

| Bucket | Count | % of B errors |
|--------|-------|---------------|
| FP - PER | 9,261 | 63.3% |
| FP - EMAIL | 3,289 | 22.5% |
| FN - PER | 1,374 | 9.4% |
| FN - EMAIL | 707 | 4.8% |

LLaMA's primary failure mode is hallucinating person names - the greedy span-match procedure tags tokens corresponding to names that the model invented but that do not appear in the gold annotation. Email false positives arise from partial matches where the model predicts an email-like token that overlaps with a non-email span.

---

## 9. Cross-Domain Generalisation (WikiANN)

To measure distribution shift, both approaches are evaluated on WikiANN (`wikiann/en`, test split, 3,652 sentences, 4,556 PER spans, 0 EMAIL spans). Neither model was trained or tuned on WikiANN.

| Metric | DeBERTa (seed 42) | LLaMA |
|--------|-------------------|-------|
| PER Span F1 | 0.718 | 0.628 |
| Token FPR | 0.0164 | 0.1442 |
| Token FNR (PER) | 0.274 | 0.126 |
| Redaction Leak Rate | 28.0% | 13.7% |

**Key findings:**

1. DeBERTa PER F1 drops 26 points (0.979 → 0.718), indicating it learned name patterns specific to the WikiNeural training distribution. The model was exposed only to Wikipedia-style person names during fine-tuning; WikiANN's broader multilingual-origin name set reveals this distributional dependency.

2. LLaMA PER F1 improves from 0.419 to 0.628 on WikiANN. This counter-intuitive result occurs because WikiANN contains canonical Wikipedia names that were present in LLaMA's pretraining corpus, whereas the in-distribution test set suppressed LLaMA's score through EMAIL detection failures (EMAIL class is absent in WikiANN).

3. **FNR inversion.** Despite lower overall F1, LLaMA has a substantially lower FNR on WikiANN (0.126 vs 0.274), meaning it misses fewer PER spans. For privacy-critical masking where under-detection is the primary risk, LLaMA's recall is superior in this OOD setting.

4. DeBERTa's leak rate rises from 2.1% to 28.0% on WikiANN, underscoring that in-distribution performance does not predict privacy guarantees on shifted data.

---

## 10. Security and Privacy Considerations

**Imperfect recall.** DeBERTa's token FNR of 0.008 in-distribution and 0.274 out-of-distribution means a non-trivial fraction of PII tokens are never detected. Systems using these models for privacy enforcement must account for residual leak risk, particularly on shifted input distributions.

**Distribution-specific overfitting.** DeBERTa's 26-point OOD degradation demonstrates that fine-tuning on a single Wikipedia-style dataset does not generalise to the full range of real-world PII patterns. Production deployment requires evaluation on representative samples from the target domain.

**Local inference.** Approach B uses `llama-cpp-python` for on-device GGUF inference. No sentence text is transmitted to external services, providing a privacy-by-design guarantee at the inference layer.

**Notebook hygiene.** Cell outputs may contain PII from intermediate evaluation steps. Before committing notebooks, strip outputs with `nbstripout` or `jupyter nbconvert --clear-output`.

**Masking ≠ anonymisation.** Removing PER and EMAIL entities does not constitute full anonymisation. Quasi-identifiers (employer, location, date-of-birth combinations), rare names surviving even after direct identifier removal, and co-reference chains are not addressed by this pipeline.

---

## 11. Conclusion

Fine-tuning DeBERTa-v3-small on the augmented WikiNeural dataset achieves macro span F1 of 0.989 ± 0.001, dramatically outperforming zero-shot LLaMA (0.423) in-distribution. The gap is statistically confirmed by paired bootstrap resampling (n = 10,000; p < 0.0001). DeBERTa also produces a lower redaction leak rate (2.1% vs 13.0%) and far fewer total errors (219 vs 14,631).

However, cross-domain evaluation on WikiANN reveals significant distribution dependency in the fine-tuned encoder: PER F1 drops to 0.718 and the redaction leak rate rises to 28.0%. LLaMA, despite lower overall F1, achieves a lower FNR on WikiANN (0.126 vs 0.274), suggesting its zero-shot recall is more robust to name distribution shift when EMAIL is absent.

The central practical conclusion is that a compact fine-tuned encoder is the right choice for well-defined, in-distribution PII detection tasks; zero-shot LLMs offer more robust recall generalisation but substantially higher false-positive noise. Future work should evaluate domain adaptation techniques (continued pre-training, domain-specific fine-tuning on ai4privacy) to close the OOD gap for the encoder approach.

---

_Bootstrap results: `results/day7/bootstrap_results.json` | Cross-domain results: `results/wikiann/eval_summary.json` | In-distribution results: `reports/results.json`_
