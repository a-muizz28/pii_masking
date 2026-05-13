"""Day 4 evaluation metrics for LLM PII detection."""

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2


def compute_llm_metrics(records: list[dict]) -> dict:
    """Compute all Day 4 metrics from predict_batch output."""
    true_seqs: list[list[str]] = []
    pred_seqs: list[list[str]] = []

    n_sentences = len(records)
    n_parse_failures = sum(1 for r in records if not r.get("parse_ok", True))

    # Token-level counters per class
    tp_per = fp_per = fn_per = tn_per = 0
    tp_email = fp_email = fn_email = tn_email = 0

    # Redaction leak counters
    total_spans = 0
    leaked_spans = 0

    for record in records:
        true_tags = record.get("ner_tags", [])
        pred_tags = record.get("predicted_tags", [])

        # Pad/truncate preds to match true length if needed
        n = len(true_tags)
        pred_tags_aligned = (pred_tags + ["O"] * n)[:n]

        true_seqs.append(true_tags)
        pred_seqs.append(pred_tags_aligned)

        # Token-level FPR/FNR
        for t_tag, p_tag in zip(true_tags, pred_tags_aligned):
            is_true_per = t_tag in ("B-PER", "I-PER")
            is_pred_per = p_tag in ("B-PER", "I-PER")
            if is_true_per and is_pred_per:
                tp_per += 1
            elif not is_true_per and is_pred_per:
                fp_per += 1
            elif is_true_per and not is_pred_per:
                fn_per += 1
            else:
                tn_per += 1

            is_true_email = t_tag in ("B-EMAIL", "I-EMAIL")
            is_pred_email = p_tag in ("B-EMAIL", "I-EMAIL")
            if is_true_email and is_pred_email:
                tp_email += 1
            elif not is_true_email and is_pred_email:
                fp_email += 1
            elif is_true_email and not is_pred_email:
                fn_email += 1
            else:
                tn_email += 1

        # Redaction leak rate
        i = 0
        while i < len(true_tags):
            tag = true_tags[i]
            if tag.startswith("B-"):
                # Start of a ground-truth span
                span_indices = [i]
                j = i + 1
                entity_type = tag[2:]
                while j < len(true_tags) and true_tags[j] == "I-" + entity_type:
                    span_indices.append(j)
                    j += 1
                total_spans += 1
                # Leaked if ANY token in span is predicted O
                if any(pred_tags_aligned[k] == "O" for k in span_indices):
                    leaked_spans += 1
                i = j
            else:
                i += 1

    # Span F1 via seqeval
    report = classification_report(
        true_seqs, pred_seqs, mode="strict", scheme=IOB2, output_dict=True, zero_division=0
    )

    per_stats = report.get("PER", {})
    email_stats = report.get("EMAIL", {})
    overall = report.get("macro avg", {})

    # seqeval overall F1
    span_f1_overall_val = report.get("macro avg", {}).get("f1-score", 0.0)
    # Also check "weighted avg" as fallback — use macro avg as primary
    span_f1_per = float(per_stats.get("f1-score", 0.0))
    span_f1_email = float(email_stats.get("f1-score", 0.0))

    fpr_per = fp_per / (fp_per + tn_per) if (fp_per + tn_per) > 0 else 0.0
    fnr_per = fn_per / (fn_per + tp_per) if (fn_per + tp_per) > 0 else 0.0
    fpr_email = fp_email / (fp_email + tn_email) if (fp_email + tn_email) > 0 else 0.0
    fnr_email = fn_email / (fn_email + tp_email) if (fn_email + tp_email) > 0 else 0.0

    leak_rate = leaked_spans / total_spans if total_spans > 0 else 0.0
    parse_failure_rate = n_parse_failures / n_sentences if n_sentences > 0 else 0.0

    return {
        "span_f1_overall": float(span_f1_overall_val),
        "span_f1_per": span_f1_per,
        "span_f1_email": span_f1_email,
        "precision_per": float(per_stats.get("precision", 0.0)),
        "recall_per": float(per_stats.get("recall", 0.0)),
        "precision_email": float(email_stats.get("precision", 0.0)),
        "recall_email": float(email_stats.get("recall", 0.0)),
        "token_fpr_per": fpr_per,
        "token_fnr_per": fnr_per,
        "token_fpr_email": fpr_email,
        "token_fnr_email": fnr_email,
        "redaction_leak_rate": leak_rate,
        "parse_failure_rate": parse_failure_rate,
        "n_sentences": n_sentences,
        "n_parse_failures": n_parse_failures,
    }
