"""Day 2 encoder smoke-test evaluation metrics."""

from __future__ import annotations

import numpy as np
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def compute_seqeval_metrics(p, id2label):
    """HuggingFace Trainer compute_metrics callback using seqeval."""
    predictions = np.argmax(p.predictions, axis=2)
    label_ids = p.label_ids

    true_labels = []
    pred_labels = []
    for pred_row, label_row in zip(predictions, label_ids):
        true_seq = [id2label[l] for l in label_row if l != -100]
        pred_seq = [id2label[pr] for pr, l in zip(pred_row, label_row) if l != -100]
        true_labels.append(true_seq)
        pred_labels.append(pred_seq)

    report = classification_report(true_labels, pred_labels, zero_division=0)
    print("\n" + report)

    # Per-entity F1: extract from seqeval type-level scores
    # f1_score with average=None + type_names gives per-class breakdown
    per_f1 = f1_score(true_labels, pred_labels, average=None, zero_division=0)
    type_names = sorted({
        lbl.split("-", 1)[1]
        for seq in true_labels for lbl in seq if lbl != "O"
    })
    per_entity_f1 = dict(zip(type_names, per_f1)) if len(type_names) == len(per_f1) else {}

    return {
        "overall_precision": precision_score(true_labels, pred_labels, zero_division=0),
        "overall_recall": recall_score(true_labels, pred_labels, zero_division=0),
        "overall_f1": f1_score(true_labels, pred_labels, zero_division=0),
        "per_f1": per_entity_f1.get("PER", 0.0),
        "email_f1": per_entity_f1.get("EMAIL", 0.0),
        "classification_report": report,
    }


def compute_token_binary_metrics(true_labels_list, pred_labels_list, entity_type="PER"):
    """Token-level binary metrics for a single entity class.

    Returns dict with fpr, fnr, precision, recall, f1, tp, fp, fn, tn.
    """
    true_flat = [l for seq in true_labels_list for l in seq]
    pred_flat = [l for seq in pred_labels_list for l in seq]

    tp = fp = fn = tn = 0
    for t, p in zip(true_flat, pred_flat):
        t_pos = entity_type in t
        p_pos = entity_type in p
        if t_pos and p_pos:
            tp += 1
        elif not t_pos and p_pos:
            fp += 1
        elif t_pos and not p_pos:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "fpr": fpr, "fnr": fnr,
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
