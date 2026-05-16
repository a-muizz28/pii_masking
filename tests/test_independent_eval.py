"""Tests for the Day 7 independent evaluation pipeline (WikiANN dataset)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pii_masking.eval_independent import (
    compute_metrics,
    compute_span_leak_rate,
    compute_token_rates,
    extract_spans,
    inspect_examples,
    wikiann_row_to_example,
)


def test_wikiann_row_drops_loc_org():
    label_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    row = {
        "tokens": ["Alice", "works", "at", "Google", "in", "London"],
        "ner_tags": [1, 0, 0, 3, 0, 5],
    }
    example = wikiann_row_to_example(row, idx=0, label_names=label_names)
    assert example is not None
    assert example["ner_tags"] == ["B-PER", "O", "O", "O", "O", "O"]


def test_wikiann_row_keeps_per_span():
    label_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    row = {
        "tokens": ["John", "Doe", "spoke"],
        "ner_tags": [1, 2, 0],
    }
    example = wikiann_row_to_example(row, idx=1, label_names=label_names)
    assert example is not None
    assert example["ner_tags"] == ["B-PER", "I-PER", "O"]
    assert example["source_dataset"] == "wikiann/en"


def test_wikiann_row_empty_tokens_returns_none():
    label_names = ["O", "B-PER", "I-PER"]
    row = {"tokens": [], "ner_tags": []}
    assert wikiann_row_to_example(row, idx=0, label_names=label_names) is None


def test_extract_spans_basic():
    tags = ["B-PER", "I-PER", "O", "B-EMAIL"]
    spans = extract_spans(tags)
    assert spans == [(0, 2, "PER"), (3, 4, "EMAIL")]


def test_extract_spans_empty():
    assert extract_spans(["O", "O", "O"]) == []


def test_independent_metrics_perfect_prediction():
    record = {
        "tokens":    ["Contact", "Jane", "Doe", "at", "jane@example.com"],
        "sequence":  "Contact Jane Doe at jane@example.com",
        "gold_tags": ["O", "B-PER", "I-PER", "O", "B-EMAIL"],
        "pred_tags": ["O", "B-PER", "I-PER", "O", "B-EMAIL"],
    }
    metrics = compute_metrics([record], email_present=True)
    assert metrics["per_f1"] == 1.0
    assert metrics["email_f1"] == 1.0
    assert metrics["overall_f1"] == 1.0
    assert metrics["fpr"] == 0.0
    assert metrics["fnr"] == 0.0
    assert metrics["redaction_leak_rate"] == 0.0


def test_independent_metrics_no_email_present_overall_equals_per():
    record = {
        "tokens":    ["Jane", "Doe", "spoke"],
        "sequence":  "Jane Doe spoke",
        "gold_tags": ["B-PER", "I-PER", "O"],
        "pred_tags": ["B-PER", "I-PER", "O"],
    }
    metrics = compute_metrics([record], email_present=False)
    assert metrics["email_f1"] is None
    assert metrics["overall_f1"] == metrics["per_f1"]


def test_compute_token_rates_all_correct():
    record = {
        "gold_tags": ["B-PER", "I-PER", "O"],
        "pred_tags": ["B-PER", "I-PER", "O"],
    }
    rates = compute_token_rates([record])
    assert rates["fpr"] == 0.0
    assert rates["fnr"] == 0.0


def test_compute_span_leak_rate_no_leak():
    record = {
        "gold_tags": ["B-PER", "I-PER", "O"],
        "pred_tags": ["B-PER", "I-PER", "O"],
    }
    result = compute_span_leak_rate([record])
    assert result["redaction_leak_rate"] == 0.0
    assert result["leaked_span_count"] == 0
    assert result["total_pii_span_count"] == 1


def test_inspect_examples_counts_spans():
    examples = [
        {
            "idx": 0,
            "lang": "en",
            "source_dataset": "wikiann/en",
            "sequence": "Alice met Bob",
            "tokens": ["Alice", "met", "Bob"],
            "ner_tags": ["B-PER", "O", "B-PER"],
        }
    ]
    stats = inspect_examples(examples, dataset_name="wikiann/en")
    assert stats["per_span_count"] == 2
    assert stats["email_span_count"] == 0
    assert stats["n_sentences"] == 1
