import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pii_masking.eval_independent import (
    ai4privacy_row_to_example,
    compute_metrics,
    inspect_examples,
)


def test_ai4privacy_converter_merges_adjacent_name_spans() -> None:
    row = {
        "source_text": "Contact Jane Doe at jane@example.com today.",
        "language": "English",
        "privacy_mask": [
            {"label": "FIRSTNAME", "start": 8, "end": 12},
            {"label": "LASTNAME", "start": 13, "end": 16},
            {"label": "EMAIL", "start": 20, "end": 36},
        ],
    }

    example = ai4privacy_row_to_example(row, idx=0)

    assert example is not None
    assert example["tokens"] == ["Contact", "Jane", "Doe", "at", "jane@example.com", "today."]
    assert example["ner_tags"] == ["O", "B-PER", "I-PER", "O", "B-EMAIL", "O"]

    stats = inspect_examples([example])
    assert stats["per_span_count"] == 1
    assert stats["email_span_count"] == 1


def test_independent_metrics_perfect_prediction() -> None:
    record = {
        "tokens": ["Contact", "Jane", "Doe", "at", "jane@example.com"],
        "sequence": "Contact Jane Doe at jane@example.com",
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
