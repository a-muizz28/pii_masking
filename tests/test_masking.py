"""Unit tests for MaskingPipeline redaction logic."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pii_masking.day5_masking import MaskingPipeline


@pytest.fixture
def pipeline():
    return MaskingPipeline()


def test_mask_single_per_span(pipeline):
    tokens = ["Hello", "John", "Smith", "here"]
    tags   = ["O", "B-PER", "I-PER", "O"]
    seq    = "Hello John Smith here"
    assert pipeline.mask_text(seq, tokens, tags) == "Hello [NAME] here"


def test_mask_email_span(pipeline):
    tokens = ["contact", "alice@example.com", "now"]
    tags   = ["O", "B-EMAIL", "O"]
    seq    = "contact alice@example.com now"
    assert pipeline.mask_text(seq, tokens, tags) == "contact [EMAIL] now"


def test_mask_multi_token_email(pipeline):
    tokens = ["email", "alice", "@", "example.com", "today"]
    tags   = ["O", "B-EMAIL", "I-EMAIL", "I-EMAIL", "O"]
    seq    = "email alice @ example.com today"
    assert pipeline.mask_text(seq, tokens, tags) == "email [EMAIL] today"


def test_mask_no_pii_unchanged(pipeline):
    tokens = ["Hello", "world"]
    tags   = ["O", "O"]
    seq    = "Hello world"
    assert pipeline.mask_text(seq, tokens, tags) == seq


def test_mask_multiple_spans(pipeline):
    tokens = ["Jane", "wrote", "to", "Bob"]
    tags   = ["B-PER", "O", "O", "B-PER"]
    seq    = "Jane wrote to Bob"
    result = pipeline.mask_text(seq, tokens, tags)
    assert "Jane" not in result
    assert "Bob" not in result
    assert result.count("[NAME]") == 2


def test_mask_length_mismatch_raises(pipeline):
    with pytest.raises(ValueError, match="pred_tags length"):
        pipeline.mask_text("Hello world", ["Hello", "world"], ["O"])


def test_compute_leak_rate_no_leak(pipeline):
    record = {
        "tokens":    ["Jane", "Doe", "called"],
        "sequence":  "Jane Doe called",
        "gold_tags": ["B-PER", "I-PER", "O"],
        "pred_tags": ["B-PER", "I-PER", "O"],
    }
    result = pipeline.compute_leak_rate([record])
    assert result["leak_rate"] == 0.0
    assert result["leaked_count"] == 0


def test_compute_leak_rate_full_leak(pipeline):
    record = {
        "tokens":    ["Jane", "Doe", "called"],
        "sequence":  "Jane Doe called",
        "gold_tags": ["B-PER", "I-PER", "O"],
        "pred_tags": ["O", "O", "O"],
    }
    result = pipeline.compute_leak_rate([record])
    assert result["leak_rate"] == 1.0
    assert result["leaked_count"] == 1


def test_compute_leak_rate_single_token_missed(pipeline):
    # Single-token entity completely missed by pred → gold string still in output.
    record = {
        "tokens":    ["alice@x.com", "replied"],
        "sequence":  "alice@x.com replied",
        "gold_tags": ["B-EMAIL", "O"],
        "pred_tags": ["O", "O"],
    }
    result = pipeline.compute_leak_rate([record])
    assert result["leak_rate"] == 1.0


def test_compute_leak_rate_partial_multi_token_miss_not_detected(pipeline):
    # Known limitation: compute_leak_rate checks full entity string presence.
    # If pred tags the first token of a multi-token span but misses later
    # tokens, the gold entity string "John Smith" is absent from the masked
    # output "[NAME] Smith replied", so no leak is reported.  Token-level
    # leak detection (day4_evaluate.py) catches this case instead.
    record = {
        "tokens":    ["John", "Smith", "replied"],
        "sequence":  "John Smith replied",
        "gold_tags": ["B-PER", "I-PER", "O"],
        "pred_tags": ["B-PER", "O", "O"],
    }
    result = pipeline.compute_leak_rate([record])
    assert result["leak_rate"] == 0.0


def test_mask_batch_adds_masked_sequence(pipeline):
    records = [
        {
            "tokens":        ["Call", "Alice", "now"],
            "sequence":      "Call Alice now",
            "predicted_tags": ["O", "B-PER", "O"],
        }
    ]
    out = pipeline.mask_batch(records)
    assert "masked_sequence" in out[0]
    assert out[0]["masked_sequence"] == "Call [NAME] now"


def test_mask_all_o_tags_leaves_sequence_intact(pipeline):
    tokens = ["Nothing", "to", "redact"]
    tags   = ["O", "O", "O"]
    seq    = "Nothing to redact"
    assert pipeline.mask_text(seq, tokens, tags) == seq
