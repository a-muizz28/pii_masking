"""Unit tests for BIO well-formedness invariants enforced by _verify_bio."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pii_masking.day1_injection import _verify_bio


def test_verify_bio_all_o():
    _verify_bio(["Hello", "world"], ["O", "O"])


def test_verify_bio_single_b_per():
    _verify_bio(["Alice", "spoke"], ["B-PER", "O"])


def test_verify_bio_valid_multi_token_per():
    _verify_bio(["John", "Doe", "spoke"], ["B-PER", "I-PER", "O"])


def test_verify_bio_valid_email_span():
    _verify_bio(["contact", "alice@x.com"], ["O", "B-EMAIL"])


def test_verify_bio_valid_multi_token_email():
    _verify_bio(["alice", "@", "x.com"], ["B-EMAIL", "I-EMAIL", "I-EMAIL"])


def test_verify_bio_orphan_i_per_raises():
    with pytest.raises(AssertionError, match="BIO violation"):
        _verify_bio(["Hello", "Doe", "there"], ["O", "I-PER", "O"])


def test_verify_bio_orphan_i_email_raises():
    with pytest.raises(AssertionError, match="BIO violation"):
        _verify_bio(["send", "example.com", "now"], ["O", "I-EMAIL", "O"])


def test_verify_bio_i_per_after_b_email_raises():
    with pytest.raises(AssertionError, match="BIO violation"):
        _verify_bio(["alice@x.com", "Smith"], ["B-EMAIL", "I-PER"])


def test_verify_bio_i_after_o_raises():
    with pytest.raises(AssertionError, match="BIO violation"):
        _verify_bio(["A", "B", "C"], ["O", "O", "I-PER"])


def test_verify_bio_multiple_valid_spans():
    _verify_bio(
        ["Jane", "Doe", "wrote", "to", "Bob"],
        ["B-PER", "I-PER", "O", "O", "B-PER"],
    )
