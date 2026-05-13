"""Unit tests for preprocessing.py — Strategy B label alignment."""

import pytest
from transformers import AutoTokenizer

from pii_masking.day2_preprocessing import tokenize_and_align_labels

TOKENIZER_NAME = "distilbert-base-cased"
LABEL2ID = {"O": 0, "B-PER": 1, "I-PER": 2, "B-EMAIL": 3, "I-EMAIL": 4}
MAX_LENGTH = 256


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def _tokenize(tokenizer, tokens, ner_tags):
    examples = {"tokens": [tokens], "ner_tags": [ner_tags]}
    return tokenize_and_align_labels(examples, tokenizer, LABEL2ID, MAX_LENGTH)


# ---------------------------------------------------------------------------
# Test 1
# ---------------------------------------------------------------------------
def test_first_subword_inherits_word_label(tokenizer):
    result = _tokenize(tokenizer, ["John", "Doe"], ["B-PER", "I-PER"])
    labels = result["labels"][0]
    word_ids = tokenizer(
        [["John", "Doe"]], is_split_into_words=True, truncation=True,
        padding="max_length", max_length=MAX_LENGTH
    ).word_ids(batch_index=0)

    first_john = next(i for i, w in enumerate(word_ids) if w == 0)
    first_doe = next(i for i, w in enumerate(word_ids) if w == 1)

    assert labels[first_john] == LABEL2ID["B-PER"], (
        f"First subword of 'John' should be B-PER (1), got {labels[first_john]}"
    )
    assert labels[first_doe] == LABEL2ID["I-PER"], (
        f"First subword of 'Doe' should be I-PER (2), got {labels[first_doe]}"
    )


# ---------------------------------------------------------------------------
# Test 2
# ---------------------------------------------------------------------------
def test_continuation_subword_is_I_not_B(tokenizer):
    # "Schwarzenegger" splits into multiple subwords with distilbert-base-cased
    token = "Schwarzenegger"
    enc = tokenizer([token], add_special_tokens=False)
    assert len(enc["input_ids"][0]) > 1, (
        f"'{token}' did not split into multiple subwords — pick a different test token"
    )

    result = _tokenize(tokenizer, [token], ["B-PER"])
    labels = result["labels"][0]
    word_ids = tokenizer(
        [[token]], is_split_into_words=True, truncation=True,
        padding="max_length", max_length=MAX_LENGTH
    ).word_ids(batch_index=0)

    positions = [i for i, w in enumerate(word_ids) if w == 0]
    assert len(positions) > 1, "Need at least 2 subword positions for this test"

    assert labels[positions[0]] == LABEL2ID["B-PER"], (
        f"First subword should be B-PER (1), got {labels[positions[0]]}"
    )
    for pos in positions[1:]:
        assert labels[pos] == LABEL2ID["I-PER"], (
            f"Continuation subword at pos {pos} should be I-PER (2), got {labels[pos]}"
        )


# ---------------------------------------------------------------------------
# Test 3
# ---------------------------------------------------------------------------
def test_no_minus100_at_first_subword(tokenizer):
    # Short sentence — no padding issues for real tokens
    result = _tokenize(tokenizer, ["Hello", "world"], ["O", "O"])
    labels = result["labels"][0]
    word_ids = tokenizer(
        [["Hello", "world"]], is_split_into_words=True, truncation=True,
        padding="max_length", max_length=MAX_LENGTH
    ).word_ids(batch_index=0)

    for i, w in enumerate(word_ids):
        if w is not None:
            assert labels[i] != -100, (
                f"Position {i} (word {w}) is a real subword but got label -100"
            )


# ---------------------------------------------------------------------------
# Test 4
# ---------------------------------------------------------------------------
def test_special_tokens_are_minus100(tokenizer):
    result = _tokenize(tokenizer, ["Hello"], ["O"])
    labels = result["labels"][0]
    word_ids = tokenizer(
        [["Hello"]], is_split_into_words=True, truncation=True,
        padding="max_length", max_length=MAX_LENGTH
    ).word_ids(batch_index=0)

    for i, w in enumerate(word_ids):
        if w is None:
            assert labels[i] == -100, (
                f"Special/pad token at pos {i} should be -100, got {labels[i]}"
            )


# ---------------------------------------------------------------------------
# Test 5
# ---------------------------------------------------------------------------
def test_email_label_continuations(tokenizer):
    tokens = ["contact", "user@example.com"]
    ner_tags = ["O", "B-EMAIL"]
    result = _tokenize(tokenizer, tokens, ner_tags)
    labels = result["labels"][0]
    word_ids = tokenizer(
        [tokens], is_split_into_words=True, truncation=True,
        padding="max_length", max_length=MAX_LENGTH
    ).word_ids(batch_index=0)

    word0_pos = [i for i, w in enumerate(word_ids) if w == 0]
    word1_pos = [i for i, w in enumerate(word_ids) if w == 1]

    assert labels[word0_pos[0]] == LABEL2ID["O"], (
        f"'contact' first subword should be O (0), got {labels[word0_pos[0]]}"
    )
    assert labels[word1_pos[0]] == LABEL2ID["B-EMAIL"], (
        f"'user@example.com' first subword should be B-EMAIL (3), got {labels[word1_pos[0]]}"
    )
    for pos in word1_pos[1:]:
        assert labels[pos] == LABEL2ID["I-EMAIL"], (
            f"'user@example.com' continuation at pos {pos} should be I-EMAIL (4), got {labels[pos]}"
        )


# ---------------------------------------------------------------------------
# Test 6
# ---------------------------------------------------------------------------
def test_label_length_matches_input_ids(tokenizer):
    sentences = [
        (["Alice", "works", "at", "Google"], ["B-PER", "O", "O", "O"]),
        (["Send", "email", "to", "bob@test.org"], ["O", "O", "O", "B-EMAIL"]),
        (["Hello", "world"], ["O", "O"]),
        (["Dr", "Smith", "replied"], ["B-PER", "I-PER", "O"]),
        (["The", "address", "is", "here"], ["O", "O", "O", "O"]),
    ]
    for tokens, tags in sentences:
        result = _tokenize(tokenizer, tokens, tags)
        assert len(result["labels"][0]) == MAX_LENGTH, (
            f"labels length {len(result['labels'][0])} != max_length {MAX_LENGTH}"
        )
        assert len(result["input_ids"][0]) == MAX_LENGTH, (
            f"input_ids length {len(result['input_ids'][0])} != max_length {MAX_LENGTH}"
        )
        assert len(result["labels"][0]) == len(result["input_ids"][0])
