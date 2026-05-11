"""Tokenization and BIO label alignment helpers."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict


LABEL2ID = {"O": 0, "B-PER": 1, "I-PER": 2, "B-EMAIL": 3, "I-EMAIL": 4}

_REMOVE_COLS = {"sequence", "lang", "email_metadata"}


def tokenize_and_align_labels(examples, tokenizer, label2id, max_length=256):
    """Tokenize word-level examples and align BIO labels using Strategy B."""
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_overflowing_tokens=False,
    )

    all_labels = []
    for i in range(len(examples["tokens"])):
        word_ids = tokenized.word_ids(batch_index=i)
        word_labels = examples["ner_tags"][i]
        previous_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(label2id[word_labels[word_id]])
            else:
                wl = word_labels[word_id]
                if wl in ("B-PER", "I-PER"):
                    label_ids.append(label2id["I-PER"])
                elif wl in ("B-EMAIL", "I-EMAIL"):
                    label_ids.append(label2id["I-EMAIL"])
                else:
                    label_ids.append(label2id["O"])
            previous_word_id = word_id
        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized


def load_and_tokenize_split(parquet_path, tokenizer, label2id, max_length=256):
    """Load a parquet split, tokenize, and return a HF Dataset in torch format."""
    df = pd.read_parquet(parquet_path)
    df["tokens"] = df["tokens"].apply(list)
    df["ner_tags"] = df["ner_tags"].apply(list)

    dataset = Dataset.from_pandas(df, preserve_index=False)

    remove_cols = [c for c in dataset.column_names if c in _REMOVE_COLS]

    fn = partial(
        tokenize_and_align_labels,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=max_length,
    )
    dataset = dataset.map(fn, batched=True, batch_size=256, remove_columns=remove_cols)
    dataset.set_format("torch")
    return dataset


def build_dataset_dict(train_path, val_path, test_path, tokenizer, label2id, max_length=256):
    """Build a DatasetDict with train/validation/test splits."""
    return DatasetDict(
        {
            "train": load_and_tokenize_split(train_path, tokenizer, label2id, max_length),
            "validation": load_and_tokenize_split(val_path, tokenizer, label2id, max_length),
            "test": load_and_tokenize_split(test_path, tokenizer, label2id, max_length),
        }
    )


def validate_alignment(dataset_split, label2id):
    """Assert Strategy B invariants on a tokenized HF Dataset split.

    Raises AssertionError with a descriptive message if any invariant is violated.
    """
    valid_ids = {-100, 0, 1, 2, 3, 4}
    b_labels = {label2id["B-PER"], label2id["B-EMAIL"]}  # {1, 3}

    for idx, row in enumerate(dataset_split):
        labels = row["labels"].tolist()
        attn = row["attention_mask"].tolist()

        # Invariant 3: all label values must be in valid set
        for pos, lbl in enumerate(labels):
            assert lbl in valid_ids, (
                f"Example {idx}, pos {pos}: label {lbl} not in {valid_ids}"
            )

        # Invariant 4: first non-(-100) label must not be I-X (i.e. not 2 or 4)
        for lbl in labels:
            if lbl != -100:
                assert lbl not in (2, 4), (
                    f"Example {idx}: first real label is I-X ({lbl}), expected O/B-X"
                )
                break

    print("Strategy B alignment validated: OK")
