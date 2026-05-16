"""Day 2 tokenization and BIO label alignment helpers.

Notes
-----
Strategy B label alignment: non-first subwords of a word inherit the word-level
span label as I-X rather than -100 (Strategy A).  This keeps every subword fully
trainable and prevents the model from seeing a B-PER followed by -100 gaps inside
a single name token.
"""

from __future__ import annotations

from functools import partial

import pandas as pd
from datasets import Dataset, DatasetDict


_REMOVE_COLS = {"sequence", "lang", "email_metadata"}


def tokenize_and_align_labels(examples, tokenizer, label2id, max_length=256, ignore_index=-100):
    """Tokenize word-level examples and align BIO labels using Strategy B.

    Special tokens and padding positions receive ignore_index so PyTorch's
    cross-entropy loss skips them.  Continuation subwords get I-X so the full
    span participates in gradient updates.
    """
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
                # [CLS], [SEP], and padding tokens are excluded from the loss.
                label_ids.append(ignore_index)
            elif word_id != previous_word_id:
                label_ids.append(label2id[word_labels[word_id]])
            else:
                # Strategy B: continuation subwords get I-X, preserving the span
                # across all subword pieces of a single word token.
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


def load_and_tokenize_split(parquet_path, tokenizer, label2id, max_length=256, ignore_index=-100):
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
        ignore_index=ignore_index,
    )
    dataset = dataset.map(fn, batched=True, batch_size=256, remove_columns=remove_cols)
    dataset.set_format("torch")
    return dataset


def build_dataset_dict(train_path, val_path, test_path, tokenizer, label2id, max_length=256, ignore_index=-100):
    """Build a DatasetDict with train/validation/test splits."""
    return DatasetDict(
        {
            "train": load_and_tokenize_split(train_path, tokenizer, label2id, max_length, ignore_index),
            "validation": load_and_tokenize_split(val_path, tokenizer, label2id, max_length, ignore_index),
            "test": load_and_tokenize_split(test_path, tokenizer, label2id, max_length, ignore_index),
        }
    )


def validate_alignment(dataset_split, label2id, ignore_index=-100):
    """Assert Strategy B invariants on a tokenized HF Dataset split."""
    valid_ids = {ignore_index} | set(label2id.values())
    i_label_ids = {v for k, v in label2id.items() if k.startswith("I-")}

    for idx, row in enumerate(dataset_split):
        labels = row["labels"].tolist()

        # Invariant 3: all label values must be in valid set
        for pos, lbl in enumerate(labels):
            assert lbl in valid_ids, (
                f"Example {idx}, pos {pos}: label {lbl} not in {valid_ids}"
            )

        # Invariant 4: first real label must not be I-X
        for lbl in labels:
            if lbl != ignore_index:
                assert lbl not in i_label_ids, (
                    f"Example {idx}: first real label is I-X ({lbl}), expected O/B-X"
                )
                break

    print("Strategy B alignment validated: OK")
