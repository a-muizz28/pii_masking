"""Tokenize parquet splits and save HuggingFace DatasetDict to disk."""

import json
import sys
from pathlib import Path

import torch  # noqa: F401 — used for tensor ops in stats
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pii_masking.preprocessing import build_dataset_dict, validate_alignment

CONFIG_PATH = Path("configs/label_config.json")
TRAIN_PATH = Path("data/processed/train_with_emails.parquet")
VAL_PATH = Path("data/processed/val_with_emails.parquet")
TEST_PATH = Path("data/processed/test_injected.parquet")
OUTPUT_PATH = Path("data/processed/hf_dataset")


def main():
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)

    label2id = cfg["label2id"]
    max_length = cfg["max_length"]
    model_name = cfg["model_name_smoke_test"]

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    assert tokenizer.is_fast, "Tokenizer must be a fast tokenizer"

    print("Building DatasetDict (tokenizing all splits)...")
    dataset_dict = build_dataset_dict(
        TRAIN_PATH, VAL_PATH, TEST_PATH, tokenizer, label2id, max_length
    )

    print("\n--- Dataset Stats ---")
    for split_name, split in dataset_dict.items():
        n = len(split)
        attn = torch.stack([split[i]["attention_mask"] for i in range(min(n, 500))])
        avg_len = attn.float().sum(dim=1).mean().item()

        labels = torch.stack([split[i]["labels"] for i in range(min(n, 500))])
        mask = labels != -100
        real_labels = labels[mask]
        total_real = real_labels.numel()

        per_frac = ((real_labels == 1) | (real_labels == 2)).sum().item() / max(total_real, 1)
        email_frac = ((real_labels == 3) | (real_labels == 4)).sum().item() / max(total_real, 1)

        print(
            f"  {split_name:12s}: {n:5d} examples | avg_len={avg_len:.1f} tokens | "
            f"PER={per_frac*100:.2f}% | EMAIL={email_frac*100:.2f}%"
        )

    print("\nRunning Strategy B alignment validation on train split...")
    validate_alignment(dataset_dict["train"], label2id)

    print(f"\nSaving DatasetDict to {OUTPUT_PATH} ...")
    dataset_dict.save_to_disk(str(OUTPUT_PATH))
    print("Preprocessing complete. DatasetDict saved.")


if __name__ == "__main__":
    main()
