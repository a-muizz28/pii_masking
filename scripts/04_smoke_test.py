"""1-epoch DistilBERT smoke test to validate the full pipeline."""

import json
import os
import sys
from pathlib import Path

import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pii_masking.metrics import compute_seqeval_metrics

MODEL_NAME = "distilbert-base-cased"
MAX_LENGTH = 256
BATCH_SIZE = 16
LR = 3e-5
NUM_EPOCHS = 1
SEED = 42
OUTPUT_DIR = "models/smoke_test_distilbert"

# Smoke test only needs pipeline validation; 2500/500 gives 156 steps.
TRAIN_SUBSET = 2500
VAL_SUBSET = 500

LABEL2ID = {"O": 0, "B-PER": 1, "I-PER": 2, "B-EMAIL": 3, "I-EMAIL": 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def main():
    print("Loading preprocessed DatasetDict...")
    dataset = load_from_disk("data/processed/hf_dataset/")
    print(f"  full: train={len(dataset['train'])} | val={len(dataset['validation'])} | test={len(dataset['test'])}")

    # Subsample for local CPU smoke test (156 train steps)
    rng = np.random.default_rng(SEED)
    train_idx = rng.choice(len(dataset["train"]), size=min(TRAIN_SUBSET, len(dataset["train"])), replace=False).tolist()
    val_idx = rng.choice(len(dataset["validation"]), size=min(VAL_SUBSET, len(dataset["validation"])), replace=False).tolist()
    train_ds = dataset["train"].select(sorted(train_idx))
    val_ds = dataset["validation"].select(sorted(val_idx))
    test_ds = dataset["test"].select(list(range(min(500, len(dataset["test"])))))
    print(f"  smoke subset: train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")

    # Truncate pre-padded 256-length tensors to 64 for CPU speed
    # (avg real seq len is 39 tokens; 64 covers >99.9% of content)
    SMOKE_LEN = 64

    def _trunc(batch):
        out = {}
        for k, vals in batch.items():
            truncated = []
            for v in vals:
                if hasattr(v, "tolist"):
                    v = v.tolist()
                truncated.append(v[:SMOKE_LEN] if isinstance(v, list) else v)
            out[k] = truncated
        return out

    train_ds = train_ds.with_format(None)
    val_ds = val_ds.with_format(None)
    test_ds = test_ds.with_format(None)
    train_ds = train_ds.map(_trunc, batched=True, desc="Truncating train")
    val_ds = val_ds.map(_trunc, batched=True, desc="Truncating val")
    test_ds = test_ds.map(_trunc, batched=True, desc="Truncating test")
    train_ds.set_format("torch")
    val_ds.set_format("torch")
    test_ds.set_format("torch")
    print(f"  sequence length truncated to {SMOKE_LEN} tokens for CPU smoke test")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=5,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    use_fp16 = torch.cuda.is_available()
    if use_fp16:
        print("GPU detected — enabling fp16.")
    else:
        print("No GPU detected — running on CPU.")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",        # skip optimizer/scheduler saves — smoke test only
        load_best_model_at_end=False,
        metric_for_best_model="overall_f1",
        greater_is_better=True,
        seed=SEED,
        fp16=use_fp16,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_seqeval_metrics(p, ID2LABEL),
    )

    print("\nStarting 1-epoch smoke test...")
    train_result = trainer.train()

    print("\nEvaluating on validation set...")
    val_metrics = trainer.evaluate()

    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=test_ds)
    print("Test metrics:", {k: f"{v:.4f}" for k, v in test_metrics.items() if isinstance(v, float)})

    print("\n=== SMOKE TEST RESULTS ===")
    print(f"Train loss: {train_result.training_loss:.4f}")
    for k, v in val_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {
        "model": MODEL_NAME,
        "epochs": NUM_EPOCHS,
        "train_loss": train_result.training_loss,
        "eval_metrics": {k: v for k, v in val_metrics.items() if isinstance(v, float)},
        "test_metrics": {k: v for k, v in test_metrics.items() if isinstance(v, float)},
    }
    out_path = f"{OUTPUT_DIR}/smoke_test_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    overall_f1 = val_metrics.get("eval_overall_f1", 0)
    print(f"\nOverall F1: {overall_f1:.4f}")
    if overall_f1 >= 0.75:
        print("Smoke test PASSED — pipeline is healthy")
    else:
        print(f"F1={overall_f1:.4f} is below 0.75 — investigate before Day 3")
        sys.exit(1)


if __name__ == "__main__":
    main()
