"""Independent Hugging Face dataset evaluation - WikiANN English test split."""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass
from typing import Any

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

from pii_masking.day1_data import validate_example
from pii_masking.day4_llm_inference import LLMPIIPipeline

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class EvalPaths:
    results_dir: pathlib.Path
    stats_path: pathlib.Path
    distilbert_path: pathlib.Path
    deberta_path: pathlib.Path
    llama_path: pathlib.Path
    summary_path: pathlib.Path
    llama_cache_path: pathlib.Path


def wikiann_paths(results_dir: str | pathlib.Path = "predictions/wikiann") -> EvalPaths:
    root = pathlib.Path(results_dir)
    return EvalPaths(
        results_dir=root,
        stats_path=root / "data_stats.json",
        distilbert_path=root / "distilbert_results.json",
        deberta_path=root / "deberta_results.json",
        llama_path=root / "llama_results.json",
        summary_path=root / "eval_summary.json",
        llama_cache_path=root / "llama_outputs.jsonl",
    )




def wikiann_row_to_example(
    row: dict[str, Any],
    idx: int,
    label_names: list[str],
) -> dict[str, Any] | None:
    """Convert one WikiANN row to project BIO tags keeping only PER; drop LOC/ORG."""
    tokens: list[str] = list(row.get("tokens") or [])
    raw_tags: list[int] = list(row.get("ner_tags") or [])
    if not tokens:
        return None

    _PER_KEEP = {"B-PER", "I-PER"}
    tags: list[str] = []
    for tag_id in raw_tags:
        decoded = label_names[tag_id] if tag_id < len(label_names) else "O"
        tags.append(decoded if decoded in _PER_KEEP else "O")

    return {
        "idx": idx,
        "source_dataset": "wikiann/en",
        "lang": "en",
        "sequence": " ".join(tokens),
        "tokens": tokens,
        "ner_tags": tags,
    }


def load_wikiann_examples(
    split: str = "test",
    min_entity_spans: int = 1,
) -> list[dict[str, Any]]:
    """Load WikiANN English split and map to project BIO labels (PER only)."""
    from datasets import load_dataset

    ds = load_dataset("wikiann", "en", split=split)
    label_names: list[str] = ds.features["ner_tags"].feature.names

    examples: list[dict[str, Any]] = []
    for idx, row in enumerate(ds):
        example = wikiann_row_to_example(row, idx, label_names)
        if example is None:
            continue
        if count_spans(example["ner_tags"], "PER") < min_entity_spans:
            continue
        examples.append(example)

    if not examples:
        raise ValueError(f"No usable examples loaded from wikiann/en split={split!r}")
    return examples


def count_spans(tags: list[str], entity: str) -> int:
    return sum(1 for tag in tags if tag == f"B-{entity}")


def extract_spans(tags: list[str]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag.startswith("B-"):
            entity = tag[2:]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{entity}":
                j += 1
            spans.append((i, j, entity))
            i = j
        else:
            i += 1
    return spans


def inspect_examples(
    examples: list[dict[str, Any]],
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """Validate and summarise a list of BIO-tagged examples.

    Returns counts of valid/invalid sentences, total tokens, and span counts
    per entity type.  Reports up to 50 BIO violations for debugging.
    """
    valid = 0
    invalid = 0
    fixes = 0
    token_count = 0
    per_spans = 0
    email_spans = 0
    violations: list[dict[str, Any]] = []

    for idx, example in enumerate(examples):
        check = validate_example(example, idx)
        if check["valid"]:
            valid += 1
        else:
            invalid += 1
            violations.append({"idx": idx, "violations": check["violations"]})
        fixes += len(check.get("fixes_applied", []))
        token_count += len(example["tokens"])
        per_spans += count_spans(example["ner_tags"], "PER")
        email_spans += count_spans(example["ner_tags"], "EMAIL")

    inferred_name = dataset_name or (
        examples[0].get("source_dataset", "unknown") if examples else "unknown"
    )
    return {
        "dataset": inferred_name,
        "n_sentences": len(examples),
        "n_valid_sentences": valid,
        "n_invalid_sentences": invalid,
        "bio_fixes_applied": fixes,
        "n_tokens": token_count,
        "per_span_count": per_spans,
        "email_span_count": email_spans,
        "email_metrics_note": None
        if email_spans
        else "Dataset contains no EMAIL spans; EMAIL metrics are reported as null.",
        "violations": violations[:50],
    }


def save_json(payload: dict[str, Any], path: str | pathlib.Path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def predict_deberta(
    examples: list[dict[str, Any]],
    model_dir: str | pathlib.Path,
    batch_size: int = 16,
    max_length: int = 256,
) -> list[dict[str, Any]]:
    """Run an encoder token classifier and return word-aligned records.

    Only the first subword of each word receives a predicted label; subsequent
    subwords are skipped via word_ids() to produce word-aligned pred_tags.
    """
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    model_dir = pathlib.Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    results: list[dict[str, Any]] = []

    for batch_start in range(0, len(examples), batch_size):
        batch = examples[batch_start : batch_start + batch_size]
        batch_tokens = [record["tokens"] for record in batch]
        encoding = tokenizer(
            batch_tokens,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        model_inputs = {
            "input_ids": encoding["input_ids"].to(device),
            "attention_mask": encoding["attention_mask"].to(device),
        }
        # token_type_ids is absent for DeBERTa-v3 but present for BERT/DistilBERT;
        # passing it when the model doesn't expect it raises an error.
        if "token_type_ids" in encoding:
            model_inputs["token_type_ids"] = encoding["token_type_ids"].to(device)

        with torch.no_grad():
            pred_ids = model(**model_inputs).logits.argmax(dim=-1).cpu().numpy()

        for batch_idx, record in enumerate(batch):
            word_ids = encoding.word_ids(batch_index=batch_idx)
            word_pred: dict[int, str] = {}
            for token_idx, word_id in enumerate(word_ids):
                if word_id is None or word_id in word_pred:
                    continue
                word_pred[word_id] = id2label.get(int(pred_ids[batch_idx][token_idx]), "O")

            pred_tags = [word_pred.get(i, "O") for i in range(len(record["tokens"]))]
            results.append(
                {
                    "idx": record.get("idx", batch_start + batch_idx),
                    "tokens": record["tokens"],
                    "sequence": record["sequence"],
                    "gold_tags": record["ner_tags"],
                    "pred_tags": pred_tags,
                }
            )

    return results


def predict_llama(
    examples: list[dict[str, Any]],
    model_path: str | pathlib.Path,
    cache_path: str | pathlib.Path,
    n_threads: int = 4,
    n_gpu_layers: int = 0,
    checkpoint_every: int = 50,
) -> list[dict[str, Any]]:
    """Run LLaMA zero-shot inference on examples with JSONL caching.

    Delegates to LLMPIIPipeline.predict_batch, then normalises the output
    schema to match the encoder record format (gold_tags / pred_tags).
    """
    pipeline = LLMPIIPipeline(
        model_path=str(model_path),
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
    )
    records = [
        {
            "tokens": example["tokens"],
            "sequence": example["sequence"],
            "ner_tags": example["ner_tags"],
        }
        for example in examples
    ]
    raw = pipeline.predict_batch(
        records,
        cache_path=str(cache_path),
        checkpoint_every=checkpoint_every,
    )
    raw.sort(key=lambda row: row.get("idx", 0))
    return [
        {
            "idx": row.get("idx", i),
            "tokens": row["tokens"],
            "sequence": row["sequence"],
            "gold_tags": row.get("ner_tags", []),
            "pred_tags": row.get("predicted_tags", []),
            "parse_ok": bool(row.get("parse_ok", False)),
        }
        for i, row in enumerate(raw)
    ]


def compute_metrics(records: list[dict[str, Any]], email_present: bool) -> dict[str, Any]:
    """Compute span-level F1, token-level FPR/FNR, and redaction leak rate.

    email_present controls whether EMAIL metrics are computed or returned as null
    (datasets with no email spans would produce misleading 0.0 EMAIL F1 otherwise).
    """
    gold_seqs: list[list[str]] = []
    pred_seqs: list[list[str]] = []
    for record in records:
        gold = list(record.get("gold_tags", []))
        pred = (list(record.get("pred_tags", [])) + ["O"] * len(gold))[: len(gold)]
        gold_seqs.append(gold)
        pred_seqs.append(pred)

    report = classification_report(
        gold_seqs,
        pred_seqs,
        mode="strict",
        scheme=IOB2,
        output_dict=True,
        zero_division=0,
    )
    token_rates = compute_token_rates(records)
    leak = compute_span_leak_rate(records)

    per_f1 = float(report.get("PER", {}).get("f1-score", 0.0))
    email_f1 = float(report.get("EMAIL", {}).get("f1-score", 0.0)) if email_present else None
    # When EMAIL is absent from the dataset, seqeval's macro avg artificially
    # includes it as 0.0, halving the headline number.  Use PER F1 directly.
    overall_f1 = (
        float(report.get("macro avg", {}).get("f1-score", 0.0))
        if email_present
        else per_f1
    )
    return {
        "per_f1": per_f1,
        "email_f1": email_f1,
        "overall_f1": overall_f1,
        "fpr": token_rates["fpr"],
        "fnr": token_rates["fnr"],
        "redaction_leak_rate": leak["redaction_leak_rate"],
        "leaked_span_count": leak["leaked_span_count"],
        "total_pii_span_count": leak["total_pii_span_count"],
    }


def compute_token_rates(records: list[dict[str, Any]]) -> dict[str, float]:
    """Compute token-level false-positive and false-negative rates across all records."""
    fp = fn = tp = tn = 0
    for record in records:
        gold = list(record.get("gold_tags", []))
        pred = (list(record.get("pred_tags", [])) + ["O"] * len(gold))[: len(gold)]
        for gold_tag, pred_tag in zip(gold, pred):
            gold_pii = gold_tag != "O"
            pred_pii = pred_tag != "O"
            if gold_pii and pred_pii:
                tp += 1
            elif not gold_pii and pred_pii:
                fp += 1
            elif gold_pii and not pred_pii:
                fn += 1
            else:
                tn += 1
    return {
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) else 0.0,
    }


def compute_span_leak_rate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the fraction of gold PII spans that have at least one token predicted O.

    A span 'leaks' if any token within it is missed by the model, meaning the
    redaction system would expose part of the entity in the masked output.
    """
    leaked = 0
    total = 0
    for record in records:
        gold = list(record.get("gold_tags", []))
        pred = (list(record.get("pred_tags", [])) + ["O"] * len(gold))[: len(gold)]
        for start, end, _entity in extract_spans(gold):
            total += 1
            if any(pred[pos] == "O" for pos in range(start, end)):
                leaked += 1
    return {
        "redaction_leak_rate": leaked / total if total else 0.0,
        "leaked_span_count": leaked,
        "total_pii_span_count": total,
    }


def _summary_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep the public summary stable and free of model-path noise."""
    return {
        "per_f1": metrics["per_f1"],
        "email_f1": metrics["email_f1"],
        "overall_f1": metrics["overall_f1"],
        "fpr": metrics["fpr"],
        "fnr": metrics["fnr"],
        "redaction_leak_rate": metrics["redaction_leak_rate"],
    }


def build_summary(
    dataset_name: str,
    stats: dict[str, Any],
    deberta_metrics: dict[str, Any],
    llama_metrics: dict[str, Any],
    distilbert_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the Day 7 summary using the final A/B/C approach labels."""
    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "n_sentences": stats["n_sentences"],
        "approaches": {
            "A": "distilbert",
            "B": "deberta",
            "C": "llama",
        },
        "deberta": _summary_metrics(deberta_metrics),
        "llama": _summary_metrics(llama_metrics)
        | {"parse_failure_rate": llama_metrics.get("parse_failure_rate", 0.0)},
    }
    if distilbert_metrics is not None:
        summary["distilbert"] = _summary_metrics(distilbert_metrics)
    return summary
