"""Independent Hugging Face dataset evaluation - WikiANN English test split."""

from __future__ import annotations

import json
import logging
import pathlib
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

from pii_masking.day1_data import validate_example
from pii_masking.day4_llm_inference import LLMPIIPipeline, _parse_failure_log

logger = logging.getLogger(__name__)

LABEL2ID = {"O": 0, "B-PER": 1, "I-PER": 2, "B-EMAIL": 3, "I-EMAIL": 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

NAME_LABELS = {
    "FIRSTNAME",
    "GIVENNAME",
    "GIVENNAME1",
    "GIVENNAME2",
    "LASTNAME",
    "LASTNAME1",
    "LASTNAME2",
    "MIDDLENAME",
    "NAME",
    "FULLNAME",
    "SURNAME",
    "USERNAME",
}
EMAIL_LABELS = {"EMAIL"}
TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class EvalPaths:
    results_dir: pathlib.Path
    stats_path: pathlib.Path
    deberta_path: pathlib.Path
    llama_path: pathlib.Path
    summary_path: pathlib.Path
    llama_cache_path: pathlib.Path


def wikiann_paths(results_dir: str | pathlib.Path = "results/wikiann") -> EvalPaths:
    root = pathlib.Path(results_dir)
    return EvalPaths(
        results_dir=root,
        stats_path=root / "data_stats.json",
        deberta_path=root / "deberta_results.json",
        llama_path=root / "llama_results.json",
        summary_path=root / "eval_summary.json",
        llama_cache_path=root / "llama_outputs.jsonl",
    )


def day7_paths(results_dir: str | pathlib.Path = "results/day7") -> EvalPaths:
    root = pathlib.Path(results_dir)
    return EvalPaths(
        results_dir=root,
        stats_path=root / "independent_data_stats.json",
        deberta_path=root / "independent_deberta.json",
        llama_path=root / "independent_llama.json",
        summary_path=root / "independent_eval_summary.json",
        llama_cache_path=root / "independent_llama_outputs.jsonl",
    )


def _tokenize_with_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    for match in TOKEN_RE.finditer(text):
        tokens.append(match.group(0))
        offsets.append((match.start(), match.end()))
    return tokens, offsets


def _normalise_mask(mask: Any) -> list[dict[str, Any]]:
    if isinstance(mask, str):
        try:
            mask = json.loads(mask)
        except json.JSONDecodeError:
            return []
    return mask if isinstance(mask, list) else []


def _map_privacy_label(label: str) -> str | None:
    normalized = str(label).upper().replace("-", "_")
    if normalized in EMAIL_LABELS:
        return "EMAIL"
    if normalized in NAME_LABELS:
        return "PER"
    return None


def ai4privacy_row_to_example(row: dict[str, Any], idx: int) -> dict[str, Any] | None:
    """Convert one ai4privacy row into project BIO tags for PER and EMAIL only."""
    text = str(row.get("source_text") or "")
    if not text.strip():
        return None

    tokens, offsets = _tokenize_with_offsets(text)
    if not tokens:
        return None

    tags = ["O"] * len(tokens)
    masks = _normalise_mask(row.get("privacy_mask", []))
    mapped_spans: list[tuple[int, int, str]] = []
    for item in masks:
        if not isinstance(item, dict):
            continue
        entity = _map_privacy_label(str(item.get("label", "")))
        if entity is None:
            continue
        try:
            start = int(item["start"])
            end = int(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        mapped_spans.append((start, end, entity))

    merged_spans = merge_adjacent_name_spans(text, mapped_spans)
    for start, end, entity in merged_spans:
        covered = [
            tok_idx
            for tok_idx, (tok_start, tok_end) in enumerate(offsets)
            if tok_start < end and tok_end > start
        ]
        if not covered:
            continue

        for pos, tok_idx in enumerate(covered):
            if tags[tok_idx] != "O":
                continue
            tags[tok_idx] = f"B-{entity}" if pos == 0 else f"I-{entity}"

    return {
        "idx": idx,
        "source_dataset": "ai4privacy/pii-masking-300k",
        "lang": row.get("language", "unknown"),
        "sequence": text,
        "tokens": tokens,
        "ner_tags": tags,
    }


def merge_adjacent_name_spans(
    text: str,
    spans: list[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    """Merge adjacent PER spans separated only by whitespace or light punctuation."""
    if not spans:
        return []

    sorted_spans = sorted(spans, key=lambda span: (span[0], span[1]))
    merged: list[tuple[int, int, str]] = []
    for start, end, entity in sorted_spans:
        if not merged:
            merged.append((start, end, entity))
            continue

        prev_start, prev_end, prev_entity = merged[-1]
        gap = text[prev_end:start]
        can_merge_name = (
            prev_entity == "PER"
            and entity == "PER"
            and re.fullmatch(r"[\s.'-]{0,4}", gap or "") is not None
        )
        if can_merge_name:
            merged[-1] = (prev_start, end, "PER")
        else:
            merged.append((start, end, entity))
    return merged


def load_ai4privacy_examples(
    dataset_name: str = "ai4privacy/pii-masking-300k",
    split: str = "train",
    max_records: int = 1000,
    language: str = "English",
    min_entity_spans: int = 1,
) -> list[dict[str, Any]]:
    """Load an external Hugging Face PII dataset and map it to project BIO labels."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split, streaming=True)
    examples: list[dict[str, Any]] = []
    for idx, row in enumerate(dataset):
        row_language = str(row.get("language", ""))
        if language and row_language.lower() != language.lower():
            continue
        example = ai4privacy_row_to_example(row, idx)
        if example is None:
            continue
        span_count = count_spans(example["ner_tags"], "PER") + count_spans(
            example["ner_tags"], "EMAIL"
        )
        if span_count < min_entity_spans:
            continue
        examples.append(example)
        if len(examples) >= max_records:
            break

    if not examples:
        raise ValueError(
            f"No usable examples loaded from {dataset_name} split={split!r} language={language!r}"
        )
    return examples


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
    _parse_failure_log.clear()
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

    return {
        "per_f1": float(report.get("PER", {}).get("f1-score", 0.0)),
        "email_f1": (
            float(report.get("EMAIL", {}).get("f1-score", 0.0)) if email_present else None
        ),
        "overall_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "fpr": token_rates["fpr"],
        "fnr": token_rates["fnr"],
        "redaction_leak_rate": leak["redaction_leak_rate"],
        "leaked_span_count": leak["leaked_span_count"],
        "total_pii_span_count": leak["total_pii_span_count"],
    }


def compute_token_rates(records: list[dict[str, Any]]) -> dict[str, float]:
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


def choose_best_deberta_model(models_dir: str | pathlib.Path = "models") -> pathlib.Path:
    """Pick the best local DeBERTa seed by result JSON, falling back to seed 42."""
    models_dir = pathlib.Path(models_dir)
    candidates: list[tuple[float, pathlib.Path]] = []
    for result_path in models_dir.glob("deberta_seed*/deberta_seed*_result.json"):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        metrics = payload.get("test_metrics") or payload.get("val_metrics") or payload
        score = (
            metrics.get("eval_overall_f1")
            or metrics.get("overall_f1")
            or metrics.get("f1")
            or 0.0
        )
        model_dir = result_path.parent / "best_model"
        if model_dir.exists():
            candidates.append((float(score), model_dir))

    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    return models_dir / "deberta_seed42" / "best_model"


def build_summary(
    dataset_name: str,
    stats: dict[str, Any],
    deberta_metrics: dict[str, Any],
    llama_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        "n_sentences": stats["n_sentences"],
        "deberta": {
            "per_f1": deberta_metrics["per_f1"],
            "email_f1": deberta_metrics["email_f1"],
            "overall_f1": deberta_metrics["overall_f1"],
            "fpr": deberta_metrics["fpr"],
            "fnr": deberta_metrics["fnr"],
            "redaction_leak_rate": deberta_metrics["redaction_leak_rate"],
        },
        "llama": {
            "per_f1": llama_metrics["per_f1"],
            "email_f1": llama_metrics["email_f1"],
            "overall_f1": llama_metrics["overall_f1"],
            "fpr": llama_metrics["fpr"],
            "fnr": llama_metrics["fnr"],
            "redaction_leak_rate": llama_metrics["redaction_leak_rate"],
            "parse_failure_rate": llama_metrics.get("parse_failure_rate", 0.0),
        },
    }
