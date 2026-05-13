"""Day 1 data loading, validation, and splitting utilities."""

import copy
import hashlib
import json
import logging
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

VALID_TAGS = {"O", "B-PER", "I-PER", "B-EMAIL", "I-EMAIL"}
PUNCT_TOKENS = {".", "!", "?", ";"}


def validate_example(example: dict, idx: int) -> dict:
    """Validate a single NER example and repair BIO violations.

    Returns a result dict with keys: valid, violations, fixes_applied,
    and optionally fixed_example if any repairs were made.
    """
    result = {"valid": True, "violations": [], "fixes_applied": []}
    tokens = example.get("tokens", [])
    ner_tags = example.get("ner_tags", [])

    # Check 1: length alignment
    if len(tokens) != len(ner_tags):
        result["valid"] = False
        result["violations"].append(
            f"Length mismatch: {len(tokens)} tokens vs {len(ner_tags)} tags"
        )
        return result

    # Check 2: unknown tags
    unknown = [t for t in ner_tags if t not in VALID_TAGS]
    if unknown:
        result["valid"] = False
        result["violations"].append(f"Unknown tags: {sorted(set(unknown))}")
        return result

    # Check 3: non-empty tokens
    if len(tokens) == 0 or any(t == "" for t in tokens):
        result["valid"] = False
        result["violations"].append("Empty token list or contains empty string tokens")
        return result

    # Check 4: BIO well-formedness — fix orphan I-X
    fixed_tags = list(ner_tags)
    for j, tag in enumerate(fixed_tags):
        if not tag.startswith("I-"):
            continue
        entity_type = tag[2:]
        prev_tag = fixed_tags[j - 1] if j > 0 else "O"
        if prev_tag not in (f"B-{entity_type}", f"I-{entity_type}"):
            fixed_tags[j] = f"B-{entity_type}"
            fix_msg = f"Converted orphan I-{entity_type} at position {j} to B-{entity_type}"
            result["fixes_applied"].append(fix_msg)
            result["violations"].append(fix_msg)

    # Check 5: language
    if example.get("lang") != "en":
        result["valid"] = False
        result["violations"].append(f"Unexpected language: {example.get('lang')!r}")

    if result["fixes_applied"]:
        fixed = copy.deepcopy(example)
        fixed["ner_tags"] = fixed_tags
        result["fixed_example"] = fixed

    return result


def _count_spans(ner_tags: list[str], entity: str) -> int:
    """Count the number of B-{entity} tags (i.e., span starts)."""
    return sum(1 for t in ner_tags if t == f"B-{entity}")


def _density_bucket(n_per: int) -> str:
    """Map PER span count to entity-density bucket label."""
    if n_per == 0:
        return "zero_per"
    if n_per == 1:
        return "one_per"
    return "multi_per"


def load_and_validate(json_path: str | Path) -> tuple[list[dict], dict]:
    """Load a WikiNeural JSON file and validate every example.

    Returns cleaned_examples (valid, fixes applied) and a detailed report dict.
    """
    json_path = Path(json_path)
    with json_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    logger.info("Processing %d examples from %s…", len(raw), json_path.name)

    cleaned: list[dict] = []
    invalid_indices: list[int] = []
    fix_log: list[dict] = []
    bio_violations_fixed = 0
    tag_counts: dict[str, int] = {}
    lengths: list[int] = []

    for idx, example in enumerate(raw):
        val = validate_example(example, idx)
        for tag in example.get("ner_tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        if not val["valid"]:
            invalid_indices.append(idx)
            continue

        effective = val.get("fixed_example", example)
        for fix in val["fixes_applied"]:
            bio_violations_fixed += 1
            fix_log.append({"idx": idx, "fix_description": fix})

        cleaned.append(effective)
        lengths.append(len(effective["tokens"]))

    per_counts = [_count_spans(e["ner_tags"], "PER") for e in cleaned]
    zero_per = sum(1 for c in per_counts if c == 0)
    one_per = sum(1 for c in per_counts if c == 1)
    two_plus_per = sum(1 for c in per_counts if c >= 2)

    email_count = sum(_count_spans(e["ner_tags"], "EMAIL") for e in cleaned)

    length_stats: dict[str, float] = {}
    if lengths:
        sorted_l = sorted(lengths)
        length_stats = {
            "min": int(min(sorted_l)),
            "max": int(max(sorted_l)),
            "mean": round(statistics.mean(sorted_l), 2),
            "median": float(statistics.median(sorted_l)),
            "p95": float(np.percentile(sorted_l, 95)),
        }

    report = {
        "source_file": str(json_path),
        "total_examples": len(raw),
        "valid_examples": len(cleaned),
        "invalid_examples": len(invalid_indices),
        "bio_violations_fixed": bio_violations_fixed,
        "tag_distribution": tag_counts,
        "length_stats": length_stats,
        "per_span_counts": {
            "0_per": zero_per,
            "1_per": one_per,
            "2plus_per": two_plus_per,
        },
        "email_span_count": email_count,
        "invalid_indices": invalid_indices,
        "fix_log": fix_log,
    }

    logger.info(
        "  valid=%d  invalid=%d  bio_fixes=%d",
        len(cleaned),
        len(invalid_indices),
        bio_violations_fixed,
    )
    return cleaned, report


def stratified_split(
    examples: list[dict],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split examples into train and val sets stratified by entity-density bucket.

    Buckets: zero_per (0 PER spans), one_per (1 PER span), multi_per (2+ PER spans).
    Uses StratifiedShuffleSplit with n_splits=1.
    """
    labels = [_density_bucket(_count_spans(e["ner_tags"], "PER")) for e in examples]
    X = list(range(len(examples)))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(sss.split(X, labels))

    train_examples = [examples[i] for i in train_idx]
    val_examples = [examples[i] for i in val_idx]

    from collections import Counter
    train_buckets = Counter(_density_bucket(_count_spans(e["ner_tags"], "PER")) for e in train_examples)
    val_buckets = Counter(_density_bucket(_count_spans(e["ner_tags"], "PER")) for e in val_examples)

    logger.info(
        "Split: train=%d %s | val=%d %s",
        len(train_examples),
        dict(train_buckets),
        len(val_examples),
        dict(val_buckets),
    )
    return train_examples, val_examples


def save_as_parquet(examples: list[dict], out_path: str | Path) -> None:
    """Convert a list of example dicts to a DataFrame and save as parquet.

    Columns: lang, sequence, tokens, ner_tags, n_tokens, n_per_spans,
    n_email_spans, entity_density_bucket.
    """
    out_path = Path(out_path)
    rows = []
    for e in examples:
        n_per = _count_spans(e["ner_tags"], "PER")
        rows.append(
            {
                "lang": e.get("lang", "en"),
                "sequence": e.get("sequence", ""),
                "tokens": e["tokens"],
                "ner_tags": e["ner_tags"],
                "n_tokens": len(e["tokens"]),
                "n_per_spans": n_per,
                "n_email_spans": _count_spans(e["ner_tags"], "EMAIL"),
                "entity_density_bucket": _density_bucket(n_per),
            }
        )
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", index=False)
    logger.info("Saved %d rows -> %s", len(df), out_path)


def compute_checksums(file_paths: list[str | Path]) -> dict[str, str]:
    """Compute SHA-256 hex digest for each file. Returns {filename: hex_digest}."""
    result: dict[str, str] = {}
    for fp in file_paths:
        fp = Path(fp)
        h = hashlib.sha256()
        with fp.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        result[fp.name] = h.hexdigest()
    return result
