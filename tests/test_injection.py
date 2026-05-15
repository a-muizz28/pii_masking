"""Pytest tests for Day 1: data validation and email injection."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pii_masking import day1_data as data_mod
from pii_masking import day1_injection as inj_mod

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "data" / "injection_config.json"


def _load_config() -> dict:
    with CONFIG_PATH.open() as f:
        return json.load(f)


def _load_parquet_examples(path: Path) -> list[dict]:
    df = pd.read_parquet(path, engine="pyarrow")
    df["tokens"] = df["tokens"].apply(list)
    df["ner_tags"] = df["ner_tags"].apply(list)
    return df.to_dict(orient="records")


def _is_bio_valid(tags: list[str]) -> bool:
    for j, tag in enumerate(tags):
        if not tag.startswith("I-"):
            continue
        entity = tag[2:]
        prev = tags[j - 1] if j > 0 else "O"
        if prev not in (f"B-{entity}", f"I-{entity}"):
            return False
    return True


def test_bio_well_formedness_after_injection() -> None:
    """50 random injected train examples must have no orphan I-X tags."""
    path = PROCESSED_DIR / "train_with_emails.parquet"
    if not path.exists():
        pytest.skip("train_with_emails.parquet not found - run 01_inject_emails.py first")

    examples = _load_parquet_examples(path)
    rng = np.random.default_rng(42)
    sample = [examples[i] for i in rng.choice(len(examples), size=min(50, len(examples)), replace=False)]

    for ex in sample:
        tags = ex["ner_tags"]
        assert _is_bio_valid(tags), (
            f"BIO violation in example tags: {tags}"
        )


def test_round_trip_email_recovery() -> None:
    """For every injected example, reconstructed email must match stored email."""
    path = PROCESSED_DIR / "train_with_emails.parquet"
    if not path.exists():
        pytest.skip("train_with_emails.parquet not found")

    df = pd.read_parquet(path, engine="pyarrow")
    df["tokens"] = df["tokens"].apply(list)
    df["ner_tags"] = df["ner_tags"].apply(list)

    # Restrict the check to rows where the injection step stored source metadata.
    has_email_col = "injected_email_str" in df.columns
    if not has_email_col:
        pytest.skip("No injected_email_str column - parquet may not contain metadata")

    injected_rows = df[df["injected_email_str"].notna()]

    for _, row in injected_rows.iterrows():
        tokens = list(row["tokens"])
        tags = list(row["ner_tags"])
        stored_email: str = row["injected_email_str"]
        multi_token: bool = bool(row.get("injected_email_multi_token", False))

        # Rebuild the injected email span from the BIO start position.
        b_email_positions = [i for i, t in enumerate(tags) if t == "B-EMAIL"]
        assert len(b_email_positions) >= 1, "Expected at least one B-EMAIL tag"
        b_pos = b_email_positions[0]

        if not multi_token:
            reconstructed = tokens[b_pos]
        else:
            # Multi-token emails are represented as local_part, "@", domain.
            assert b_pos + 2 < len(tokens), "Multi-token email requires 3 slots"
            reconstructed = f"{tokens[b_pos]}@{tokens[b_pos + 2]}"

        assert reconstructed == stored_email, (
            f"Email mismatch: reconstructed={reconstructed!r} stored={stored_email!r}"
        )


def test_disjoint_domain_pools() -> None:
    """Train and test domain pools must be completely disjoint."""
    config = _load_config()
    train_personal = set(config["train"]["personal_domains"])
    test_personal = set(config["test"]["personal_domains"])
    train_corporate = set(config["train"]["corporate_domains"])
    test_corporate = set(config["test"]["corporate_domains"])

    assert train_personal & test_personal == set(), (
        f"Shared personal domains: {train_personal & test_personal}"
    )
    assert train_corporate & test_corporate == set(), (
        f"Shared corporate domains: {train_corporate & test_corporate}"
    )


def test_injection_rate() -> None:
    """Injection rate for train set must be within [0.55, 0.65]."""
    path = PROCESSED_DIR / "train_with_emails.parquet"
    if not path.exists():
        pytest.skip("train_with_emails.parquet not found")

    df = pd.read_parquet(path, engine="pyarrow")
    df["ner_tags"] = df["ner_tags"].apply(list)

    total_candidates = sum(
        1 for tags in df["ner_tags"] if any(t == "B-PER" for t in tags)
    )
    injected_count = sum(
        1 for tags in df["ner_tags"] if any(t == "B-EMAIL" for t in tags)
    )

    # Include validation data when present so the configured split rate is tested.
    val_path = PROCESSED_DIR / "val_with_emails.parquet"
    if val_path.exists():
        val_df = pd.read_parquet(val_path, engine="pyarrow")
        val_df["ner_tags"] = val_df["ner_tags"].apply(list)
        total_candidates += sum(
            1 for tags in val_df["ner_tags"] if any(t == "B-PER" for t in tags)
        )
        injected_count += sum(
            1 for tags in val_df["ner_tags"] if any(t == "B-EMAIL" for t in tags)
        )

    assert total_candidates > 0, "No PER-containing candidates found"
    rate = injected_count / total_candidates
    assert 0.55 <= rate <= 0.65, (
        f"Injection rate {rate:.3f} outside [0.55, 0.65]"
    )


def test_single_vs_multi_token_ratio() -> None:
    """Single-token emails must constitute 85-95% of all injected emails."""
    path = PROCESSED_DIR / "train_with_emails.parquet"
    if not path.exists():
        pytest.skip("train_with_emails.parquet not found")

    df = pd.read_parquet(path, engine="pyarrow")
    if "injected_email_multi_token" not in df.columns:
        pytest.skip("injected_email_multi_token column missing")

    injected = df[df["injected_email_str"].notna()]
    single = int(injected["injected_email_multi_token"].eq(False).sum())
    multi = int(injected["injected_email_multi_token"].eq(True).sum())
    total = single + multi
    assert total > 0, "No injected examples found"

    ratio = single / total
    assert 0.85 <= ratio <= 0.95, (
        f"Single-token ratio {ratio:.3f} outside [0.85, 0.95] "
        f"(single={single}, multi={multi})"
    )


def test_validate_example_fixes_orphan_i_per() -> None:
    """validate_example must detect and repair an orphan I-PER tag."""
    example = {
        "lang": "en",
        "tokens": ["X", "Y", "Z"],
        "ner_tags": ["O", "I-PER", "O"],
        "sequence": "X Y Z",
    }
    result = data_mod.validate_example(example, idx=0)

    assert any("Converted orphan I-PER" in fix for fix in result["fixes_applied"]), (
        f"Expected orphan I-PER fix, got: {result['fixes_applied']}"
    )
    assert result["fixed_example"]["ner_tags"] == ["O", "B-PER", "O"], (
        f"Expected ['O', 'B-PER', 'O'], got: {result['fixed_example']['ner_tags']}"
    )


def test_length_mismatch_is_invalid() -> None:
    """An example with mismatched token and tag counts must be marked invalid."""
    example = {
        "lang": "en",
        "tokens": ["A", "B", "C"],
        "ner_tags": ["O", "B-PER"],
        "sequence": "A B C",
    }
    result = data_mod.validate_example(example, idx=0)
    assert result["valid"] is False, "Expected valid=False for length mismatch"
