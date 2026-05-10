"""Inject synthetic email addresses into cleaned parquet datasets."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pii_masking import data as data_mod
from pii_masking import injection as inj_mod

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def parquet_to_examples(path: Path) -> list[dict]:
    """Load a parquet file and return a list of example dicts with Python lists."""
    df = pd.read_parquet(path, engine="pyarrow")
    df["tokens"] = df["tokens"].apply(list)
    df["ner_tags"] = df["ner_tags"].apply(list)
    return df.to_dict(orient="records")


def examples_to_parquet(examples: list[dict], out_path: Path) -> None:
    """Save examples list (possibly with injected_email key) as parquet."""
    rows = []
    for e in examples:
        n_per = sum(1 for t in e["ner_tags"] if t == "B-PER")
        n_email = sum(1 for t in e["ner_tags"] if t == "B-EMAIL")
        bucket = (
            "zero_per" if n_per == 0 else "one_per" if n_per == 1 else "multi_per"
        )
        row = {
            "lang": e.get("lang", "en"),
            "sequence": e.get("sequence", ""),
            "tokens": e["tokens"],
            "ner_tags": e["ner_tags"],
            "n_tokens": len(e["tokens"]),
            "n_per_spans": n_per,
            "n_email_spans": n_email,
            "entity_density_bucket": bucket,
        }
        if "injected_email" in e:
            row["injected_email_str"] = e["injected_email"]["email"]
            row["injected_email_context"] = e["injected_email"]["context"]
            row["injected_email_multi_token"] = e["injected_email"]["multi_token"]
            row["injected_email_inserted_at"] = e["injected_email"]["inserted_at"]
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", index=False)
    print(f"  Saved {len(df)} rows -> {out_path}")


def _safe(text: str) -> str:
    """Encode to ASCII replacing non-ASCII chars for terminal-safe display."""
    return text.encode("ascii", errors="replace").decode("ascii")


def print_sample_injections(examples: list[dict], n: int = 10, seed: int = 42) -> None:
    """Print n randomly sampled injected examples in aligned table format."""
    rng = np.random.default_rng(seed)
    injected = [e for e in examples if "injected_email" in e]
    if not injected:
        print("  (no injected examples found)")
        return

    sample_size = min(n, len(injected))
    indices = rng.choice(len(injected), size=sample_size, replace=False)
    chosen = [injected[i] for i in sorted(indices)]

    print("\n" + "=" * 70)
    print("  Sample injected examples")
    print("=" * 70)
    for ex in chosen:
        meta = ex["injected_email"]
        mode = "multi-token" if meta["multi_token"] else "single-token"
        print(f"\n[context={meta['context']} - {mode}]")
        safe_tokens = [_safe(t) for t in ex["tokens"]]
        col_w = [max(len(tok), len(tag)) + 2 for tok, tag in zip(safe_tokens, ex["ner_tags"])]
        tok_line = "".join(tok.ljust(w) for tok, w in zip(safe_tokens, col_w))
        tag_line = "".join(tag.ljust(w) for tag, w in zip(ex["ner_tags"], col_w))
        print(f"Tokens: {tok_line}")
        print(f"Tags:   {tag_line}")


def main() -> None:
    # 1. Load config
    config_path = DATA_DIR / "injection_config.json"
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)

    # 2. Load parquets — track original split indices
    print("Loading train_clean.parquet …")
    train_examples = parquet_to_examples(PROCESSED_DIR / "train_clean.parquet")
    print("Loading val_clean.parquet …")
    val_examples = parquet_to_examples(PROCESSED_DIR / "val_clean.parquet")
    print("Loading test_clean.parquet …")
    test_examples = parquet_to_examples(PROCESSED_DIR / "test_clean.parquet")

    n_train_orig = len(train_examples)
    n_val_orig = len(val_examples)

    # 3. Inject into train+val combined (same config/domains)
    combined = train_examples + val_examples
    print(f"\nInjecting emails into train+val combined ({len(combined)} examples) …")
    injected_combined, train_report = inj_mod.inject_dataset(
        combined, config, split="train", seed=42
    )

    # 4. Inject into test
    print(f"Injecting emails into test ({len(test_examples)} examples) …")
    injected_test, test_report = inj_mod.inject_dataset(
        test_examples, config, split="test", seed=123
    )

    # 5. Re-split injected combined back by original proportions
    # The injection shuffled the list; re-split by position count to maintain ratio
    total_combined = len(injected_combined)
    # Recompute split sizes to match original ratio
    val_ratio = n_val_orig / (n_train_orig + n_val_orig)
    n_val_new = round(total_combined * val_ratio)
    n_train_new = total_combined - n_val_new

    injected_train = injected_combined[:n_train_new]
    injected_val = injected_combined[n_train_new:]

    print(f"\n  Re-split: train={len(injected_train)}  val={len(injected_val)}")

    # 6. Save parquets
    examples_to_parquet(injected_train, PROCESSED_DIR / "train_with_emails.parquet")
    examples_to_parquet(injected_val, PROCESSED_DIR / "val_with_emails.parquet")
    examples_to_parquet(injected_test, PROCESSED_DIR / "test_injected.parquet")

    # 7. Save injection report
    combined_report = {"train_val": train_report, "test": test_report}
    report_path = PROCESSED_DIR / "injection_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(combined_report, f, indent=2)
    print(f"  Report -> {report_path}")

    # Print summary
    print(f"\n  Train+val injection: {train_report['injected']} emails injected")
    print(f"    single-token: {train_report['single_token_count']}")
    print(f"    multi-token : {train_report['multi_token_count']}")
    print(f"    contexts    : {train_report['context_distribution']}")
    print(f"  Test injection : {test_report['injected']} emails injected")
    print(f"    single-token: {test_report['single_token_count']}")
    print(f"    multi-token : {test_report['multi_token_count']}")
    print(f"    contexts    : {test_report['context_distribution']}")

    # 8. Print 10 sample injected sentences
    print_sample_injections(injected_train + injected_val, n=10, seed=42)

    print("\nInjection complete.")


if __name__ == "__main__":
    main()
