"""Validate raw WikiNeural JSON files, split into train/val, save parquets and report."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pii_masking import day1_data as data_mod


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"


def print_summary(name: str, report: dict) -> None:
    """Print a human-readable validation summary for one split."""
    print("\n" + "=" * 60)
    print(f"  {name.upper()}")
    print("=" * 60)
    print(f"  Source       : {report['source_file']}")
    print(f"  Total        : {report['total_examples']}")
    print(f"  Valid        : {report['valid_examples']}")
    print(f"  Invalid      : {report['invalid_examples']}")
    print(f"  BIO fixes    : {report['bio_violations_fixed']}")
    ls = report["length_stats"]
    print(
        f"  Token lengths: min={ls['min']}  max={ls['max']}  "
        f"mean={ls['mean']:.1f}  median={ls['median']:.1f}  p95={ls['p95']:.1f}"
    )
    psc = report["per_span_counts"]
    print(
        f"  PER spans    : 0-span={psc['0_per']}  "
        f"1-span={psc['1_per']}  2+-span={psc['2plus_per']}"
    )
    print(f"  EMAIL spans  : {report['email_span_count']}")
    if report["invalid_indices"]:
        print(f"  Invalid idx  : {report['invalid_indices'][:10]} (showing first 10)")
    if report["fix_log"]:
        print("  Sample fixes :")
        for entry in report["fix_log"][:5]:
            print(f"    [{entry['idx']}] {entry['fix_description']}")


def save_histogram(
    train_lengths: list,
    test_lengths: list,
    out_path: Path,
) -> None:
    """Save a dual token-length histogram with a 256-token cutoff marker."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, lengths, title in zip(
        axes,
        [train_lengths, test_lengths],
        [f"Train (n={len(train_lengths):,})", f"Test (n={len(test_lengths):,})"],
    ):
        ax.hist(lengths, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(256, color="crimson", linestyle="--", linewidth=1.5, label="max_length=256")
        ax.set_xlabel("Token count per sentence")
        ax.set_ylabel("Frequency")
        ax.set_title(title)
        ax.legend()

    fig.suptitle(
        "Token length distribution - WikiNeural English"
        f" (n={len(train_lengths) + len(test_lengths):,})",
        fontsize=13,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Histogram -> {out_path}")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and validate
    train_examples, train_report = data_mod.load_and_validate(RAW_DIR / "train_data.json")
    test_examples, test_report = data_mod.load_and_validate(RAW_DIR / "test_data.json")

    print_summary("train", train_report)
    print_summary("test", test_report)

    # 2. Save merged validation report
    merged_report = {"train": train_report, "test": test_report}
    report_path = PROCESSED_DIR / "data_validation_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(merged_report, f, indent=2)
    print(f"\n  Report -> {report_path}")

    # 3. Stratified split on train
    train_split, val_split = data_mod.stratified_split(train_examples, val_ratio=0.15, seed=42)
    print(f"\n  Train split : {len(train_split)} train  {len(val_split)} val")

    # 4. Save parquets
    data_mod.save_as_parquet(train_split, PROCESSED_DIR / "train_clean.parquet")
    data_mod.save_as_parquet(val_split, PROCESSED_DIR / "val_clean.parquet")
    data_mod.save_as_parquet(test_examples, PROCESSED_DIR / "test_clean.parquet")

    # 5. Checksums
    checksums = data_mod.compute_checksums([
        RAW_DIR / "train_data.json",
        RAW_DIR / "test_data.json",
    ])
    checksum_path = DATA_DIR / "checksums.txt"
    with checksum_path.open("w", encoding="utf-8") as f:
        for fname, digest in checksums.items():
            f.write(f"{digest}  {fname}\n")
    print(f"  Checksums  -> {checksum_path}")
    for fname, digest in checksums.items():
        print(f"    {fname}: {digest}")

    # 6. Token-length histogram
    train_lengths = [len(ex["tokens"]) for ex in train_examples]
    test_lengths = [len(ex["tokens"]) for ex in test_examples]
    save_histogram(train_lengths, test_lengths, FIGURES_DIR / "token_length_distribution.png")

    # Coverage note
    max_len = max(train_lengths + test_lengths)
    all_lengths = train_lengths + test_lengths
    coverage_pct = 100 * sum(1 for length in all_lengths if length <= 256) / len(all_lengths)
    if coverage_pct < 99:
        print(
            f"  NOTE: max_length=256 covers only {coverage_pct:.1f}% of sentences "
            f"(max observed={max_len})"
        )
    else:
        print(f"  max_length=256 covers {coverage_pct:.2f}% of all sentences (max={max_len})")

    print("\nValidation complete. Report saved to data/processed/data_validation_report.json")


if __name__ == "__main__":
    main()
