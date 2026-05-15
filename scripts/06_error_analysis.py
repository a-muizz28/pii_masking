"""Day 6 - Error analysis pipeline (CLI entrypoint).

Orchestrates all Day 6 artefacts by delegating to the library module
`src.pii_masking.day6_error_analysis`.  Run this script after Day 5
(05_evaluate_all.py) has produced the encoder and LLM prediction files.

Outputs
-------
  results/day6/error_buckets.json
  errors/manual_A.tsv             
  errors/manual_B.tsv             
  reports/figures/fig1_error_analysis.png
"""

import json
import logging
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pii_masking.day6_error_analysis import (
    BucketResult,
    build_tsv_rows,
    collect_error_instances,
    compute_buckets,
    load_predictions,
    markdown_table_str,
    plot_error_distribution,
    sample_distribution_str,
    save_tsv,
    stratified_sample,
    summary_table_str,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


APPROACH_A_FILE    = ROOT / "predictions" / "encoder" / "deberta_s42_predictions.jsonl"
APPROACH_B_FILE    = ROOT / "predictions" / "llm"     / "raw_outputs.jsonl"
RESULTS_DIR        = ROOT / "results"  / "day6"
ERROR_BUCKETS_JSON = RESULTS_DIR / "error_buckets.json"
ERRORS_DIR         = ROOT / "errors"
FIGURES_DIR        = ROOT / "reports" / "figures"
FIGURE_PNG         = FIGURES_DIR / "fig1_error_analysis.png"


def main() -> None:
    
    logger.info("Loading predictions...")
    try:
        records_a, records_b = load_predictions(APPROACH_A_FILE, APPROACH_B_FILE)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    logger.info("  Approach A: %d sentences", len(records_a))
    logger.info("  Approach B: %d sentences", len(records_b))

    # 2. Compute error buckets
    logger.info("Computing error buckets...")
    buckets_a = compute_buckets(records_a)
    buckets_b = compute_buckets(records_b)

    print("\n" + summary_table_str(buckets_a, buckets_b))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "approach_A": buckets_a.to_dict(),
        "approach_B": buckets_b.to_dict(),
    }
    ERROR_BUCKETS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved -> %s", ERROR_BUCKETS_JSON)

    # 3. Stratified error sampling
    logger.info("Collecting and sampling errors (seed=42, max=30 per approach)...")
    instances_a = collect_error_instances(records_a)
    instances_b = collect_error_instances(records_b)

    samples_a = stratified_sample(instances_a, max_total=30, seed=42)
    samples_b = stratified_sample(instances_b, max_total=30, seed=42)

    print()
    print(sample_distribution_str("Approach A", instances_a, samples_a))
    print()
    print(sample_distribution_str("Approach B", instances_b, samples_b))
    print()

    ERRORS_DIR.mkdir(parents=True, exist_ok=True)
    out_a = ERRORS_DIR / "manual_A.tsv"
    out_b = ERRORS_DIR / "manual_B.tsv"
    save_tsv(build_tsv_rows(samples_a, "A"), out_a)
    save_tsv(build_tsv_rows(samples_b, "B"), out_b)
    print(f"Saved {len(samples_a)} rows -> {out_a}")
    print(f"Saved {len(samples_b)} rows -> {out_b}")
    print()
    print("Valid sub_labels for annotation:")
    print("  FN: partial_span | rare_name | context_ambiguous | email_format_unseen")
    print("  FP: org_misclassified_as_per | common_noun | hallucinated_email")

    # 4. Error distribution plot
    logger.info("Generating error distribution figure...")
    plot_error_distribution(buckets_a, buckets_b, FIGURE_PNG, dpi=150)
    print(f"\nFigure saved -> {FIGURE_PNG}")

    # 5. Markdown table
    print("\nMarkdown table (copy and paste into report):\n")
    print(markdown_table_str(buckets_a, buckets_b))

    # 6. Final checklist
    print("""
Day 6 checklist
---------------
[x] results/day6/error_buckets.json  saved
[x] errors/manual_A.tsv             saved  (needs manual sub_label annotation)
[x] errors/manual_B.tsv             saved  (needs manual sub_label annotation)
[x] reports/figures/fig1_error_analysis.png  saved
[ ] MANUAL: open errors/manual_A.tsv & manual_B.tsv, fill sub_label column
[ ] MANUAL: update error analysis section of report with sub_label counts
[ ] Day 7: ai4privacy cross-domain check and final report review
""")


if __name__ == "__main__":
    main()
