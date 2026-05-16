"""Day 6 error-analysis pipeline.

This CLI mirrors notebook 06 and compares the final three reporting
approaches:

  A: DistilBERT-s42
  B: DeBERTa-s7
  C: LLaMA

Run it after Day 5 has produced encoder predictions and Day 4 has produced
the LLaMA prediction cache.

Outputs
-------
  results/day6/error_buckets.json
  errors/manual_A.tsv
  errors/manual_B.tsv
  errors/manual_C.tsv
  reports/figures/fig1_error_analysis.png
"""

import json
import logging
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pii_masking.day6_error_analysis import (
    build_tsv_rows,
    collect_error_instances,
    compute_buckets,
    load_encoder_predictions,
    load_llm_predictions,
    markdown_table_str,
    plot_error_distribution,
    sample_distribution_str,
    save_tsv,
    stratified_sample,
    summary_table_str,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


APPROACH_A_FILE    = ROOT / "predictions" / "encoder" / "distilbert_s42_predictions.jsonl"
APPROACH_B_FILE    = ROOT / "predictions" / "encoder" / "deberta_s7_predictions.jsonl"
APPROACH_C_FILE    = ROOT / "predictions" / "llm"     / "raw_outputs.jsonl"
RESULTS_DIR        = ROOT / "results"  / "day6"
ERROR_BUCKETS_JSON = RESULTS_DIR / "error_buckets.json"
ERRORS_DIR         = ROOT / "errors"
FIGURES_DIR        = ROOT / "reports" / "figures"
FIGURE_PNG         = FIGURES_DIR / "fig1_error_analysis.png"


def main() -> None:
    logger.info("Loading predictions...")
    try:
        records_a = load_encoder_predictions(APPROACH_A_FILE)
        records_b = load_encoder_predictions(APPROACH_B_FILE)
        records_c = load_llm_predictions(APPROACH_C_FILE)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    logger.info("  Approach A (DistilBERT-s42): %d sentences", len(records_a))
    logger.info("  Approach B (DeBERTa-s7): %d sentences", len(records_b))
    logger.info("  Approach C (LLaMA): %d sentences", len(records_c))

    logger.info("Computing error buckets...")
    models = {
        "DistilBERT-s42": compute_buckets(records_a),
        "DeBERTa-s7": compute_buckets(records_b),
        "LLaMA": compute_buckets(records_c),
    }

    print("\n" + summary_table_str(models))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "approach_A_distilbert_s42": models["DistilBERT-s42"].to_dict(),
        "approach_B_deberta_s7": models["DeBERTa-s7"].to_dict(),
        "approach_C_llama": models["LLaMA"].to_dict(),
    }
    ERROR_BUCKETS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved -> %s", ERROR_BUCKETS_JSON)

    logger.info("Collecting and sampling errors (seed=42, max=30 per approach)...")
    sample_inputs = {
        "A": ("Approach A (DistilBERT-s42)", collect_error_instances(records_a)),
        "B": ("Approach B (DeBERTa-s7)", collect_error_instances(records_b)),
        "C": ("Approach C (LLaMA)", collect_error_instances(records_c)),
    }

    ERRORS_DIR.mkdir(parents=True, exist_ok=True)
    print()
    for approach_key, (label, instances) in sample_inputs.items():
        samples = stratified_sample(instances, max_total=30, seed=42)
        out_path = ERRORS_DIR / f"manual_{approach_key}.tsv"
        print(sample_distribution_str(label, instances, samples))
        save_tsv(build_tsv_rows(samples, approach_key), out_path)
        print(f"Saved {len(samples)} rows -> {out_path}\n")

    print()
    print("Valid sub_labels for annotation:")
    print("  FN: partial_span | rare_name | context_ambiguous | email_format_unseen")
    print("  FP: org_misclassified_as_per | common_noun | hallucinated_email")

    logger.info("Generating error distribution figure...")
    plot_error_distribution(models, FIGURE_PNG, dpi=150)
    print(f"\nFigure saved -> {FIGURE_PNG}")

    print("\nMarkdown table (copy and paste into report):\n")
    print(markdown_table_str(models))

    print("""
Day 6 checklist
---------------
[x] results/day6/error_buckets.json  saved
[x] errors/manual_A.tsv              saved  (needs manual sub_label annotation)
[x] errors/manual_B.tsv              saved  (needs manual sub_label annotation)
[x] errors/manual_C.tsv              saved  (needs manual sub_label annotation)
[x] reports/figures/fig1_error_analysis.png  saved
[ ] MANUAL: open errors/manual_*.tsv and fill sub_label column
[ ] MANUAL: update error analysis section of report with sub_label counts
[ ] Day 7: WikiANN cross-domain evaluation and final report review
""")


if __name__ == "__main__":
    main()
