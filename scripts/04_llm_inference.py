"""Day 4: LLaMA zero-shot PII inference on the test set."""

import argparse
import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# Resolve project root so imports work when run from any cwd
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
src_path = os.path.join(_project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pii_masking.day4_llm_inference import LLMPIIPipeline, _parse_failure_log
from pii_masking.day4_evaluate import compute_llm_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Day 4: LLaMA zero-shot PII inference")
    p.add_argument("--model-path", default="models/llama/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
    p.add_argument("--data-path", default="data/processed/test_injected.parquet")
    p.add_argument("--hf-dataset-path", default="data/processed/hf_dataset")
    p.add_argument("--cache-path", default="predictions/llm/raw_outputs.jsonl")
    p.add_argument("--results-dir", default="results/day4_llm_inference")
    p.add_argument("--n-threads", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-smoke-test", action="store_true")
    return p.parse_args()



def ensure_model(model_path: str, skip_download: bool) -> None:
    if os.path.exists(model_path):
        print(f"Model found: {model_path}")
        return
    if skip_download:
        raise FileNotFoundError(
            f"Model not found at {model_path} and --skip-download was set."
        )
    print("Model not found. Downloading from HuggingFace …")
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
        filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        local_dir="models/llama/",
    )
    print("Download complete.")



def load_test_records(data_path: str, hf_dataset_path: str) -> list[dict]:
    records = []

    # Try parquet first
    if os.path.exists(data_path):
        try:
            import pandas as pd
            df = pd.read_parquet(data_path)
            for _, row in df.iterrows():
                tokens = row["tokens"] if isinstance(row["tokens"], list) else list(row["tokens"])
                ner_tags = row["ner_tags"] if isinstance(row["ner_tags"], list) else list(row["ner_tags"])
                sequence = str(row["sequence"]) if "sequence" in row and row["sequence"] else " ".join(tokens)
                records.append({"tokens": tokens, "ner_tags": ner_tags, "sequence": sequence})
            print(f"Loaded {len(records)} test records from parquet.")
            return records
        except Exception as e:
            print(f"Parquet load failed ({e}), falling back to HF DatasetDict.")

    # Fallback: HF DatasetDict
    from datasets import load_from_disk
    ds = load_from_disk(hf_dataset_path)
    split = ds["test"]
    for row in split:
        tokens = list(row["tokens"])
        ner_tags = list(row["ner_tags"]) if "ner_tags" in row else []
        sequence = str(row["sequence"]) if "sequence" in row else " ".join(tokens)
        records.append({"tokens": tokens, "ner_tags": ner_tags, "sequence": sequence})
    print(f"Loaded {len(records)} test records from HF DatasetDict.")
    return records



def run_smoke_test(pipeline: LLMPIIPipeline, records: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("SMOKE TEST — first 3 records")
    print("=" * 60)
    n_ok = 0
    for i in range(min(3, len(records))):
        rec = records[i]
        prompt = pipeline._build_prompt(rec["sequence"])
        print(f"\n--- Record {i} ---")
        print(f"Sentence: {rec['sequence']}")
        print(f"Prompt (last 300 chars): ...{prompt[-300:]}")

        result = pipeline.predict_sentence(rec["tokens"], rec["sequence"])
        print(f"Raw response: {result['raw_response']!r}")
        print(f"Parse OK: {result['parse_ok']}")
        print(f"Names: {result['parsed_names']}")
        print(f"Emails: {result['parsed_emails']}")
        print(f"Predicted tags: {result['predicted_tags']}")
        if rec.get("ner_tags"):
            print(f"Ground truth:   {rec['ner_tags']}")
        if result["parse_ok"]:
            n_ok += 1

    print(f"\nSmoke test: {n_ok}/3 records parsed successfully.")
    if n_ok == 0:
        raise RuntimeError(
            "Smoke test failed: 0/3 records parsed successfully. Check prompt format."
        )
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Helpers for the comparison table
# ---------------------------------------------------------------------------
def _fmt_f1(val: float) -> str:
    return f"{val:.3f}"


def _fmt_pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def _extract_deberta_stats(summary: dict) -> dict:
    """Read whatever structure training_summary.json has and return per-seed test metrics."""
    runs = summary.get("runs", [])
    deberta_runs = [r for r in runs if "deberta" in r.get("model_name", "").lower()]
    distilbert_runs = [r for r in runs if "distilbert" in r.get("model_name", "").lower()]

    def _avg(values):
        return sum(values) / len(values) if values else 0.0

    def _std(values):
        if len(values) < 2:
            return 0.0
        m = _avg(values)
        return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5

    deb = {
        "span_f1_overall": [r["test_metrics"].get("eval_overall_f1", 0) for r in deberta_runs],
        "span_f1_per":     [r["test_metrics"].get("eval_per_f1", 0) for r in deberta_runs],
        "span_f1_email":   [r["test_metrics"].get("eval_email_f1", 0) for r in deberta_runs],
        "fpr":             [r["test_metrics"].get("eval_token_fpr", 0) for r in deberta_runs],
        "fnr":             [r["test_metrics"].get("eval_token_fnr", 0) for r in deberta_runs],
    }

    dis = {}
    if distilbert_runs:
        dr = distilbert_runs[0]["test_metrics"]
        dis = {
            "span_f1_overall": dr.get("eval_overall_f1", 0),
            "span_f1_per":     dr.get("eval_per_f1", 0),
            "span_f1_email":   dr.get("eval_email_f1", 0),
            "fpr_per":         dr.get("eval_token_fpr", 0),
            "fnr_per":         dr.get("eval_token_fnr", 0),
        }

    return {
        "deberta_mean_overall": _avg(deb["span_f1_overall"]),
        "deberta_std_overall":  _std(deb["span_f1_overall"]),
        "deberta_mean_per":     _avg(deb["span_f1_per"]),
        "deberta_std_per":      _std(deb["span_f1_per"]),
        "deberta_mean_email":   _avg(deb["span_f1_email"]),
        "deberta_std_email":    _std(deb["span_f1_email"]),
        "deberta_mean_fpr":     _avg(deb["fpr"]),
        "deberta_std_fpr":      _std(deb["fpr"]),
        "deberta_mean_fnr":     _avg(deb["fnr"]),
        "deberta_std_fnr":      _std(deb["fnr"]),
        "distilbert": dis,
    }



def build_summary_md(llm_metrics: dict, deb: dict, n_sentences: int, elapsed: float) -> str:
    dis = deb.get("distilbert", {})

    def deb_cell(mean, std):
        return f"{mean:.3f} ± {std:.3f}"

    def dis_cell(val):
        return f"{val:.3f}" if val else "N/A"

    llm_leak = _fmt_pct(llm_metrics["redaction_leak_rate"])
    dis_leak = "N/A"  # not computed from day3 summary (no token-level span data)
    deb_leak = "N/A"

    parse_fail = _fmt_pct(llm_metrics["parse_failure_rate"])
    n_fail = llm_metrics["n_parse_failures"]

    lines = [
        "# Day 4 Results — LLaMA Zero-Shot vs DeBERTa Fine-Tuned",
        "",
        "## Inference Setup",
        "- Model: Llama-3.2-1B-Instruct (Q4_K_M GGUF, CPU-only, i7-11gen)",
        "- Prompt: Template C (zero-shot, definition-heavy, no fine-tuning)",
        "- Temperature: 0.0 (deterministic)",
        f"- Test sentences: {n_sentences}",
        f"- Elapsed inference time: {elapsed:.1f}s ({elapsed / max(n_sentences, 1):.2f}s/sent)",
        "",
        "## Model Comparison",
        "",
        "| Metric                   | DeBERTa-v3-small (mean ± std, 3 seeds) | DistilBERT-cased (1 seed) | LLaMA-3.2-1B zero-shot |",
        "|--------------------------|----------------------------------------|--------------------------|------------------------|",
        "| Span F1 — PER            | "
        + deb_cell(deb["deberta_mean_per"], deb["deberta_std_per"]).ljust(38)
        + " | " + dis_cell(dis.get("span_f1_per", 0)).ljust(24)
        + " | " + _fmt_f1(llm_metrics["span_f1_per"]).ljust(22) + " |",
        "| Span F1 — EMAIL          | "
        + deb_cell(deb["deberta_mean_email"], deb["deberta_std_email"]).ljust(38)
        + " | " + dis_cell(dis.get("span_f1_email", 0)).ljust(24)
        + " | " + _fmt_f1(llm_metrics["span_f1_email"]).ljust(22) + " |",
        "| Span F1 — Overall        | "
        + deb_cell(deb["deberta_mean_overall"], deb["deberta_std_overall"]).ljust(38)
        + " | " + dis_cell(dis.get("span_f1_overall", 0)).ljust(24)
        + " | " + _fmt_f1(llm_metrics["span_f1_overall"]).ljust(22) + " |",
        "| Token FPR (PER)          | "
        + _fmt_pct(deb["deberta_mean_fpr"]).ljust(38)
        + " | " + _fmt_pct(dis.get("fpr_per", 0)).ljust(24)
        + " | " + _fmt_pct(llm_metrics["token_fpr_per"]).ljust(22) + " |",
        "| Token FNR (PER)          | "
        + _fmt_pct(deb["deberta_mean_fnr"]).ljust(38)
        + " | " + _fmt_pct(dis.get("fnr_per", 0)).ljust(24)
        + " | " + _fmt_pct(llm_metrics["token_fnr_per"]).ljust(22) + " |",
        "| Token FPR (EMAIL)        | "
        + _fmt_pct(deb["deberta_mean_fpr"]).ljust(38)
        + " | " + _fmt_pct(dis.get("fpr_per", 0)).ljust(24)
        + " | " + _fmt_pct(llm_metrics["token_fpr_email"]).ljust(22) + " |",
        "| Token FNR (EMAIL)        | "
        + _fmt_pct(deb["deberta_mean_fnr"]).ljust(38)
        + " | " + _fmt_pct(dis.get("fnr_per", 0)).ljust(24)
        + " | " + _fmt_pct(llm_metrics["token_fnr_email"]).ljust(22) + " |",
        "| Redaction Leak Rate      | "
        + deb_leak.ljust(38)
        + " | " + dis_leak.ljust(24)
        + " | " + llm_leak.ljust(22) + " |",
        "| Parse Failure Rate       | "
        + "N/A".ljust(38)
        + " | " + "N/A".ljust(24)
        + " | " + parse_fail.ljust(22) + " |",
        "",
        "## Notes",
        "- LLaMA inference is ~100× slower than encoder (CPU, ~0.5 sent/sec vs ~50 sent/sec)",
        f"- Parse failures: {n_fail} out of {n_sentences} sentences (see parse_failures.jsonl)",
        "- No fine-tuning was applied to LLaMA; results reflect zero-shot generalization at 1B scale",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # STEP 0 — environment check
    ensure_model(args.model_path, args.skip_download)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.cache_path)), exist_ok=True)

    # STEP 1 — load data
    all_records = load_test_records(args.data_path, args.hf_dataset_path)
    if args.limit:
        records = all_records[: args.limit]
    else:
        records = all_records
    print(f"Loaded {len(records)} test records.")

    # Initialise pipeline
    print("Loading LLaMA model …")
    pipeline = LLMPIIPipeline(
        model_path=args.model_path,
        n_threads=args.n_threads,
    )
    print("Model loaded.")

    # STEP 2 — smoke test
    if not args.skip_smoke_test:
        run_smoke_test(pipeline, all_records[:3])

    # STEP 3 — full inference
    print(f"\nRunning inference on {len(records)} records …")
    t0 = time.time()
    results = pipeline.predict_batch(
        records,
        cache_path=args.cache_path,
        checkpoint_every=args.checkpoint_every,
    )
    elapsed = time.time() - t0
    print(f"Inference complete in {elapsed:.1f}s ({elapsed / max(len(records), 1):.2f}s/sent)")

    # STEP 4 — compute metrics
    metrics = compute_llm_metrics(results)
    metrics_path = os.path.join(args.results_dir, "llm_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {metrics_path}")

    # STEP 5 — save parse failures
    failures_path = os.path.join(args.results_dir, "parse_failures.jsonl")
    with open(failures_path, "w", encoding="utf-8") as f:
        for entry in _parse_failure_log:
            f.write(json.dumps(entry) + "\n")
    print(f"Parse failures saved → {failures_path} ({len(_parse_failure_log)} entries)")

    # STEP 6 — load DeBERTa results
    day3_summary_path = os.path.join("results", "day3_encoder_training", "training_summary.json")
    deb_stats: dict = {}
    if os.path.exists(day3_summary_path):
        with open(day3_summary_path, "r", encoding="utf-8") as f:
            day3_summary = json.load(f)
        deb_stats = _extract_deberta_stats(day3_summary)
    else:
        print(f"Warning: {day3_summary_path} not found; DeBERTa columns will show N/A.")
        deb_stats = {
            "deberta_mean_overall": 0.0, "deberta_std_overall": 0.0,
            "deberta_mean_per": 0.0,     "deberta_std_per": 0.0,
            "deberta_mean_email": 0.0,   "deberta_std_email": 0.0,
            "deberta_mean_fpr": 0.0,     "deberta_std_fpr": 0.0,
            "deberta_mean_fnr": 0.0,     "deberta_std_fnr": 0.0,
            "distilbert": {},
        }

    # STEP 7 — write summary markdown
    summary_md = build_summary_md(metrics, deb_stats, len(records), elapsed)
    summary_path = os.path.join(args.results_dir, "day4_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"\n{summary_md}")
    print(f"Summary saved → {summary_path}")

    # STEP 8 — final print
    print("\n=== Day 4 Complete ===")
    print(f"LLaMA span F1 overall: {metrics['span_f1_overall']:.3f}")
    print(f"LLaMA span F1 PER:     {metrics['span_f1_per']:.3f}")
    print(f"LLaMA span F1 EMAIL:   {metrics['span_f1_email']:.3f}")
    print(f"Redaction leak rate:   {metrics['redaction_leak_rate'] * 100:.1f}%")
    print(f"Parse failure rate:    {metrics['parse_failure_rate'] * 100:.1f}%")
    print(f"Results  → {metrics_path}")
    print(f"Summary  → {summary_path}")
    print(f"Raw cache → {args.cache_path}")


if __name__ == "__main__":
    main()
