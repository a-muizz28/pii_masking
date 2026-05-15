"""Day 7: independent Hugging Face PII evaluation."""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pii_masking.eval_independent import (
    build_summary,
    choose_best_deberta_model,
    compute_metrics,
    day7_paths,
    inspect_examples,
    load_ai4privacy_examples,
    predict_deberta,
    predict_llama,
    save_json,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day 7 independent PII evaluation")
    parser.add_argument("--dataset", default="ai4privacy/pii-masking-300k")
    parser.add_argument("--split", default="train")
    parser.add_argument("--language", default="English")
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--results-dir", default="results/day7")
    parser.add_argument("--deberta-model-dir", default=None)
    parser.add_argument("--llama-model-path", default="models/llama/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
    parser.add_argument("--skip-deberta", action="store_true")
    parser.add_argument("--skip-llama", action="store_true")
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = day7_paths(args.results_dir)
    paths.results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading external dataset: %s", args.dataset)
    examples = load_ai4privacy_examples(
        dataset_name=args.dataset,
        split=args.split,
        max_records=args.max_records,
        language=args.language,
    )
    stats = inspect_examples(examples)
    stats.update(
        {
            "dataset": args.dataset,
            "split": args.split,
            "language_filter": args.language,
            "max_records": args.max_records,
        }
    )
    save_json(stats, paths.stats_path)
    logger.info("Saved data inspection summary -> %s", paths.stats_path)

    email_present = stats["email_span_count"] > 0
    deberta_metrics = None
    llama_metrics = None

    if not args.skip_deberta:
        model_dir = (
            pathlib.Path(args.deberta_model_dir)
            if args.deberta_model_dir
            else choose_best_deberta_model(ROOT / "models")
        )
        logger.info("Running DeBERTa inference from %s", model_dir)
        deberta_records = predict_deberta(examples, model_dir=model_dir)
        deberta_metrics = compute_metrics(deberta_records, email_present=email_present)
        deberta_metrics["model_dir"] = str(model_dir)
        save_json(deberta_metrics, paths.deberta_path)
        logger.info("Saved DeBERTa metrics -> %s", paths.deberta_path)

    if not args.skip_llama:
        logger.info("Running LLaMA inference from %s", args.llama_model_path)
        llama_records = predict_llama(
            examples,
            model_path=args.llama_model_path,
            cache_path=paths.llama_cache_path,
            n_threads=args.n_threads,
            n_gpu_layers=args.n_gpu_layers,
        )
        llama_metrics = compute_metrics(llama_records, email_present=email_present)
        parse_failures = sum(1 for row in llama_records if not row.get("parse_ok", False))
        llama_metrics["parse_failure_rate"] = parse_failures / len(llama_records)
        llama_metrics["parse_failure_count"] = parse_failures
        llama_metrics["model_path"] = args.llama_model_path
        save_json(llama_metrics, paths.llama_path)
        logger.info("Saved LLaMA metrics -> %s", paths.llama_path)

    if deberta_metrics is not None and llama_metrics is not None:
        summary = build_summary(args.dataset, stats, deberta_metrics, llama_metrics)
        save_json(summary, paths.summary_path)
        logger.info("Saved consolidated summary -> %s", paths.summary_path)


if __name__ == "__main__":
    main()
