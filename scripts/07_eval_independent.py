"""Day 7: cross-domain evaluation on WikiANN English test split."""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pii_masking.day7_eval_independent import (
    build_summary,
    compute_metrics,
    inspect_examples,
    load_wikiann_examples,
    predict_deberta,
    predict_llama,
    save_json,
    wikiann_paths,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day 7 cross-domain evaluation on WikiANN")
    parser.add_argument("--split", default="test")
    parser.add_argument("--results-dir", default="predictions/wikiann")
    parser.add_argument("--distilbert-model-dir", default="models/distilbert_seed42/best_model")
    parser.add_argument("--deberta-model-dir", default="models/deberta_seed7/best_model")
    parser.add_argument("--llama-model-path", default="models/llama/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
    parser.add_argument("--skip-distilbert", action="store_true")
    parser.add_argument("--skip-deberta", action="store_true")
    parser.add_argument("--skip-llama", action="store_true")
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = wikiann_paths(args.results_dir)
    paths.results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading WikiANN English %s split", args.split)
    examples = load_wikiann_examples(split=args.split)
    stats = inspect_examples(examples, dataset_name="wikiann/en")
    save_json(stats, paths.stats_path)
    logger.info("Saved data inspection summary -> %s", paths.stats_path)

    email_present = stats["email_span_count"] > 0
    distilbert_metrics = None
    deberta_metrics = None
    llama_metrics = None

    if not args.skip_distilbert:
        model_dir = pathlib.Path(args.distilbert_model_dir)
        logger.info("Running Approach A (DistilBERT-s42) inference from %s", model_dir)
        distilbert_records = predict_deberta(examples, model_dir=model_dir)
        distilbert_metrics = compute_metrics(distilbert_records, email_present=email_present)
        distilbert_metrics["model_dir"] = str(model_dir)
        save_json(distilbert_metrics, paths.distilbert_path)
        logger.info("Saved DistilBERT metrics -> %s", paths.distilbert_path)

    if not args.skip_deberta:
        model_dir = pathlib.Path(args.deberta_model_dir)
        logger.info("Running Approach B (DeBERTa-s7) inference from %s", model_dir)
        deberta_records = predict_deberta(examples, model_dir=model_dir)
        deberta_metrics = compute_metrics(deberta_records, email_present=email_present)
        deberta_metrics["model_dir"] = str(model_dir)
        save_json(deberta_metrics, paths.deberta_path)
        logger.info("Saved DeBERTa metrics -> %s", paths.deberta_path)

    if not args.skip_llama:
        logger.info("Running Approach C (LLaMA) inference from %s", args.llama_model_path)
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
        summary = build_summary(
            "wikiann/en",
            stats,
            deberta_metrics,
            llama_metrics,
            distilbert_metrics=distilbert_metrics,
        )
        save_json(summary, paths.summary_path)
        logger.info("Saved consolidated summary -> %s", paths.summary_path)


if __name__ == "__main__":
    main()
