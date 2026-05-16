"""Paired bootstrap significance test: DeBERTa-s42 vs LLaMA macro-F1."""

import json
import os
import random
from pathlib import Path


def _span_f1(gold_tags: list[str], pred_tags: list[str]) -> float:
    """Compute span-level macro-F1 (PER + EMAIL) for a single sentence."""
    def _spans(tags, label_prefix):
        spans = set()
        start = None
        for i, t in enumerate(tags):
            if t == f"B-{label_prefix}":
                start = i
            elif t == f"I-{label_prefix}" and start is None:
                start = i  # tolerate missing B
            elif t == "O" or (t.startswith("B-") and t != f"B-{label_prefix}"):
                if start is not None:
                    spans.add((start, i - 1))
                    start = None
        if start is not None:
            spans.add((start, len(tags) - 1))
        return spans

    f1_scores = []
    for label in ("PER", "EMAIL"):
        g = _spans(gold_tags, label)
        p = _spans(pred_tags, label)
        if not g and not p:
            continue
        tp = len(g & p)
        prec = tp / len(p) if p else 0.0
        rec = tp / len(g) if g else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_bootstrap(
    deberta_path: str,
    llama_path: str,
    out_path: str,
    n_bootstrap: int = 10_000,
    random_state: int = 42,
) -> dict:
    """Run a paired bootstrap significance test comparing DeBERTa and LLaMA F1.

    Args:
        deberta_path: Path to JSONL predictions with ``gold_tags`` / ``pred_tags``.
        llama_path: Path to JSONL predictions with ``ner_tags`` / ``predicted_tags``.
        out_path: Where to write the JSON result summary.
        n_bootstrap: Number of bootstrap resamples (default 10 000).
        random_state: Seed for the Python ``random`` module.

    Returns:
        Dict with keys: n_sentences, deberta_mean_f1, llama_mean_f1,
        observed_mean_diff, ci_lower_2_5, ci_upper_97_5, p_value,
        significant_at_0_05.
    """
    deberta_rows = _load_jsonl(deberta_path)
    llama_rows = _load_jsonl(llama_path)

    if len(deberta_rows) != len(llama_rows):
        raise ValueError(
            f"Row count mismatch: deberta={len(deberta_rows)} llama={len(llama_rows)}"
        )
    paired: list[tuple[float, float]] = []
    for dr, lr in zip(deberta_rows, llama_rows):
        d_f1 = _span_f1(dr["gold_tags"], dr["pred_tags"])
        l_f1 = _span_f1(lr["ner_tags"], lr["predicted_tags"])
        paired.append((d_f1, l_f1))

    n = len(paired)
    d_scores = [p[0] for p in paired]
    l_scores = [p[1] for p in paired]

    deberta_mean = sum(d_scores) / n
    llama_mean = sum(l_scores) / n
    observed_diff = deberta_mean - llama_mean

    rng = random.Random(random_state)
    boot_diffs = []
    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        bd = sum(d_scores[i] for i in indices) / n
        bl = sum(l_scores[i] for i in indices) / n
        boot_diffs.append(bd - bl)

    boot_diffs.sort()
    ci_lower = boot_diffs[int(0.025 * n_bootstrap)]
    ci_upper = boot_diffs[int(0.975 * n_bootstrap)]
    # One-sided p-value: fraction of bootstrap samples where DeBERTa does NOT beat LLaMA.
    # Under H0 (no difference), this should be ~0.5; values < 0.05 reject H0.
    p_value = sum(1 for d in boot_diffs if d <= 0) / n_bootstrap

    result = {
        "n_sentences": n,
        "n_bootstrap": n_bootstrap,
        "deberta_mean_f1": round(deberta_mean, 6),
        "llama_mean_f1": round(llama_mean, 6),
        "observed_mean_diff": round(observed_diff, 6),
        "ci_lower_2_5": round(ci_lower, 6),
        "ci_upper_97_5": round(ci_upper, 6),
        "p_value": p_value if p_value > 0 else f"<{1/n_bootstrap:.5f}",
        "significant_at_0_05": p_value < 0.05,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    result = run_bootstrap(
        deberta_path=str(root / "predictions/encoder/deberta_s42_predictions.jsonl"),
        llama_path=str(root / "predictions/llm/raw_outputs.jsonl"),
        out_path=str(root / "results/day5/bootstrap_results.json"),
        n_bootstrap=10_000,
        random_state=42,
    )
    for k, v in result.items():
        print(f"  {k}: {v}")
