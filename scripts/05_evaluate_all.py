"""Day 5: Full evaluation + masking pipeline + paired bootstrap.

Adapted to actual project layout:
  - DeBERTa seeds: 0, 7, 42  (not 42/43/44 as in spec template)
  - Encoder predictions generated from data/processed/test_injected.parquet
  - LLM predictions:  predictions/llm/raw_outputs.jsonl
    keys: tokens, sequence, ner_tags (gold), predicted_tags (pred)
"""

import datetime
import json
import logging
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

from src.pii_masking.day5_masking import MaskingPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


PREDICTIONS_DIR = ROOT / "predictions"
ENCODER_DIR     = PREDICTIONS_DIR / "encoder"
LLM_FILE        = PREDICTIONS_DIR / "llm" / "raw_outputs.jsonl"
TEST_DATA_FILE  = ROOT / "data" / "processed" / "test_injected.parquet"
MODELS_DIR      = ROOT / "models"
REPORTS_DIR     = ROOT / "reports"
RESULTS_DIR     = ROOT / "results" / "day5"
BOOTSTRAP_OUT   = PREDICTIONS_DIR / "bootstrap_results.json"
RESULTS_JSON    = REPORTS_DIR / "results.json"
SUMMARY_MD      = RESULTS_DIR / "day5_summary.md"

# Encoder model definitions (adapt as models are added/renamed)
ENCODER_MODELS: dict[str, pathlib.Path] = {
    "deberta_s0":     MODELS_DIR / "deberta_seed0"  / "best_model",
    "deberta_s7":     MODELS_DIR / "deberta_seed7"  / "best_model",
    "deberta_s42":    MODELS_DIR / "deberta_seed42" / "best_model",
    "distilbert_s42": MODELS_DIR / "distilbert_seed42" / "best_model",
}



def _load_jsonl(path: pathlib.Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _load_predictions(path: pathlib.Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return _load_jsonl(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        raise ValueError(f"{path} does not contain a JSON list")
    raise ValueError(f"Unknown format: {path.suffix}")


def _load_test_records() -> list[dict]:
    """Load Day 5 test records from processed data, with LLM output as fallback."""
    if TEST_DATA_FILE.exists():
        import pandas as pd

        df = pd.read_parquet(TEST_DATA_FILE)
        records: list[dict] = []
        for row in df.to_dict("records"):
            tokens = row.get("tokens", [])
            ner_tags = row.get("ner_tags", [])
            records.append({
                "tokens": list(tokens),
                "sequence": row.get("sequence") or " ".join(tokens),
                "ner_tags": list(ner_tags),
            })
        return records

    if LLM_FILE.exists():
        records = _load_jsonl(LLM_FILE)
        records.sort(key=lambda r: r.get("idx", 0))
        return records

    raise FileNotFoundError(
        f"No test records found. Expected {TEST_DATA_FILE} or fallback {LLM_FILE}"
    )


def _run_encoder_inference(
    model, tokenizer, records: list[dict], id2label: dict, device: str,
    batch_size: int = 16, max_length: int = 256,
) -> list[dict]:
    import torch

    results: list[dict] = []
    n_total = len(records)

    for batch_start in range(0, n_total, batch_size):
        batch = records[batch_start : batch_start + batch_size]
        batch_tokens = [r["tokens"] for r in batch]

        encoding = tokenizer(
            batch_tokens,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        # token_type_ids present for DistilBERT-style models; some DeBERTa configs want it
        if "token_type_ids" in encoding:
            model_inputs["token_type_ids"] = encoding["token_type_ids"].to(device)

        with torch.no_grad():
            outputs = model(**model_inputs)

        pred_ids = outputs.logits.argmax(dim=-1).cpu().numpy()

        for j, record in enumerate(batch):
            word_ids = encoding.word_ids(batch_index=j)
            word_pred: dict[int, str] = {}
            for token_idx, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                if word_id not in word_pred:
                    label_id = pred_ids[j][token_idx]
                    word_pred[word_id] = id2label.get(label_id, "O")

            n_words = len(record["tokens"])
            pred_tags = [word_pred.get(k, "O") for k in range(n_words)]

            results.append({
                "tokens":    record["tokens"],
                "sequence":  record.get("sequence", " ".join(record["tokens"])),
                "gold_tags": record.get("ner_tags", []),
                "pred_tags": pred_tags,
            })

        if (batch_start // batch_size + 1) % 20 == 0:
            done = min(batch_start + batch_size, n_total)
            logger.info("  %d / %d sentences processed", done, n_total)

    return results


def generate_encoder_predictions() -> None:
    """Run each trained encoder on the test set; save to predictions/encoder/."""
    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError:
        logger.error("transformers / torch not installed — cannot generate encoder predictions")
        return

    ENCODER_DIR.mkdir(parents=True, exist_ok=True)

    try:
        test_records = _load_test_records()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return

    logger.info("Loaded %d test records for encoder inference", len(test_records))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Encoder inference device: %s", device)

    for model_key, model_path in ENCODER_MODELS.items():
        out_file = ENCODER_DIR / f"{model_key}_predictions.jsonl"
        if out_file.exists():
            logger.info("Skipping %s — prediction file already exists", model_key)
            continue

        if not model_path.exists():
            logger.warning("Model directory not found, skipping %s: %s", model_key, model_path)
            continue

        logger.info("Generating predictions for %s ...", model_key)
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForTokenClassification.from_pretrained(str(model_path))
            model.eval()
            model.to(device)
            id2label: dict[int, str] = {int(k): v for k, v in model.config.id2label.items()}

            batch_size = 4 if device == "cpu" else 16
            preds = _run_encoder_inference(
                model, tokenizer, test_records, id2label, device, batch_size=batch_size
            )

            with open(out_file, "w", encoding="utf-8") as fh:
                for rec in preds:
                    fh.write(json.dumps(rec) + "\n")

            logger.info("  Saved %d records → %s", len(preds), out_file)
            del model  # free memory before next model
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to generate predictions for %s: %s", model_key, exc)


# ── Step 1: load all predictions ─────────────────────────────────────────────

def load_all_predictions() -> tuple[dict[str, list[dict]], list[str]]:
    model_predictions: dict[str, list[dict]] = {}
    warnings: list[str] = []

    # Encoder predictions
    if not ENCODER_DIR.exists():
        warnings.append("predictions/encoder/ not found — encoder models were skipped")
    else:
        for pfile in sorted(ENCODER_DIR.glob("*.json*")):
            key = pfile.stem.replace("_predictions", "")
            try:
                raw = _load_predictions(pfile)
                normalized: list[dict] = []
                recon_count = 0
                for r in raw:
                    seq = r.get("sequence")
                    if seq is None:
                        seq = " ".join(r.get("tokens", []))
                        recon_count += 1
                    normalized.append({
                        "tokens":    r.get("tokens", []),
                        "sequence":  seq,
                        "gold_tags": r.get("gold_tags") or r.get("ner_tags", []),
                        "pred_tags": r.get("pred_tags") or r.get("predicted_tags", []),
                    })
                if recon_count:
                    warnings.append(
                        f"{key}: {recon_count} sequences reconstructed from tokens"
                    )
                model_predictions[key] = normalized
                logger.info("Loaded %d records for %s", len(normalized), key)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Failed to load {pfile.name}: {exc}")

    # LLM predictions
    if LLM_FILE.exists():
        llm_raw = _load_jsonl(LLM_FILE)
        llm_raw.sort(key=lambda r: r.get("idx", 0))
        llm_norm: list[dict] = []
        recon_llm = 0
        for r in llm_raw:
            seq = r.get("sequence")
            if seq is None:
                seq = " ".join(r.get("tokens", []))
                recon_llm += 1
            llm_norm.append({
                "tokens":    r.get("tokens", []),
                "sequence":  seq,
                "gold_tags": r.get("ner_tags", []),
                "pred_tags": r.get("predicted_tags", []),
            })
        if recon_llm:
            warnings.append(f"llm_templateC: {recon_llm} sequences reconstructed from tokens")
        model_predictions["llm_templateC"] = llm_norm
        logger.info("Loaded %d LLM records", len(llm_norm))
    else:
        warnings.append(f"LLM predictions not found: {LLM_FILE}")

    return model_predictions, warnings


# ── Step 2: seqeval span F1 ───────────────────────────────────────────────────

def run_seqeval(model_predictions: dict) -> dict:
    results: dict[str, dict] = {}
    for model_key, records in model_predictions.items():
        gold_seqs: list[list[str]] = []
        pred_seqs: list[list[str]] = []
        for r in records:
            g = r.get("gold_tags", [])
            p = r.get("pred_tags", [])
            n = len(g)
            p_aligned = (list(p) + ["O"] * n)[:n]
            gold_seqs.append(g)
            pred_seqs.append(p_aligned)

        try:
            report = classification_report(
                gold_seqs, pred_seqs,
                mode="strict",
                scheme=IOB2,
                output_dict=True,
                zero_division=0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("seqeval failed for %s: %s", model_key, exc)
            results[model_key] = {}
            continue

        per_stats   = report.get("PER", {})
        email_stats = report.get("EMAIL", None)
        macro       = report.get("macro avg", {})

        results[model_key] = {
            "PER": {
                "precision": float(per_stats.get("precision", 0.0)),
                "recall":    float(per_stats.get("recall", 0.0)),
                "f1":        float(per_stats.get("f1-score", 0.0)),
            },
            "EMAIL": (
                {
                    "precision": float(email_stats.get("precision", 0.0)),
                    "recall":    float(email_stats.get("recall", 0.0)),
                    "f1":        float(email_stats.get("f1-score", 0.0)),
                }
                if email_stats else None
            ),
            "macro_avg": {
                "precision": float(macro.get("precision", 0.0)),
                "recall":    float(macro.get("recall", 0.0)),
                "f1":        float(macro.get("f1-score", 0.0)),
            },
        }

    return results


# ── Step 3: token-level FPR / FNR ────────────────────────────────────────────

def compute_token_rates(model_predictions: dict) -> dict:
    results: dict[str, dict] = {}
    for model_key, records in model_predictions.items():
        tp = fp = fn = tn = 0
        tp_per = fp_per = fn_per = tn_per = 0
        tp_email = fp_email = fn_email = tn_email = 0

        for r in records:
            gold = r.get("gold_tags", [])
            pred = r.get("pred_tags", [])
            n = len(gold)
            pred = (list(pred) + ["O"] * n)[:n]

            for g, p in zip(gold, pred):
                g_bin = int(g != "O")
                p_bin = int(p != "O")
                if g_bin and p_bin:           tp += 1
                elif not g_bin and p_bin:     fp += 1
                elif g_bin and not p_bin:     fn += 1
                else:                         tn += 1

                g_per = g in ("B-PER", "I-PER")
                p_per = p in ("B-PER", "I-PER")
                if g_per and p_per:           tp_per += 1
                elif not g_per and p_per:     fp_per += 1
                elif g_per and not p_per:     fn_per += 1
                else:                         tn_per += 1

                g_email = g in ("B-EMAIL", "I-EMAIL")
                p_email = p in ("B-EMAIL", "I-EMAIL")
                if g_email and p_email:       tp_email += 1
                elif not g_email and p_email: fp_email += 1
                elif g_email and not p_email: fn_email += 1
                else:                         tn_email += 1

        fpr       = fp       / (fp       + tn)       if (fp       + tn)       > 0 else 0.0
        fnr       = fn       / (fn       + tp)       if (fn       + tp)       > 0 else 0.0
        per_fpr   = fp_per   / (fp_per   + tn_per)   if (fp_per   + tn_per)   > 0 else 0.0
        per_fnr   = fn_per   / (fn_per   + tp_per)   if (fn_per   + tp_per)   > 0 else 0.0
        email_fpr = fp_email / (fp_email + tn_email) if (fp_email + tn_email) > 0 else 0.0
        email_fnr = fn_email / (fn_email + tp_email) if (fn_email + tp_email) > 0 else 0.0

        results[model_key] = {
            "fpr": fpr, "fnr": fnr,
            "per_fpr": per_fpr, "per_fnr": per_fnr,
            "email_fpr": email_fpr, "email_fnr": email_fnr,
        }

    return results


# ── Step 4: masking pipeline + leak rate ─────────────────────────────────────

def compute_leak_rates(model_predictions: dict) -> dict:
    masker = MaskingPipeline()
    return {key: masker.compute_leak_rate(records) for key, records in model_predictions.items()}


# ── Step 5: paired bootstrap ─────────────────────────────────────────────────

def _sentence_f1(gold_tags: list[str], pred_tags: list[str]) -> float:
    """Span F1 for a single sentence; returns 0.0 when no entities on either side."""
    if not any(t != "O" for t in gold_tags) and not any(t != "O" for t in pred_tags):
        return 0.0
    try:
        report = classification_report(
            [gold_tags], [pred_tags],
            mode="strict",
            scheme=IOB2,
            output_dict=True,
            zero_division=0,
        )
        return float(report.get("macro avg", {}).get("f1-score", 0.0))
    except Exception:  # noqa: BLE001
        return 0.0


def run_bootstrap(model_predictions: dict) -> dict | None:
    key_deberta = "deberta_s42"
    key_llm     = "llm_templateC"

    if key_deberta not in model_predictions:
        logger.warning("Bootstrap skipped: %s not found in predictions", key_deberta)
        return None
    if key_llm not in model_predictions:
        logger.warning("Bootstrap skipped: %s not found in predictions", key_llm)
        return None

    deb_records = model_predictions[key_deberta]
    llm_records = model_predictions[key_llm]
    n = min(len(deb_records), len(llm_records))
    logger.info("Bootstrap: %d aligned sentences", n)

    deb_f1s: list[float] = []
    llm_f1s: list[float] = []
    for i in range(n):
        g  = deb_records[i]["gold_tags"]
        dp = deb_records[i]["pred_tags"]
        lp = llm_records[i]["pred_tags"]
        ng = len(g)
        dp = (list(dp) + ["O"] * ng)[:ng]
        lp = (list(lp) + ["O"] * ng)[:ng]
        deb_f1s.append(_sentence_f1(g, dp))
        llm_f1s.append(_sentence_f1(g, lp))

    observed_diff = float(np.mean(deb_f1s) - np.mean(llm_f1s))

    N_BOOTSTRAP = 1000
    np.random.seed(42)
    boot_diffs: list[float] = []
    for _ in range(N_BOOTSTRAP):
        idx = np.random.choice(n, size=n, replace=True)
        d_boot = np.mean([deb_f1s[i] for i in idx])
        l_boot = np.mean([llm_f1s[i] for i in idx])
        boot_diffs.append(float(d_boot - l_boot))

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    p_value  = float(np.mean([d <= 0.0 for d in boot_diffs]))

    interpretation = (
        "DeBERTa significantly outperforms LLaMA (p < 0.05)"
        if p_value < 0.05
        else f"No significant difference at α=0.05 (p = {p_value:.3f})"
    )

    return {
        "comparison":         "deberta_s42_vs_llm_templateC",
        "observed_diff_f1":   observed_diff,
        "ci_lower_95":        ci_lower,
        "ci_upper_95":        ci_upper,
        "p_value_one_sided":  p_value,
        "n_bootstrap":        N_BOOTSTRAP,
        "n_sentences":        n,
        "interpretation":     interpretation,
    }


# ── Step 6: build reports/results.json ───────────────────────────────────────

def build_results_json(
    model_predictions: dict,
    seqeval_res: dict,
    token_res: dict,
    leak_res: dict,
    bootstrap: dict | None,
) -> dict:
    deberta_keys = sorted(
        [k for k in seqeval_res if k.startswith("deberta_s")],
        key=lambda k: int(k.split("_s")[1]),
    )
    seeds = [int(k.split("_s")[1]) for k in deberta_keys]

    def _per_seed(entity: str, field: str) -> list:
        vals = []
        for k in deberta_keys:
            ent = seqeval_res.get(k, {}).get(entity)
            vals.append(ent.get(field) if ent else None)
        return vals

    def safe_mean(vals: list) -> float | None:
        v = [x for x in vals if x is not None]
        return float(np.mean(v)) if v else None

    def safe_std(vals: list) -> float | None:
        v = [x for x in vals if x is not None]
        return float(np.std(v)) if v else None

    per_f1s   = _per_seed("PER",   "f1")
    email_f1s = _per_seed("EMAIL", "f1")
    macro_f1s = [seqeval_res.get(k, {}).get("macro_avg", {}).get("f1") for k in deberta_keys]

    ref_key = "deberta_s42"
    tr = token_res.get(ref_key, {})
    lr = leak_res.get(ref_key, {})

    deberta_section: dict = {
        "seeds": seeds,
        "span_f1": {
            "PER":   {"mean": safe_mean(per_f1s),   "std": safe_std(per_f1s),   "per_seed": per_f1s},
            "EMAIL": {"mean": safe_mean(email_f1s), "std": safe_std(email_f1s), "per_seed": email_f1s},
            "macro_avg": {"mean": safe_mean(macro_f1s), "std": safe_std(macro_f1s)},
        },
        "token_fpr":    tr.get("fpr"),
        "token_fnr":    tr.get("fnr"),
        "per_fpr":      tr.get("per_fpr"),
        "per_fnr":      tr.get("per_fnr"),
        "email_fpr":    tr.get("email_fpr"),
        "email_fnr":    tr.get("email_fnr"),
        "leak_rate":    lr.get("leak_rate"),
        "leaked_count": lr.get("leaked_count"),
        "total_count":  lr.get("total_count"),
    }

    def _single_model_section(key: str) -> dict:
        sr = seqeval_res.get(key, {})
        tr_ = token_res.get(key, {})
        lr_ = leak_res.get(key, {})
        email = sr.get("EMAIL")
        return {
            "span_f1": {
                "PER":       sr.get("PER", {}).get("f1"),
                "EMAIL":     email.get("f1") if email else None,
                "macro_avg": sr.get("macro_avg", {}).get("f1"),
            },
            "token_fpr":    tr_.get("fpr"),
            "token_fnr":    tr_.get("fnr"),
            "per_fpr":      tr_.get("per_fpr"),
            "per_fnr":      tr_.get("per_fnr"),
            "email_fpr":    tr_.get("email_fpr"),
            "email_fnr":    tr_.get("email_fnr"),
            "leak_rate":    lr_.get("leak_rate"),
            "leaked_count": lr_.get("leaked_count"),
            "total_count":  lr_.get("total_count"),
        }

    test_size = (
        len(next(iter(model_predictions.values()))) if model_predictions else 0
    )

    return {
        "deberta_v3_small":      deberta_section,
        "distilbert_base_cased": _single_model_section("distilbert_s42"),
        "llama_3b_templateC":    _single_model_section("llm_templateC"),
        "bootstrap":             bootstrap,
        "metadata": {
            "test_set_size":    test_size,
            "evaluation_date":  datetime.date.today().isoformat(),
            "seqeval_mode":     "strict",
            "seqeval_scheme":   "IOB2",
        },
    }


# ── Step 7: build results/day5/day5_summary.md ───────────────────────────────

def build_summary_md(
    results: dict,
    bootstrap: dict | None,
    warnings: list[str],
) -> str:
    deb  = results["deberta_v3_small"]
    dist = results["distilbert_base_cased"]
    llm  = results["llama_3b_templateC"]

    def _f(v, d: int = 3) -> str:
        return "N/A" if v is None else f"{v:.{d}f}"

    def _ms(mean, std, d: int = 3) -> str:
        if mean is None:
            return "N/A"
        return f"{mean:.{d}f}±{std:.{d}f}" if std is not None else f"{mean:.{d}f}"

    sf = deb["span_f1"]
    lines = [
        "# Day 5 Evaluation Summary",
        "",
        f"DeBERTa seeds evaluated: {deb['seeds']}  "
        f"(token-level rates and leak rate reported for seed 42)",
        "",
        "## Main Comparison Table",
        "",
        "| Metric | DeBERTa-v3-small (mean±std) | DistilBERT-cased (1 seed) | LLaMA-3.2-1B-Instruct (Template C) |",
        "|-------------------------|------------------------------|---------------------------|--------------------------------------|",
        f"| Span F1 — PER           | {_ms(sf['PER']['mean'], sf['PER']['std'])} | {_f(dist['span_f1']['PER'])} | {_f(llm['span_f1']['PER'])} |",
        f"| Span F1 — EMAIL         | {_ms(sf['EMAIL']['mean'], sf['EMAIL']['std'])} | {_f(dist['span_f1']['EMAIL'])} | {_f(llm['span_f1']['EMAIL'])} |",
        f"| Macro Span F1           | {_ms(sf['macro_avg']['mean'], sf['macro_avg']['std'])} | {_f(dist['span_f1']['macro_avg'])} | {_f(llm['span_f1']['macro_avg'])} |",
        f"| Token FPR (binary)      | {_f(deb['token_fpr'])} | {_f(dist['token_fpr'])} | {_f(llm['token_fpr'])} |",
        f"| Token FNR (binary)      | {_f(deb['token_fnr'])} | {_f(dist['token_fnr'])} | {_f(llm['token_fnr'])} |",
        f"| PER FNR                 | {_f(deb['per_fnr'])} | {_f(dist['per_fnr'])} | {_f(llm['per_fnr'])} |",
        f"| EMAIL FNR               | {_f(deb['email_fnr'])} | {_f(dist['email_fnr'])} | {_f(llm['email_fnr'])} |",
        f"| Redaction Leak Rate     | {_f(deb['leak_rate'])} | {_f(dist['leak_rate'])} | {_f(llm['leak_rate'])} |",
        "",
    ]

    if bootstrap:
        sig = "significant" if bootstrap["p_value_one_sided"] < 0.05 else "not significant"
        lines += [
            "## Statistical Comparison (DeBERTa vs LLaMA)",
            "",
            f"- Observed F1 difference: {bootstrap['observed_diff_f1']:.3f}",
            f"- 95% CI: [{bootstrap['ci_lower_95']:.3f}, {bootstrap['ci_upper_95']:.3f}]",
            f"- p-value (one-sided): {bootstrap['p_value_one_sided']:.3f}",
            f"- Conclusion: {sig} at α=0.05",
            "",
        ]

    if warnings:
        lines += ["## Notes", ""]
        lines += [f"- {w}" for w in warnings]
        lines.append("")

    return "\n".join(lines)


# ── stdout tables ─────────────────────────────────────────────────────────────

def _print_seqeval_table(seqeval_res: dict) -> None:
    print("\n── Span F1 (seqeval strict IOB2) ──")
    print(f"{'Model':<22} {'PER F1':>8} {'EMAIL F1':>10} {'Macro F1':>10}")
    print("─" * 54)
    for key in sorted(seqeval_res):
        m = seqeval_res[key]
        per_f1   = (m.get("PER")       or {}).get("f1", 0.0) or 0.0
        email_f1 = (m.get("EMAIL")     or {}).get("f1", 0.0) or 0.0
        macro_f1 = (m.get("macro_avg") or {}).get("f1", 0.0) or 0.0
        print(f"{key:<22} {per_f1:>8.4f} {email_f1:>10.4f} {macro_f1:>10.4f}")
    print()


def _print_token_table(token_res: dict) -> None:
    print("── Token FPR / FNR ──")
    header = f"{'Model':<22} {'FPR':>7} {'FNR':>7} {'PER_FPR':>9} {'PER_FNR':>9} {'EM_FPR':>8} {'EM_FNR':>8}"
    print(header)
    print("─" * len(header))
    for key in sorted(token_res):
        m = token_res[key]
        print(
            f"{key:<22}"
            f" {m.get('fpr',0):>7.4f}"
            f" {m.get('fnr',0):>7.4f}"
            f" {m.get('per_fpr',0):>9.4f}"
            f" {m.get('per_fnr',0):>9.4f}"
            f" {m.get('email_fpr',0):>8.4f}"
            f" {m.get('email_fnr',0):>8.4f}"
        )
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Step 0 — generate encoder predictions if absent
    print("[DAY 5] Checking encoder predictions...")
    missing_encoder_files = [
        ENCODER_DIR / f"{model_key}_predictions.jsonl"
        for model_key, model_path in ENCODER_MODELS.items()
        if model_path.exists()
        and not (ENCODER_DIR / f"{model_key}_predictions.jsonl").exists()
    ]
    need_gen = bool(missing_encoder_files)
    if need_gen:
        print(
            "[DAY 5] Generating missing encoder predictions from "
            f"{TEST_DATA_FILE.name} ({len(missing_encoder_files)} files)..."
        )
        generate_encoder_predictions()
    else:
        print("[DAY 5] Encoder predictions found — skipping generation.")

    # Step 1
    print("[DAY 5] Loading predictions...")
    model_preds, warnings = load_all_predictions()
    if not model_preds:
        print("[DAY 5] ERROR: No predictions loaded. Aborting.")
        sys.exit(1)
    print(f"        Models loaded: {sorted(model_preds.keys())}")

    # Step 2
    print("[DAY 5] Running seqeval evaluation...")
    seqeval_res = run_seqeval(model_preds)
    _print_seqeval_table(seqeval_res)

    # Step 3
    print("[DAY 5] Computing token-level FPR/FNR...")
    token_res = compute_token_rates(model_preds)
    _print_token_table(token_res)

    # Step 4
    print("[DAY 5] Running masking pipeline + leak rate...")
    leak_res = compute_leak_rates(model_preds)
    for k, v in sorted(leak_res.items()):
        print(
            f"  {k:<22}  leak_rate={v['leak_rate']:.4f}"
            f"  ({v['leaked_count']}/{v['total_count']})"
        )
    print()

    # Step 5
    print("[DAY 5] Running paired bootstrap (N=1000)...")
    bootstrap = run_bootstrap(model_preds)
    if bootstrap:
        print(f"  Observed diff : {bootstrap['observed_diff_f1']:.4f}")
        print(f"  95% CI        : [{bootstrap['ci_lower_95']:.4f}, {bootstrap['ci_upper_95']:.4f}]")
        print(f"  p-value       : {bootstrap['p_value_one_sided']:.4f}")
        print(f"  {bootstrap['interpretation']}")
        BOOTSTRAP_OUT.parent.mkdir(parents=True, exist_ok=True)
        BOOTSTRAP_OUT.write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
        print(f"  Saved → {BOOTSTRAP_OUT}")
    print()

    # Step 6
    print("[DAY 5] Saving reports/results.json...")
    full_results = build_results_json(model_preds, seqeval_res, token_res, leak_res, bootstrap)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(full_results, indent=2), encoding="utf-8")
    print(f"  Saved → {RESULTS_JSON}")

    # Step 7
    print("[DAY 5] Saving results/day5/day5_summary.md...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary_md(full_results, bootstrap, warnings)
    SUMMARY_MD.write_text(summary, encoding="utf-8")
    print(f"  Saved → {SUMMARY_MD}")

    print()
    print("[DAY 5] DONE. All outputs written.")


if __name__ == "__main__":
    main()
