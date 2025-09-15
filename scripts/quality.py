import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import time


def _load_pd() -> Optional[object]:
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception:
        return None


def _load_sacrebleu():
    try:
        import sacrebleu  # type: ignore
        return sacrebleu
    except Exception:
        return None


def _load_comet_model(model_name: str = "Unbabel/wmt22-comet-da"):
    try:
        # unbabel-comet exposes `comet` top-level
        from comet import download_model, load_from_checkpoint  # type: ignore
        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)
        model.eval()
        return model
    except Exception:
        return None


def _load_spacy_model():
    try:
        import spacy  # type: ignore
        try:
            # Only load if already installed; do not auto-download to avoid long hangs
            return spacy.load("en_core_web_sm")
        except Exception:
            return None
    except Exception:
        return None


def _load_rouge():
    try:
        from rouge_score import rouge_scorer  # type: ignore
        return rouge_scorer
    except Exception:
        return None


def _load_bertscore_metric():
    try:
        import evaluate  # type: ignore
        metric = evaluate.load("bertscore")
        return metric
    except Exception:
        return None


def _read_frame(json_path: Path):
    """Read a pandas-style JSON DataFrame written by `DataFrame.to_json`.
    Falls back to simple JSON parsing if pandas is unavailable.
    """
    pd = _load_pd()
    if pd is not None:
        return pd.read_json(json_path)

    # Fallback: minimally reconstruct columnar dict to row dicts
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # Expect columns -> {index -> value}
    # Determine max length
    keys = list(obj.keys())
    n = 0
    for k in keys:
        try:
            n = max(n, len(obj[k]))
        except Exception:
            pass
    rows = []
    for i in range(n):
        row = {}
        for k in keys:
            vmap = obj.get(k, {})
            # pandas to_json with default orient uses string indices
            v = vmap.get(str(i)) if isinstance(vmap, dict) else None
            row[k] = v
        rows.append(row)
    return rows  # list of dicts if pandas is missing


def _extract_column(series_or_rows, col: str) -> List[str]:
    try:
        # pandas Series access
        values = series_or_rows[col].astype(str).tolist()
        return [v if v is not None else "" for v in values]
    except Exception:
        # list[dict]
        return [str((r.get(col) or "")) for r in series_or_rows]


def compute_sacrebleu(refs: List[str], hyps: List[str]) -> Optional[Dict[str, float]]:
    sacrebleu = _load_sacrebleu()
    if sacrebleu is None:
        return None
    # Filter pairs with non-empty strings
    pairs = [(r.strip(), h.strip()) for r, h in zip(refs, hyps) if r and h]
    if not pairs:
        return {"score": 0.0}
    refs_f = [r for r, _ in pairs]
    hyps_f = [h for _, h in pairs]
    bleu = sacrebleu.corpus_bleu(hyps_f, [refs_f])
    return {
        "score": float(bleu.score),
        "sys_len": int(bleu.sys_len),
        "ref_len": int(bleu.ref_len),
    }


def compute_rouge(refs: List[str], hyps: List[str]) -> Optional[Dict[str, Dict[str, float]]]:
    rouge_scorer = _load_rouge()
    if rouge_scorer is None:
        return None
    pairs = [(r.strip(), h.strip()) for r, h in zip(refs, hyps) if r and h]
    if not pairs:
        return {"rouge1": {"p": 0.0, "r": 0.0, "f1": 0.0}, "rouge2": {"p": 0.0, "r": 0.0, "f1": 0.0}, "rougeL": {"p": 0.0, "r": 0.0, "f1": 0.0}}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {
        "rouge1": {"p": 0.0, "r": 0.0, "f1": 0.0},
        "rouge2": {"p": 0.0, "r": 0.0, "f1": 0.0},
        "rougeL": {"p": 0.0, "r": 0.0, "f1": 0.0},
    }
    for r, h in pairs:
        s = scorer.score(r, h)
        for k in ["rouge1", "rouge2", "rougeL"]:
            totals[k]["p"] += float(s[k].precision)
            totals[k]["r"] += float(s[k].recall)
            totals[k]["f1"] += float(s[k].fmeasure)
    n = float(len(pairs))
    for k in totals:
        totals[k]["p"] /= n
        totals[k]["r"] /= n
        totals[k]["f1"] /= n
    return totals


def compute_bertscore(refs: List[str], hyps: List[str], metric, model_type: str = "bert-base-uncased", lang: str = "en", batch_size: int = 32) -> Optional[Dict[str, float]]:
    if metric is None:
        return None
    pairs = [(r.strip(), h.strip()) for r, h in zip(refs, hyps) if r and h]
    if not pairs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0}
    refs_f = [r for r, _ in pairs]
    hyps_f = [h for _, h in pairs]
    try:
        out = metric.compute(predictions=hyps_f, references=refs_f, model_type=model_type, lang=lang, batch_size=batch_size, device="cpu")
        # out keys: precision, recall, f1
        p = out.get("precision", [])
        r = out.get("recall", [])
        f = out.get("f1", [])
        if not f:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0}
        n = float(len(f))
        return {
            "precision": float(sum(p) / n) if p else 0.0,
            "recall": float(sum(r) / n) if r else 0.0,
            "f1": float(sum(f) / n),
            "n": int(n),
        }
    except Exception:
        return None


def compute_comet(refs: List[str], hyps: List[str], comet_model) -> Optional[Dict[str, float]]:
    if comet_model is None:
        return None
    data = []
    for r, h in zip(refs, hyps):
        r = (r or "").strip()
        h = (h or "").strip()
        if not r or not h:
            continue
        data.append({"src": None, "mt": h, "ref": r})
    if not data:
        return {"mean": 0.0}
    out = comet_model.predict(data, batch_size=16, gpus=0)
    scores = out["scores"] if isinstance(out, dict) else out.scores
    if not scores:
        return {"mean": 0.0}
    return {
        "mean": float(sum(scores) / len(scores)),
        "n": int(len(scores)),
    }


def _extract_ents(nlp, text: str) -> List[Tuple[str, str]]:
    doc = nlp((text or "").strip())
    return [(ent.label_, ent.text.strip().lower()) for ent in doc.ents]


def compute_ner_overlap(refs: List[str], hyps: List[str], nlp) -> Optional[Dict[str, float]]:
    if nlp is None:
        return None
    tp = fp = fn = 0
    for r, h in zip(refs, hyps):
        rset = set(_extract_ents(nlp, r))
        hset = set(_extract_ents(nlp, h))
        tp += len(rset & hset)
        fp += len(hset - rset)
        fn += len(rset - hset)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate_file(
    clsa_path: Path,
    *,
    comet_model=None,
    nlp=None,
    enable_bleu: bool = True,
    enable_rouge: bool = True,
    enable_bertscore: bool = False,
    enable_comet: bool = False,
    enable_ner: bool = False,
    bertscore_metric=None,
    bertscore_model: str = "bert-base-uncased",
) -> Dict:
    frame = _read_frame(clsa_path)
    refs = _extract_column(frame, "generated_text")

    results: Dict[str, Dict] = {}

    # Backtranslation vs original
    backs = _extract_column(frame, "backtranslation") if isinstance(frame, list) or "backtranslation" in getattr(frame, "columns", []) else []
    if backs:
        results["backtranslation"] = {}
        if enable_bleu:
            results["backtranslation"]["sacrebleu"] = compute_sacrebleu(refs, backs)
        if enable_rouge:
            results["backtranslation"]["rouge"] = compute_rouge(refs, backs)
        if enable_bertscore:
            results["backtranslation"]["bertscore"] = compute_bertscore(refs, backs, bertscore_metric, model_type=bertscore_model)
        if enable_comet:
            results["backtranslation"]["comet"] = compute_comet(refs, backs, comet_model)
        if enable_ner:
            results["backtranslation"]["ner_overlap"] = compute_ner_overlap(refs, backs, nlp)

    # Paraphrase vs original
    paras = _extract_column(frame, "paraphrase") if isinstance(frame, list) or "paraphrase" in getattr(frame, "columns", []) else []
    if paras:
        results["paraphrase"] = {}
        if enable_bleu:
            results["paraphrase"]["sacrebleu"] = compute_sacrebleu(refs, paras)
        if enable_rouge:
            results["paraphrase"]["rouge"] = compute_rouge(refs, paras)
        if enable_bertscore:
            results["paraphrase"]["bertscore"] = compute_bertscore(refs, paras, bertscore_metric, model_type=bertscore_model)
        if enable_comet:
            results["paraphrase"]["comet"] = compute_comet(refs, paras, comet_model)
        if enable_ner:
            results["paraphrase"]["ner_overlap"] = compute_ner_overlap(refs, paras, nlp)

    # XLSum (reference, non-English) vs generated English: compute BLEU/ROUGE if requested
    xlsums = _extract_column(frame, "xlsum") if isinstance(frame, list) or "xlsum" in getattr(frame, "columns", []) else []
    if xlsums:
        results["xlsum"] = {}
        if enable_bleu:
            results["xlsum"]["sacrebleu"] = compute_sacrebleu(refs, xlsums)
        if enable_rouge:
            results["xlsum"]["rouge"] = compute_rouge(refs, xlsums)

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute quality metrics for CLSA outputs.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to data directory (default: <repo>/data)")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path (default: <data-dir>/quality.json)")
    parser.add_argument("--pattern", type=str, default="*_clsa.json", help="Glob pattern for CLSA files")
    # Metrics toggles: BLEU on by default; COMET/NER opt-in if available
    parser.add_argument("--no-bleu", action="store_true", help="Disable SacreBLEU scoring (enabled by default)")
    parser.add_argument("--no-rouge", action="store_true", help="Disable ROUGE scoring (enabled by default)")
    parser.add_argument("--enable-bertscore", action="store_true", help="Enable BERTScore scoring (may download model)")
    parser.add_argument("--bertscore-model", type=str, default="bert-base-uncased", help="HF model name for BERTScore")
    parser.add_argument("--enable-comet", action="store_true", help="Enable COMET scoring (no auto-download)")
    parser.add_argument("--enable-ner", action="store_true", help="Enable NER overlap via spaCy (no auto-download)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files processed (0 = no limit)")
    parser.add_argument("--comet-model", type=str, default="Unbabel/wmt22-comet-da", help="COMET model name")
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else (repo_dir / "data")
    out_path = Path(args.out) if args.out else (data_dir / "quality.json")

    clsa_files = sorted(data_dir.glob(args.pattern))
    if args.limit:
        clsa_files = clsa_files[: args.limit]

    results: Dict[str, Dict] = {}

    # Load heavy deps at most once and only if enabled
    enable_bleu = not args.no_bleu
    enable_rouge = not args.no_rouge
    comet_model = None
    nlp = None
    bertscore_metric = None
    if args.enable_comet:
        print("[quality] Loading COMET model...", flush=True)
        t0 = time.time()
        comet_model = _load_comet_model(args.comet_model)
        if comet_model is None:
            print("[quality] COMET not available; skipping.")
        else:
            print(f"[quality] COMET ready in {time.time()-t0:.1f}s.")
    if args.enable_ner:
        print("[quality] Loading spaCy model...", flush=True)
        t0 = time.time()
        nlp = _load_spacy_model()
        if nlp is None:
            print("[quality] spaCy model not available; skipping.")
        else:
            print(f"[quality] spaCy ready in {time.time()-t0:.1f}s.")
    if args.enable_bertscore:
        print("[quality] Loading BERTScore metric...", flush=True)
        t0 = time.time()
        bertscore_metric = _load_bertscore_metric()
        if bertscore_metric is None:
            print("[quality] BERTScore not available; skipping.")
        else:
            print(f"[quality] BERTScore ready in {time.time()-t0:.1f}s.")

    total = len(clsa_files)
    for idx, fp in enumerate(clsa_files, 1):
        stem = fp.stem  # e.g., Unigram_spanish_clsa
        base = stem[:-5] if stem.endswith("_clsa") else stem
        parts = base.split("_")
        lang = parts[-1] if len(parts) >= 2 else None
        algo = "_".join(parts[:-1]) if len(parts) >= 2 else base

        print(f"[quality] ({idx}/{total}) Processing {fp.name}...", flush=True)

        entry: Dict[str, object] = {
            "file": str(fp.relative_to(repo_dir)),
            "algorithm": algo,
            "language": lang,
        }

        # Compute quality metrics
        entry["quality"] = evaluate_file(
            fp,
            comet_model=comet_model,
            nlp=nlp,
            enable_bleu=enable_bleu,
            enable_rouge=enable_rouge,
            enable_bertscore=args.enable_bertscore,
            enable_comet=args.enable_comet,
            enable_ner=args.enable_ner,
            bertscore_metric=bertscore_metric,
            bertscore_model=args.bertscore_model,
        )

        # Attach detection metric files if present (copied into this JSON for convenience)
        clsa_metrics = data_dir / f"{base}_clsa_metrics.json"
        back_metrics = data_dir / f"{base}_back_metrics.json"
        if clsa_metrics.exists():
            try:
                entry["detection_metrics_clsa"] = json.loads(clsa_metrics.read_text(encoding="utf-8"))
            except Exception:
                entry["detection_metrics_clsa"] = None
        if back_metrics.exists():
            try:
                entry["detection_metrics_back"] = json.loads(back_metrics.read_text(encoding="utf-8"))
            except Exception:
                entry["detection_metrics_back"] = None

        results[base] = entry

    # Add baseline quality metrics using baseline paraphrase files if present
    for algo in ["KGW", "XSIR", "Unigram", "SIR"]:
        base_para = data_dir / f"{algo}_paraphrase.json"
        if not base_para.exists():
            continue
        print(f"[quality] Computing baseline paraphrase quality for {algo}...", flush=True)
        frame = _read_frame(base_para)
        refs = _extract_column(frame, "generated_text")
        paras = _extract_column(frame, "paraphrase")
        bq: Dict[str, Dict] = {"paraphrase": {}}
        if enable_bleu:
            bq["paraphrase"]["sacrebleu"] = compute_sacrebleu(refs, paras)
        if enable_rouge:
            bq["paraphrase"]["rouge"] = compute_rouge(refs, paras)
        if args.enable_bertscore:
            bq["paraphrase"]["bertscore"] = compute_bertscore(refs, paras, bertscore_metric, model_type=args.bertscore_model)
        if args.enable_comet:
            bq["paraphrase"]["comet"] = compute_comet(refs, paras, comet_model)
        if args.enable_ner:
            bq["paraphrase"]["ner_overlap"] = compute_ner_overlap(refs, paras, nlp)

        results[f"{algo}_baseline"] = {
            "file": str(base_para.relative_to(repo_dir)),
            "algorithm": algo,
            "language": None,
            "quality": bq,
        }

    # Also list the raw metrics files encountered (without modifying them)
    for mfp in sorted(data_dir.glob("*_metrics.json")):
        # Skip ones that were already included above
        if any(str(mfp.name).startswith(k) for k in results.keys()):
            continue
        try:
            mdata = json.loads(mfp.read_text(encoding="utf-8"))
        except Exception:
            mdata = None
        results[str(mfp.stem)] = {
            "file": str(mfp.relative_to(repo_dir)),
            "detection_metrics": mdata,
        }

    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved quality metrics to {out_path.resolve()}")


if __name__ == "__main__":
    main()
