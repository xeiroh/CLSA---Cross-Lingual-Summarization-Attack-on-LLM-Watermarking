from __future__ import annotations

import os
from typing import Iterable

import pandas as pd
import torch
from tqdm import tqdm

from datasets import Dataset
from transformers import LogitsProcessorList

from evaluation import DATA_PATH, evaluate_detection
from tools import load_data, split_dataset

# Watermarking utils (same library used in tools.load_model)
from markllm.watermark.auto_watermark import AutoWatermark
from markllm.utils.transformers_config import TransformersConfig

# Use the XLSum mT5 summarization model (already configured in pipeline.py)
from pipeline import _get_xlsum_mt5_models


def _generate_xlsum(
    model_components,
    dataset: Iterable[dict] | Dataset,
    watermark: bool,
    *,
    max_src_len: int = 1024,
    max_new_tokens: int = 256,
    min_new_tokens: int = 64,
):
    """Generate summaries with mT5 XLSum, optionally watermarked.

    Returns a list of detection dicts with keys including:
      - generated_text
      - score, is_watermarked (from detector)
      - true_label (1 if watermark else 0)
    """
    tok, model, wm = model_components
    model.eval()

    results = []

    # Iterate sequentially to control VRAM usage
    try:
        total = len(dataset)
    except Exception:
        total = None

    for ex in tqdm(dataset, total=total, desc=f"XLSum gen ({'wm' if watermark else 'no-wm'})", unit="ex"):
        src = ex.get("text", None)
        if not isinstance(src, str) or not src.strip():
            continue

        enc = tok(
            src,
            return_tensors="pt",
            truncation=True,
            max_length=max_src_len,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        logits_proc = None
        if watermark and hasattr(wm, "logits_processor") and wm.logits_processor is not None:
            logits_proc = LogitsProcessorList([wm.logits_processor])

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                num_beams=2,
                do_sample=False,
                length_penalty=1.0,
                early_stopping=True,
                logits_processor=logits_proc,
                pad_token_id=getattr(tok, "pad_token_id", None),
                eos_token_id=getattr(tok, "eos_token_id", None),
            )

        generated_text = tok.decode(out[0], skip_special_tokens=True).strip()

        det = wm.detect_watermark(generated_text)
        det["generated_text"] = generated_text
        det["true_label"] = 1 if watermark else 0
        results.append(det)

    return results


def _load_xlsum_with_watermark(algorithm: str, *, max_new_tokens: int = 256):
    """Load XLSum mT5 + a MarkLLM watermark model for the given algorithm."""
    tok, model = _get_xlsum_mt5_models()

    # Build watermark wrapper for seq2seq generation
    tf_cfg = TransformersConfig(model=model, tokenizer=tok, max_new_tokens=max_new_tokens)
    wm = AutoWatermark.load(algorithm_name=algorithm, transformers_config=tf_cfg)
    return tok, model, wm


def cwra_chinese(algorithm: str, samples: int = 200, max_new_tokens: int = 256):
    """Run CWRA baseline using XLSum (Chinese) with half WM and half no-WM.

    Returns detections DataFrame and metrics dict.
    """
    # Load XLSum test split for Chinese
    dataset = load_data("chinese_simplified")

    # Model + watermark for the requested algorithm
    model_components = _load_xlsum_with_watermark(algorithm, max_new_tokens=max_new_tokens)

    # Split into equal halves for watermarked vs non-watermarked generation
    wm_samples, uwm_samples = split_dataset(dataset, sample_size=samples)

    det_wm = _generate_xlsum(model_components, wm_samples, watermark=True, max_new_tokens=max_new_tokens)
    det_uwm = _generate_xlsum(model_components, uwm_samples, watermark=False, max_new_tokens=max_new_tokens)

    detections = det_wm + det_uwm
    df = pd.DataFrame(detections)

    metrics = evaluate_detection(df)
    return df, metrics


if __name__ == "__main__":

    # Run for both algorithms and save to DATA_PATH
    for algo, out_name in (("XSIR", "XSIR_chinese.json"), ("KGW", "KGW_chinese.json")):
        print(f"Running CWRA Chinese with {algo}...")
        try:
            df, metrics = cwra_chinese(algorithm=algo, samples=200, max_new_tokens=256)
            out_path = os.path.join(DATA_PATH, out_name)
            df.to_json(out_path)
            print(f"Saved detections to: {out_path}")
            print("Metrics:", metrics)
        except FileNotFoundError as e:
            print(f"Skipping {algo}: {e}")
