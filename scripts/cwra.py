from __future__ import annotations

import os
import numpy as np
from typing import Iterable, List

import torch
from tqdm import tqdm

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList

from evaluation import DATA_PATH
from tools import load_data, save_file, detect

# Watermarking utils
from markllm.watermark.auto_watermark import AutoWatermark
from markllm.utils.transformers_config import TransformersConfig


def _get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _load_llama_with_watermark(algorithm: str, *, max_new_tokens: int = 256):
    """Load Llama 2 7B + MarkLLM watermark wrapper for the given algorithm."""
    model_name = "meta-llama/Llama-2-7b-hf"
    print(f"[cwra] Loading tokenizer: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    # Left padding is typically better for decoder-only models with KV cache
    try:
        tok.padding_side = "left"
    except Exception:
        pass
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[cwra] Loading model: {model_name}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            trust_remote_code=True,
            device_map="auto",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        model = model.to(_get_device())

    try:
        if len(tok) != model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tok))
    except Exception:
        pass

    tf_cfg = TransformersConfig(model=model, tokenizer=tok, max_new_tokens=max_new_tokens)
    wm = AutoWatermark.load(algorithm_name=algorithm, transformers_config=tf_cfg)
    return tok, model, wm


def _select_samples(dataset: Dataset, n: int, seed: int = 42) -> List[dict]:
    n = min(n, len(dataset))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=n, replace=False)
    return [dataset[int(i)] for i in idx]


def _paraphrase_chinese(
    model_components,
    dataset: Iterable[dict] | Dataset,
    *,
    watermark: bool = True,
    max_src_len: int = 1024,
    max_new_tokens: int = 256,
    min_new_tokens: int = 64,
    batch_size: int = 8,
) -> list[dict]:
    """Generate Chinese paraphrases with Llama 2 7B and (optionally) watermark them.

    Returns list of dicts with keys: generated_text, score, is_watermarked, true_label.
    """
    tok, model, wm = model_components
    model.eval()

    # Prepare prompts
    items = [ex for ex in dataset]
    prompts: List[str] = []
    for ex in items:
        src = ex.get("text", None)
        if isinstance(src, str) and src.strip():
            prompt = (
                "请将下面的中文文本进行同义改写，保持原意但使用不同且自然的表达方式：\n\n"
                f"文本：\n{src.strip()}\n\n改写："
            )
        else:
            prompt = ""
        prompts.append(prompt)

    outputs: List[str] = [""] * len(prompts)
    logits_proc = None
    if watermark and hasattr(wm, "logits_processor") and wm.logits_processor is not None:
        logits_proc = LogitsProcessorList([wm.logits_processor])

    # Batched generation
    for i in tqdm(range(0, len(prompts), max(1, batch_size)), desc=f"Llama2 paraphrase ({'wm' if watermark else 'no-wm'})", unit="batch"):
        batch = prompts[i:i + batch_size]
        # Filter empties while preserving indices
        idxs = [j for j, p in enumerate(batch) if p]
        if not idxs:
            continue
        enc = tok(
            [batch[j] for j in idxs],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_src_len,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            # Enable AMP on CUDA for speed
            if model.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    out = model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.95,
                        num_beams=1,
                        logits_processor=logits_proc,
                        use_cache=True,
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id,
                    )
            else:
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    num_beams=1,
                    logits_processor=logits_proc,
                    use_cache=True,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )

        attn = enc.get("attention_mask")
        for k, j in enumerate(idxs):
            in_len = int(attn[k].sum().item()) if attn is not None else int(enc["input_ids"].shape[1])
            gen_ids = out[k][in_len:]
            outputs[i + j] = tok.decode(gen_ids, skip_special_tokens=True).strip()

    # Parallelize detection using tools.detect for speed
    import pandas as _pd
    tmp_df = _pd.DataFrame({"generated_text": outputs})
    det_df = detect(tmp_df, wm, column="generated_text")

    results: list[dict] = []
    for idx in range(len(outputs)):
        row = det_df.iloc[idx].to_dict() if idx < len(det_df) else {}
        row["generated_text"] = outputs[idx]
        row["true_label"] = 1 if watermark else 0
        results.append(row)

    return results


def cwra_chinese_llama(
    algorithm: str,
    samples: int = 200,
    max_new_tokens: int = 256,
    batch_size: int = 8,
    watermark: bool = False,
) -> list[dict]:
    """Run CWRA with Llama 2 7B paraphrasing on Chinese XLSum texts.

    By default generates UNWATERMARKED paraphrases (watermark=False).
    """
    dataset = load_data("chinese_simplified")
    picked = _select_samples(dataset, samples)

    model_components = _load_llama_with_watermark(algorithm, max_new_tokens=max_new_tokens)

    detections = _paraphrase_chinese(
        model_components,
        picked,
        watermark=watermark,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    return detections


if __name__ == "__main__":
    import json

    algo = "XSIR"
    print(f"[cwra] Running Chinese CWRA (unwatermarked) with Llama2 7B and {algo} detector...")
    try:
        detections = cwra_chinese_llama(
            algorithm=algo,
            samples=200,
            max_new_tokens=256,
            batch_size=8,
            watermark=False,
        )

        # Append to existing XSIR_cwra_chinese.json if present
        out_name = f"{algo}_cwra_chinese.json"
        out_path = os.path.join(DATA_PATH, out_name)
        existing: list[dict] = []
        try:
            if os.path.exists(out_path):
                with open(out_path, "r") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        existing = loaded
        except Exception as e:
            print(f"[cwra] Warning: failed to read existing file {out_path}: {e}")

        combined = existing + detections
        save_file(combined, filename=out_path)
        print(f"[cwra] Appended {len(detections)} unwatermarked samples. Saved to: {out_path}")
    except Exception as e:
        print(f"[cwra] Failed for {algo}: {e}")
