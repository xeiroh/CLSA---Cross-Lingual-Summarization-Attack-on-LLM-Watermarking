import os
from functools import lru_cache
from typing import List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from evaluation import DATA_PATH
from markllm.watermark.auto_watermark import AutoWatermark
from markllm.utils.transformers_config import TransformersConfig

from tools import detect  # reuse existing parallel detection helper


@lru_cache(None)
def _get_flan_t5() -> tuple:
    """Load and cache FLAN-T5 base for paraphrasing and detection tokenization."""
    name = "google/flan-t5-base"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    # Align vocab if needed
    try:
        if len(tok) != model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tok))
    except Exception:
        pass
    # Ensure special tokens present
    try:
        if getattr(model.config, "pad_token_id", None) is None and getattr(tok, "pad_token_id", None) is not None:
            model.config.pad_token_id = tok.pad_token_id
        if getattr(model.config, "eos_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
            model.config.eos_token_id = tok.eos_token_id
        if getattr(model.config, "decoder_start_token_id", None) is None and getattr(tok, "pad_token_id", None) is not None:
            model.config.decoder_start_token_id = tok.pad_token_id
    except Exception:
        pass
    # Prefer CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    try:
        dev = next(model.parameters()).device
        print(f"[paraphrase] FLAN-T5 device: {dev}")
    except Exception:
        pass
    return tok, model


def paraphrase_batch(texts: List[str], max_src_len: int = 512, max_tgt_len: int = 256) -> List[str]:
    """Paraphrase a batch of English texts with FLAN-T5 base."""
    tok, model = _get_flan_t5()
    device = next(model.parameters()).device

    prompts = [
        (
            "Paraphrase the following text in English, preserving meaning and naturalness.\n\n" + t.strip()
        )
        if isinstance(t, str) and t.strip() else ""
        for t in texts
    ]

    # Identify non-empty items to avoid wasted compute
    idxs = [i for i, p in enumerate(prompts) if p]
    if not idxs:
        return ["" for _ in texts]

    enc = tok(
        [prompts[i] for i in idxs],
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_src_len,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_length=max_tgt_len,
            min_length=min(64, max_tgt_len),
            num_beams=2,
            do_sample=True,
            top_p=0.92,
            temperature=0.8,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

    dec = tok.batch_decode(out, skip_special_tokens=True)
    results = ["" for _ in texts]
    for j, i in enumerate(idxs):
        results[i] = dec[j].strip() if j < len(dec) else ""
    return results


def paraphrase_df(
    df: pd.DataFrame,
    source_col: str = "generated_text",
    out_col: str = "paraphrase",
    batch_size: int = 8,
    *,
    max_src_len: int = 512,
    max_tgt_len: int = 256,
) -> pd.DataFrame:
    """Add a paraphrase column to the DataFrame using FLAN-T5."""
    if source_col not in df.columns:
        raise ValueError(f"DataFrame missing required column: {source_col}")
    texts = df[source_col].astype(str).tolist()
    n = len(texts)
    out: List[str] = ["" for _ in range(n)]

    for i in tqdm(range(0, n, max(1, batch_size)), desc="Paraphrasing", unit="batch"):
        batch = texts[i:i + batch_size]
        res = paraphrase_batch(batch, max_src_len=max_src_len, max_tgt_len=max_tgt_len)
        out[i:i + batch_size] = res

    df[out_col] = out
    return df


def detect_on_paraphrase(df: pd.DataFrame, algorithm: str, column: str = "paraphrase") -> pd.DataFrame:
    """Run watermark detection on the paraphrased column and return a DataFrame of detections."""
    # Build a watermark detector using the FLAN-T5 tokenizer/model config
    tok, model = _get_flan_t5()
    tf_cfg = TransformersConfig(model=model, tokenizer=tok, max_new_tokens=256)
    wm = AutoWatermark.load(algorithm_name=algorithm, transformers_config=tf_cfg)
    det_df = detect(df, wm, column)
    return det_df


def process_algorithms(algorithms: List[str]) -> None:
    for algo in algorithms:
        in_path = os.path.join(DATA_PATH, f"{algo}.json")
        if not os.path.exists(in_path):
            print(f"[paraphrase] Skip {algo}: file not found at {in_path}")
            continue
        print(f"[paraphrase] Loading: {in_path}")
        df = pd.read_json(in_path)

        # Paraphrase generated text
        df = paraphrase_df(df, source_col="generated_text", out_col="paraphrase", batch_size=8)

        # Detect watermark on paraphrases
        det = detect_on_paraphrase(df, algo, column="paraphrase")
        # Attach detection columns
        if "score" in det.columns:
            df["paraphrase_score"] = det["score"].values
        else:
            df["paraphrase_score"] = None
        if "is_watermarked" in det.columns:
            df["paraphrase_watermarked"] = det["is_watermarked"].values
        else:
            df["paraphrase_watermarked"] = None

        out_path = os.path.join(DATA_PATH, f"{algo}_paraphrase.json")
        df.to_json(out_path)
        print(f"[paraphrase] Saved: {out_path}")


if __name__ == "__main__":
    # Default set based on user request; de-duplicate while preserving order
    algos = ["Unigram", "KGW", "XSIR", "SIR"]
    process_algorithms(algos)

