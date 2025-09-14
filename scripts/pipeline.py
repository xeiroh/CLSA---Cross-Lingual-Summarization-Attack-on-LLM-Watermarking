from functools import lru_cache
import torch
import pandas as pd
from tqdm import tqdm
from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
)


def _lang_to_code(lang: str) -> str:
    """Map language name to ISO code used by M2M100/mT5.

    Defaults to English if unknown.
    """
    if not lang:
        return "en"
    l = lang.strip().lower()
    mapping = {
        "english": "en",
        "en": "en",
        "swahili": "sw",
        "sw": "sw",
        "amharic": "am",
        "am": "am",
        "spanish": "es",
        "es": "es",
        "french": "fr",
        "fr": "fr",
        "german": "de",
        "de": "de",
        "arabic": "ar",
        "ar": "ar",
        "bengali": "bn",
        "bn": "bn",
        "hindi": "hi",
        "hi": "hi",
        "chinese": "zh",
        "zh": "zh",
    }
    return mapping.get(l, "en")


@lru_cache(None)
def _get_m2m_models():
    """Cache and return M2M100 tokenizer and model for translation."""
    t = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    m = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    try:
        if torch.cuda.is_available():
            m = m.to("cuda")
    except Exception:
        pass
    return t, m


@lru_cache(None)
def _get_xlsum_mt5_models():
    """Cache and return the mT5 model fine-tuned on XLSum for summarization."""
    name = "csebuetnlp/mT5_multilingual_XLSum"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    try:
        if torch.cuda.is_available():
            model = model.to("cuda")
    except Exception:
        pass
    return tok, model


@lru_cache(None)
def _get_pegasus_paraphrase_models():
    """Cache and return Pegasus paraphrasing tokenizer and model."""
    name = "tuner007/pegasus_paraphrase"
    tok = PegasusTokenizer.from_pretrained(name)
    model = PegasusForConditionalGeneration.from_pretrained(name)
    try:
        if torch.cuda.is_available():
            model = model.to("cuda")
    except Exception:
        pass
    return tok, model


def xlsum_attack(text: str, language: str = "swahili") -> str:
    """Translate `text` to `language` and summarize using mT5 XLSum.

    1) Translate with M2M100 (assumes source is English by default).
    2) Summarize with csebuetnlp/mT5_multilingual_XLSum in the target language.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    src_code = "en"  # assume input articles are English by default (XLSum-English)
    tgt_code = _lang_to_code(language)

    # 1) Translate to target language using M2M100
    try:
        trans_tok, trans_model = _get_m2m_models()
        trans_tok.src_lang = src_code
        enc = trans_tok(text, return_tensors="pt", truncation=True, max_length=1024)
        if torch.cuda.is_available():
            enc = {k: v.to("cuda") for k, v in enc.items()}
        forced_id = trans_tok.get_lang_id(tgt_code)
        with torch.no_grad():
            gen_ids = trans_model.generate(
                **enc,
                forced_bos_token_id=forced_id,
                max_length=1024,
                num_beams=4,
            )
        translated = trans_tok.batch_decode(gen_ids, skip_special_tokens=True)[0]
    except Exception:
        translated = text  # fallback

    # 2) Summarize translated text using mT5 XLSum
    sum_tok, sum_model = _get_xlsum_mt5_models()
    # mT5 XLSum typically works without a prefix, but T5-style prefixes are safe
    input_text = translated
    enc = sum_tok(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    if torch.cuda.is_available():
        enc = {k: v.to("cuda") for k, v in enc.items()}

    with torch.no_grad():
        out = sum_model.generate(
            **enc,
            max_length=200,
            min_length=30,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True,
        )
    summary = sum_tok.decode(out[0], skip_special_tokens=True).strip()
    return summary


def xlsum(
    df: pd.DataFrame,
    language: str = "swahili",
    source_col: str = "generated_text",
    out_col: str = "xlsum",
    back_col: str = "backtranslation",
    batch_size: int = 8,
    *,
    max_src_len: int = 1024,
    max_sum_len: int = 200,
    min_sum_len: int = 30,
) -> pd.DataFrame:
    """Add translated+summarized text to a detections DataFrame.

    - Reads source from `source_col` (defaults to 'generated_text').
    - Writes results to `out_col` (defaults to 'xlsum').
    - Returns the same DataFrame for convenience.
    """
    if source_col not in df.columns:
        raise ValueError(f"DataFrame missing required column: {source_col}")

    texts = df[source_col].astype(str).tolist()
    n = len(texts)
    results: list[str] = [""] * n
    back_results: list[str] = [""] * n

    # Prefetch cached models once
    trans_tok, trans_model = _get_m2m_models()
    sum_tok, sum_model = _get_xlsum_mt5_models()
    src_code = "en"
    tgt_code = _lang_to_code(language)
    trans_tok.src_lang = src_code
    forced_id = trans_tok.get_lang_id(tgt_code)
    en_id = trans_tok.get_lang_id("en")

    device = next(sum_model.parameters()).device if hasattr(sum_model, 'parameters') else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    # Indices with non-empty text
    idxs = [i for i, t in enumerate(texts) if t.strip()]

    # Process in batches
    for bi in tqdm(range(0, len(idxs), max(1, batch_size)), desc=f"XLSum {language}", unit="batch"):
        batch_ids = idxs[bi: bi + max(1, batch_size)]
        batch_texts = [texts[i] for i in batch_ids]

        # 1) Translate batch to target language
        try:
            enc_t = trans_tok(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_src_len,
            )
            if torch.cuda.is_available():
                enc_t = {k: v.to("cuda") for k, v in enc_t.items()}
            with torch.no_grad():
                gen_ids_t = trans_model.generate(
                    **enc_t,
                    forced_bos_token_id=forced_id,
                    max_length=max_src_len,
                    num_beams=4,
                )
            translated_batch = trans_tok.batch_decode(gen_ids_t, skip_special_tokens=True)
        except Exception:
            translated_batch = batch_texts

        # 2) Summarize batch with mT5 XLSum
        enc_s = sum_tok(
            translated_batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_src_len,
        )
        if torch.cuda.is_available():
            enc_s = {k: v.to("cuda") for k, v in enc_s.items()}
        with torch.no_grad():
            out = sum_model.generate(
                **enc_s,
                max_length=max_sum_len,
                min_length=min_sum_len,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True,
            )
        summaries = sum_tok.batch_decode(out, skip_special_tokens=True)

        for j, idx in enumerate(batch_ids):
            results[idx] = summaries[j].strip() if j < len(summaries) else ""

        # 3) Backtranslate summaries to English
        try:
            # Set source language to the target code for backtranslation
            trans_tok.src_lang = tgt_code
            enc_bt = trans_tok(
                [results[i] for i in batch_ids],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_src_len,
            )
            if torch.cuda.is_available():
                enc_bt = {k: v.to("cuda") for k, v in enc_bt.items()}
            with torch.no_grad():
                gen_ids_bt = trans_model.generate(
                    **enc_bt,
                    forced_bos_token_id=en_id,
                    max_length=max_src_len,
                    num_beams=4,
                )
            back_batch = trans_tok.batch_decode(gen_ids_bt, skip_special_tokens=True)
        except Exception:
            back_batch = [results[i] for i in batch_ids]

        for j, idx in enumerate(batch_ids):
            back_results[idx] = back_batch[j].strip() if j < len(back_batch) else ""

    df[out_col] = results
    df[back_col] = back_results
    return df


def paraphrase(
    df: pd.DataFrame,
    source_col: str = "generated_text",
    out_col: str = "paraphrase",
    batch_size: int = 8,
    *,
    max_src_len: int = 256,
    max_tgt_len: int = 128,
) -> pd.DataFrame:
    """Paraphrase `source_col` using Pegasus and store in `out_col`.

    Uses cached "tuner007/pegasus_paraphrase" model.
    """
    if source_col not in df.columns:
        raise ValueError(f"DataFrame missing required column: {source_col}")

    texts = df[source_col].astype(str).tolist()
    n = len(texts)
    outs: list[str] = [""] * n

    tok, model = _get_pegasus_paraphrase_models()

    # Indices with non-empty text
    idxs = [i for i, t in enumerate(texts) if t.strip()]

    for bi in tqdm(range(0, len(idxs), max(1, batch_size)), desc="Paraphrase", unit="batch"):
        batch_ids = idxs[bi: bi + max(1, batch_size)]
        batch_texts = [texts[i] for i in batch_ids]

        enc = tok(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_src_len,
        )
        if torch.cuda.is_available():
            enc = {k: v.to("cuda") for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_length=max_tgt_len,
                num_beams=8,
                num_return_sequences=1,
                length_penalty=1.0,
                early_stopping=True,
            )
        dec = tok.batch_decode(out, skip_special_tokens=True)
        for j, idx in enumerate(batch_ids):
            outs[idx] = dec[j].strip() if j < len(dec) else ""

    df[out_col] = outs
    return df
