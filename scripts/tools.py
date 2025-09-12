# minimal example sketch (adapt paths/models to your GPU/CPU)
import os
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # or AutoModelForCausalLM
from transformers import LogitsProcessorList
from markllm.watermark.auto_watermark import AutoWatermark, WATERMARK_MAPPING_NAMES
from markllm.utils.transformers_config import TransformersConfig
import warnings
from functools import lru_cache

warnings.filterwarnings("ignore")  # transformers logging

# device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device, "cuda available:", torch.cuda.is_available())
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "results")

def _normalize_algorithm(name: str | None) -> str | None:
    """Normalize user-provided algorithm names to library-supported keys.

    - Case-insensitive match of known keys from WATERMARK_MAPPING_NAMES
    - Common aliases mapped to canonical names (e.g., "Unbiased"/"UW" -> "Unigram")
    """
    if name is None:
        return None
    s = str(name).strip()
    if not s:
        return None
    for key in WATERMARK_MAPPING_NAMES.keys():
        if s.lower() == key.lower():
            return key
    aliases = {
        "uw": "Unigram",
        "unbiased": "Unigram",
    }
    return aliases.get(s.lower())

@lru_cache(maxsize=8)
def load_model(max_tokens=512, algorithm="KGW"):

	# load summarization model (use a small checkpoint for quick tests)
	tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
	model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/mT5_multilingual_XLSum").to(
		device
	)

	# Use the model's actual output vocab size to avoid mask/logits mismatch
	try:
		model_vocab_size = model.get_output_embeddings().weight.shape[0]
	except Exception:
		model_vocab_size = getattr(getattr(model, "config", object()), "vocab_size", None)
	
	if not model_vocab_size:
		# Fallback to tokenizer length, but this may cause mismatch for some models
		model_vocab_size = len(tokenizer)

	transformers_config = TransformersConfig(
		model=model,
		tokenizer=tokenizer,
		vocab_size=model_vocab_size,
		device=device,
		max_new_tokens=max_tokens,
		do_sample=True,
	)

	# normalize and validate algorithm name
	alg = _normalize_algorithm(algorithm)
	if not alg:
		supported = ", ".join(sorted(WATERMARK_MAPPING_NAMES.keys()))
		raise ValueError(f"Unsupported algorithm '{algorithm}'. Supported: {supported}. Aliases: UW/Unbiased -> Unigram.")

	# load watermark algorithm (e.g., KGW)
	try:
		wm = AutoWatermark.load(alg, transformers_config=transformers_config)
	except FileNotFoundError as e:
		if alg == "XSIR":
			raise FileNotFoundError(
				"XSIR assets missing. Provide a valid XSIR config with: "
				"embedding_model_path (e.g., 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'), "
				"transform_model_name (.pth), and mapping_name (.json); or exclude XSIR. "
				f"Original error: {e}"
			)
		raise
	return wm

def load_data(language="english"):
	dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset = load_dataset("csebuetnlp/xlsum", language, split="test")
	return dataset

def _json_default(o):
	"""Best-effort JSON serializer for numpy/torch objects."""
	try:
		import numpy as np
		if isinstance(o, (np.integer,)):
			return int(o)
		if isinstance(o, (np.floating,)):
			return float(o)
		if isinstance(o, (np.ndarray,)):
			return o.tolist()
	except Exception:
		pass
	try:
		import torch as _torch
		if isinstance(o, _torch.Tensor):
			return o.detach().cpu().tolist()
	except Exception:
		pass
	return str(o)


def save_file(data, filename: str | None = None, out_dir: str | None = None, name: str | None = None, with_timestamp: bool = True) -> str:
	"""
	Save `data` (dict/list/str) to JSON file.

	- If `filename` is provided, writes to that path (adds .json if missing).
	  If `filename` is relative with no directory, it is placed under `RESULTS_PATH`.
	- Else writes to `{out_dir or RESULTS_PATH}`/`{name or 'results'}_[timestamp].json`.
	- Creates parent directories as needed.
	- Returns the absolute path written.
	"""
	from pathlib import Path
	import json
	import time

	base_dir = Path(out_dir) if out_dir else Path(RESULTS_PATH)

	if filename:
		path = Path(filename)
		# If filename is relative and has no explicit parent, place under RESULTS_PATH
		if not path.is_absolute() and path.parent == Path('.'):
			path = base_dir / path
		if path.suffix == "":
			path = path.with_suffix(".json")
	else:
		base = name or "results"
		stamp = time.strftime("%Y%m%d_%H%M%S") if with_timestamp else None
		fname = f"{base}_{stamp}.json" if stamp else f"{base}.json"
		path = base_dir / fname

	# Ensure directory exists
	path.parent.mkdir(parents=True, exist_ok=True)

	# If data is not a string, dump as JSON with sensible defaults
	if isinstance(data, (dict, list)):
		with open(path, "w") as f:
			json.dump(data, f, indent=2, default=_json_default)
	else:
		# Write raw text
		with open(path, "w") as f:
			f.write(str(data))

	return str(path.resolve())


def load_file(filename: str, as_json: bool | None = None):
	"""
	Load content from a file. If `as_json` is True, parse JSON.
	If `as_json` is None, infer from file extension:
	- .json -> json.load
	- .jsonl/.ndjson -> parse each line as JSON and return a list
	- otherwise -> return raw text
	"""
	from pathlib import Path
	import json

	path = Path(filename)
	# If given a bare filename or non-existent relative path, try under RESULTS_PATH
	if not path.is_absolute() and not path.exists():
		candidate = Path(RESULTS_PATH) / path
		if candidate.exists():
			path = candidate
	if not path.exists():
		# raise FileNotFoundError(f"No such file: {path}")
		return None

	ext = path.suffix.lower()
	if as_json is True or (as_json is None and ext in {".json", ".jsonl", ".ndjson"}):
		text = path.read_text()
		if ext in {".jsonl", ".ndjson"}:
			return [json.loads(line) for line in text.splitlines() if line.strip()]
		return json.loads(text)
	else:
		return path.read_text()

def _make_prompt(text, max_chars=2000):
	prompt = "Summarize the following article:\n\n" + text[:max_chars]
	return prompt

def generate(model, dataset, max_chars=2000, workers=4, batch_size: int = 8):
	"""Generate watermarked texts and run detection.

	- Uses efficient batched generation when the watermark algorithm exposes a
	  `logits_processor` and `config` with `generation_model/tokenizer`.
	- Falls back to per-sample generation (optionally threaded) otherwise.
	"""
	prompts = [_make_prompt(item["text"], max_chars) for item in dataset]

	# Prefer batched path when available (handles KGW, UW, X-SIR, SWEET, etc.)
	can_batch = hasattr(model, "logits_processor") and hasattr(model, "config") \
		and hasattr(model.config, "generation_model") and hasattr(model.config, "generation_tokenizer")

	detections: list[dict] = []

	if can_batch and (batch_size or 0) > 1:
		gen_model = model.config.generation_model
		tokenizer = model.config.generation_tokenizer
		mdl_device = getattr(model.config, "device", None) or globals().get("device", "cpu")

		try:
			gen_model.eval()
		except Exception:
			pass
		try:
			torch.set_grad_enabled(False)
		except Exception:
			pass

		for start in tqdm(range(0, len(prompts), batch_size)):
			batch_prompts = prompts[start:start + batch_size]
			enc = tokenizer(
				batch_prompts,
				return_tensors="pt",
				padding=True,
				truncation=True,
				add_special_tokens=True,
			).to(mdl_device)

			outputs = gen_model.generate(
				**enc,
				logits_processor=LogitsProcessorList([model.logits_processor]),
				**getattr(model.config, "gen_kwargs", {}),
			)
			texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

			for text in texts:
				det = model.detect_watermark(text)
				det["watermarked_text"] = text
				det["true_label"] = True
				detections.append(det)
		return detections

	# Fallback: per-sample generation (optionally threaded)
	def work(p):
		watermarked = model.generate_watermarked_text(p)
		det = model.detect_watermark(watermarked)
		det["watermarked_text"] = watermarked
		det["true_label"] = True  # all generated texts are watermarked
		detections.append(det)

	if workers is None or workers <= 1:
		for p in tqdm(prompts):
			work(p)
	else:
		with ThreadPoolExecutor(max_workers=workers) as executor:
			futures = [executor.submit(work, p) for p in tqdm(prompts)]
			for future in tqdm(as_completed(futures), total=len(futures)):
				future.result()

	return detections

	
	
	

def detect(samples, model, workers=4):
	detections = []
	def work(p):
		detect = model.detect_watermark(p)
		detections.append(detect)
	
	with ThreadPoolExecutor(max_workers=workers) as executor:
		futures = [executor.submit(work, p) for p in tqdm(samples)]
		for future in tqdm(as_completed(futures), total=len(futures)):
			future.result()
	return detections


if __name__ == "__main__":
	pass
