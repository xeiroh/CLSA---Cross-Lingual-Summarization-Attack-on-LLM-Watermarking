# minimal example sketch (adapt paths/models to your GPU/CPU)
import os
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
from regex import T
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
import numpy as np
from functools import lru_cache
from pipeline import translate

warnings.filterwarnings("ignore")  # transformers logging

# device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device, "cuda available:", torch.cuda.is_available())
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "results")

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

	# load watermark algorithm (e.g., KGW)
	wm = AutoWatermark.load(algorithm, transformers_config=transformers_config)
	return (tokenizer, model, wm)

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

def _make_prompt(text, max_chars=2000, language="english"):
	trans = translate("Summarize the following text:", language)
	return trans + "\n\n" + text[:max_chars] 

def split_dataset(dataset, sample_size=100):
	sample_size = min(sample_size, len(dataset))
	idx = np.random.choice(len(dataset), size=sample_size, replace=False)
	watermark_idx = set(idx[: sample_size // 2])
	non_watermark_idx = set(idx[sample_size // 2 :])
	watermark_samples = dataset.select(list(watermark_idx))
	non_watermark_samples = dataset.select(list(non_watermark_idx))
	return watermark_samples, non_watermark_samples

def split_and_generate(model_components, dataset, language="swahili", sample_size=100, max_chars=2000, workers=4, batch_size: int = 8):
	"""Split dataset into two halves and generate watermarked and unwatermarked outputs."""
	watermark_samples, non_watermark_samples = split_dataset(dataset, sample_size=sample_size)
	det_wm = generate(model_components, watermark_samples, watermark=True, max_chars=max_chars, workers=workers, batch_size=batch_size, language=language)
	det_uwm = generate(model_components, non_watermark_samples, watermark=False, max_chars=max_chars, workers=workers, batch_size=batch_size, language=language)
	return det_wm + det_uwm

def generate(model_components, dataset, watermark: bool, language='swahili',max_chars=2000, workers=4, batch_size: int = 8):
	"""Generate either watermarked or unwatermarked outputs for the dataset.

	Uses true batched generation if the wrapped watermark model exposes either
	`config.generation_model` or `config.model`. Falls back to per-sample threads otherwise.
	"""
	prompts = [_make_prompt(item["text"], max_chars, language) for item in dataset]
	n = len(prompts)
	results: list[dict | None] = [None] * n
	tokenizer, gen_model, wm = model_components
	model = wm  # for clarity

	cfg = getattr(wm, "config", object())
	logits_proc = getattr(wm, "logits_processor", None)
	can_batch = gen_model is not None and tokenizer is not None

	if can_batch:
		gen_model.eval()
		# Ensure model is on the right device (HF models use .to)
		try:
			gen_model.to(device)
		except Exception:
			pass
		max_new = int(getattr(cfg, "max_new_tokens", 128))

		for start in tqdm(range(0, n, batch_size), desc="Batched generation"):
			end = min(start + batch_size, n)
			batch_idx = list(range(start, end))
			texts = [prompts[i] for i in batch_idx]

			enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
			enc = {k: v.to(device) for k, v in enc.items()}

			use_lp = (watermark and logits_proc) and LogitsProcessorList([logits_proc]) or None

			with torch.inference_mode():
				out = gen_model.generate(
					**enc,
					max_new_tokens=max_new,
					do_sample=True,
					logits_processor=use_lp,
					use_cache=True,
					num_beams=1,
				)

			dec = tokenizer.batch_decode(out, skip_special_tokens=True)
			for i, text in zip(batch_idx, dec):
				det = model.detect_watermark(text)
				det["generated_text"] = text
				det["true_label"] = watermark
				results[i] = det

		return [r for r in results if r is not None]

	# Fallback threaded path
	def work(idx: int):
		p = prompts[idx]
		gen = (
			model.generate_watermarked_text(p)
			if watermark
			else model.generate_unwatermarked_text(p)
		)
		det = model.detect_watermark(gen)
		det["generated_text"] = gen
		det["true_label"] = watermark
		return idx, det

	with ThreadPoolExecutor(max_workers=workers) as ex:
		futures = [ex.submit(work, i) for i in range(n)]
		for fut in tqdm(as_completed(futures), total=len(futures), desc="Threaded generation"):
			idx, det = fut.result()
			results[idx] = det

	return [r for r in results if r is not None]
	
	
	

def detect(samples, model, column='generated_text', workers=4):
	detections = []
	column = column
	def work(p):
		detect = model.detect_watermark(p)
		detections.append(detect)
	
	with ThreadPoolExecutor(max_workers=workers) as executor:
		futures = [executor.submit(work, p[column]) for p in tqdm(samples)]
		for future in tqdm(as_completed(futures), total=len(futures)):
			future.result()
	return detections
