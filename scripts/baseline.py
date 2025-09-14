from tools import load_model, load_data, generate, detect, save_file, load_file, split_and_generate
from evaluation import evaluate_detection
import numpy as np
import os, warnings, logging

os.environ.setdefault("PYTHONWARNINGS", "ignore")              # silence Python warnings
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_DATASETS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass
try:
    from datasets.utils.logging import set_verbosity_error as ds_set_verbosity_error
    ds_set_verbosity_error()
except Exception:
    pass

def baseline(language="english", algorithm="KGW", samples=100, max_tokens=256):
	model = load_model(max_tokens=max_tokens, algorithm=algorithm)
	dataset = load_data(language)

	# Efficient random sample using indices to avoid materializing the whole dataset

	# Interleave to form a mixed list (order not important for metrics)
	# Sequential generation for lower VRAM usage
	detections = split_and_generate(model, dataset, sample_size=samples, max_tokens=max_tokens)

	metrics = evaluate_detection(detections)
	return detections, metrics

if __name__ == "__main__":
	# Use "Unbiased" as the label; tools maps it to the correct implementation
	algorithms = ["XSIR", "KGW", "Unigram", "SIR"] # XSIR add later

	lang = "english"
	for algo in algorithms:
		try:
			detections, metrics = baseline(language=lang, algorithm=algo, samples=500, max_tokens=512)
			save_file(detections, f"baseline_detections_{algo}.json")
			save_file(metrics, f"baseline_metrics_{algo}.json")
			print("EVALUATION METRICS:", metrics)
		except FileNotFoundError as e:
			print(f"Skipping {algo} for {lang}: {e}")
