from tools import load_model, load_data, generate, detect, save_file, load_file, split_and_generate
from evaluation import evaluate_detection
import numpy as np

def baseline(language="english",  algorithm="KGW", samples=100, max_tokens=256):
	model = load_model(max_tokens=max_tokens, algorithm=algorithm)
	dataset = load_data(language)

	# Efficient random sample using indices to avoid materializing the whole dataset

	# Interleave to form a mixed list (order not important for metrics)
	detections = split_and_generate(model, dataset, sample_size=samples)

	metrics = evaluate_detection(detections)
	return detections, metrics

if __name__ == "__main__":
	# Use "Unbiased" as the label; tools maps it to the correct implementation
	algorithms = ["KGW", "Unigram", "SIR", "XSIR"]
	languages = ["english", "swahili", "spanish", "amharic"]

	for lang in languages:
		for algo in algorithms:
			try:
				detections, metrics = baseline(language=lang, algorithm=algo, samples=100, max_tokens=256)
				save_file(detections, f"baseline_detections_{algo}_{lang}.json")
				save_file(metrics, f"baseline_metrics_{algo}_{lang}.json")
				print("EVALUATION METRICS:", metrics)
			except FileNotFoundError as e:
				print(f"Skipping {algo} for {lang}: {e}")
