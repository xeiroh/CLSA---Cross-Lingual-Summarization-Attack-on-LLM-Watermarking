from tools import load_model, load_data, generate, detect, save_file, load_file
from evaluation import evaluate_detection
import numpy as np

def baseline(language="english",  algorithm="KGW", samples=100, max_tokens=256):
	model = load_model(max_tokens=max_tokens, algorithm=algorithm)
	dataset = load_data(language)

	# Efficient random sample using indices to avoid materializing the whole dataset
	samples = min(samples, len(dataset))
	idx = np.random.choice(len(dataset), size=samples, replace=False)
	sample = dataset.select(idx)

	detections = generate(model, sample, max_chars=2000, workers=8)
	metrics = evaluate_detection(detections)
	return detections, metrics

if __name__ == "__main__":
	detections, metrics = baseline(language="english", algorithm="KGW", samples=100, max_tokens=256)
	save_file(detections, "results/baseline_detections_KGW.json")
	save_file(metrics, "results/baseline_metrics_KGW.json")

	print("EVALUATION METRICS:", metrics)

	detections, metrics = baseline(language="english", algorithm="UW", samples=100, max_tokens=256)
	save_file(detections, "results/baseline_detections_UW.json")
	save_file(metrics, "results/baseline_metrics_UW.json")	

	print("EVALUATION METRICS:", metrics)

	detections, metrics = baseline(language="english", algorithm="X-SIR", samples=100, max_tokens=256)
	save_file(detections, "results/baseline_detections_X-SIR.json")
	save_file(metrics, "results/baseline_metrics_X-SIR.json")
	print("EVALUATION METRICS:", metrics)