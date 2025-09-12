from tools import load_model, load_data, generate, detect, save_file, load_file, split_and_generate
from evaluation import evaluate_detection
import numpy as np

def _get_prompt(text, max_chars=2000, language="swahili"):
	return f"Translate this to {language} and summarize:\n\n" + text[:max_chars]

def CWRA(samples=100, max_tokens=256, max_chars=2000, language="english"):
	model = load_model(max_tokens=max_tokens)
	dataset = load_data(language)
	prompts = [_get_prompt(text=item["text"], max_chars=max_chars, language=language) for item in dataset]


	# Efficient random sample using indices to avoid materializing the whole dataset
	detections = split_and_generate(model, dataset, sample_size=samples, max_chars=max_chars)

	metrics = evaluate_detection(detections)
	return detections, metrics



if __name__ == "__main__":
	detections, metrics = CWRA(samples=1000, max_tokens=256, max_chars=None, language="swahili")
	save_file(detections, "cwra_detections.json")
	save_file(metrics, "cwra_metrics.json")
	print("EVALUATION METRICS:", metrics)