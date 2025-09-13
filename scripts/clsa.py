from tools import load_model, load_data, generate, detect, save_file, load_file, split_and_generate
from evaluation import evaluate_detection
from pipeline import translate, summarize
import numpy as np


def CLSA(samples=100, max_tokens=256, max_chars=2000, language="amharic"):
	model = load_model(max_tokens=max_tokens)
	dataset = load_data("english")

	# Efficient random sample using indices to avoid materializing the whole dataset
	detections = split_and_generate(model, dataset, language=language, sample_size=samples, max_chars=max_chars)
	for detection in detections:
		detection["translated_text"] = translate(detection["generated_text"], target_lang=language, source_lang="english")[0]
		detection["summarized_text"] = summarize(detection["translated_text"], max_length=250)[0]
	# Re-detect on the translated/summarized text
	base_metrics = evaluate_detection(detections)
	translate_detections = detect(detections, model, column="translated_text")
	summary_detections = detect(detections, model, column="summarized_text")
	translate_metrics = evaluate_detection(translate_detections)
	summary_metrics = evaluate_detection(summary_detections)
	return (base_metrics, detections), (translate_metrics, translate_detections), (summary_metrics, summary_detections)



if __name__ == "__main__":
	(detections, metrics), (translate_metrics, translate_detections), (summary_metrics, summary_detections) = CLSA(samples=200, max_tokens=256, max_chars=2000, language="amharic")
	print(metrics)
	save_file(detections, "clsa_detections.json", as_json=True)
	save_file(metrics, "clsa_metrics.json", as_json=True)
	