from tools import load_model, load_data, generate, detect, save_file, load_file, split_and_generate
from evaluation import DATA_PATH, evaluate_detection, evaluate_from_file
from pipeline import xlsum
from paraphrase import paraphrase_df, detect_on_paraphrase

import numpy as np
import os
import pandas as pd
from main import print_results

def clsa(detections, algorithm, language):
	# XLSum + backtranslation
	xlsum_generations = xlsum(detections, language=language)
	# Paraphrase base generations into English (monolingual paraphrase)
	detections = paraphrase_df(detections, source_col="generated_text", out_col="paraphrase")

	# Load watermark model and run detections
	tokenizer, gen_model, wm_model = load_model(algorithm=algorithm)

	# Detection on XLSum summaries
	clsa_detections = detect(xlsum_generations, wm_model, "xlsum")
	clsa_detections["true_label"] = detections["true_label"].astype(int).values
	clsa_evaluation = evaluate_detection(clsa_detections)
	detections["xlsum_score"] = clsa_detections["score"]
	detections["xlsum_watermarked"] = clsa_detections["is_watermarked"]

	# Detection on backtranslated English summaries
	back_detections = detect(xlsum_generations, wm_model, "backtranslation")
	back_detections["true_label"] = detections["true_label"].astype(int).values
	back_evaluation = evaluate_detection(back_detections)
	detections["backtranslate_score"] = back_detections["score"]
	detections["backtranslate_watermarked"] = back_detections["is_watermarked"]

	# Detection on paraphrased base generations (use FLAN-T5 tokenizer for detection)
	para_detections = detect_on_paraphrase(detections, algorithm, column="paraphrase")
	para_detections["true_label"] = detections["true_label"].astype(int).values
	detections["paraphrase_score"] = para_detections["score"]
	detections["paraphrase_watermarked"] = para_detections["is_watermarked"]

	return detections, clsa_evaluation, back_evaluation #, para_evaluation






if __name__ == "__main__":

	languages = ["spanish", "chinese", "hindi", "swahili", "amharic"]
	algorithms = ["Unigram", "KGW", "XSIR", "SIR"]

	# languages = ["spanish"]
	# algorithms = ["Unigram"]

	for lang in languages:
		for algo in algorithms:
			print(f"Processing {lang} with {algo}")
			# file_path = os.path.join(DATA_PATH, f"{algo}.json")
			file_path = os.path.join(DATA_PATH, f"{algo}.json")

			#detections, clsa_evaluation, back_evaluation, para_evaluation = clsa(load_file(file_path), algo, lang)
			
			detections, clsa_evaluation, back_evaluation = clsa(load_file(file_path), algo, lang)

			detections.to_json(os.path.join(DATA_PATH, f"{algo}_{lang}_clsa.json"))

			save_file(clsa_evaluation, os.path.join(DATA_PATH, f"{algo}_{lang}_clsa_metrics.json"))
			save_file(back_evaluation, os.path.join(DATA_PATH, f"{algo}_{lang}_back_metrics.json"))
			# Save paraphrase-specific outputs

			# save_file(para_evaluation, os.path.join(DATA_PATH, f"{algo}_{lang}_paraphrase_metrics.json"))
			
			# Save a detection file variant for paraphrase for convenience 
			detections.to_json(os.path.join(DATA_PATH, f"{algo}_{lang}_paraphrase.json"))
			# print(detections.head())
			print(f"EVALUATION CLSA for {algo}:{lang}:", clsa_evaluation)
			print(f"EVALUATION BACKTRANSLATE for {algo}:{lang}:", back_evaluation)
			
			# print(f"EVALUATION PARAPHRASE for {algo}:{lang}:", para_evaluation)

	# file_path= os.path.join(DATA_PATH, filename)
	# detections, evaluation = clsa(load_file(file_path), "Unigram")
	# detections.to_json(os.path.join(DATA_PATH, test_file))
	# save_file(evaluation, os.path.join(DATA_PATH, test_metrics))
	# print(detections.head())
	# print("EVALUATION METRICS:", evaluation)
