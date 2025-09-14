#!/usr/bin/env python3
import select
from tools import load_model, load_data, generate, detect, save_file, load_file, split_and_generate
from evaluation import evaluate_detection, evaluate_from_file
import pandas as pd
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

def print_results(name, find_by_algorithm=True, select_mode='youden', target_fpr=0.01):
	results = evaluate_from_file(name, find_by_algorithm=find_by_algorithm, select_mode=select_mode, target_fpr=target_fpr)
	print(f"Results for {name}:")
	for key, value in results.items():
		print(f"{key}: {value}")

if __name__ == "__main__":
	filename = 'baseline_detections_KGW_english.json'
	if len(sys.argv) > 2:
		if sys.argv[1] == 'algo':
			print_results(sys.argv[2])
		else:
			print_results(sys.argv[2], find_by_algorithm=False, select_mode='youden', target_fpr=0.01)
	else:
		print_results("XSIR", select_mode='target_fpr', target_fpr=0.01)