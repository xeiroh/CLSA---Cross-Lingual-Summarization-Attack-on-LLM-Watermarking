from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

def evaluate_detection(detections):
	y_true = [item['is_watermarked'] for item in detections]
	y_pred = [item['true_label'] for item in detections]
	y_scores = [item['score'] for item in detections]

	accuracy = accuracy_score(y_true, y_pred)
	precision = precision_score(y_true, y_pred)
	recall = recall_score(y_true, y_pred)
	f1 = f1_score(y_true, y_pred)
	roc_auc = roc_auc_score(y_true, y_scores)

	metrics = {
		'accuracy': accuracy,
		'precision': precision,
		'recall': recall,
		'f1_score': f1,
		'roc_auc': roc_auc
	}

	return metrics
