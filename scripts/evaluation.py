from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import numpy as np

def evaluate_detection(detections):
	y_true = [item['is_watermarked'] for item in detections]
	y_pred = [item['true_label'] for item in detections]
	y_scores = [item['score'] for item in detections]

	roc_auc = roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else float('nan')
	precision = average_precision_score(y_true, y_scores) if len(set(y_true)) > 1 else float('nan')

	fpr, tpr, thresholds = roc_curve(y_true, y_scores) if len(set(y_true)) > 1 else ([], [], [])
	best = np.argmax(tpr - fpr) if len(fpr) > 0 else None
	threshold = thresholds[best] if best is not None else None
	y_pred_thresh = [1 if score >= threshold else 0 for score in y_scores]

	metrics = {
		'accuracy': accuracy_score(y_true, y_pred_thresh),
		'precision': precision,
		'recall': recall_score(y_true, y_pred_thresh),
		'f1_score': f1_score(y_true, y_pred_thresh),
		'roc_auc': roc_auc
	}

	return metrics
