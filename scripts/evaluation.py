from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
import numpy as np

def evaluate_detection(detections):
	"""Evaluate detection scores and choose an operating threshold.

	Expected fields per detection dict:
	- 'true_label': ground-truth (bool: watermarked or not)
	- 'score': detector score (higher = more likely watermarked)
	- 'is_watermarked' may exist from the detector's internal default threshold,
	  but it is NOT used as ground-truth here.
	"""

	# Ground truth and scores
	y_true = [1 if bool(item.get('true_label', False)) else 0 for item in detections]
	y_scores = [float(item.get('score', 0.0)) for item in detections]

	has_both_classes = len(set(y_true)) > 1

	# Ranking metrics (defined only if both classes are present)
	roc_auc = roc_auc_score(y_true, y_scores) if has_both_classes else float('nan')
	avg_precision = average_precision_score(y_true, y_scores) if has_both_classes else float('nan')

	# Threshold selection via Youden's J (tpr - fpr) on ROC
	if has_both_classes:
		fpr, tpr, thresholds = roc_curve(y_true, y_scores)
		best_idx = int(np.argmax(tpr - fpr)) if len(thresholds) else None
		threshold = float(thresholds[best_idx]) if best_idx is not None else None
	else:
		fpr, tpr, thresholds = [], [], []
		threshold = None

	# Thresholded metrics (only meaningful if a threshold can be chosen)
	if threshold is not None:
		y_pred_thresh = [1 if s >= threshold else 0 for s in y_scores]
		acc = accuracy_score(y_true, y_pred_thresh)
		prec = precision_score(y_true, y_pred_thresh, zero_division=0)
		rec = recall_score(y_true, y_pred_thresh, zero_division=0)
		f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
	else:
		y_pred_thresh = None
		acc = prec = rec = f1 = float('nan')

	metrics = {
		'threshold': threshold,
		'accuracy': acc,
		# precision below is the thresholded precision; keep AP separately
		'precision': prec,
		'recall': rec,
		'f1_score': f1,
		'roc_auc': roc_auc,
		'average_precision': avg_precision,
	}

	return metrics
