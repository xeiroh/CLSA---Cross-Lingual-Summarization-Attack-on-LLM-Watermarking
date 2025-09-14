import os
import numpy as np
import pandas as pd
DATA_PATH=os.path.join(os.path.dirname(__file__), "..", "data")

import numpy as np
from sklearn.metrics import (
	roc_auc_score, average_precision_score, roc_curve,
	precision_recall_fscore_support, accuracy_score
)


def evaluate_detection(
	test_df: pd.DataFrame,
	val_df: pd.DataFrame | None = None,
	*,
	select_mode: str = "youden",   # "youden" or "target_fpr"
	target_fpr: float = 0.01,
	flip_if_needed: bool = False
) -> dict:
	"""
	Evaluate watermark detection using pandas DataFrames only.

	Required columns in each DataFrame:
	  - 'true_label' : {0,1} or bool  (1 = watermarked, 0 = unwatermarked)
	  - 'score'    : float (higher => more watermarked)
	Optional:
	  - 'is_watermarked' : detector's internal binary output (ignored for metrics)

	Behavior:
	  - If `val_df` is provided, selects the operating threshold on val, then reports metrics on test.
	  - If `val_df` is None, selects & evaluates on test (optimistic; for debugging).
	  - If AUROC < 0.5 on the split used for selection and `flip_if_needed=True`, flips score sign.

	Returns a dict with AUROC, AUPRC, EER@test, TPR@FPR=target_fpr@test,
	and Accuracy/Precision/Recall/F1 at the chosen threshold.
	"""
	
	req_cols = {"true_label", "score"}
	for name, df in (("test_df", test_df), ("val_df", val_df)):
		if df is None:
			continue
		missing = req_cols - set(df.columns)
		if missing:
			raise ValueError(f"{name} missing required columns: {sorted(missing)}")

	def _prep(df: pd.DataFrame):
		# Coerce and clean
		y_true = df["true_label"].astype(int).to_numpy()
		score = pd.to_numeric(df["score"], errors="coerce").to_numpy(dtype=float)
		m = np.isfinite(score)
		return y_true[m], score[m]

	def _pick_threshold_from_roc(y_true, score, mode="youden", target_fpr=0.01):
		fpr, tpr, thr = roc_curve(y_true, score)
		fnr = 1 - tpr
		# EER on the selection split (useful to log)
		eer_idx = int(np.argmin(np.abs(fpr - fnr)))
		eer_val = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
		thr_eer = float(thr[eer_idx])

		if mode == "youden":
			best_idx = int(np.argmax(tpr - fpr))
			return float(thr[best_idx]), {"eer_sel": eer_val, "thr_eer_sel": thr_eer}
		elif mode == "target_fpr":
			idx = int(np.argmin(np.abs(fpr - target_fpr)))
			return float(thr[idx]), {
				"eer_sel": eer_val,
				"thr_eer_sel": thr_eer,
				"tpr_at_target_fpr_sel": float(tpr[idx]),
				"thr_at_target_fpr_sel": float(thr[idx]),
				"target_fpr_sel": float(fpr[idx]),
			}
		else:
			raise ValueError("select_mode must be 'youden' or 'target_fpr'")

	# --- Load & clean splits ---
	y_true_test, score_test = _prep(test_df)
	if len(y_true_test) == 0:
		raise ValueError("No valid test rows after filtering non-finite score.")
	if len(set(y_true_test.tolist())) < 2:
		raise ValueError("Test set must contain both classes (true_label 0 and 1).")

	if val_df is not None:
		y_true_val, score_val = _prep(val_df)
		if len(y_true_val) == 0:
			raise ValueError("No valid validation rows after filtering non-finite score.")
		if len(set(y_true_val.tolist())) < 2:
			raise ValueError("Validation set must contain both classes.")

		# Direction check on validation
		auroc_val = roc_auc_score(y_true_val, score_val)
		flip = (auroc_val < 0.5) if flip_if_needed else False
		if flip:
			score_val = -score_val
			score_test = -score_test

		thr, sel_extra = _pick_threshold_from_roc(
			y_true_val, score_val, mode=select_mode, target_fpr=target_fpr
		)
		threshold_source = "validation"
	else:
		# Single-set (optimistic)
		auroc_tmp = roc_auc_score(y_true_test, score_test)
		flip = (auroc_tmp < 0.5) if flip_if_needed else False
		if flip:
			score_test = -score_test
		thr, sel_extra = _pick_threshold_from_roc(
			y_true_test, score_test, mode=select_mode, target_fpr=target_fpr
		)
		threshold_source = "test (optimistic)"

	# --- Threshold-free metrics on TEST ---
	auroc = roc_auc_score(y_true_test, score_test)
	ap = average_precision_score(y_true_test, score_test)

	# --- Thresholded metrics on TEST ---
	y_pred = (score_test >= thr).astype(int)
	acc = float(accuracy_score(y_true_test, y_pred))
	prec, rec, f1, _ = precision_recall_fscore_support(
		y_true_test, y_pred, average="binary", zero_division=0
	)

	# EER and TPR@targetFPR on TEST
	fpr_t, tpr_t, thr_t = roc_curve(y_true_test, score_test)
	fnr_t = 1 - tpr_t
	eer_idx_t = int(np.argmin(np.abs(fpr_t - fnr_t)))
	eer_t = float((fpr_t[eer_idx_t] + fnr_t[eer_idx_t]) / 2.0)
	idx_t = int(np.argmin(np.abs(fpr_t - target_fpr)))
	tpr_at_target = float(tpr_t[idx_t])

	if flip:
		return {
		"n_test": int(len(y_true_test)),
		"n_pos_test": int(int(y_true_test.sum())),
		"n_neg_test": int(len(y_true_test) - int(y_true_test.sum())),
		"n_val": int(len(y_true_val)) if val_df is not None else None,
		"n_pos_val": int(int(y_true_val.sum())) if val_df is not None else None,
		"n_neg_val": int(len(y_true_val) - int(y_true_val.sum())) if val_df is not None else None,
		"flip_score": bool(flip),
		"threshold": float(thr),
		"threshold_source": threshold_source,
		"score_flipped": bool(flip),
		"auroc@test": float(auroc),
		"auprc@test": float(ap),
		"accuracy@thr": acc,
		"precision@thr": float(prec),
		"recall@thr": float(rec),
		"f1@thr": float(f1),
		"eer@test": eer_t,
		f"tpr@fpr={target_fpr:.3f}@test": tpr_at_target,
		"selection_details": sel_extra,
		}
	return {
		"sample_sizes": {"n_test":int(len(y_true_test)),
		"n_pos_test": int(int(y_true_test.sum())),
		"n_neg_test": int(len(y_true_test) - int(y_true_test.sum())),
		"n_val": int(len(y_true_val)) if val_df is not None else None,
		"n_pos_val": int(int(y_true_val.sum())) if val_df is not None else None,
		"n_neg_val": int(len(y_true_val) - int(y_true_val.sum())) if val_df is not None else None},
		"threshold": float(thr),
		"threshold_source": threshold_source,
		"auroc@test": float(auroc),
		"auprc@test": float(ap),
		"accuracy@thr": acc,
		"precision@thr": float(prec),
		"recall@thr": float(rec),
		"f1@thr": float(f1),
		"eer@test": eer_t,
		f"tpr@fpr={target_fpr:.3f}@test": tpr_at_target,
		"selection_details": sel_extra,
		}

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
	# Keep only relevant columns
	needed_cols = {"true_label", "score"}
	missing = needed_cols - set(df.columns)
	if missing:
		raise ValueError(f"Input DataFrame missing required columns: {sorted(missing)}")
	df["true_label"] = df["true_label"].dropna().astype(int)
	df["score"] = df["score"].dropna().astype(float)
	return df

def evaluate_from_file(name, find_by_algorithm=True, select_mode='youden', target_fpr=0.01):
	if find_by_algorithm:
		filename = os.path.join(DATA_PATH, f'{name}_detections.json')
	else:
		filename = os.path.join(DATA_PATH, name)
		
	data = pd.read_json(filename)
	n = len(data)
	val_data = pd.concat([data.iloc[:n//5], data.iloc[4*n//5:]], ignore_index=True)
	test_data = data.iloc[n//5:4*n//5].reset_index(drop=True)
	val_data, test_data = clean_data(val_data), clean_data(test_data)
	# print(val_data)
	# print(test_data)
	results = evaluate_detection(test_df=test_data, val_df=val_data, select_mode=select_mode, target_fpr=target_fpr)
	return results

	