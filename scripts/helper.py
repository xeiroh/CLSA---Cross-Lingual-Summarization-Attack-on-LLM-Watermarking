#!/usr/bin/env python3
import os
import json
from collections import defaultdict
from typing import Optional, Tuple

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "baseline_all.json")

def parse_fname(fname: str) -> Optional[Tuple[str, str]]:
    """
    Expected patterns (no language suffix):
      baseline_metrics_{ALGO}.json
      baseline_detections_{ALGO}.json

    Returns:
      (kind, algo) or None if not matching.

    Notes:
      - {ALGO} may contain underscores; we parse from the left and take everything
        after the second token as the algo.
    """
    if not fname.endswith(".json") or not fname.startswith("baseline_"):
        return None
    if fname == "baseline_all.json":
        return None

    stem = fname[:-5]  # drop ".json"
    parts = stem.split("_")
    # minimal: baseline + (metrics|detections) + algo
    if len(parts) < 3 or parts[0] != "baseline":
        return None

    kind = parts[1]
    if kind not in ("metrics", "detections"):
        return None

    algo = "_".join(parts[2:])  # supports underscores in algo
    if not algo:
        return None
    return kind, algo

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    files = [f for f in os.listdir(RESULTS_DIR)
             if f.startswith("baseline_") and f.endswith(".json") and f != "baseline_all.json"]

    # Collate by algorithm
    bucket = defaultdict(lambda: {
        "algorithm": None,
        "metrics": None,     # dict
        "detections": []     # list of dicts
    })

    for fname in sorted(files):
        parsed = parse_fname(fname)
        if not parsed:
            continue

        kind, algo = parsed
        path = os.path.join(RESULTS_DIR, fname)

        try:
            data = load_json(path)
        except Exception as e:
            print(f"Skip {fname}: {e}")
            continue

        entry = bucket[algo]
        if entry["algorithm"] is None:
            entry["algorithm"] = algo

        if kind == "metrics":
            if isinstance(data, dict):
                entry["metrics"] = data
            else:
                print(f"Skip {fname}: metrics file is not a dict")
        else:  # detections
            if isinstance(data, list):
                entry["detections"].extend(data)
            else:
                print(f"Skip {fname}: detections file is not a list")

    # Emit only algorithms that have at least one of metrics or detections
    combined = []
    for algo, entry in sorted(bucket.items(), key=lambda x: x[0]):
        if entry["metrics"] is None and not entry["detections"]:
            continue
        # ensure key order: algorithm -> metrics -> detections
        combined.append({
            "algorithm": entry["algorithm"],
            "metrics": entry["metrics"],
            "detections": entry["detections"],
        })

    with open(OUTPUT_PATH, "w") as out:
        json.dump(combined, out, indent=2, ensure_ascii=False)

    print(f"Wrote {OUTPUT_PATH} with {len(combined)} algorithms.")

if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()