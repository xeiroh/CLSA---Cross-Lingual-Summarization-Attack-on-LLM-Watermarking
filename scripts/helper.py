#!/usr/bin/env python3
import os
import json
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "baseline_all.json")

def parse_fname(fname: str):
    """
    Expect: baseline_{detections|metrics}_{ALGO}_{LANG}.json
    Returns: (kind, algo, lang) or None if not matching.
    """
    if not fname.endswith(".json"):
        return None
    parts = fname[:-5].split("_")  # drop .json
    if len(parts) < 4:
        return None
    if parts[0] != "baseline":
        return None
    kind = parts[1]  # "detections" or "metrics"
    # allow algo with internal dashes/words merged by underscores beyond the 3rd token
    # but per your convention it's exactly 4 tokens; keep strict to avoid surprises
    algo = parts[2]
    lang = parts[3]
    if kind not in ("detections", "metrics"):
        return None
    return kind, algo, lang

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("baseline_") and f.endswith(".json")]

    bucket = defaultdict(lambda: {"algorithm": None, "language": None, "metrics": None, "detections": None})

    for fname in files:
        parsed = parse_fname(fname)
        if not parsed:
            continue
        kind, algo, lang = parsed
        key = (algo, lang)
        path = os.path.join(RESULTS_DIR, fname)

        if bucket[key]["algorithm"] is None:
            bucket[key]["algorithm"] = algo
        if bucket[key]["language"] is None:
            bucket[key]["language"] = lang

        try:
            data = load_json(path)
        except Exception as e:
            print(f"Skip {fname}: {e}")
            continue

        if kind == "metrics":
            # metrics should be a dict like {"accuracy": ..., "precision": ...}
            bucket[key]["metrics"] = data
        else:
            # detections should be a list of dicts
            bucket[key]["detections"] = data

    combined = []
    for (algo, lang), entry in sorted(bucket.items(), key=lambda x: (x[0][0], x[0][1])):
        # only include pairs where at least one of metrics or detections exists
        if entry["metrics"] is None and entry["detections"] is None:
            continue
        combined.append(entry)

    with open(OUTPUT_PATH, "w") as out:
        json.dump(combined, out, indent=2, ensure_ascii=False)

    print(f"Wrote {OUTPUT_PATH} with {len(combined)} entries.")

if __name__ == "__main__":
    main()