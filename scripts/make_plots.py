#!/usr/bin/env python3
import json, re, sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 160, "font.size": 10})

DATA_DIR = Path("../data")
OUT_DIR = Path("./plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- 1) load and normalize ----
rows = []
pat = re.compile(r"^(?P<algo>[^_]+)_(?P<lang>[^_]+)_(?P<attack>[^_]+)_metrics\.json$")

key_map = {
    "auroc@test": "AUROC",
    "auprc@test": "AUPRC",
    "accuracy@thr": "ACC@thr",
    "precision@thr": "Prec@thr",
    "recall@thr": "Rec@thr",
    "f1@thr": "F1@thr",
    "eer@test": "EER",
    "tpr@fpr=0.010@test": "TPR@FPR=0.01",
    "threshold": "Thr",
}
def normalize(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            continue
        out[key_map.get(k, k)] = v
    return out

for fp in DATA_DIR.glob("*_metrics.json"):
    m = pat.match(fp.name)
    if not m:
        # tolerate names like baseline_* if any show up later
        continue
    meta = m.groupdict()
    with open(fp, "r", encoding="utf-8") as f:
        raw = json.load(f)
    flat = normalize(raw)
    flat.update(meta)
    rows.append(flat)

if not rows:
    print(f"No matching *_metrics.json in {DATA_DIR.resolve()}", file=sys.stderr)
    sys.exit(1)

df = pd.DataFrame(rows)
# enforce categorical ordering for readability
df["attack"] = pd.Categorical(df["attack"], categories=sorted(df["attack"].unique()), ordered=True)
df["algo"]   = pd.Categorical(df["algo"],   categories=sorted(df["algo"].unique()),   ordered=True)
df["lang"]   = pd.Categorical(df["lang"],   categories=sorted(df["lang"].unique()),   ordered=True)

# save the aggregated CSV for convenience
df.sort_values(["attack","algo","lang"]).to_csv(OUT_DIR/"all_metrics.csv", index=False)

# ---- 2) plotting helpers ----
def bar_group(dfm, metric, title, fname, by="attack"):
    """Grouped bar chart: x=lang, group=algo, one panel per attack (or vice versa)."""
    # panel facet on 'by'
    panels = list(dfm[by].cat.categories if hasattr(dfm[by], "cat") else sorted(dfm[by].unique()))
    n = len(panels)
    fig_h = 3.2 if n == 1 else max(3.2, 2.6*n)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, fig_h), constrained_layout=True, sharex=True)
    if n == 1:
        axes = [axes]

    for ax, val in zip(axes, panels):
        sub = dfm[dfm[by] == val]
        # pivot to wide for grouped bars
        pivot = sub.pivot_table(index="lang", columns="algo", values=metric)
        langs = list(pivot.index)
        algos = list(pivot.columns)
        x = range(len(langs))
        width = 0.8 / max(1, len(algos))

        for i, algo in enumerate(algos):
            y = pivot[algo].values
            ax.bar([xx + i*width - 0.4 + width/2 for xx in x], y, width=width, label=algo)

        ax.set_ylabel(metric)
        ax.set_title(f"{title} • {by}={val}")
        ax.set_ylim(0, 1.0 if metric in ("AUROC","AUPRC","ACC@thr","Prec@thr","Rec@thr","F1@thr","TPR@FPR=0.01") else ax.get_ylim()[1])
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xticks(range(len(langs)), [str(l) for l in langs], rotation=0)
    axes[0].legend(ncols=min(4, len(algos)), fontsize=9, frameon=False)
    fig.suptitle(title, y=1.02, fontsize=12)
    fig.savefig(OUT_DIR/fname, bbox_inches="tight")
    plt.close(fig)

def table_image(dfm, metric, fname):
    """Save a compact table image of metric by (algo,lang,attack)."""
    show = dfm.pivot_table(index=["algo","lang"], columns="attack", values=metric)
    fig, ax = plt.subplots(figsize=(min(12, 2+1.1*len(show.columns)), 0.4+0.35*len(show)))
    ax.axis("off")
    tbl = ax.table(cellText=show.round(3).values,
                   rowLabels=[f"{i[0]} | {i[1]}" for i in show.index],
                   colLabels=show.columns.tolist(),
                   loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.2)
    fig.savefig(OUT_DIR/fname, bbox_inches="tight")
    plt.close(fig)

# ---- 3) make plots ----
metrics_to_plot = [
    ("AUROC", "AUROC by language and algorithm", "auroc_bars.png"),
    ("AUPRC", "AUPRC by language and algorithm", "auprc_bars.png"),
    ("ACC@thr", "Accuracy@thr by language and algorithm", "acc_bars.png"),
    ("F1@thr", "F1@thr by language and algorithm", "f1_bars.png"),
    ("EER", "Equal Error Rate (lower is better)", "eer_bars.png"),
]

for metric, title, fname in metrics_to_plot:
    if metric not in df.columns:
        continue
    bar_group(df, metric, title, fname, by="attack")
    table_image(df, metric, f"{Path(fname).stem}_table.png")

# ---- 4) quick textual summary to stdout ----
def best_by(group_cols, metric, higher_is_better=True):
    sign = -1 if higher_is_better else 1
    return (df.assign(_val = sign*df[metric])
              .sort_values(group_cols+["_val"])
              .groupby(group_cols, as_index=False)
              .first()
              .drop(columns=["_val"]))

print("\nTop AUROC per (lang, attack):")
print(best_by(["lang","attack"], "AUROC", higher_is_better=True)
      .sort_values(["attack","lang"])[["lang","attack","algo","AUROC"]]
      .to_string(index=False))

print("\nWorst EER per (lang, attack)  (i.e., most effective attack if EER high):")
print(best_by(["lang","attack"], "EER", higher_is_better=True)
      .sort_values(["attack","lang"])[["lang","attack","algo","EER"]]
      .to_string(index=False))

print(f"\nSaved plots and CSV to: {OUT_DIR.resolve()}")