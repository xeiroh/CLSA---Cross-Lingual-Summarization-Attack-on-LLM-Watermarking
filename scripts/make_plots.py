#!/usr/bin/env python3
"""Generate comparison tables and plots for detection metrics."""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 160, "font.size": 10})

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
OUT_DIR = BASE_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMPARISON_FILE = DATA_DIR / "comparison.json"
CLSA_FILE = DATA_DIR / "clsa.json"

SCENARIO_LABELS = {
    "baseline": "Baseline",
    "paraphrase": "Paraphrase",
    "cwra": "CWRA",
    "clsa": "CLSA",
    "backtranslation": "CLSA+Backtranslation",
}
SCENARIO_ORDER_KEYS = ["baseline", "paraphrase", "cwra", "clsa", "backtranslation"]
FOCUS_SCENARIOS = [SCENARIO_LABELS[key] for key in ["paraphrase", "cwra", "clsa", "backtranslation"]]
SUMMARY_SCENARIO_KEYS = ["baseline", "paraphrase", "clsa"]
SUMMARY_METRICS = ["AUROC", "Precision@thr", "Recall@thr", "F1@thr"]

METRIC_FIELDS: List[tuple[str, str]] = [
    ("auroc@test", "AUROC"),
    ("auprc@test", "AUPRC"),
    ("accuracy@thr", "ACC@thr"),
    ("precision@thr", "Precision@thr"),
    ("recall@thr", "Recall@thr"),
    ("f1@thr", "F1@thr"),
    ("eer@test", "EER"),
    ("tpr@fpr=0.010@test", "TPR@FPR=0.01"),
    ("threshold", "Threshold"),
]
SELECTION_RENAME = {
    "eer_sel": "Selection EER",
    "thr_eer_sel": "Selection thr@EER",
    "tpr_at_target_fpr_sel": "Selection TPR@targetFPR",
    "thr_at_target_fpr_sel": "Selection thr@targetFPR",
    "target_fpr_sel": "Selection target FPR",
}


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_language(lang: Optional[str]) -> str:
    if lang is None:
        return "Overall"
    text = str(lang).strip()
    if not text or text.lower() == "none":
        return "Overall"
    if text.lower() == "overall":
        return "Overall"
    return text.replace("_", " ").title()


def flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for src, dst in METRIC_FIELDS:
        if src in metrics:
            flat[dst] = metrics[src]
    if "threshold_source" in metrics:
        flat["Threshold source"] = metrics["threshold_source"]
    if "sample_sizes" in metrics and isinstance(metrics["sample_sizes"], dict):
        for key, value in metrics["sample_sizes"].items():
            flat[f"Samples {key}"] = value
    if "selection_details" in metrics and isinstance(metrics["selection_details"], dict):
        for key, value in metrics["selection_details"].items():
            label = SELECTION_RENAME.get(key, f"Selection {key}")
            flat[label] = value
    for bool_key in ("flip_score", "score_flipped"):
        if bool_key in metrics:
            flat[bool_key] = metrics[bool_key]
    return flat


def build_row(
    algo: str,
    language: Optional[str],
    scenario_key: str,
    metrics: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    scenario = SCENARIO_LABELS.get(scenario_key)
    if scenario is None:
        raise KeyError(f"Unexpected scenario key: {scenario_key}")

    row: Dict[str, Any] = {
      "Algorithm": algo,
      "Language": normalize_language(language),
      "Scenario": scenario,
      "Source": scenario_key,
    }

    if not metrics:
        row["Error"] = "Missing metrics"
        return row
    if "error" in metrics:
        row["Error"] = metrics["error"]
        return row

    row.update(flatten_metrics(metrics))
    return row


def append_comparison_rows(rows: List[Dict[str, Any]], data: Dict[str, Any]) -> None:
    for algo, sections in data.items():
        if "baseline" in sections:
            rows.append(build_row(algo, None, "baseline", sections["baseline"]))
        if "paraphrase" in sections and isinstance(sections["paraphrase"], dict):
            for lang, metrics in sections["paraphrase"].items():
                rows.append(build_row(algo, lang, "paraphrase", metrics))
        if "cwra" in sections and isinstance(sections["cwra"], dict):
            for lang, metrics in sections["cwra"].items():
                rows.append(build_row(algo, lang, "cwra", metrics))


def append_clsa_rows(rows: List[Dict[str, Any]], data: Dict[str, Any]) -> None:
    for algo, languages in data.items():
        for lang, metrics in languages.items():
            if "clsa" in metrics:
                rows.append(build_row(algo, lang, "clsa", metrics["clsa"]))
            if "backtranslation" in metrics:
                rows.append(build_row(algo, lang, "backtranslation", metrics["backtranslation"]))


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def prepare_dataframe() -> pd.DataFrame:
    comparison_data = load_json(COMPARISON_FILE)
    clsa_data = load_json(CLSA_FILE)

    rows: List[Dict[str, Any]] = []
    append_comparison_rows(rows, comparison_data)
    append_clsa_rows(rows, clsa_data)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No metrics collected from input files.")

    algo_order = sorted(df["Algorithm"].dropna().unique())
    df["Algorithm"] = pd.Categorical(df["Algorithm"], categories=algo_order, ordered=True)

    languages = sorted(df["Language"].dropna().unique(), key=lambda x: (0, "") if x == "Overall" else (1, x))
    df["Language"] = pd.Categorical(df["Language"], categories=languages, ordered=True)

    scenario_order = [SCENARIO_LABELS[key] for key in SCENARIO_ORDER_KEYS if SCENARIO_LABELS[key] in df["Scenario"].unique()]
    df["Scenario"] = pd.Categorical(df["Scenario"], categories=scenario_order, ordered=True)

    return df


def save_master_csv(df: pd.DataFrame) -> Path:
    csv_path = OUT_DIR / "detection_metrics_full.csv"
    printable = df.copy()
    for col in ("Algorithm", "Language", "Scenario"):
        printable[col] = printable[col].astype(str)
    printable.sort_values(["Algorithm", "Language", "Scenario"]).to_csv(csv_path, index=False)
    return csv_path


def plot_metric_bars(df: pd.DataFrame, metric: str) -> Optional[Path]:
    if metric not in df.columns:
        return None
    data = df[df["Scenario"].isin(FOCUS_SCENARIOS)].dropna(subset=[metric])
    if data.empty:
        return None

    algorithms = [algo for algo in df["Algorithm"].cat.categories if algo in data["Algorithm"].unique()]
    if not algorithms:
        return None

    fig_height = max(3.0, 2.6 * len(algorithms))
    fig, axes = plt.subplots(len(algorithms), 1, figsize=(10, fig_height), sharex=True, constrained_layout=True)
    if len(algorithms) == 1:
        axes = [axes]

    width = 0.8 / max(1, len(FOCUS_SCENARIOS))
    base_positions = np.arange(len(df["Language"].cat.categories))
    lang_labels = [str(l) for l in df["Language"].cat.categories]

    for idx, (ax, algo) in enumerate(zip(axes, algorithms)):
        subset = data[data["Algorithm"] == algo]
        if subset.empty:
            ax.axis("off")
            continue
        pivot = subset.pivot_table(index="Language", columns="Scenario", values=metric, observed=False)
        pivot = pivot.reindex(index=df["Language"].cat.categories)
        positions = np.arange(len(pivot.index))

        for s_idx, scenario in enumerate(FOCUS_SCENARIOS):
            if scenario not in pivot.columns:
                continue
            values = pivot[scenario].to_numpy(dtype=float)
            mask = np.isfinite(values)
            pos = positions[mask] + (-0.4 + width / 2.0) + s_idx * width
            label = scenario if idx == 0 else None
            ax.bar(pos, values[mask], width=width, label=label)

        ax.set_ylabel(metric)
        ax.set_title(f"{algo}")
        if metric in {"AUROC", "AUPRC", "ACC@thr", "Precision@thr", "Recall@thr", "F1@thr", "EER", "TPR@FPR=0.01"}:
            ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xticks(base_positions, lang_labels, rotation=0)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, frameon=False, ncols=len(labels))

    fig.suptitle(f"{metric} comparison across attacks", y=1.02)
    out_path = OUT_DIR / f"{slugify(metric)}_bars.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_metric_table(df: pd.DataFrame, metric: str) -> Optional[Path]:
    if metric not in df.columns:
        return None
    subset = df[df["Scenario"].isin(FOCUS_SCENARIOS)]
    table = subset.pivot_table(index=["Algorithm", "Language"], columns="Scenario", values=metric, observed=False)
    if table.empty:
        return None
    table = table.sort_index()

    fig_width = min(12, 2 + 1.3 * len(table.columns))
    fig_height = max(2.5, 0.6 * len(table))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    tbl = ax.table(
        cellText=np.round(table.values, 3),
        rowLabels=[f"{idx[0]} | {idx[1]}" for idx in table.index],
        colLabels=list(table.columns),
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.2)

    out_path = OUT_DIR / f"{slugify(metric)}_table.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_summary_metrics(df: pd.DataFrame) -> Optional[Path]:
    scenarios = [SCENARIO_LABELS[key] for key in SUMMARY_SCENARIO_KEYS if SCENARIO_LABELS[key] in df["Scenario"].cat.categories]
    metrics = [metric for metric in SUMMARY_METRICS if metric in df.columns]
    if not scenarios or not metrics:
        return None

    subset = df[df["Scenario"].isin(scenarios)]
    subset = subset.dropna(subset=metrics, how="all")
    if subset.empty:
        return None

    grouped = subset.groupby(["Algorithm", "Scenario"], observed=False)[metrics].mean()
    grouped = grouped.reset_index().set_index(["Algorithm", "Scenario"])

    algorithms = [algo for algo in df["Algorithm"].cat.categories if algo in grouped.index.get_level_values("Algorithm").unique()]
    if not algorithms:
        return None

    entries: List[str] = []
    values: Dict[str, List[float]] = {algo: [] for algo in algorithms}

    for scenario in scenarios:
        for metric in metrics:
            entries.append(f"{scenario}\n{metric}")
            for algo in algorithms:
                val = np.nan
                key = (algo, scenario)
                if key in grouped.index:
                    val = grouped.loc[key, metric]
                values[algo].append(val)

    x_positions = np.arange(len(entries))
    bar_width = min(0.2, 0.8 / max(1, len(algorithms)))
    offsets = [bar_width * (idx - (len(algorithms) - 1) / 2.0) for idx in range(len(algorithms))]

    fig_width = max(10.0, 0.9 * len(entries))
    fig, ax = plt.subplots(figsize=(fig_width, 6.0))
    for algo, offset in zip(algorithms, offsets):
        heights = np.asarray(values[algo], dtype=float)
        ax.bar(x_positions + offset, heights, width=bar_width, label=algo)

    ax.set_xticks(x_positions, entries, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Baseline vs Paraphrase vs CLSA detection metrics")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, ncols=len(algorithms))

    out_path = OUT_DIR / "summary_metrics_bars.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def summarize_best(df: pd.DataFrame, metric: str, higher_is_better: bool = True) -> pd.DataFrame:
    if metric not in df.columns:
        return pd.DataFrame()
    subset = df[df["Scenario"].isin(FOCUS_SCENARIOS)].dropna(subset=[metric])
    if subset.empty:
        return pd.DataFrame()
    sign = 1 if higher_is_better else -1
    ordered = subset.assign(_score=sign * subset[metric])
    best = ordered.sort_values(["Algorithm", "Language", "_score"], ascending=[True, True, False])
    best = best.groupby(["Algorithm", "Language"], as_index=False, observed=False).first()
    return best[["Algorithm", "Language", "Scenario", metric]]


def main() -> None:
    df = prepare_dataframe()
    csv_path = save_master_csv(df)

    metrics_for_plots = [label for _, label in METRIC_FIELDS if label in df.columns and label != "Threshold"]
    generated_plots: List[Path] = []
    generated_tables: List[Path] = []

    for metric in metrics_for_plots:
        path = plot_metric_bars(df, metric)
        if path:
            generated_plots.append(path)
        table_path = save_metric_table(df, metric)
        if table_path:
            generated_tables.append(table_path)

    summary_path = plot_summary_metrics(df)
    if summary_path:
        generated_plots.append(summary_path)

    best_auroc = summarize_best(df, "AUROC", higher_is_better=True)
    best_eer = summarize_best(df, "EER", higher_is_better=False)

    print("Saved master CSV:", csv_path)
    if generated_plots:
        print("Generated plots:")
        for path in generated_plots:
            print("  -", path)
    if generated_tables:
        print("Generated tables:")
        for path in generated_tables:
            print("  -", path)

    if not best_auroc.empty:
        print("\nBest AUROC per (algorithm, language) within focus scenarios:")
        print(best_auroc.to_string(index=False))
    if not best_eer.empty:
        print("\nLowest EER per (algorithm, language) within focus scenarios:")
        print(best_eer.to_string(index=False))


if __name__ == "__main__":
    main()
