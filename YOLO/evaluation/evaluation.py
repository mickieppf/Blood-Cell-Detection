"""
Compare YOLO training validation metrics against held-out test metrics.

This script reads:
  - train_results*/results.csv for the best training-run validation metrics
  - YOLO/test/runs/best_pt_validation_summary_*.csv for held-out test metrics

It writes comparison tables, a markdown report, and graph images to
YOLO/evaluation/outputs by default.

Example:
    python evaluation.py
    python evaluation.py --test-report ../test/runs/best_pt_validation_summary_20260513_005420.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - matplotlib fallback is enough for the script.
    sns = None


TRAIN_METRIC_COLUMNS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}

TEST_METRIC_COLUMNS = ["precision", "recall", "map50", "map50_95", "f1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create graphs comparing YOLO train validation and held-out test reports."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Research workspace root. Default: two folders above this script.",
    )
    parser.add_argument(
        "--test-report",
        type=Path,
        default=None,
        help="Path to best_pt_validation_summary_*.csv. Default: latest in YOLO/test/runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory where plots and summary files will be written.",
    )
    parser.add_argument(
        "--sort-by",
        default="test_map50_95",
        choices=(
            "test_map50_95",
            "test_map50",
            "test_f1",
            "train_best_map50_95",
            "generalization_gap_map50_95",
        ),
        help="Column used to sort models in tables and most plots.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="Limit plots to the top N rows after sorting. Default 0 means all runs.",
    )
    return parser.parse_args()


def parse_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: Any) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def model_family_from_run_name(run_name: str) -> str:
    match = re.search(r"\(([^)]+)\)", run_name)
    if not match:
        return "unknown"
    model = match.group(1)
    return model.replace("yolov11", "yolo11").replace("yolov26", "yolo26")


def experiment_group_from_run_name(run_name: str) -> str:
    if "150_epochs" in run_name:
        return "150_epochs"
    if "500epochs_with_optimizer" in run_name:
        return "500epochs_with_optimizer"
    if "500epochs" in run_name:
        return "500epochs"
    return "unknown"


def short_run_name(run_name: str) -> str:
    group = experiment_group_from_run_name(run_name)
    model = model_family_from_run_name(run_name)
    if group == "150_epochs":
        prefix = "150"
    elif group == "500epochs":
        prefix = "500"
    elif group == "500epochs_with_optimizer":
        prefix = "500+opt"
    else:
        prefix = group
    return f"{model} ({prefix})"


def read_train_results(root: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for results_csv in sorted(root.glob("train_results*/results.csv")):
        rows = read_csv_dicts(results_csv)
        if not rows:
            continue

        best_row = max(
            rows,
            key=lambda row: parse_float(row.get(TRAIN_METRIC_COLUMNS["map50_95"])) or -math.inf,
        )
        final_row = rows[-1]
        run_name = results_csv.parent.name

        record = {
            "run_name": run_name,
            "short_name": short_run_name(run_name),
            "model_family": model_family_from_run_name(run_name),
            "experiment_group": experiment_group_from_run_name(run_name),
            "train_results_csv": str(results_csv),
            "train_epochs_recorded": len(rows),
            "train_best_epoch": parse_int(best_row.get("epoch")),
            "train_final_epoch": parse_int(final_row.get("epoch")),
            "train_final_map50_95": parse_float(final_row.get(TRAIN_METRIC_COLUMNS["map50_95"])),
        }

        for metric, column in TRAIN_METRIC_COLUMNS.items():
            record[f"train_best_{metric}"] = parse_float(best_row.get(column))
            record[f"train_final_{metric}"] = parse_float(final_row.get(column))

        records.append(record)

    return pd.DataFrame(records)


def read_test_report(test_report: Path) -> pd.DataFrame:
    rows = read_csv_dicts(test_report)
    records: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") and row["status"] != "ok":
            continue

        run_name = row["run_name"]
        record = {
            "run_name": run_name,
            "test_rank": parse_int(row.get("rank")),
            "test_report_csv": str(test_report),
        }

        for metric in TEST_METRIC_COLUMNS:
            record[f"test_{metric}"] = parse_float(row.get(metric))

        records.append(record)

    return pd.DataFrame(records)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def latest_test_report(root: Path) -> Path:
    report_dir = root / "YOLO" / "test" / "runs"
    candidates = sorted(
        report_dir.glob("best_pt_validation_summary_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No test summary CSV found in {report_dir}")
    return candidates[0]


def build_comparison(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    comparison = train_df.merge(test_df, on="run_name", how="inner")
    if comparison.empty:
        raise ValueError("No matching run_name values between train results and test report.")

    comparison["generalization_gap_map50_95"] = (
        comparison["train_best_map50_95"] - comparison["test_map50_95"]
    )
    comparison["generalization_gap_map50"] = comparison["train_best_map50"] - comparison["test_map50"]
    comparison["test_minus_train_map50_95"] = (
        comparison["test_map50_95"] - comparison["train_best_map50_95"]
    )
    comparison["overfit_signal"] = comparison["generalization_gap_map50_95"].apply(
        lambda value: "higher train" if value > 0 else "higher test"
    )
    return comparison


def sorted_plot_df(comparison: pd.DataFrame, sort_by: str, top_n: int) -> pd.DataFrame:
    ascending = sort_by == "generalization_gap_map50_95"
    data = comparison.sort_values(sort_by, ascending=ascending).copy()
    if top_n > 0:
        data = data.head(top_n)
    return data.reset_index(drop=True)


def setup_style() -> None:
    if sns is not None:
        sns.set_theme(style="whitegrid", context="talk")
    else:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["font.size"] = 10


def save_train_vs_test_bar(data: pd.DataFrame, output_dir: Path) -> Path:
    plot_data = data.melt(
        id_vars=["short_name"],
        value_vars=["train_best_map50_95", "test_map50_95"],
        var_name="split",
        value_name="mAP50-95",
    )
    plot_data["split"] = plot_data["split"].map(
        {"train_best_map50_95": "Train validation best", "test_map50_95": "Held-out test"}
    )

    fig, ax = plt.subplots(figsize=(max(12, len(data) * 0.8), 7))
    if sns is not None:
        sns.barplot(data=plot_data, x="short_name", y="mAP50-95", hue="split", ax=ax)
    else:
        plot_grouped_bars(ax, plot_data, "short_name", "split", "mAP50-95")
    ax.set_title("Train Validation vs Held-Out Test mAP50-95")
    ax.set_xlabel("Model run")
    ax.set_ylabel("mAP50-95")
    ax.set_ylim(0, max(0.75, plot_data["mAP50-95"].max() + 0.05))
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="")
    fig.tight_layout()

    path = output_dir / "train_vs_test_map50_95.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_generalization_gap(data: pd.DataFrame, output_dir: Path) -> Path:
    sorted_data = data.sort_values("generalization_gap_map50_95", ascending=False)
    colors = ["#c44e52" if value > 0 else "#55a868" for value in sorted_data["generalization_gap_map50_95"]]

    fig, ax = plt.subplots(figsize=(max(12, len(data) * 0.75), 7))
    ax.bar(sorted_data["short_name"], sorted_data["generalization_gap_map50_95"], color=colors)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("Generalization Gap: Train Best mAP50-95 minus Test mAP50-95")
    ax.set_xlabel("Model run")
    ax.set_ylabel("mAP50-95 gap")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()

    path = output_dir / "generalization_gap_map50_95.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_scatter(data: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 8))
    if sns is not None:
        sns.scatterplot(
            data=data,
            x="train_best_map50_95",
            y="test_map50_95",
            hue="model_family",
            style="experiment_group",
            s=130,
            ax=ax,
        )
    else:
        ax.scatter(data["train_best_map50_95"], data["test_map50_95"], s=110)

    min_value = min(data["train_best_map50_95"].min(), data["test_map50_95"].min()) - 0.02
    max_value = max(data["train_best_map50_95"].max(), data["test_map50_95"].max()) + 0.02
    ax.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="#555555")
    ax.set_xlim(min_value, max_value)
    ax.set_ylim(min_value, max_value)
    ax.set_title("Train Validation mAP50-95 vs Test mAP50-95")
    ax.set_xlabel("Train validation best mAP50-95")
    ax.set_ylabel("Held-out test mAP50-95")

    for _, row in data.iterrows():
        ax.annotate(row["short_name"], (row["train_best_map50_95"], row["test_map50_95"]), fontsize=8)

    fig.tight_layout()
    path = output_dir / "train_test_map50_95_scatter.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_test_metric_heatmap(data: pd.DataFrame, output_dir: Path) -> Path:
    metric_columns = [f"test_{metric}" for metric in TEST_METRIC_COLUMNS]
    heatmap_data = data.set_index("short_name")[metric_columns].copy()
    heatmap_data.columns = ["Precision", "Recall", "mAP50", "mAP50-95", "F1"]

    fig, ax = plt.subplots(figsize=(10, max(7, len(data) * 0.45)))
    if sns is not None:
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            linewidths=0.5,
            vmin=0.35,
            vmax=1.0,
            ax=ax,
        )
    else:
        image = ax.imshow(heatmap_data.values, aspect="auto", cmap="viridis", vmin=0.35, vmax=1.0)
        ax.set_xticks(range(len(heatmap_data.columns)), heatmap_data.columns)
        ax.set_yticks(range(len(heatmap_data.index)), heatmap_data.index)
        fig.colorbar(image, ax=ax)
    ax.set_title("Held-Out Test Metrics")
    ax.set_xlabel("")
    ax.set_ylabel("Model run")
    fig.tight_layout()

    path = output_dir / "test_metrics_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_model_group_boxplot(comparison: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    if sns is not None:
        sns.boxplot(data=comparison, x="model_family", y="test_map50_95", ax=axes[0])
        sns.stripplot(data=comparison, x="model_family", y="test_map50_95", ax=axes[0], color="black", size=6)
        sns.boxplot(data=comparison, x="experiment_group", y="test_map50_95", ax=axes[1])
        sns.stripplot(
            data=comparison,
            x="experiment_group",
            y="test_map50_95",
            ax=axes[1],
            color="black",
            size=6,
        )
    else:
        comparison.boxplot(column="test_map50_95", by="model_family", ax=axes[0])
        comparison.boxplot(column="test_map50_95", by="experiment_group", ax=axes[1])

    axes[0].set_title("Test mAP50-95 by Model Family")
    axes[0].set_xlabel("Model family")
    axes[0].set_ylabel("Test mAP50-95")
    axes[1].set_title("Test mAP50-95 by Experiment Group")
    axes[1].set_xlabel("Experiment group")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("")
    fig.tight_layout()

    path = output_dir / "test_map50_95_by_group.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_grouped_bars(ax: plt.Axes, data: pd.DataFrame, x_col: str, hue_col: str, y_col: str) -> None:
    x_values = list(data[x_col].drop_duplicates())
    hue_values = list(data[hue_col].drop_duplicates())
    width = 0.8 / len(hue_values)
    centers = range(len(x_values))

    for hue_index, hue in enumerate(hue_values):
        subset = data[data[hue_col] == hue]
        heights = [
            subset.loc[subset[x_col] == x_value, y_col].iloc[0]
            if not subset.loc[subset[x_col] == x_value, y_col].empty
            else 0
            for x_value in x_values
        ]
        offsets = [center - 0.4 + width / 2 + hue_index * width for center in centers]
        ax.bar(offsets, heights, width=width, label=hue)

    ax.set_xticks(list(centers), x_values)


def write_outputs(
    comparison: pd.DataFrame,
    sorted_data: pd.DataFrame,
    figure_paths: list[Path],
    output_dir: Path,
    test_report: Path,
) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"train_test_comparison_{timestamp}.csv"
    json_path = output_dir / f"train_test_comparison_{timestamp}.json"
    report_path = output_dir / f"evaluation_report_{timestamp}.md"

    comparison.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(json.loads(comparison.to_json(orient="records")), indent=2),
        encoding="utf-8",
    )

    best = sorted_data.iloc[0]
    report_lines = [
        "# YOLO Train vs Test Evaluation",
        "",
        f"Test report: `{test_report}`",
        f"Runs compared: {len(comparison)}",
        "",
        "## Best Held-Out Test Model",
        "",
        f"- Run: `{best['run_name']}`",
        f"- Test mAP50-95: {best['test_map50_95']:.4f}",
        f"- Test mAP50: {best['test_map50']:.4f}",
        f"- Test precision: {best['test_precision']:.4f}",
        f"- Test recall: {best['test_recall']:.4f}",
        f"- Train best mAP50-95: {best['train_best_map50_95']:.4f}",
        f"- Generalization gap: {best['generalization_gap_map50_95']:.4f}",
        "",
        "## Top Runs",
        "",
        "| Rank | Run | Train best mAP50-95 | Test mAP50-95 | Test mAP50 | Test F1 | Gap |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]

    for rank, (_, row) in enumerate(sorted_data.iterrows(), start=1):
        report_lines.append(
            "| "
            f"{rank} | `{row['run_name']}` | "
            f"{row['train_best_map50_95']:.4f} | "
            f"{row['test_map50_95']:.4f} | "
            f"{row['test_map50']:.4f} | "
            f"{row['test_f1']:.4f} | "
            f"{row['generalization_gap_map50_95']:.4f} |"
        )

    report_lines.extend(["", "## Figures", ""])
    for figure in figure_paths:
        report_lines.append(f"- `{figure.name}`")

    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return csv_path, report_path


def print_summary(sorted_data: pd.DataFrame, figure_paths: list[Path], csv_path: Path, report_path: Path) -> None:
    best = sorted_data.iloc[0]
    print("\nBest held-out test model")
    print(f"  Run:         {best['run_name']}")
    print(f"  Test mAP50-95: {best['test_map50_95']:.6f}")
    print(f"  Test mAP50:    {best['test_map50']:.6f}")
    print(f"  Test F1:       {best['test_f1']:.6f}")
    print(f"  Train best:    {best['train_best_map50_95']:.6f}")
    print(f"  Gap:           {best['generalization_gap_map50_95']:.6f}")
    print("\nSaved outputs")
    print(f"  Table:  {csv_path}")
    print(f"  Report: {report_path}")
    for figure in figure_paths:
        print(f"  Figure: {figure}")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    test_report = args.test_report.resolve() if args.test_report else latest_test_report(root)
    train_df = read_train_results(root)
    test_df = read_test_report(test_report)
    comparison = build_comparison(train_df, test_df)
    sorted_data = sorted_plot_df(comparison, args.sort_by, args.top_n)

    figure_paths = [
        save_train_vs_test_bar(sorted_data, output_dir),
        save_generalization_gap(sorted_data, output_dir),
        save_scatter(sorted_data, output_dir),
        save_test_metric_heatmap(sorted_data, output_dir),
        save_model_group_boxplot(comparison, output_dir),
    ]
    csv_path, report_path = write_outputs(comparison, sorted_data, figure_paths, output_dir, test_report)
    print_summary(sorted_data, figure_paths, csv_path, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
