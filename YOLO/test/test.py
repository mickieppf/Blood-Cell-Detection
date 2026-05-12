"""
Validate every YOLO best.pt checkpoint in this research workspace.

The script discovers all train_results*/weights/best.pt files, runs
Ultralytics validation on the same data config, and writes ranked CSV/JSON
reports showing which checkpoint performs best.

Example:
    python test.py --test-path C:/path/to/test --device 0

If --data is omitted, the script prefers the local
blood-cell-detection-datatset/data.yaml and automatically adds a test split
that points at blood-cell-detection-datatset/test/images.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_METRIC = "map50_95"


@dataclass
class RunInfo:
    run_name: str
    model_family: str
    experiment_group: str
    weights: str
    args_yaml: str | None
    train_best_epoch: int | None
    train_best_map50_95: float | None


@dataclass
class ValidationResult:
    rank: int | None
    run_name: str
    model_family: str
    experiment_group: str
    weights: str
    status: str
    error: str
    precision: float | None
    recall: float | None
    map50: float | None
    map50_95: float | None
    f1: float | None
    fitness: float | None
    train_best_epoch: int | None
    train_best_map50_95: float | None
    val_output_dir: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate all train_results*/weights/best.pt checkpoints and rank them."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Research workspace root. Default: two folders above this script.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help=(
            "Path to YOLO data.yaml. If omitted, use the local dataset data.yaml "
            "when available, then fall back to train_results*/args.yaml."
        ),
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=None,
        help=(
            "Path to the test split folder or images folder. Default: "
            "blood-cell-detection-datatset/test under the research root."
        ),
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument(
        "--device",
        default=None,
        help="Validation device, for example 0, cpu, or cuda:0. Default lets Ultralytics choose.",
    )
    parser.add_argument("--conf", type=float, default=None, help="Optional confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        choices=("map50_95", "map50", "f1", "precision", "recall", "fitness"),
        help="Primary metric used for ranking.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for validation outputs and reports. Default: YOLO/test/runs.",
    )
    parser.add_argument(
        "--pattern",
        default="train_results*",
        help="Folder glob used to discover training runs.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable Ultralytics validation plots.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow Ultralytics to reuse existing validation output folders.",
    )
    return parser.parse_args()


def read_simple_yaml_value(path: Path, key: str) -> str | None:
    if not path.exists():
        return None

    prefix = f"{key}:"
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            value = stripped[len(prefix) :].strip()
            return value.strip("'\"") or None
    return None


def resolve_data_path(
    root: Path,
    args: argparse.Namespace,
    output_dir: Path,
    args_files: list[Path],
) -> Path:
    local_dataset_yaml = root / "blood-cell-detection-datatset" / "data.yaml"
    local_test_path = root / "blood-cell-detection-datatset" / "test"

    test_path = args.test_path.expanduser().resolve() if args.test_path else None
    if test_path is None and local_test_path.exists():
        test_path = local_test_path.resolve()

    if args.data is not None:
        cli_data = args.data.expanduser().resolve()
        if cli_data.is_dir():
            return build_eval_data_yaml(root, output_dir, cli_data.parent / "data.yaml", cli_data)
        if test_path is not None and args.split == "test":
            return build_eval_data_yaml(root, output_dir, cli_data, test_path)
        return cli_data

    if local_dataset_yaml.exists() and test_path is not None and args.split == "test":
        return build_eval_data_yaml(root, output_dir, local_dataset_yaml, test_path)

    for args_yaml in args_files:
        raw_data = read_simple_yaml_value(args_yaml, "data")
        if raw_data:
            candidate = Path(raw_data)
            if not candidate.is_absolute():
                candidate = root / candidate
            if candidate.exists():
                return candidate.resolve()

    if local_dataset_yaml.exists():
        return local_dataset_yaml.resolve()

    raise FileNotFoundError(
        "Could not find a data path in args.yaml. Pass one explicitly with --data."
    )


def as_yaml_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def split_images_dir(split_path: Path) -> Path:
    images_dir = split_path / "images"
    return images_dir if images_dir.is_dir() else split_path


def first_existing_path(candidates: list[Path], fallback: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return fallback


def build_eval_data_yaml(
    root: Path,
    output_dir: Path,
    source_yaml: Path,
    test_path: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = test_path.parent if (test_path / "images").is_dir() else test_path.parent.parent
    test_images = split_images_dir(test_path)
    train_images = first_existing_path(
        [dataset_root / "train" / "images"],
        test_images,
    )
    val_images = first_existing_path(
        [dataset_root / "valid" / "images", dataset_root / "val" / "images"],
        test_images,
    )

    nc = read_simple_yaml_value(source_yaml, "nc") if source_yaml.exists() else None
    names = read_simple_yaml_value(source_yaml, "names") if source_yaml.exists() else None
    nc = nc or "3"
    names = names or "['Platelets', 'RBC', 'WBC']"

    eval_yaml = output_dir / "data_test_autogen.yaml"
    eval_yaml.write_text(
        "\n".join(
            [
                "# Auto-generated by YOLO/test/test.py for fair best.pt evaluation.",
                f"path: {as_yaml_path(root)}",
                f"train: {as_yaml_path(train_images)}",
                f"val: {as_yaml_path(val_images)}",
                f"test: {as_yaml_path(test_images)}",
                f"nc: {nc}",
                f"names: {names}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return eval_yaml


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


def best_training_metric(results_csv: Path) -> tuple[int | None, float | None]:
    if not results_csv.exists():
        return None, None

    with results_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    best_epoch: int | None = None
    best_map: float | None = None
    for row in rows:
        value = parse_float(row.get("metrics/mAP50-95(B)"))
        if value is None:
            continue
        if best_map is None or value > best_map:
            best_map = value
            best_epoch = parse_int(row.get("epoch"))
    return best_epoch, best_map


def discover_runs(root: Path, pattern: str) -> list[RunInfo]:
    runs: list[RunInfo] = []
    for run_dir in sorted(path for path in root.glob(pattern) if path.is_dir()):
        weights = run_dir / "weights" / "best.pt"
        if not weights.exists():
            continue

        args_yaml = run_dir / "args.yaml"
        best_epoch, best_map = best_training_metric(run_dir / "results.csv")
        runs.append(
            RunInfo(
                run_name=run_dir.name,
                model_family=model_family_from_run_name(run_dir.name),
                experiment_group=experiment_group_from_run_name(run_dir.name),
                weights=str(weights),
                args_yaml=str(args_yaml) if args_yaml.exists() else None,
                train_best_epoch=best_epoch,
                train_best_map50_95=best_map,
            )
        )
    return runs


def parse_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def metric_from_object(obj: Any, names: tuple[str, ...]) -> float | None:
    current = obj
    for name in names:
        current = getattr(current, name, None)
        if current is None:
            return None
    return parse_float(current)


def extract_metrics(metrics: Any) -> dict[str, float | None]:
    box = getattr(metrics, "box", None)
    precision = metric_from_object(metrics, ("box", "mp"))
    recall = metric_from_object(metrics, ("box", "mr"))
    map50 = metric_from_object(metrics, ("box", "map50"))
    map50_95 = metric_from_object(metrics, ("box", "map"))
    fitness = None

    fitness_value = getattr(metrics, "fitness", None)
    if callable(fitness_value):
        fitness = parse_float(fitness_value())
    else:
        fitness = parse_float(fitness_value)

    if box is None and hasattr(metrics, "results_dict"):
        results_dict = getattr(metrics, "results_dict")
        precision = parse_float(results_dict.get("metrics/precision(B)", precision))
        recall = parse_float(results_dict.get("metrics/recall(B)", recall))
        map50 = parse_float(results_dict.get("metrics/mAP50(B)", map50))
        map50_95 = parse_float(results_dict.get("metrics/mAP50-95(B)", map50_95))
        fitness = parse_float(results_dict.get("fitness", fitness))

    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
        "f1": f1,
        "fitness": fitness,
    }


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return cleaned.strip("_") or "run"


def validate_run(
    run: RunInfo,
    data: Path,
    output_dir: Path,
    args: argparse.Namespace,
    yolo_class: Any,
) -> ValidationResult:
    run_output_name = safe_name(run.run_name)
    try:
        model = yolo_class(run.weights)
        kwargs: dict[str, Any] = {
            "data": str(data),
            "imgsz": args.imgsz,
            "batch": args.batch,
            "split": args.split,
            "iou": args.iou,
            "plots": not args.no_plots,
            "project": str(output_dir),
            "name": run_output_name,
            "exist_ok": args.exist_ok,
        }
        if args.device is not None:
            kwargs["device"] = args.device
        if args.conf is not None:
            kwargs["conf"] = args.conf

        metrics = model.val(**kwargs)
        extracted = extract_metrics(metrics)
        val_output_dir = getattr(metrics, "save_dir", None)

        return ValidationResult(
            rank=None,
            run_name=run.run_name,
            model_family=run.model_family,
            experiment_group=run.experiment_group,
            weights=run.weights,
            status="ok",
            error="",
            precision=extracted["precision"],
            recall=extracted["recall"],
            map50=extracted["map50"],
            map50_95=extracted["map50_95"],
            f1=extracted["f1"],
            fitness=extracted["fitness"],
            train_best_epoch=run.train_best_epoch,
            train_best_map50_95=run.train_best_map50_95,
            val_output_dir=str(val_output_dir) if val_output_dir is not None else None,
        )
    except Exception as exc:
        return ValidationResult(
            rank=None,
            run_name=run.run_name,
            model_family=run.model_family,
            experiment_group=run.experiment_group,
            weights=run.weights,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            precision=None,
            recall=None,
            map50=None,
            map50_95=None,
            f1=None,
            fitness=None,
            train_best_epoch=run.train_best_epoch,
            train_best_map50_95=run.train_best_map50_95,
            val_output_dir=None,
        )


def rank_results(results: list[ValidationResult], metric: str) -> list[ValidationResult]:
    def score(result: ValidationResult) -> tuple[int, float, float, float]:
        primary = getattr(result, metric)
        primary_value = primary if primary is not None else -1.0
        map50 = result.map50 if result.map50 is not None else -1.0
        f1 = result.f1 if result.f1 is not None else -1.0
        ok = 1 if result.status == "ok" else 0
        return ok, primary_value, map50, f1

    ranked = sorted(results, key=score, reverse=True)
    rank = 1
    for result in ranked:
        if result.status == "ok":
            result.rank = rank
            rank += 1
        else:
            result.rank = None
    return ranked


def write_reports(
    output_dir: Path,
    ranked_results: list[ValidationResult],
    runs: list[RunInfo],
    data: Path,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"best_pt_validation_summary_{timestamp}.csv"
    json_path = output_dir / f"best_pt_validation_summary_{timestamp}.json"

    fieldnames = list(asdict(ranked_results[0]).keys()) if ranked_results else []
    if fieldnames:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for result in ranked_results:
                writer.writerow(asdict(result))

    payload = {
        "created_at": timestamp,
        "root": str(args.root),
        "data": str(data),
        "split": args.split,
        "ranking_metric": args.metric,
        "runs_discovered": [asdict(run) for run in runs],
        "results": [asdict(result) for result in ranked_results],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nSaved CSV report:  {csv_path}")
    print(f"Saved JSON report: {json_path}")


def print_summary(ranked_results: list[ValidationResult], metric: str) -> None:
    ok_results = [result for result in ranked_results if result.status == "ok"]
    failed_results = [result for result in ranked_results if result.status != "ok"]

    if not ok_results:
        print("\nNo validation runs completed successfully.")
        return

    best = ok_results[0]
    print("\nBest checkpoint")
    print(f"  Rank metric: {metric}")
    print(f"  Run:         {best.run_name}")
    print(f"  Weights:     {best.weights}")
    print(f"  mAP50-95:    {format_metric(best.map50_95)}")
    print(f"  mAP50:       {format_metric(best.map50)}")
    print(f"  Precision:   {format_metric(best.precision)}")
    print(f"  Recall:      {format_metric(best.recall)}")
    print(f"  F1:          {format_metric(best.f1)}")

    print("\nRanking")
    print(
        "rank,run,model,group,map50_95,map50,precision,recall,f1,train_best_epoch"
    )
    for result in ok_results:
        print(
            ",".join(
                [
                    str(result.rank),
                    result.run_name,
                    result.model_family,
                    result.experiment_group,
                    format_metric(result.map50_95),
                    format_metric(result.map50),
                    format_metric(result.precision),
                    format_metric(result.recall),
                    format_metric(result.f1),
                    str(result.train_best_epoch or ""),
                ]
            )
        )

    if failed_results:
        print("\nFailed runs")
        for result in failed_results:
            print(f"  {result.run_name}: {result.error}")


def format_metric(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (args.output_dir or (Path(__file__).resolve().parent / "runs")).resolve()
    config_dir = Path(__file__).resolve().parent / "ultralytics_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir.resolve()))

    runs = discover_runs(root, args.pattern)
    if not runs:
        print(f"No best.pt checkpoints found under {root} using pattern {args.pattern!r}.")
        return 1

    data = resolve_data_path(
        root,
        args,
        output_dir,
        [Path(run.args_yaml) for run in runs if run.args_yaml],
    )
    if not data.exists():
        print(f"Data YAML not found: {data}")
        print("Pass the correct dataset path with --data C:/path/to/data.yaml")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("The ultralytics package is not installed in this Python environment.")
        print("Install it with: pip install ultralytics")
        return 1

    print(f"Research root: {root}")
    print(f"Data YAML:     {data}")
    print(f"Output dir:    {output_dir}")
    print(f"Found {len(runs)} checkpoints.")

    results = []
    for index, run in enumerate(runs, start=1):
        print(f"\n[{index}/{len(runs)}] Validating {run.run_name}")
        results.append(validate_run(run, data, output_dir, args, YOLO))

    ranked_results = rank_results(results, args.metric)
    write_reports(output_dir, ranked_results, runs, data, args)
    print_summary(ranked_results, args.metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
