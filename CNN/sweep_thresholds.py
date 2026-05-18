#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

from evaluate import DEFAULT_CHECKPOINT, evaluate
from train_detector import DEFAULT_DATASET_ROOT, normalize_path


DEFAULT_OUTPUT_JSON = r"C:\Users\usEr\Downloads\Research\CNN\outputs\sweep_results.json"
DEFAULT_OUTPUT_CSV = r"C:\Users\usEr\Downloads\Research\CNN\outputs\sweep_results.csv"


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search score/NMS thresholds for Faster R-CNN evaluator.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to trained checkpoint.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help="YOLO-format dataset root.")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"], help="Dataset split.")
    parser.add_argument("--score-values", default="0.60,0.62,0.63,0.64,0.65", help="Comma-separated thresholds.")
    parser.add_argument("--nms-values", default="0.18,0.20,0.205,0.21,0.22,0.25", help="Comma-separated NMS IoUs.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="Matching IoU for TP/FP.")
    parser.add_argument("--model-score-threshold", type=float, default=0.05, help="Internal model score threshold.")
    parser.add_argument("--max-detections", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--batch-size", type=int, default=2, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--objective",
        default="micro_f1",
        choices=["micro_f1", "micro_precision", "micro_recall", "mAP50"],
        help="Metric used to rank the best configuration.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="How many top runs to print.")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON, help="Where to save JSON results.")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, help="Where to save CSV results.")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    return parser.parse_args()


def run_single_eval(
    checkpoint: str,
    dataset_root: str,
    split: str,
    score_threshold: float,
    nms_iou_threshold: float,
    iou_threshold: float,
    model_score_threshold: float,
    max_detections: int,
    batch_size: int,
    num_workers: int,
    device: str,
) -> Dict:
    args = SimpleNamespace(
        checkpoint=checkpoint,
        dataset_root=dataset_root,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        model_score_threshold=model_score_threshold,
        nms_iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
        save_json=None,
        device=device,
    )
    return evaluate(args)


def main() -> None:
    args = parse_args()
    score_values = parse_float_list(args.score_values)
    nms_values = parse_float_list(args.nms_values)

    rows: List[Dict] = []
    total = len(score_values) * len(nms_values)
    done = 0

    for nms in nms_values:
        for score in score_values:
            done += 1
            print(f"[{done}/{total}] score={score:.4f} nms={nms:.4f}")
            result = run_single_eval(
                checkpoint=args.checkpoint,
                dataset_root=args.dataset_root,
                split=args.split,
                score_threshold=score,
                nms_iou_threshold=nms,
                iou_threshold=args.iou_threshold,
                model_score_threshold=args.model_score_threshold,
                max_detections=args.max_detections,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device,
            )
            metrics = result["metrics"]
            rows.append(
                {
                    "score_threshold": score,
                    "nms_iou_threshold": nms,
                    "micro_precision": metrics["micro_precision"],
                    "micro_recall": metrics["micro_recall"],
                    "micro_f1": metrics["micro_f1"],
                    "mAP50": metrics["mAP50"],
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                }
            )

    rows_sorted = sorted(rows, key=lambda r: r[args.objective], reverse=True)
    top_k = min(args.top_k, len(rows_sorted))

    print("")
    print(f"Top {top_k} by {args.objective}:")
    for i, row in enumerate(rows_sorted[:top_k], start=1):
        print(
            f"{i:2d}. score={row['score_threshold']:.4f} nms={row['nms_iou_threshold']:.4f} | "
            f"P={row['micro_precision']:.6f} R={row['micro_recall']:.6f} "
            f"F1={row['micro_f1']:.6f} mAP50={row['mAP50']:.6f} "
            f"FP={row['fp']} FN={row['fn']}"
        )

    best = rows_sorted[0]
    print("")
    print(
        f"Best config by {args.objective}: "
        f"score={best['score_threshold']:.4f}, nms={best['nms_iou_threshold']:.4f}"
    )

    output_json_path = normalize_path(args.output_json)
    output_csv_path = normalize_path(args.output_csv)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "objective": args.objective,
        "score_values": score_values,
        "nms_values": nms_values,
        "best": best,
        "results": rows_sorted,
    }
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    fieldnames = [
        "score_threshold",
        "nms_iou_threshold",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "mAP50",
        "tp",
        "fp",
        "fn",
    ]
    with output_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(row)

    print(f"Saved JSON: {output_json_path}")
    print(f"Saved CSV: {output_csv_path}")


if __name__ == "__main__":
    main()
