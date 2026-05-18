#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from train_detector import (
    DEFAULT_DATASET_ROOT,
    collate_fn,
    create_model,
    normalize_path,
    read_class_names,
    YoloDetectionDataset,
)


DEFAULT_CHECKPOINT = r"C:\Users\usEr\Downloads\Research\CNN\outputs\best_model.pt"
DEFAULT_JSON_OUTPUT = r"C:\Users\usEr\Downloads\Research\CNN\outputs\eval_test_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Faster R-CNN blood cell detector on a dataset split.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to best_model.pt or last_model.pt")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help="Path to YOLO-format dataset root.")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"], help="Dataset split.")
    parser.add_argument("--batch-size", type=int, default=2, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Threshold for counting detections.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for TP/FP matching.")
    parser.add_argument("--model-score-threshold", type=float, default=0.05, help="Internal detector score threshold.")
    parser.add_argument("--nms-iou-threshold", type=float, default=0.3, help="NMS IoU threshold.")
    parser.add_argument("--max-detections", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--save-json", default=DEFAULT_JSON_OUTPUT, help="Optional path to save JSON metrics.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cuda or cpu")
    return parser.parse_args()


def compute_ap(scores: List[float], tp_flags: List[int], fp_flags: List[int], gt_count: int) -> float:
    if gt_count == 0:
        return float("nan")
    if not scores:
        return 0.0

    scores_t = torch.tensor(scores, dtype=torch.float32)
    tp_t = torch.tensor(tp_flags, dtype=torch.float32)
    fp_t = torch.tensor(fp_flags, dtype=torch.float32)

    order = torch.argsort(scores_t, descending=True)
    tp_sorted = tp_t[order]
    fp_sorted = fp_t[order]

    tp_cum = torch.cumsum(tp_sorted, dim=0)
    fp_cum = torch.cumsum(fp_sorted, dim=0)

    recalls = tp_cum / float(gt_count)
    precisions = tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-9)

    mrec = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])

    for i in range(mpre.numel() - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])

    changing_points = torch.where(mrec[1:] != mrec[:-1])[0]
    ap = torch.sum((mrec[changing_points + 1] - mrec[changing_points]) * mpre[changing_points + 1]).item()
    return float(ap)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Dict:
    dataset_root = normalize_path(args.dataset_root)
    checkpoint_path = normalize_path(args.checkpoint)
    output_json_path = normalize_path(args.save_json) if args.save_json else None

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    class_names = read_class_names(dataset_root)
    num_classes = len(class_names) + 1

    dataset = YoloDetectionDataset(dataset_root, split=args.split)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = create_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.roi_heads.score_thresh = args.model_score_threshold
    model.roi_heads.nms_thresh = args.nms_iou_threshold
    model.roi_heads.detections_per_img = args.max_detections

    device = torch.device(args.device)
    model.to(device)
    model.eval()

    per_class: Dict[int, Dict[str, List]] = {
        c: {"scores": [], "tp": [], "fp": [], "gt_count": 0} for c in range(1, num_classes)
    }

    for batch_images, batch_targets in dataloader:
        images = [img.to(device) for img in batch_images]
        predictions = model(images)

        for pred, target in zip(predictions, batch_targets):
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]

            pred_boxes = pred["boxes"].detach().cpu()
            pred_labels = pred["labels"].detach().cpu()
            pred_scores = pred["scores"].detach().cpu()

            for class_id in range(1, num_classes):
                cls_gt_mask = gt_labels == class_id
                cls_pred_mask = (pred_labels == class_id) & (pred_scores >= args.score_threshold)

                cls_gt_boxes = gt_boxes[cls_gt_mask]
                cls_pred_boxes = pred_boxes[cls_pred_mask]
                cls_pred_scores = pred_scores[cls_pred_mask]

                class_state = per_class[class_id]
                class_state["gt_count"] += int(cls_gt_boxes.shape[0])

                if cls_pred_boxes.numel() == 0:
                    continue

                order = torch.argsort(cls_pred_scores, descending=True)
                cls_pred_boxes = cls_pred_boxes[order]
                cls_pred_scores = cls_pred_scores[order]

                matched_gt = torch.zeros((cls_gt_boxes.shape[0],), dtype=torch.bool)

                for det_box, det_score in zip(cls_pred_boxes, cls_pred_scores):
                    if cls_gt_boxes.shape[0] == 0:
                        tp_flag = 0
                        fp_flag = 1
                    else:
                        ious = box_iou(det_box.unsqueeze(0), cls_gt_boxes).squeeze(0)
                        best_iou, best_idx = torch.max(ious, dim=0)
                        if best_iou.item() >= args.iou_threshold and not matched_gt[best_idx]:
                            matched_gt[best_idx] = True
                            tp_flag = 1
                            fp_flag = 0
                        else:
                            tp_flag = 0
                            fp_flag = 1

                    class_state["scores"].append(float(det_score.item()))
                    class_state["tp"].append(tp_flag)
                    class_state["fp"].append(fp_flag)

    class_results = []
    sum_tp = 0
    sum_fp = 0
    sum_fn = 0
    ap_values = []

    for class_id in range(1, num_classes):
        class_name = class_names[class_id - 1]
        state = per_class[class_id]

        tp_total = int(sum(state["tp"]))
        fp_total = int(sum(state["fp"]))
        gt_count = int(state["gt_count"])
        fn_total = max(gt_count - tp_total, 0)

        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        recall = tp_total / gt_count if gt_count > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        ap50 = compute_ap(state["scores"], state["tp"], state["fp"], gt_count)

        if not math.isnan(ap50):
            ap_values.append(ap50)

        class_results.append(
            {
                "class_id": class_id - 1,
                "class_name": class_name,
                "gt_count": gt_count,
                "tp": tp_total,
                "fp": fp_total,
                "fn": fn_total,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "ap50": round(ap50, 6) if not math.isnan(ap50) else None,
            }
        )

        sum_tp += tp_total
        sum_fp += fp_total
        sum_fn += fn_total

    micro_precision = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
    micro_recall = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0.0
    micro_f1 = (
        (2 * micro_precision * micro_recall / (micro_precision + micro_recall))
        if (micro_precision + micro_recall) > 0
        else 0.0
    )
    map50 = float(sum(ap_values) / len(ap_values)) if ap_values else 0.0

    results = {
        "checkpoint": str(checkpoint_path),
        "dataset_root": str(dataset_root),
        "split": args.split,
        "score_threshold": args.score_threshold,
        "iou_threshold": args.iou_threshold,
        "nms_iou_threshold": args.nms_iou_threshold,
        "model_score_threshold": args.model_score_threshold,
        "num_images": len(dataset),
        "metrics": {
            "micro_precision": round(micro_precision, 6),
            "micro_recall": round(micro_recall, 6),
            "micro_f1": round(micro_f1, 6),
            "mAP50": round(map50, 6),
            "tp": sum_tp,
            "fp": sum_fp,
            "fn": sum_fn,
        },
        "per_class": class_results,
    }

    if output_json_path:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with output_json_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


def main() -> None:
    args = parse_args()
    results = evaluate(args)

    print(f"Split: {results['split']} | Images: {results['num_images']}")
    print(
        "Overall: "
        f"Precision={results['metrics']['micro_precision']:.4f} "
        f"Recall={results['metrics']['micro_recall']:.4f} "
        f"F1={results['metrics']['micro_f1']:.4f} "
        f"mAP50={results['metrics']['mAP50']:.4f}"
    )
    for item in results["per_class"]:
        print(
            f"[{item['class_name']}] "
            f"GT={item['gt_count']} TP={item['tp']} FP={item['fp']} FN={item['fn']} "
            f"P={item['precision']:.4f} R={item['recall']:.4f} F1={item['f1']:.4f} "
            f"AP50={(item['ap50'] if item['ap50'] is not None else float('nan')):.4f}"
        )
    if args.save_json:
        print(f"Saved JSON metrics: {normalize_path(args.save_json)}")


if __name__ == "__main__":
    main()
