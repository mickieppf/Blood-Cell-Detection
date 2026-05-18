#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List

import torch
from PIL import Image, ImageDraw
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor


DEFAULT_OUTPUT_IMAGE = r"C:\Users\usEr\Downloads\Research\CNN\prediction.jpg"
DEFAULT_CHECKPOINT = r"C:\Users\usEr\Downloads\Research\CNN\outputs\best_model.pt"
DEFAULT_IMAGE = r"C:\Users\usEr\Downloads\Research\blood-cell-detection-datatset\test\images\BloodImage_00038_jpg.rf.ffa23e4b5b55b523367f332af726eae8.jpg"


def normalize_path(path_str: str) -> Path:
    if os.name == "nt":
        return Path(path_str)
    if re.match(r"^[A-Za-z]:\\", path_str):
        drive_letter = path_str[0].lower()
        rest = path_str[2:].replace("\\", "/").lstrip("/")
        return Path(f"/mnt/{drive_letter}/{rest}")
    return Path(path_str)


def create_model(num_classes: int):
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained Faster R-CNN checkpoint.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to best_model.pt or last_model.pt")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Input image path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_IMAGE, help="Output image path with drawn boxes")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Minimum confidence score")
    parser.add_argument("--model-score-threshold", type=float, default=0.05, help="Internal detector score threshold.")
    parser.add_argument("--nms-iou-threshold", type=float, default=0.3, help="NMS IoU threshold.")
    parser.add_argument("--max-detections", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cuda or cpu")
    return parser.parse_args()


def draw_predictions(image: Image.Image, boxes, labels, scores, class_names: List[str], threshold: float):
    draw = ImageDraw.Draw(image)
    kept = 0
    for box, label, score in zip(boxes, labels, scores):
        if float(score) < threshold:
            continue
        kept += 1
        x1, y1, x2, y2 = [float(v) for v in box]
        class_idx = int(label) - 1
        class_name = class_names[class_idx] if 0 <= class_idx < len(class_names) else str(class_idx)
        text = f"{class_name}: {float(score):.2f}"
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
        draw.text((x1 + 2, y1 + 2), text, fill="red")
    return image, kept


def main() -> None:
    args = parse_args()
    checkpoint_path = normalize_path(args.checkpoint)
    image_path = normalize_path(args.image)
    output_path = normalize_path(args.output)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names = checkpoint.get("class_names")
    if not class_names:
        raise ValueError("Checkpoint does not contain class_names.")

    num_classes = len(class_names) + 1
    model = create_model(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.roi_heads.score_thresh = args.model_score_threshold
    model.roi_heads.nms_thresh = args.nms_iou_threshold
    model.roi_heads.detections_per_img = args.max_detections
    model.to(args.device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = to_tensor(image).to(args.device)

    with torch.no_grad():
        prediction = model([image_tensor])[0]

    result, kept = draw_predictions(
        image.copy(),
        prediction["boxes"].cpu(),
        prediction["labels"].cpu(),
        prediction["scores"].cpu(),
        class_names,
        args.score_threshold,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    print(f"Detections kept after score threshold {args.score_threshold:.2f}: {kept}")
    print(f"Saved prediction image: {output_path}")


if __name__ == "__main__":
    main()
