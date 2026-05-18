#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor


DEFAULT_DATASET_ROOT = r"C:\Users\usEr\Downloads\Research\blood-cell-detection-datatset"
DEFAULT_OUTPUT_DIR = r"C:\Users\usEr\Downloads\Research\CNN\outputs"


def normalize_path(path_str: str) -> Path:
    if os.name == "nt":
        return Path(path_str)
    if re.match(r"^[A-Za-z]:\\", path_str):
        drive_letter = path_str[0].lower()
        rest = path_str[2:].replace("\\", "/").lstrip("/")
        return Path(f"/mnt/{drive_letter}/{rest}")
    return Path(path_str)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_class_names(dataset_root: Path) -> List[str]:
    data_yaml = dataset_root / "data.yaml"
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [name for _, name in sorted(names.items(), key=lambda item: int(item[0]))]
    if not isinstance(names, list) or not names:
        raise ValueError(f"Invalid classes in {data_yaml}")
    return [str(name) for name in names]


class YoloDetectionDataset(Dataset):
    def __init__(self, dataset_root: Path, split: str) -> None:
        self.dataset_root = dataset_root
        self.split = split
        self.images_dir = dataset_root / split / "images"
        self.labels_dir = dataset_root / split / "labels"
        if not self.images_dir.exists() or not self.labels_dir.exists():
            raise FileNotFoundError(f"Missing expected folders: {self.images_dir} and/or {self.labels_dir}")

        image_patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        image_paths: List[Path] = []
        for pattern in image_patterns:
            image_paths.extend(self.images_dir.glob(pattern))
        self.image_paths = sorted(image_paths)
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        label_path = self.labels_dir / f"{image_path.stem}.txt"
        boxes, labels = self._read_yolo_labels(label_path, width, height)

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": self._compute_area(boxes),
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
        }
        return to_tensor(image), target

    @staticmethod
    def _compute_area(boxes: torch.Tensor) -> torch.Tensor:
        if boxes.numel() == 0:
            return torch.zeros((0,), dtype=torch.float32)
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    @staticmethod
    def _read_yolo_labels(label_path: Path, width: int, height: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not label_path.exists():
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.int64),
            )

        boxes_list: List[List[float]] = []
        labels_list: List[int] = []

        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, xc, yc, bw, bh = map(float, parts)
                x_center = xc * width
                y_center = yc * height
                box_w = bw * width
                box_h = bh * height

                x1 = max(0.0, x_center - (box_w / 2.0))
                y1 = max(0.0, y_center - (box_h / 2.0))
                x2 = min(float(width), x_center + (box_w / 2.0))
                y2 = min(float(height), y_center + (box_h / 2.0))

                if x2 <= x1 or y2 <= y1:
                    continue
                boxes_list.append([x1, y1, x2, y2])
                # Reserve class 0 for background as required by Faster R-CNN.
                labels_list.append(int(class_id) + 1)

        if not boxes_list:
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.int64),
            )

        return (
            torch.tensor(boxes_list, dtype=torch.float32),
            torch.tensor(labels_list, dtype=torch.int64),
        )


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def create_model(num_classes: int, pretrained: bool = True) -> FasterRCNN:
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def move_to_device(images, targets, device: torch.device):
    images = [image.to(device) for image in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return images, targets


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    print_freq: int,
) -> float:
    model.train()
    running_loss = 0.0

    for step, (images, targets) in enumerate(dataloader, start=1):
        images, targets = move_to_device(images, targets, device)
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if step % print_freq == 0 or step == len(dataloader):
            print(f"[Epoch {epoch}] step {step}/{len(dataloader)} train_loss={loss.item():.4f}")

    return running_loss / max(len(dataloader), 1)


@torch.no_grad()
def validate_loss(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    # Detection models only return losses in training mode.
    model.train()
    total_loss = 0.0
    for images, targets in dataloader:
        images, targets = move_to_device(images, targets, device)
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        total_loss += loss.item()
    return total_loss / max(len(dataloader), 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Faster R-CNN blood cell detector from YOLO labels.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help="Path to YOLO-format dataset root.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Folder to save checkpoints.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--lr", type=float, default=0.005, help="Initial learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum.")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay.")
    parser.add_argument("--step-size", type=int, default=6, help="StepLR step size.")
    parser.add_argument("--gamma", type=float, default=0.1, help="StepLR gamma.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--print-freq", type=int, default=20, help="Print every N training steps.")
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="Use COCO-pretrained Faster R-CNN weights before fine-tuning.",
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Disable COCO-pretrained initialization.",
    )
    parser.set_defaults(pretrained=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cuda or cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_root = normalize_path(args.dataset_root)
    output_dir = normalize_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    class_names = read_class_names(dataset_root)
    num_classes = len(class_names) + 1
    print(f"Classes: {class_names}")
    print(f"Using dataset: {dataset_root}")

    train_dataset = YoloDetectionDataset(dataset_root, split="train")
    val_dataset = YoloDetectionDataset(dataset_root, split="valid")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    device = torch.device(args.device)
    model = create_model(num_classes=num_classes, pretrained=args.pretrained).to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args.print_freq)
        val_loss = validate_loss(model, val_loader, device)
        scheduler.step()

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_names": class_names,
            "args": vars(args),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        last_path = output_dir / "last_model.pt"
        torch.save(checkpoint, last_path)

        if val_loss < best_val:
            best_val = val_loss
            best_path = output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

    print(f"Training complete. Best validation loss: {best_val:.4f}")
    print(f"Checkpoints saved in: {output_dir}")


if __name__ == "__main__":
    main()
