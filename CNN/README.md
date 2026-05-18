# Blood Cell Detection CNN (Faster R-CNN)

This project trains a CNN-based object detector on your YOLO-format blood-cell dataset:

- Dataset: `C:\Users\usEr\Downloads\Research\blood-cell-detection-datatset`
- Project: `C:\Users\usEr\Downloads\Research\CNN`

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

If you use GPU, install the CUDA-enabled PyTorch build that matches your CUDA version from the official PyTorch install page.

## 2) Train

```bash
python train_detector.py ^
  --dataset-root "C:\Users\usEr\Downloads\Research\blood-cell-detection-datatset" ^
  --output-dir "C:\Users\usEr\Downloads\Research\CNN\outputs" ^
  --epochs 20 ^
  --batch-size 2 ^
  --pretrained
```

On PowerShell, replace `^` with backtick `` ` `` or put everything on one line.

Training saves:

- `outputs\last_model.pt`
- `outputs\best_model.pt`

## 3) Run prediction on one image

Default quick run:

```bash
python predict.py
```

Tuned run (helps reduce duplicate RBC boxes):

```bash
python predict.py ^
  --checkpoint "C:\Users\usEr\Downloads\Research\CNN\outputs\best_model.pt" ^
  --image "C:\Users\usEr\Downloads\Research\blood-cell-detection-datatset\test\images\BloodImage_00038_jpg.rf.ffa23e4b5b55b523367f332af726eae8.jpg" ^
  --output "C:\Users\usEr\Downloads\Research\CNN\prediction.jpg" ^
  --score-threshold 0.6 ^
  --nms-iou-threshold 0.3
```

## 4) Evaluate on full test set

```bash
python evaluate.py ^
  --checkpoint "C:\Users\usEr\Downloads\Research\CNN\outputs\best_model.pt" ^
  --dataset-root "C:\Users\usEr\Downloads\Research\blood-cell-detection-datatset" ^
  --split test ^
  --score-threshold 0.6 ^
  --iou-threshold 0.5 ^
  --nms-iou-threshold 0.3
```

## 5) Auto-tune score/NMS thresholds

```bash
python sweep_thresholds.py ^
  --objective micro_f1
```

This runs a grid search over threshold pairs and saves:

- `outputs\sweep_results.json`
- `outputs\sweep_results.csv`

## Notes

- The script reads `data.yaml` and uses classes: `Platelets`, `RBC`, `WBC`.
- YOLO labels are automatically converted into Faster R-CNN bounding boxes.
- For lower VRAM, reduce `--batch-size` to `1`.
