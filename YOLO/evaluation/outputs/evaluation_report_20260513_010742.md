# YOLO Train vs Test Evaluation

Test report: `C:\Users\usEr\Downloads\Research\YOLO\test\runs\best_pt_validation_summary_20260513_005420.csv`
Runs compared: 12

## Best Held-Out Test Model

- Run: `train_results_150_epochs(yolov26m)`
- Test mAP50-95: 0.6111
- Test mAP50: 0.8921
- Test precision: 0.7942
- Test recall: 0.8901
- Train best mAP50-95: 0.6456
- Generalization gap: 0.0345

## Top Runs

| Rank | Run | Train best mAP50-95 | Test mAP50-95 | Test mAP50 | Test F1 | Gap |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `train_results_150_epochs(yolov26m)` | 0.6456 | 0.6111 | 0.8921 | 0.8394 | 0.0345 |
| 2 | `train_results_500epochs(yolov8m)` | 0.6472 | 0.6008 | 0.8824 | 0.8530 | 0.0463 |
| 3 | `train_results_500epochs_with_optimizer(yolov8m)` | 0.6472 | 0.6008 | 0.8824 | 0.8530 | 0.0463 |
| 4 | `train_results_150_epochs(yolov11m)` | 0.6583 | 0.5968 | 0.8849 | 0.8589 | 0.0615 |
| 5 | `train_results_150_epochs(yolov8m)` | 0.6519 | 0.5959 | 0.8790 | 0.8467 | 0.0559 |
| 6 | `train_results_150_epochs(yolov8l)` | 0.6544 | 0.5923 | 0.8746 | 0.8307 | 0.0621 |
| 7 | `train_results_500epochs(yolov11m)` | 0.6553 | 0.5918 | 0.8817 | 0.8468 | 0.0635 |
| 8 | `train_results_500epochs_with_optimizer(yolo11m)` | 0.6553 | 0.5918 | 0.8817 | 0.8468 | 0.0635 |
| 9 | `train_results_500epochs(yolo26m)` | 0.6485 | 0.5878 | 0.8697 | 0.8213 | 0.0607 |
| 10 | `train_results_500epochs_with_optimizer(yolo26m)` | 0.6485 | 0.5878 | 0.8697 | 0.8213 | 0.0607 |
| 11 | `train_results_500epochs(yolov8l)` | 0.6507 | 0.5873 | 0.8657 | 0.8395 | 0.0634 |
| 12 | `train_results_500epochs_with_optimizer(yolov8l)` | 0.6507 | 0.5873 | 0.8657 | 0.8395 | 0.0634 |

## Figures

- `train_vs_test_map50_95.png`
- `generalization_gap_map50_95.png`
- `train_test_map50_95_scatter.png`
- `test_metrics_heatmap.png`
- `test_map50_95_by_group.png`
