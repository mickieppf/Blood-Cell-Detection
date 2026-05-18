# CNN Blood Cell Detection - Result Record

Use this file to track each experiment side-by-side.

## Run Comparison

| Item | Baseline (Run 1) | Run 2 | Run 3 | Run 4 | Run 5 | Run 6 |
|---|---:|---:|---:|---:|---:|---:|
| Checkpoint | `outputs/best_model.pt` | `outputs/best_model.pt` | `outputs/best_model.pt` | `outputs/best_model.pt` | `outputs/best_model.pt` | `outputs/best_model.pt` |
| Split | `test` (36 images) | `test` (36 images) | `test` (36 images) | `test` (36 images) | `test` (36 images) | `test` (36 images) |
| Score Threshold | 0.50 | 0.60 | 0.65 | 0.62 | 0.63 | 0.64 |
| IoU Threshold | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| NMS IoU Threshold | 0.30 | 0.25 | 0.20 | 0.22 | 0.21 | 0.205 |
| Model Score Threshold | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |
| Micro Precision | 0.617689 | 0.656394 | 0.684385 | 0.664025 | 0.671521 | 0.679803 |
| Micro Recall | 0.919321 | 0.904459 | 0.874735 | 0.889597 | 0.881104 | 0.878981 |
| Micro F1 | 0.738908 | 0.760714 | 0.767940 | 0.760436 | 0.762167 | 0.766667 |
| mAP50 | 0.870394 | 0.861847 | 0.853943 | 0.857855 | 0.855651 | 0.855112 |
| TP | 433 | 426 | 412 | 419 | 415 | 414 |
| FP | 268 | 223 | 190 | 212 | 203 | 195 |
| FN | 38 | 45 | 59 | 52 | 56 | 57 |

## Per-Class Comparison

| Class | Run 1 P | Run 1 R | Run 1 F1 | Run 1 AP50 | Run 2 P | Run 2 R | Run 2 F1 | Run 2 AP50 | Run 3 P | Run 3 R | Run 3 F1 | Run 3 AP50 | Run 4 P | Run 4 R | Run 4 F1 | Run 4 AP50 | Run 5 P | Run 5 R | Run 5 F1 | Run 5 AP50 | Run 6 P | Run 6 R | Run 6 F1 | Run 6 AP50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Platelets | 0.622642 | 0.916667 | 0.741573 | 0.813582 | 0.653061 | 0.888889 | 0.752941 | 0.796286 | 0.653061 | 0.888889 | 0.752941 | 0.796286 | 0.653061 | 0.888889 | 0.752941 | 0.796286 | 0.653061 | 0.888889 | 0.752941 | 0.796286 | 0.653061 | 0.888889 | 0.752941 | 0.796286 |
| RBC | 0.595745 | 0.914573 | 0.721506 | 0.825359 | 0.635879 | 0.899497 | 0.745057 | 0.817012 | 0.666667 | 0.864322 | 0.752735 | 0.793301 | 0.644037 | 0.881910 | 0.744433 | 0.805035 | 0.652256 | 0.871859 | 0.746237 | 0.798423 | 0.661568 | 0.869347 | 0.751357 | 0.796806 |
| WBC | 0.972973 | 0.972973 | 0.972973 | 0.972243 | 0.972973 | 0.972973 | 0.972973 | 0.972243 | 0.972973 | 0.972973 | 0.972973 | 0.972243 | 0.972973 | 0.972973 | 0.972973 | 0.972243 | 0.972973 | 0.972973 | 0.972973 | 0.972243 | 0.972973 | 0.972973 | 0.972973 | 0.972243 |

## Delta (Run 6 - Run 1)

| Metric | Change |
|---|---:|
| Micro Precision | +0.062114 |
| Micro Recall | -0.040340 |
| Micro F1 | +0.027759 |
| mAP50 | -0.015282 |
| TP | -19 |
| FP | -73 |
| FN | +19 |

## Delta (Run 6 - Run 5)

| Metric | Change |
|---|---:|
| Micro Precision | +0.008282 |
| Micro Recall | -0.002123 |
| Micro F1 | +0.004500 |
| mAP50 | -0.000539 |
| TP | -1 |
| FP | -8 |
| FN | +1 |

## Notes

- Baseline issue: high false positives in `RBC` (247 FP).
- Run 3 still has the best strict precision/F1 overall (`P=0.684385`, `F1=0.767940`).
- Run 6 is close to Run 3 F1 with slightly lower precision, but higher recall than Run 3.
- Run 6 improves over Run 5 in precision/F1 and further lowers false positives.
- Final decision should be based on an automatic sweep and then lock the best config by chosen objective.
- Current default pick (before sweep): **Run 3** (`score=0.65`, `nms=0.20`) for best precision/F1.

## Next Commands (Auto Sweep)

```bash
python "C:\Users\usEr\Downloads\Research\CNN\sweep_thresholds.py" --objective micro_f1
```

Then use the best pair from sweep output:

```bash
python "C:\Users\usEr\Downloads\Research\CNN\evaluate.py" --score-threshold <best_score> --nms-iou-threshold <best_nms>
python "C:\Users\usEr\Downloads\Research\CNN\predict.py" --score-threshold <best_score> --nms-iou-threshold <best_nms>
```
