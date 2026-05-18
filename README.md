# Blood Cell Detection

This repository contains utilities and saved reports for blood-cell object detection experiments. It includes YOLO evaluation utilities plus a Faster R-CNN CNN baseline with threshold tuning outputs.

## Repository Contents

- `dataloader.py` downloads the KaggleHub dataset `adhoppin/blood-cell-detection-datatset`.
- `YOLO/test/test.py` validates discovered `train_results*/weights/best.pt` checkpoints with Ultralytics and writes ranked CSV/JSON reports.
- `YOLO/evaluation/evaluation.py` compares training validation metrics with held-out test metrics and generates tables, Markdown reports, and plots.
- `YOLO/test/runs/` contains generated validation summary files and an auto-generated test data YAML.
- `YOLO/evaluation/outputs/` contains generated comparison reports and figures.
- `CNN/train_detector.py` trains a Faster R-CNN detector from the same YOLO-format labels.
- `CNN/evaluate.py` evaluates the CNN detector on a selected split.
- `CNN/sweep_thresholds.py` searches score and NMS threshold pairs.
- `CNN/record.md` records the CNN threshold-tuning history.
- `CNN/outputs/` tracks lightweight metrics and sweep summaries. Model checkpoints remain local.

## Dataset

The raw dataset is intentionally excluded from Git. Download it with:

```bash
python dataloader.py
```

By default, the project expects the dataset directory to be named:

```text
blood-cell-detection-datatset/
```

> Note: The directory name preserves the spelling used by the KaggleHub dataset slug.

## Expected Local Artifacts

YOLO training outputs are expected to follow this structure when running validation or evaluation locally:

```text
train_results*/
  args.yaml
  results.csv
  weights/
    best.pt
```

CNN checkpoints are expected locally in `CNN/outputs/best_model.pt` and `CNN/outputs/last_model.pt`.

Large local artifacts, model checkpoints, caches, virtual environments, downloaded datasets, and local presentation exports are ignored through `.gitignore`.

## Validate YOLO Checkpoints

From the repository root, run:

```bash
python YOLO/test/test.py
```

Useful options include:

```bash
python YOLO/test/test.py --test-path /path/to/test --device 0
python YOLO/test/test.py --data /path/to/data.yaml --batch 16 --imgsz 640
python YOLO/test/test.py --metric map50_95 --no-plots
```

The validation script writes ranked summaries to `YOLO/test/runs/`.

## Train And Evaluate The CNN Detector

Install the CNN dependencies:

```bash
pip install -r CNN/requirements.txt
```

Train the Faster R-CNN detector:

```bash
python CNN/train_detector.py --dataset-root blood-cell-detection-datatset --output-dir CNN/outputs
```

Evaluate the test split:

```bash
python CNN/evaluate.py --checkpoint CNN/outputs/best_model.pt --dataset-root blood-cell-detection-datatset --split test
```

Run the threshold sweep:

```bash
python CNN/sweep_thresholds.py --objective micro_f1
```

The tracked CNN outputs include `CNN/outputs/eval_test_metrics.json`, `CNN/outputs/sweep_results.csv`, and `CNN/outputs/sweep_results.json`.

## Generate Train-vs-Test Evaluation Reports

After validation summaries exist, run:

```bash
python YOLO/evaluation/evaluation.py
```

To evaluate a specific validation summary:

```bash
python YOLO/evaluation/evaluation.py --test-report YOLO/test/runs/best_pt_validation_summary_YYYYMMDD_HHMMSS.csv
```

Evaluation outputs are written to `YOLO/evaluation/outputs/` and include:

- train/test comparison CSV and JSON files
- a Markdown evaluation report
- plots for held-out test metrics, generalization gap, train-vs-test scatter, and model group comparisons

## Python Dependencies

The scripts use these Python packages:

- `kagglehub`
- `ultralytics`
- `pandas`
- `matplotlib`
- `seaborn` (optional; the evaluation script falls back to Matplotlib when unavailable)

Install them in your preferred virtual environment, for example:

```bash
python -m pip install kagglehub ultralytics pandas matplotlib seaborn
```

## Typical Workflow

1. Install the Python dependencies.
2. Download the dataset with `python dataloader.py`.
3. Add or generate local `train_results*/` experiment folders with `weights/best.pt` checkpoints.
4. Validate checkpoints with `python YOLO/test/test.py`.
5. Generate comparison reports with `python YOLO/evaluation/evaluation.py`.
6. Train or evaluate the CNN baseline with the scripts in `CNN/`.

## Git Hygiene

Keep datasets, checkpoints, generated caches, local runtime files, and presentation exports out of commits. The existing `.gitignore` covers the common local artifacts for this project.
