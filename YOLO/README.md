# Blood Cell Detection

This repository contains utilities and saved reports for evaluating YOLO-based blood cell detection experiments. It is set up for workflows that download the Blood Cell Detection dataset, validate trained YOLO checkpoints, and compare training validation metrics against held-out test metrics.

## Repository Contents

- `dataloader.py` downloads the KaggleHub dataset `adhoppin/blood-cell-detection-datatset`.
- `YOLO/test/test.py` validates discovered `train_results*/weights/best.pt` checkpoints with Ultralytics and writes ranked CSV/JSON reports.
- `YOLO/evaluation/evaluation.py` compares training validation metrics with held-out test metrics and generates tables, Markdown reports, and plots.
- `YOLO/test/runs/` contains generated validation summary files and an auto-generated test data YAML.
- `YOLO/evaluation/outputs/` contains generated comparison reports and figures.

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

Training outputs are expected to follow this structure when running validation or evaluation locally:

```text
train_results*/
  args.yaml
  results.csv
  weights/
    best.pt
```

Large local artifacts, model checkpoints, caches, virtual environments, and downloaded datasets are ignored through `.gitignore`.

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

## Git Hygiene

Keep datasets, checkpoints, generated caches, and local runtime files out of commits. The existing `.gitignore` covers the common local artifacts for this project.
