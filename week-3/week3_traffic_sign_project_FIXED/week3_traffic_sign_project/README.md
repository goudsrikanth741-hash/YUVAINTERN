# Week 3 AI Project — AI-Based Traffic Sign Recognition System

A complete Python experiment framework for designing, training, evaluating, comparing and analyzing traffic-sign recognition models using the **GTSRB (German Traffic Sign Recognition Benchmark)**.

## Scope

This project is deliberately different from Pneumonia Detection and Fake News Detection. It treats traffic-sign recognition as a 43-class image-classification problem and implements four controlled experiments:

1. **Baseline CNN** — no augmentation, randomly initialized CNN.
2. **CNN + Data Augmentation** — same CNN with controlled image augmentation.
3. **ResNet18 Transfer Learning** — ImageNet-pretrained ResNet18 with augmentation.
4. **ResNet18 + Label Smoothing** — same transfer-learning setup with label smoothing.

The framework automatically produces metrics, confusion matrices, ROC curves, training/validation plots, experiment comparisons, predictions and an experiment-design summary.

## Dataset

Recommended public dataset: **GTSRB**. The loader uses `torchvision.datasets.GTSRB`, so the dataset can be downloaded automatically when internet access is available. It contains 43 traffic-sign classes.

If automatic download is unavailable, download GTSRB manually and adapt the dataset loader in `src/data.py` to your local directory.

## Project structure

```text
week3_traffic_sign_project/
├── configs/experiments.yaml
├── data/                         # created/downloaded dataset
├── outputs/                     # generated artifacts
├── scripts/
│   ├── run_experiments.py       # train all experiments
│   └── predict.py               # predict one image
├── src/
│   ├── config.py
│   ├── data.py
│   ├── models.py
│   ├── engine.py
│   ├── metrics.py
│   ├── visualization.py
│   ├── experiments.py
│   └── utils.py
├── tests/test_smoke.py
├── requirements.txt
└── README.md
```

## Setup

### Windows PowerShell

```powershell
cd path\to\week3_traffic_sign_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run a smoke test

```bash
python -m unittest discover -s tests -v
```

## Run all four experiments

```bash
python scripts/run_experiments.py --config configs/experiments.yaml
```

The first run can take substantial time because GTSRB must be downloaded and the four models are trained. For a quick test, copy the YAML and reduce `epochs` to 1, `batch_size` to 16 and set `num_workers` to 0.

Artifacts are written under `outputs/`:

- `results.csv` — one row per experiment
- `experiment_comparison.csv`
- `experiment_design.json`
- `outcomes_prediction.json`
- per-experiment `best_model.pt`
- `history.csv`
- `metrics.json`
- `confusion_matrix.png`
- `roc_auc.png`
- `training_curves.png`
- `test_predictions.csv`
- `experiment_comparison.png`

## Predict a single image

After training, use the saved model and class mapping:

```bash
python scripts/predict.py --checkpoint outputs/resnet18_label_smoothing/best_model.pt --image path/to/sign.png
```

## Experiment design

### Independent variables

- Model architecture: baseline CNN vs ResNet18.
- Data augmentation: off vs on.
- Transfer learning: off vs on.
- Label smoothing: 0.0 vs 0.1.

### Dependent variables

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Multiclass ROC-AUC (one-vs-rest, macro)
- Test loss
- Training time

### Control variables

The following are held constant unless explicitly listed as an independent variable:

- Dataset and class definitions
- Train/validation/test split
- Image size (64×64)
- Batch size
- Optimizer (Adam)
- Base learning rate
- Weight decay
- Epoch budget and early-stopping patience
- Evaluation code
- Random seed (42)
- Test set and preprocessing at evaluation time

### Expected outcomes

The baseline CNN establishes a reference point. Augmentation is expected to improve robustness to realistic changes such as small rotations and translations. ResNet18 transfer learning is expected to outperform the smaller CNN because of stronger learned visual features. Label smoothing is expected to reduce overconfidence and may improve macro-F1/ROC-AUC, although it can slightly reduce raw accuracy if over-regularized.

These are hypotheses, not guaranteed results. The generated experiment output is the authoritative result.

## Simulation environment and realistic scenarios

The test environment models several realistic variations through augmentation and the held-out test set:

- small rotations
- translation/cropping
- brightness/contrast variation
- mild perspective changes
- normalization differences
- class imbalance and visually similar signs

The core evaluation remains on the untouched test distribution. Augmentation is applied only to training images so test leakage is avoided.

## Data collection method

GTSRB is a public benchmark. Each image is assigned one of 43 sign classes. The framework records image paths/indices, labels, split membership, predictions, probabilities and evaluation metrics. No user-identifying information is required.

## Evaluation procedure

1. Fix the seed.
2. Create the same stratified train/validation split for every experiment.
3. Apply only the experiment's specified augmentation during training.
4. Train on the training split.
5. Select the checkpoint with the best validation accuracy.
6. Evaluate that checkpoint once on the held-out test set.
7. Calculate accuracy, macro precision, macro recall, macro F1 and macro one-vs-rest ROC-AUC.
8. Save the confusion matrix, ROC curve, training curves and predictions.
9. Compare all experiments in a common table and chart.

## Risks and challenges

- **Class imbalance:** accuracy alone can hide poor performance on rare classes; macro metrics are therefore included.
- **Visually similar signs:** confusion between related speed-limit/warning signs is expected.
- **Domain shift:** benchmark images may not represent every camera, country, weather or lighting condition.
- **Compute cost:** ResNet18 training can be slow on CPU; GPU is recommended.
- **Download/network issues:** automatic GTSRB download requires network access.
- **Overfitting:** early stopping, augmentation and regularization are used.
- **ROC-AUC edge cases:** if a class is absent from a particular test subset, ROC-AUC can become undefined; the implementation safely reports NaN for invalid cases.

## Reproducibility

The seed is fixed at 42 and applied to Python, NumPy and PyTorch. Deterministic settings are enabled where practical. GPU hardware, PyTorch version and runtime can still cause small numerical differences.

## Experiment workflow

```text
GTSRB
  │
  ▼
Dataset download/load
  │
  ▼
Stratified train/validation split ───────────────┐
  │                                              │
  ▼                                              │
Experiment configuration                          │
  │                                              │
  ├── Baseline CNN                               │
  ├── CNN + augmentation                         │
  ├── ResNet18 transfer learning                 │
  └── ResNet18 + label smoothing                 │
  │                                              │
  ▼                                              │
Training + validation ──► best checkpoint         │
  │                                              │
  ▼                                              │
Held-out test evaluation ◄────────────────────────┘
  │
  ├── Accuracy / Precision / Recall / F1
  ├── ROC-AUC
  ├── Confusion matrix
  ├── Training curves
  └── Predictions
  │
  ▼
Experiment comparison + outcome analysis
```

## References

- Stallkamp, J., Schlipsing, M., Salmen, J., Igel, C. (2012). *Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition*. Neural Networks.
- GTSRB benchmark: https://benchmark.ini.rub.de/gtsrb_news.html
- PyTorch: https://pytorch.org/
- Torchvision GTSRB dataset API: https://pytorch.org/vision/stable/generated/torchvision.datasets.GTSRB.html
- He, K. et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.

## Important note

This repository is the **Python project only**. No DOC/DOCX report is generated by this project, as requested.
