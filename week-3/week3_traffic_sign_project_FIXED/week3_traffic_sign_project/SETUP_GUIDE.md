# Week 3 AI Project — AI-Based Traffic Sign Recognition System

A complete Python experiment framework for designing, training, evaluating, comparing and analyzing traffic-sign recognition models using the **GTSRB (German Traffic Sign Recognition Benchmark)** with an interactive **Streamlit web application**.

## Scope

This project implements four controlled experiments for traffic-sign recognition (43 classes):

1. **Baseline CNN** — no augmentation, randomly initialized CNN
2. **CNN + Data Augmentation** — same CNN with controlled image augmentation
3. **ResNet18 Transfer Learning** — ImageNet-pretrained ResNet18 with augmentation
4. **ResNet18 + Label Smoothing** — transfer-learning setup with label smoothing (0.1)

The framework automatically produces metrics, confusion matrices, ROC curves, training plots, and an interactive web application for predictions and analysis.

## Dataset

**GTSRB** (German Traffic Sign Recognition Benchmark) - automatically downloaded via `torchvision.datasets.GTSRB`.
- **43 traffic sign classes**
- **50,000+ training samples**
- **12,600 test samples**
- **64×64 RGB images**

## Project Structure

```
week3_traffic_sign_project/
├── app.py                          # Streamlit web application
├── train_models.py                 # Training script (generates models)
├── configs/
│   └── experiments.yaml            # Experiment configuration
├── data/                           # Dataset directory (auto-created)
├── outputs/                        # Results, models, visualizations
├── scripts/
│   ├── run_experiments.py         # Alternative training script
│   └── predict.py                 # Single image prediction CLI
├── src/
│   ├── config.py                  # Configuration loading
│   ├── data.py                    # Dataset and transforms
│   ├── models.py                  # CNN and ResNet models
│   ├── engine.py                  # Training loop
│   ├── experiments.py             # Experiment runner
│   ├── metrics.py                 # Evaluation metrics
│   ├── visualization.py           # Plotting functions
│   ├── utils.py                   # Helper functions
│   ├── training.py               # Training utilities
│   ├── validation.py              # Validation utilities
│   ├── testing.py                 # Testing utilities
│   └── __init__.py
├── tests/
│   └── test_smoke.py
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Setup Environment

#### Windows PowerShell
```powershell
cd path\to\week3_traffic_sign_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Linux/macOS
```bash
cd week3_traffic_sign_project
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Train Models

Run all experiments and generate visualizations:

```bash
python train_models.py
```

This will:
- Download GTSRB dataset (first run only, ~500MB)
- Train 4 experiments with different configurations
- Generate training curves, confusion matrices, ROC curves
- Save trained models to `outputs/<experiment_name>/best_model.pt`
- Create comparison metrics and visualizations
- Estimated time: 30-60 minutes (varies by GPU)

**For quick testing** (reduce training time):
```bash
python train_models.py --config configs/experiments.yaml
# Then edit configs/experiments.yaml:
#   epochs: 1-2
#   batch_size: 32-64
#   num_workers: 0
```

### 3. Launch Web Application

After training completes, start the Streamlit app:

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Web Application Pages

### Home 🏠
- Project overview and statistics
- Experiment descriptions
- System information
- Dataset status

### Prediction 🎯
- Upload traffic sign images
- Real-time model inference
- Top-5 predictions with confidence scores
- Visual confidence chart
- **Requires at least one trained model**

### Experiment Results 📊
- Comparison table of all experiments
- Key metrics for each model:
  - Accuracy
  - Precision (macro)
  - Recall (macro)
  - F1-Score (macro)
  - ROC-AUC (macro)
- Interactive performance charts
- **Populated after training completes**

### Training Graphs 📈
- Training & validation loss curves
- Confusion matrices
- ROC curves (one-vs-rest)
- Training history tables
- Experiment comparison chart
- **Viewable after training**

## Alternative: Command-Line Training

For headless or automated training:

```bash
python scripts/run_experiments.py --config configs/experiments.yaml
```

## Single Image Prediction (CLI)

After training, predict on a single image:

```bash
python scripts/predict.py \
  --checkpoint outputs/resnet18_label_smoothing/best_model.pt \
  --image path/to/traffic_sign.png \
  --model resnet18 \
  --image-size 64
```

Output:
```
Top-5 predictions:
class 38: 0.9543
class 37: 0.0312
class 12: 0.0089
class 25: 0.0042
class 5: 0.0014
```

## Configuration

Edit `configs/experiments.yaml` to customize:

```yaml
# Model training parameters
epochs: 10              # Number of training epochs
batch_size: 64         # Batch size for training
learning_rate: 0.001   # Learning rate (Adam optimizer)
weight_decay: 0.0001   # L2 regularization

# Dataset parameters
image_size: 64         # Input image size (64×64)
val_split: 0.15        # Validation split (15%)
seed: 42               # Random seed for reproducibility

# Optimization
patience: 3            # Early stopping patience
num_workers: 2         # Data loading workers
device: auto           # 'auto', 'cuda', or 'cpu'

# Experiments to run
experiments:
  baseline_cnn:
    model: baseline_cnn
    augmentation: false
    label_smoothing: 0.0
    pretrained: false
  # ... more experiments
```

## Experiment Design

### Independent Variables
- **Model architecture**: Baseline CNN vs ResNet18
- **Data augmentation**: Off vs on
- **Transfer learning**: Off vs on  
- **Label smoothing**: 0.0 vs 0.1

### Dependent Variables
- Accuracy
- Precision (macro average)
- Recall (macro average)
- F1-Score (macro average)
- ROC-AUC (one-vs-rest, macro)
- Training time
- Test loss

### Control Variables (held constant)
- Dataset: GTSRB, 43 classes
- Image size: 64×64
- Optimizer: Adam
- Loss function: CrossEntropyLoss
- Train/val/test split: 80/5/15
- Evaluation metrics and code
- Random seed: 42

### Expected Outcomes

- **Baseline CNN**: Establishes reference performance
- **With Augmentation**: Improved robustness to rotations, translations, color shifts
- **ResNet18 Transfer**: Higher accuracy due to ImageNet pretrained features
- **Label Smoothing**: Improved calibration, potentially better macro-F1

## Output Artifacts

After `python train_models.py`, the `outputs/` directory contains:

### Per-Experiment (in each experiment folder)
```
outputs/<experiment_name>/
├── best_model.pt              # Trained model (PyTorch)
├── metrics.json               # Evaluation metrics
├── history.csv                # Training history (loss, accuracy)
├── training_curves.png        # Train/val loss plot
├── confusion_matrix.png       # Confusion matrix heatmap
├── roc_auc.png                # ROC curves (one-vs-rest)
└── test_predictions.csv       # Predictions on test set
```

### Overall Results
```
outputs/
├── results.csv                # Summary of all experiments
├── experiment_comparison.csv   # Experiment metrics table
├── experiment_comparison.png   # Bar chart comparing experiments
├── experiment_design.json      # Experiment design documentation
└── outcomes_prediction.json    # Expected outcomes and interpretation
```

## Testing

Run smoke tests:
```bash
python -m pytest tests/ -v
```

Or using unittest:
```bash
python -m unittest discover -s tests -v
```

## Troubleshooting

### "No trained model was found"
- **Solution**: Run `python train_models.py` first to generate models

### GTSRB dataset download fails
- **Cause**: No internet connection or torchvision issue
- **Solution**: Download GTSRB manually or set `num_workers: 0` in config

### Out of memory errors
- **Solution**: Reduce `batch_size` in `configs/experiments.yaml` (e.g., 32 or 16)

### Training is slow
- **Solution**: 
  - Reduce `epochs` (2-3 for testing)
  - Reduce `batch_size` if GPU is idle
  - Set `num_workers: 0` if data loading is slow

### Streamlit port 8501 already in use
```bash
streamlit run app.py --server.port 8502
```

## System Requirements

- **Python 3.8+**
- **4GB RAM minimum** (8GB+ recommended for GPU)
- **GPU optional** but recommended (NVIDIA CUDA 11.8+)
- **Storage**: ~2GB for dataset + models

## Performance Expectations

Training times on different hardware:

| Hardware | Config | Time/Experiment | Total (4 exp) |
|----------|--------|-----------------|---------------|
| CPU (i7) | Full   | 20-30 min       | 90-120 min    |
| NVIDIA A100 | Full | 2-3 min         | 10-15 min     |
| NVIDIA T4 | Full  | 5-8 min         | 25-35 min     |
| NVIDIA RTX 3080 | Full | 3-5 min    | 15-20 min     |

## Code Quality

- Type hints throughout
- Comprehensive error handling
- Logging at each step
- Reproducible experiments (seeding)
- Modular architecture

## License

This project is provided as-is for educational purposes.

## Citation

If using this project, please reference:

```
@misc{week3_traffic_signs_2024,
  title={AI-Based Traffic Sign Recognition System},
  author={AI Project Team},
  year={2024},
  publisher={GitHub}
}
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review error messages in terminal output
3. Check `outputs/` directory for generated logs
4. Verify configuration in `configs/experiments.yaml`

---

**Last Updated**: 2024
**Framework**: PyTorch + Streamlit
**Dataset**: GTSRB
**Classes**: 43 traffic sign types
