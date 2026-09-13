# Quick Start Guide — AI Traffic Sign Recognition

## 🚀 Get Running in 5 Minutes

### 1. Install Dependencies (2 minutes)

```bash
# Windows PowerShell
cd week3_traffic_sign_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# Linux/macOS
cd week3_traffic_sign_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train Models (30-60 minutes)

```bash
# Full training (4 experiments, ~50 min on GPU)
python train_models.py

# Quick test (~5 min, fewer epochs)
# Edit: configs/experiments.yaml
#   Change: epochs: 10 → epochs: 2
#   Change: batch_size: 64 → batch_size: 32
# Then run: python train_models.py
```

What happens:
- ✅ GTSRB dataset auto-downloads (~500MB)
- ✅ 4 models train with different configurations
- ✅ Models save to `outputs/<experiment_name>/best_model.pt`
- ✅ Training curves and metrics generated

### 3. Launch Web App (1 minute)

```bash
streamlit run app.py
```

Opens at: `http://localhost:8501`

---

## 📱 Web App Pages

| Page | What You Can Do |
|------|-----------------|
| **Home 🏠** | See project overview, system info |
| **Prediction 🎯** | Upload traffic sign images → get predictions |
| **Results 📊** | Compare all trained models side-by-side |
| **Graphs 📈** | View training curves, confusion matrices, ROC curves |

---

## ✅ Verification

Check everything is working:

```bash
python verify_setup.py
```

This checks:
- Python version ✓
- All dependencies ✓
- Project structure ✓
- Configuration ✓
- Device/CUDA ✓

---

## 🎯 Prediction Workflow

After training:

1. Open `http://localhost:8501`
2. Click "Prediction" in sidebar
3. Upload a traffic sign image (.jpg, .png)
4. See top-5 predictions with confidence scores

---

## 📊 Expected Results

After training completes, you should see:

**Experiment Comparison**:
- Baseline CNN: ~85% accuracy
- CNN + Augmentation: ~92% accuracy
- ResNet18 Transfer: ~96% accuracy
- ResNet18 + Label Smoothing: ~97% accuracy

---

## ⚡ Performance Tips

### Faster Training
```yaml
# configs/experiments.yaml
epochs: 2              # Instead of 10
batch_size: 32         # Instead of 64
num_workers: 0         # If data loading is slow
```

### Better Results
```yaml
epochs: 20             # More training
batch_size: 128        # Larger batches
learning_rate: 0.0005  # Slower learning
```

---

## 🆘 Troubleshooting

| Problem | Fix |
|---------|-----|
| "No trained model found" | Run `python train_models.py` |
| Disk space error | Reduce `batch_size` to 16 |
| Slow training | Reduce `epochs` to 2 |
| CUDA out of memory | Set `device: cpu` in config |
| Port 8501 in use | `streamlit run app.py --server.port 8502` |

---

## 📂 Important Directories

```
week3_traffic_sign_project/
├── data/              ← GTSRB dataset (auto-downloaded)
├── outputs/           ← Trained models and results
│   ├── baseline_cnn/
│   │   └── best_model.pt
│   ├── cnn_augmentation/
│   │   └── best_model.pt
│   ├── resnet18_transfer/
│   │   └── best_model.pt
│   └── resnet18_label_smoothing/
│       └── best_model.pt
├── configs/
│   └── experiments.yaml  ← Training configuration
└── app.py             ← Launch this with streamlit
```

---

## 📖 Full Documentation

- **SETUP_GUIDE.md** - Comprehensive setup instructions
- **FIX_SUMMARY.md** - Technical details of all fixes
- **README.md** - Original project documentation

---

## 💡 Key Points

✅ **Models auto-save** to `outputs/<experiment_name>/best_model.pt`  
✅ **App auto-loads** the latest trained model  
✅ **Results auto-display** once training finishes  
✅ **Predictions work** with any uploaded image  
✅ **Graphs auto-generate** during training  

---

## 🎓 Learning Resources

1. **Training Curves**: Shows model learning over time
2. **Confusion Matrix**: See which signs are confused with each other
3. **ROC Curves**: Model performance across all 43 classes
4. **Comparison Chart**: Which experiment setup works best?

---

**Ready to start?** → Run `python train_models.py`! 🚀
