"""
AI-Based Traffic Sign Recognition — Streamlit App
==================================================
Pipeline: Data Collection -> Preprocessing -> Augmentation -> Model Training
          -> Validation -> Testing -> Prediction -> Experiment Comparison

Every stage above is its own page in the sidebar. Every stage that *does*
something (download data, preview a transform, train, validate, test,
predict) is triggered by an explicit button — nothing runs silently just
because you opened a page or uploaded a file.

Experiments are NOT hardcoded: the Model Training page lets you tick which
of the experiments defined in configs/experiments.yaml you actually want to
run, and every other page that needs a trained model lets you pick which
trained experiment to use.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config
from src.utils import seed_everything, choose_device
from src.data import (
    GTSRB_MEAN, GTSRB_STD, NUM_CLASSES, get_class_name,
    collect_dataset, sample_raw_images, preview_preprocessing,
    preview_augmentations, make_dataloaders,
)
from src.experiments import run_all
from src.validation import validate
from src.testing import test_model
from src.inference import list_trained_experiments, load_experiment_model, predict_image
from src.visualization import save_confusion_matrix, save_roc_curve

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="AI Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main { padding: 1.5rem 2rem; }
.metric-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0; }
.success-box { background-color: #d4edda; padding: 15px; border-radius: 5px; border-left: 4px solid #28a745; }
.error-box { background-color: #f8d7da; padding: 15px; border-radius: 5px; border-left: 4px solid #dc3545; }
.info-box { background-color: #d1ecf1; padding: 15px; border-radius: 5px; border-left: 4px solid #17a2b8; }
.stage-badge { display:inline-block; padding: 3px 10px; border-radius: 12px; background:#e9ecef; font-size:0.8rem; margin-right:6px; }
</style>
""", unsafe_allow_html=True)

PIPELINE_STAGES = [
    ("Home", "🏠"),
    ("Data Collection", "📥"),
    ("Preprocessing", "🔧"),
    ("Augmentation", "🎨"),
    ("Model Training", "🏋️"),
    ("Validation", "✅"),
    ("Testing", "🧪"),
    ("Prediction", "🎯"),
    ("Experiment Comparison", "📊"),
]
STAGE_LABELS = [f"{icon} {name}" for name, icon in PIPELINE_STAGES]

CONFIG_PATH = "configs/experiments.yaml"
OUTPUT_DIR = "outputs"
DATA_DIR = "data"


# ============================================================
# Cached / session helpers
# ============================================================
@st.cache_resource
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_base_cfg():
    """Load the experiment config fresh (cheap YAML read) and normalize the
    device field so every page works from the same settings."""
    cfg = load_config(CONFIG_PATH)
    cfg["device"] = choose_device(cfg.get("device", "auto"))
    return cfg


def load_results_df():
    results_csv = Path(OUTPUT_DIR) / "results.csv"
    if results_csv.exists():
        return pd.read_csv(results_csv)
    return pd.DataFrame()


def experiment_metrics(exp_name):
    path = Path(OUTPUT_DIR) / exp_name / "metrics.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def goto(stage_name):
    st.session_state["nav"] = stage_name
    st.rerun()


if "nav" not in st.session_state:
    st.session_state["nav"] = "Home"

# ============================================================
# Sidebar navigation
# ============================================================
st.sidebar.markdown("# 🚦 Traffic Sign AI")
st.sidebar.caption("Data Collection → Preprocessing → Augmentation → Model Training → Validation → Testing → Prediction → Experiment Comparison")

current_index = [n for n, _ in PIPELINE_STAGES].index(st.session_state["nav"]) if st.session_state["nav"] in [n for n, _ in PIPELINE_STAGES] else 0
selected_label = st.sidebar.radio("Pipeline stage", STAGE_LABELS, index=current_index)
page = selected_label.split(" ", 1)[1]
st.session_state["nav"] = page

st.sidebar.divider()
device = get_device()
st.sidebar.markdown(f"**Device:** `{device}`")
trained = list_trained_experiments(OUTPUT_DIR)
st.sidebar.markdown(f"**Trained experiments:** {len(trained)}")
if trained:
    for t in trained:
        st.sidebar.markdown(f"- ✅ {t}")

# ============================================================
# HOME
# ============================================================
if page == "Home":
    st.markdown("# 🚦 AI-Based Traffic Sign Recognition System")
    st.markdown("### Week 3 — Simulation Environment & Experiment Design")
    st.write("An end-to-end deep learning pipeline for GTSRB traffic sign classification, "
             "built with PyTorch and served through this Streamlit app.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Traffic Sign Classes", NUM_CLASSES)
    col2.metric("Dataset", "GTSRB")
    col3.metric("Framework", "PyTorch")

    st.divider()
    st.markdown("## Project Pipeline")
    st.write("Click a stage below to jump straight to it.")

    cols = st.columns(4)
    for i, (name, icon) in enumerate(PIPELINE_STAGES[1:]):
        with cols[i % 4]:
            if st.button(f"{icon}  {name}", use_container_width=True, key=f"home_jump_{name}"):
                goto(name)

    st.divider()
    st.markdown("## Implemented Experiments")
    cfg = get_base_cfg()
    rows = []
    for name, exp in cfg["experiments"].items():
        rows.append({
            "Experiment": name,
            "Model": exp["model"],
            "Augmentation": "Yes" if exp["augmentation"] else "No",
            "Pretrained": "Yes" if exp["pretrained"] else "No",
            "Label Smoothing": exp["label_smoothing"],
            "Status": "✅ Trained" if name in trained else "⬜ Not trained yet",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.caption("Experiments are configurable in `configs/experiments.yaml` and selectable "
               "(you don't have to run all of them) on the **Model Training** page.")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### System Info")
        st.success(f"Using device: {device}")
    with col2:
        st.markdown("### Dataset Location")
        data_dir = Path(DATA_DIR)
        if data_dir.exists() and any(data_dir.iterdir()):
            st.success(f"✓ Data directory exists: {data_dir}")
        else:
            st.info("Dataset not downloaded yet — use the **Data Collection** page.")

# ============================================================
# 1. DATA COLLECTION
# ============================================================
elif page == "Data Collection":
    st.markdown("# 📥 Stage 1 — Data Collection")
    st.write("Download (or verify) the GTSRB dataset and inspect its class balance and sample images.")

    if st.button("📥 Download / Verify Dataset", type="primary"):
        with st.spinner("Downloading / verifying GTSRB (first run can take a few minutes)…"):
            try:
                stats = collect_dataset(DATA_DIR, download=True)
                st.session_state["dataset_stats"] = stats
                st.success("Dataset ready.")
            except Exception as e:
                st.error(f"Could not download/verify the dataset: {e}")

    stats = st.session_state.get("dataset_stats")
    if stats:
        c1, c2, c3 = st.columns(3)
        c1.metric("Training images", stats["num_train"])
        c2.metric("Test images", stats["num_test"])
        c3.metric("Classes", stats["num_classes"])

        st.divider()
        st.markdown("### Class Distribution (training set)")
        dist = stats["class_distribution"]
        dist_df = pd.DataFrame({
            "class": list(dist.keys()),
            "count": list(dist.values()),
        })
        st.bar_chart(dist_df.set_index("class"))

        st.divider()
        st.markdown("### Sample Images")
        if st.button("🔀 Show New Random Samples"):
            st.session_state["data_sample_seed"] = int(time.time())
        seed = st.session_state.get("data_sample_seed", 42)
        try:
            samples = sample_raw_images(DATA_DIR, n=6, seed=seed)
            cols = st.columns(6)
            for i, (img, label) in enumerate(samples):
                with cols[i]:
                    st.image(img, use_container_width=True)
                    st.caption(f"{get_class_name(label)} ({label})")
        except Exception as e:
            st.warning(f"Could not load sample images: {e}")
    else:
        st.info("Click **Download / Verify Dataset** above to fetch GTSRB and see stats here.")

# ============================================================
# 2. PREPROCESSING
# ============================================================
elif page == "Preprocessing":
    st.markdown("# 🔧 Stage 2 — Preprocessing")
    st.write("Preview how a raw image is resized, converted to a tensor, and normalized "
             "before it reaches the model.")

    image_size = st.selectbox("Target image size", [32, 48, 64, 128], index=2)
    st.caption(f"Normalization — mean: `{GTSRB_MEAN}`, std: `{GTSRB_STD}`")

    source = st.radio("Preview image source", ["Random dataset sample", "Upload my own image"], horizontal=True)

    preview_image = None
    if source == "Random dataset sample":
        if st.button("🔀 Pick Random Sample"):
            st.session_state["preprocess_seed"] = int(time.time())
        seed = st.session_state.get("preprocess_seed", 7)
        try:
            samples = sample_raw_images(DATA_DIR, n=1, seed=seed)
            preview_image = samples[0][0]
        except Exception as e:
            st.info("Dataset not downloaded yet — go to **Data Collection** first, "
                    "or upload your own image instead.")
    else:
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            preview_image = Image.open(uploaded)

    if preview_image is not None and st.button("🔧 Preview Preprocessing", type="primary"):
        resized, denorm = preview_preprocessing(preview_image, image_size=image_size)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**1. Original**")
            st.image(preview_image, use_container_width=True)
        with c2:
            st.markdown(f"**2. Resized ({image_size}×{image_size})**")
            st.image(resized, use_container_width=True)
        with c3:
            st.markdown("**3. Tensor → Normalized → De-normalized for display**")
            st.image(denorm, use_container_width=True)
        st.markdown("""
**Pipeline applied:**
1. `Resize((image_size, image_size))`
2. `ToTensor()` — scales pixel values to `[0, 1]`
3. `Normalize(mean, std)` — per-channel standardization using GTSRB statistics
        """)

# ============================================================
# 3. AUGMENTATION
# ============================================================
elif page == "Augmentation":
    st.markdown("# 🎨 Stage 3 — Data Augmentation")
    st.write("Toggle augmentation operations and preview several randomly-augmented "
             "versions of a sample image. These are the operations applied to the "
             "**training set only** when an experiment has `augmentation: true`.")

    col1, col2, col3 = st.columns(3)
    rotation = col1.checkbox("Random rotation (±12°)", value=True)
    affine = col2.checkbox("Random affine (translate/scale)", value=True)
    color_jitter = col3.checkbox("Color jitter (brightness/contrast)", value=True)
    image_size = st.selectbox("Image size", [32, 48, 64, 128], index=2, key="aug_size")
    n_previews = st.slider("Number of augmented previews", 3, 12, 6)

    source = st.radio("Base image source", ["Random dataset sample", "Upload my own image"], horizontal=True, key="aug_source")
    base_image = None
    if source == "Random dataset sample":
        if st.button("🔀 Pick Random Base Image"):
            st.session_state["aug_seed"] = int(time.time())
        seed = st.session_state.get("aug_seed", 3)
        try:
            samples = sample_raw_images(DATA_DIR, n=1, seed=seed)
            base_image = samples[0][0]
        except Exception:
            st.info("Dataset not downloaded yet — go to **Data Collection** first, "
                    "or upload your own image instead.")
    else:
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="aug_upload")
        if uploaded is not None:
            base_image = Image.open(uploaded)

    if base_image is not None:
        st.markdown("**Base image**")
        st.image(base_image, width=160)

        if st.button("🎨 Generate Augmented Previews", type="primary"):
            if not (rotation or affine or color_jitter):
                st.warning("All augmentation operations are off — previews will only show the resize step.")
            previews = preview_augmentations(
                base_image, image_size=image_size, n=n_previews,
                rotation=rotation, affine=affine, color_jitter=color_jitter,
            )
            cols = st.columns(min(6, n_previews))
            for i, img in enumerate(previews):
                with cols[i % len(cols)]:
                    st.image(img, use_container_width=True)
                    st.caption(f"Sample {i + 1}")

# ============================================================
# 4. MODEL TRAINING
# ============================================================
elif page == "Model Training":
    st.markdown("# 🏋️ Stage 4 — Model Training")
    st.write("Select which experiment(s) to train — you are **not** required to run all of them.")

    cfg = get_base_cfg()
    exp_names = list(cfg["experiments"].keys())

    st.markdown("### 1. Choose experiments to run")
    cols = st.columns(len(exp_names))
    selected = []
    for i, name in enumerate(exp_names):
        exp = cfg["experiments"][name]
        default_checked = name not in trained
        with cols[i]:
            checked = st.checkbox(
                name, value=default_checked, key=f"train_pick_{name}",
                help=f"model={exp['model']}, augmentation={exp['augmentation']}, "
                     f"pretrained={exp['pretrained']}, label_smoothing={exp['label_smoothing']}",
            )
            st.caption(("✅ already trained" if name in trained else "⬜ not trained yet"))
            if checked:
                selected.append(name)

    st.markdown("### 2. Training settings")
    c1, c2, c3, c4 = st.columns(4)
    epochs = c1.number_input("Epochs", min_value=1, max_value=200, value=int(cfg["epochs"]))
    batch_size = c2.number_input("Batch size", min_value=4, max_value=512, value=int(cfg["batch_size"]), step=4)
    image_size = c3.selectbox("Image size", [32, 48, 64, 128], index=[32, 48, 64, 128].index(cfg["image_size"]) if cfg["image_size"] in [32, 48, 64, 128] else 2)
    patience = c4.number_input("Early-stop patience", min_value=1, max_value=50, value=int(cfg["patience"]))

    st.caption("⚠️ Training downloads the ~300MB GTSRB dataset on first run and can take a while "
               "per experiment, especially on CPU.")

    start = st.button("🚀 Start Training", type="primary", disabled=(len(selected) == 0))
    if len(selected) == 0:
        st.info("Select at least one experiment above to enable training.")

    if start:
        cfg["epochs"] = int(epochs)
        cfg["batch_size"] = int(batch_size)
        cfg["image_size"] = int(image_size)
        cfg["patience"] = int(patience)
        seed_everything(cfg["seed"])

        st.markdown("### Training progress")
        placeholders = {}
        progress_bars = {}
        for name in selected:
            with st.expander(f"📌 {name}", expanded=True):
                progress_bars[name] = st.progress(0.0, text="Waiting to start…")
                placeholders[name] = st.empty()

        def progress_cb(exp_name, exp_idx, exp_total, epoch, epoch_total, row):
            frac = epoch / epoch_total
            progress_bars[exp_name].progress(
                min(frac, 1.0),
                text=f"Epoch {epoch}/{epoch_total} — train_loss={row['train_loss']:.4f}, "
                     f"val_loss={row['val_loss']:.4f}, val_accuracy={row['val_accuracy']:.2%}"
                     f"{'  ⭐ new best' if row.get('improved') else ''}",
            )
            placeholders[exp_name].markdown(
                f"Epoch **{epoch}/{epoch_total}** · train_loss **{row['train_loss']:.4f}** · "
                f"val_loss **{row['val_loss']:.4f}** · val_accuracy **{row['val_accuracy']:.2%}** · "
                f"best_val_accuracy **{row['best_val_accuracy']:.2%}**"
            )

        try:
            with st.spinner(f"Training {len(selected)} experiment(s)…"):
                df = run_all(cfg, experiment_names=selected, progress_cb=progress_cb)
            for name in selected:
                progress_bars[name].progress(1.0, text="Done ✅")
            st.success(f"Training complete for: {', '.join(selected)}")
            st.dataframe(df, use_container_width=True)
            st.cache_resource.clear()
        except Exception as e:
            st.error(f"Training failed: {e}")

# ============================================================
# 5. VALIDATION
# ============================================================
elif page == "Validation":
    st.markdown("# ✅ Stage 5 — Validation")
    st.write("Inspect the per-epoch validation curve recorded during training, or re-run "
             "validation on the current best checkpoint.")

    if not trained:
        st.info("No trained experiments yet. Go to **Model Training** first.")
    else:
        exp_choice = st.selectbox("Select a trained experiment", trained)
        history_path = Path(OUTPUT_DIR) / exp_choice / "history.csv"

        if history_path.exists():
            hist_df = pd.read_csv(history_path)
            best_row = hist_df.loc[hist_df["val_accuracy"].idxmax()]
            c1, c2, c3 = st.columns(3)
            c1.metric("Epochs trained", int(hist_df["epoch"].max()))
            c2.metric("Best val accuracy", f"{best_row['val_accuracy']:.2%}")
            c3.metric("Best epoch", int(best_row["epoch"]))

            st.markdown("### Training vs Validation Loss")
            st.line_chart(hist_df.set_index("epoch")[["train_loss", "val_loss"]])
            st.markdown("### Validation Accuracy per Epoch")
            st.line_chart(hist_df.set_index("epoch")[["val_accuracy"]])
        else:
            st.warning("No history.csv found for this experiment.")

        st.divider()
        if st.button("▶️ Run Validation Now (fresh pass on the val split)", type="primary"):
            cfg = get_base_cfg()
            with st.spinner("Loading validation data and evaluating…"):
                try:
                    exp_cfg = cfg["experiments"][exp_choice]
                    _, val_loader, _, nclasses = make_dataloaders(
                        cfg["data_dir"], cfg["image_size"], cfg["batch_size"], cfg["num_workers"],
                        cfg["val_split"], exp_cfg["augmentation"], cfg["seed"],
                    )
                    model, dev = load_experiment_model(exp_choice, OUTPUT_DIR, nclasses)
                    result = validate(model, val_loader, dev)
                    st.success(f"Validation accuracy: {result['accuracy']:.2%}  |  Validation loss: {result['loss']:.4f}")
                except Exception as e:
                    st.error(f"Validation run failed: {e}")

# ============================================================
# 6. TESTING
# ============================================================
elif page == "Testing":
    st.markdown("# 🧪 Stage 6 — Testing")
    st.write("Evaluate a trained experiment on the held-out GTSRB test split.")

    if not trained:
        st.info("No trained experiments yet. Go to **Model Training** first.")
    else:
        exp_choice = st.selectbox("Select a trained experiment", trained, key="test_exp_choice")
        metrics = experiment_metrics(exp_choice)

        if metrics:
            st.markdown("### Saved test metrics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
            c2.metric("Precision (macro)", f"{metrics.get('precision_macro', 0):.2%}")
            c3.metric("Recall (macro)", f"{metrics.get('recall_macro', 0):.2%}")
            c4.metric("F1 (macro)", f"{metrics.get('f1_macro', 0):.2%}")

            img_c1, img_c2 = st.columns(2)
            cm_path = Path(OUTPUT_DIR) / exp_choice / "confusion_matrix.png"
            roc_path = Path(OUTPUT_DIR) / exp_choice / "roc_auc.png"
            if cm_path.exists():
                img_c1.image(str(cm_path), caption="Confusion Matrix", use_container_width=True)
            if roc_path.exists():
                img_c2.image(str(roc_path), caption="ROC Curves (One-vs-Rest)", use_container_width=True)
        else:
            st.info("No saved metrics.json for this experiment yet — run the test evaluation below.")

        st.divider()
        if st.button("🧪 Run Test Evaluation Now", type="primary"):
            cfg = get_base_cfg()
            with st.spinner("Loading test data and evaluating…"):
                try:
                    exp_cfg = cfg["experiments"][exp_choice]
                    _, _, test_loader, nclasses = make_dataloaders(
                        cfg["data_dir"], cfg["image_size"], cfg["batch_size"], cfg["num_workers"],
                        cfg["val_split"], exp_cfg["augmentation"], cfg["seed"],
                    )
                    model, dev = load_experiment_model(exp_choice, OUTPUT_DIR, nclasses)
                    result, y_true, y_pred, probs = test_model(model, test_loader, dev, nclasses)
                    cm = result.pop("confusion_matrix")
                    exp_dir = Path(OUTPUT_DIR) / exp_choice
                    save_confusion_matrix(np.array(cm), exp_dir / "confusion_matrix.png")
                    save_roc_curve(y_true, probs, exp_dir / "roc_auc.png", nclasses)
                    # Merge into any existing metrics.json instead of overwriting it, so
                    # fields like "model" (used to pick the right architecture when
                    # loading this checkpoint later) and best_val_accuracy survive.
                    merged = dict(metrics or {})
                    merged.update(result)
                    merged["experiment"] = exp_choice
                    merged["model"] = exp_cfg["model"]
                    merged["augmentation"] = exp_cfg["augmentation"]
                    merged["pretrained"] = exp_cfg["pretrained"]
                    merged["label_smoothing"] = exp_cfg["label_smoothing"]
                    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
                        json.dump(merged, f, indent=2, default=str)
                    st.success(f"Test accuracy: {result['accuracy']:.2%}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Test evaluation failed: {e}")

# ============================================================
# 7. PREDICTION
# ============================================================
elif page == "Prediction":
    st.markdown("# 🎯 Stage 7 — Prediction")

    if not trained:
        st.markdown("""
        <div class="error-box"><b>⚠️ No trained model was found in the outputs folder.</b></div>
        """, unsafe_allow_html=True)
        st.info("Go to **Model Training**, select at least one experiment, and click **Start Training** first. "
                "Models are saved to `outputs/<experiment_name>/best_model.pt`.")
    else:
        exp_choice = st.selectbox("Model to use for prediction", trained, key="predict_exp_choice")

        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_file = st.file_uploader("Upload a traffic sign image", type=["jpg", "jpeg", "png"])
        with col2:
            image_size = st.selectbox("Image size", [32, 48, 64, 128], index=2, key="predict_size")
            top_k = st.slider("Top-K predictions", 1, 10, 5)

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Traffic Sign", width=280)

            if st.button("🔍 Predict", type="primary"):
                with st.spinner("Running inference…"):
                    try:
                        model, dev = load_experiment_model(exp_choice, OUTPUT_DIR, NUM_CLASSES)
                        results = predict_image(model, dev, image, image_size=image_size, top_k=top_k)

                        st.divider()
                        st.markdown(f"### Top-{top_k} Predictions ({exp_choice})")
                        pred_df = pd.DataFrame([
                            {"Rank": i + 1, "Class": f"{name} ({idx})", "Confidence": conf}
                            for i, (idx, name, conf) in enumerate(results)
                        ])
                        display_df = pred_df.copy()
                        display_df["Confidence"] = display_df["Confidence"].map(lambda v: f"{v:.2%}")
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        st.bar_chart(pred_df.set_index("Class")[["Confidence"]])

                        top_idx, top_name, top_conf = results[0]
                        st.success(f"**Predicted: {top_name}** (class {top_idx}) — confidence {top_conf:.2%}")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
        else:
            st.info("Upload an image, then click **Predict** to run inference.")

# ============================================================
# 8. EXPERIMENT COMPARISON
# ============================================================
elif page == "Experiment Comparison":
    st.markdown("# 📊 Stage 8 — Experiment Comparison")

    df_results = load_results_df()
    if df_results.empty:
        st.markdown("""
        <div class="error-box"><b>⚠️ Experiment comparison results are not available yet.</b></div>
        """, unsafe_allow_html=True)
        st.info("Train experiments on the **Model Training** page first — results are saved to "
                "`outputs/results.csv` automatically after each run.")
    else:
        all_exps = df_results["experiment"].tolist()
        chosen = st.multiselect("Experiments to compare", all_exps, default=all_exps)
        filtered = df_results[df_results["experiment"].isin(chosen)] if chosen else df_results

        st.success(f"Showing {len(filtered)} of {len(df_results)} experiment(s)")
        st.markdown("### Results Table")
        st.dataframe(filtered, use_container_width=True)

        metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro_ovr"]
        available_metrics = [m for m in metrics if m in filtered.columns]

        if available_metrics and not filtered.empty:
            st.markdown("### Metric Comparison")
            chart_data = filtered[["experiment"] + available_metrics].set_index("experiment")
            st.bar_chart(chart_data)

            st.markdown("### Summary Statistics")
            st.dataframe(filtered[available_metrics].describe(), use_container_width=True)

        comparison_img = Path(OUTPUT_DIR) / "experiment_comparison.png"
        if comparison_img.exists():
            st.divider()
            st.markdown("### Comparison Chart")
            st.image(str(comparison_img), use_container_width=True)

        st.divider()
        st.download_button(
            "⬇️ Download results.csv",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="experiment_comparison.csv",
            mime="text/csv",
        )

# ============================================================
# Footer
# ============================================================
st.divider()
st.markdown("""
---
**AI-Based Traffic Sign Recognition System** | Python + PyTorch + Streamlit | GTSRB Dataset
""")
