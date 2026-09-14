# Plant Disease Detection — Week 4: Data Preprocessing & Feature Engineering Strategies

## 1. Project Purpose

This project implements the **Week 4** stage of an AI course project: a complete, reproducible
**data preprocessing and feature-engineering pipeline** for plant-disease detection from leaf
images, a baseline classifier that validates the pipeline produces usable, class-separable model
inputs, and a small **Flask dashboard** to browse the results.

It is **not** a final production disease-classifier — the focus is the data pipeline (loading,
cleaning, preprocessing, augmentation, feature engineering, visualization) and demonstrating,
end-to-end, that it works and can be inspected in a browser.

## 2. Project Structure

```text
plant_disease_week4/
├── dataset/                     # Sample image dataset (see Section 3)
│   ├── Apple___Black_rot/
│   ├── Apple___healthy/
│   ├── Potato___Early_blight/
│   ├── Potato___Late_blight/
│   └── Tomato___healthy/
├── outputs/                     # Everything main.py generates
│   ├── figures/                 # EDA, preprocessing, augmentation, feature & result plots
│   ├── features/                # Engineered feature CSVs + preprocessed_arrays.npz
│   └── results/                 # metrics.json, classification_report.txt, (cnn_model.keras)
├── src/
│   ├── main.py                  # Pipeline entry point (run this first)
│   ├── app.py                   # Flask site that displays the outputs/ above
│   ├── model.py                 # Baseline (RandomForest) + CNN (tf.keras) model code
│   ├── data_loader.py           # Dataset scanning, corrupt/duplicate detection
│   ├── data_analysis.py         # EDA plots (class distribution, dimensions, sample grid)
│   ├── preprocessing.py         # Load/resize/denoise/normalize + leak-free split
│   ├── augmentation.py          # Training-only OpenCV/NumPy augmentation
│   ├── feature_extraction.py    # Color/HSV/histogram/texture/edge/shape features
│   ├── evaluation.py            # Metrics, confusion matrix, classification report
│   └── demo_dataset.py          # Synthetic fallback dataset generator (see Section 3)
├── requirements.txt
└── README.md
```

## 3. Dataset

The project targets the public **[PlantVillage dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)**
(also available via TensorFlow Datasets as `plant_village`), which contains leaf images across
many crop species (Apple, Potato, Tomato, Corn, Grape, etc.) with per-class folders such as
`Tomato___healthy` and `Tomato___Late_blight`.

`dataset/` in this repo ships with a **small labeled sample** (5 classes, a few dozen images per
class) so the pipeline can be cloned and run immediately without first downloading the full
~2GB Kaggle dataset. It intentionally also includes a handful of **corrupt files** and
**duplicate images** per class, so that `src/data_loader.py`'s corrupt-file handling and
duplicate detection can be verified as actually working rather than just assumed.

Expected folder layout (one sub-folder per class):

```text
dataset/
├── Apple___healthy/
│   ├── image1.jpg
│   └── ...
├── Apple___Black_rot/
├── Potato___Early_blight/
├── Potato___Late_blight/
└── Tomato___healthy/
```

Any image formats in `{.jpg, .jpeg, .png, .bmp, .tif, .tiff}` are picked up. If `dataset/` is
ever completely empty, `src/main.py` automatically generates a small synthetic placeholder
dataset (`src/demo_dataset.py`) so the pipeline never fails to run — but this repo's bundled
`dataset/` already contains real sample images, so that fallback won't normally trigger.

**To run this on the full real PlantVillage dataset:** download it from Kaggle and replace/extend
`dataset/` with the same one-folder-per-class structure above. No code changes are required.

## 4. Setup Instructions

### 4.1 Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 4.2 Install dependencies

```bash
pip install -r requirements.txt
```

Core dependencies: `numpy`, `pandas`, `opencv-python`, `scikit-image`, `scikit-learn`, `Pillow`,
`matplotlib`, `scipy`, `flask`, `imageio`, `tifffile`.

`tensorflow` is **optional** (commented out in `requirements.txt`). If it is installed,
`src/main.py` automatically trains a CNN (`src/model.py: build_cnn_model`) directly on the
preprocessed image tensors. If it is **not** installed, `src/main.py` automatically falls back to
a **scikit-learn RandomForest** classifier trained on the engineered features from Section 7 —
no crashes, no missing outputs, just a clearly logged fallback message.

### 4.3 Run the pipeline (generates everything under `outputs/`)

```bash
python src/main.py
```

Useful options:

```bash
python src/main.py \
  --dataset_root dataset \   # where the class folders live (default: <repo>/dataset)
  --img_size 224 \           # square resize target
  --max_per_class 40 \       # cap images/class for a fast run; 0 = use ALL images
  --val_size 0.15 \
  --test_size 0.15 \
  --aug_multiplier 1 \       # augmented copies added per training image
  --cnn_epochs 8
```

`--max_per_class` exists specifically so this can run quickly on a laptop or CI machine, while
remaining fully compatible with the complete PlantVillage dataset (`--max_per_class 0` uses every
image).

### 4.4 Launch the Flask results dashboard

After `src/main.py` has populated `outputs/` at least once:

```bash
python src/app.py
```

Then open **http://127.0.0.1:5000/** in a browser. The site shows:

- **A "Scan Image" upload form** at the top of the page (see Section 4.5 below) — the main way
  to test the model interactively.
- The evaluation metrics from `outputs/results/metrics.json` as cards (accuracy, precision,
  recall, F1 — macro and weighted).
- Every figure from `outputs/figures/` (class distribution, sample images, before/after
  preprocessing, augmentation examples, color analysis, feature distributions, texture/edge
  examples, confusion matrix), each with a caption.
- The full text of `outputs/results/classification_report.txt`, plus a link to
  download/view it directly at `/results/classification_report.txt`.

If `outputs/` doesn't exist yet, the page shows a friendly message telling you to run
`src/main.py` first, instead of crashing.

### 4.5 Use the "Scan Image" upload button to detect disease on a new photo

`src/main.py` (Step 6 of the pipeline) automatically saves the trained model right after
evaluation:

- If TensorFlow was available, it saves `outputs/results/cnn_model.keras`.
- Otherwise (the default in most environments), it saves
  `outputs/results/baseline_model.joblib` + `baseline_scaler.joblib` (the RandomForest + its
  feature scaler).
- Either way, it also saves `outputs/results/inference_meta.json`, which records which model was
  trained, the class names, the image size used, and (for the baseline) the exact feature-column
  order — so `src/app.py` always knows how to reproduce the training-time preprocessing for a
  brand-new uploaded image.

On the homepage:

1. Click **Choose File** under "Scan a Leaf Image" and pick a `.jpg`, `.jpeg`, `.png`, `.bmp`,
   `.tif`, or `.tiff` photo of a plant leaf (max 10 MB).
2. Click **🔍 Scan Image**.
3. The page reloads showing:
   - the uploaded image,
   - the **predicted class** (e.g. `Tomato___healthy`, `Potato___Late_blight`),
   - a **confidence score** (%) with a progress bar,
   - the **top-3 predicted classes** ranked by probability.

Under the hood, `POST /scan` in `src/app.py`:

1. Saves the upload to `outputs/uploads/`.
2. Runs it through the **identical** preprocessing used during training
   (`src/preprocessing.py: preprocess_image` — load → RGB → denoise → resize → normalize).
3. If using the baseline model: extracts the same 45 engineered features
   (`src/feature_extraction.py`), reorders them to match the training-time column order, scales
   them with the saved `StandardScaler`, and calls `clf.predict_proba(...)`.
   If using the CNN: feeds the preprocessed 224×224×3 tensor directly into the saved Keras model.
4. Returns the top class + confidence + top-3 breakdown, or a friendly error message (e.g. "not a
   readable image file") instead of crashing if the upload can't be processed.

If you haven't run `src/main.py` yet, the upload form is replaced with a message asking you to
run it first (no trained model to predict with).

## 5. Preprocessing Steps (`src/preprocessing.py`)

1. **Load** each image with OpenCV, convert **BGR → RGB**.
2. **Denoise** with a small median blur (removes sensor/JPEG noise while preserving lesion edges
   better than a Gaussian blur would).
3. **Resize** to a fixed size (default **224×224**, the standard input size for most CNN
   backbones) using area interpolation.
4. **Normalize** pixel values from `[0, 255]` to `[0, 1]` (float32).
5. **Stratified train/val/test split** (default 70/15/15) performed on the **raw file list**,
   before any augmentation — this is what prevents data leakage: no augmented derivative of a
   validation/test image can ever end up in the training set, and no file appears in more than
   one split.
6. **Class-imbalance analysis**: per-class counts + imbalance ratio (max/min class count) are
   computed and logged.
7. **Class-imbalance handling**: rather than duplicating minority-class images (risking
   overfitting on small datasets), the pipeline computes `sklearn`-style **balanced class
   weights** and passes them into the model's loss function (both the CNN path and the
   RandomForest baseline use `class_weight="balanced"`).

## 6. Data Augmentation (`src/augmentation.py`, training split only)

Implemented directly with OpenCV/NumPy (framework-agnostic): random horizontal flip, random
rotation (±25°), random brightness/contrast jitter, random zoom/crop, and light Gaussian pixel
noise. `augment_training_set()` is only ever called on the **training** split.

## 7. Feature Engineering (`src/feature_extraction.py`)

For every preprocessed image, **45 hand-crafted features across 6 groups** are extracted (well
above the 10-feature minimum), chosen because they plausibly separate healthy vs. diseased leaf
tissue — no filler features were added just to inflate the count:

| # | Group | Features (named columns) | Why |
|---|---|---|---|
| 1–6 | RGB color stats | `rgb_r_mean`, `rgb_r_std`, `rgb_g_mean`, `rgb_g_std`, `rgb_b_mean`, `rgb_b_std` | Overall color shift |
| 7–12 | HSV color stats | `hsv_h_mean`, `hsv_h_std`, `hsv_s_mean`, `hsv_s_std`, `hsv_v_mean`, `hsv_v_std` | Hue is more lighting-robust than RGB; useful for discoloration |
| 13–36 | Color histogram | `hist_{r,g,b}_0..7` (8 bins × 3 channels) | Coarse color distribution shape |
| 37–40 | Texture (GLCM) | `texture_contrast`, `texture_homogeneity`, `texture_energy`, `texture_correlation` | Lesions/spots change local texture |
| 41 | Edge | `edge_density` (Canny) | Lesion boundaries increase local edge density |
| 42–45 | Shape / leaf region | `shape_area_ratio`, `shape_perimeter`, `shape_extent`, `shape_solidity` (Otsu mask on saturation channel) | Leaf shape irregularity / damage |

Features are saved as CSV (`outputs/features/{train,val,test}_features.csv`), the preprocessed
image tensors are saved as a compressed NumPy archive (`outputs/features/preprocessed_arrays.npz`),
and the exact column order is saved in `outputs/results/inference_meta.json` so the Flask app's
image-upload scanner (Section 4.5) can reproduce the same feature vector for a brand-new image.

## 8. Generated Outputs (`outputs/`)

```text
outputs/
├── figures/
│   ├── 01_class_distribution.png
│   ├── 02_image_dimensions.png
│   ├── 03_sample_images.png
│   ├── 04_before_after_preprocessing.png
│   ├── 05_augmentation_examples.png
│   ├── 06_color_analysis.png
│   ├── 07_feature_distributions.png
│   ├── 08_texture_edge_examples.png
│   └── 09_confusion_matrix.png
├── features/
│   ├── train_features.csv / val_features.csv / test_features.csv
│   └── preprocessed_arrays.npz
└── results/
    ├── classification_report.txt
    ├── metrics.json
    ├── inference_meta.json         # model type, class names, img size, feature-column order
    ├── baseline_model.joblib       # trained RandomForest (produced when TensorFlow is absent)
    ├── baseline_scaler.joblib      # matching StandardScaler for the baseline features
    └── cnn_model.keras             # trained CNN (only produced if TensorFlow is installed)
```

`inference_meta.json`, and either the `baseline_*.joblib` files or `cnn_model.keras`, are what
`src/app.py`'s "Scan Image" upload feature loads to run predictions on new photos (Section 4.5).

## 9. Model / Evaluation Process (`src/model.py`, `src/evaluation.py`)

- **Primary path (CNN):** a small Conv→Pool ×3 → GlobalAveragePooling → Dense CNN
  (`build_cnn_model`), trained with `tf.keras`, `class_weight` applied, Adam optimizer, sparse
  categorical cross-entropy. Used automatically when TensorFlow is available.
- **Fallback path (baseline):** a `RandomForestClassifier` (`class_weight="balanced"`) trained on
  the standardized engineered feature vectors from Section 7. Used automatically when TensorFlow
  is not installed, so the pipeline is still fully runnable and its output still genuinely
  demonstrates that Sections 5–7 produce usable, class-separable model inputs.
- **Evaluation** (`src/evaluation.py`): accuracy, macro/weighted precision, recall, F1-score, a
  full `sklearn.metrics.classification_report`, and a confusion matrix plot — computed on the
  held-out **test** split only.

## 10. Case Study: Applying This Pipeline to the Full Kaggle PlantVillage Dataset

This is a **hypothetical walkthrough** of how the same code in this repo would be used against
the full public dataset, for anyone who wants to scale it up beyond the bundled sample:

1. **Download**: grab the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
   from Kaggle (~54,000 images across 38 class/crop-disease combinations).
2. **Drop-in replace**: extract it so each class becomes a sub-folder of `dataset/`, matching the
   layout in Section 3 — no renaming or code changes needed, `src/data_loader.py` discovers
   classes purely from folder names.
3. **Run at scale**: `python src/main.py --max_per_class 0 --img_size 224 --aug_multiplier 2`
   would preprocess and feature-engineer the full dataset, with the leak-free stratified split
   keeping evaluation honest across all 38 classes.
4. **Expect**: `src/data_loader.py`'s corrupt-file and duplicate detection to flag the small
   number of known bad/duplicate files the public dataset ships with; `src/preprocessing.py`'s
   class-imbalance analysis to reveal that some classes (e.g. `Tomato___healthy`) are better
   represented than rarer disease classes, which is why balanced class weights (Section 5,
   point 7) matter here rather than being optional.
5. **Compare model paths**: with `tensorflow` installed, the CNN path in `src/model.py` would
   train directly on the 224×224 tensors and, given 38 classes, would be expected to outperform
   the RandomForest baseline on the hand-crafted features — the baseline exists precisely to give
   a fast, dependency-light lower bound to compare the CNN against.
6. **Review results**: run `python src/app.py` and open `http://127.0.0.1:5000/` to browse the
   resulting confusion matrix (38×38 at full scale), per-class metrics, and all EDA/feature
   figures, exactly as with the bundled sample dataset.
7. **Scan real-world photos**: with the full 38-class model trained, the same "Scan Image" upload
   button (Section 4.5) would let a user photograph a leaf from any of those 38 crop/disease
   combinations and get back a predicted class + confidence — no code changes needed, since
   `app.py` reads the class list and feature schema from `inference_meta.json` rather than having
   them hard-coded.

This case study is illustrative — the actual `outputs/` in this repo come from running the
pipeline against the small bundled sample dataset in `dataset/`, not the full Kaggle download.

## 11. Reproducibility

All random operations (dataset split, augmentation, class-weight computation, model
initialization) are seeded (`SEED = 42` in `src/main.py`), so re-running `python src/main.py`
with the same dataset and arguments reproduces the same split, augmentation, and (for the
RandomForest baseline) the same metrics.

## 12. Notes on What Actually Ran

Everything under `outputs/` in this repo — dataset scanning, corrupt/duplicate detection, all
preprocessing steps, all augmentation, all feature extraction, all visualizations, and the
RandomForest baseline model with real accuracy/precision/recall/F1/confusion matrix — was
genuinely executed against the bundled sample `dataset/`, and the Flask app in `src/app.py` was
verified to correctly serve those exact metrics, figures, and report. The **image-upload scanner**
was also tested end-to-end: uploading a real sample leaf photo through `POST /scan` correctly
returned its true class with a real confidence score computed by the saved RandomForest model.
TensorFlow was not installed in the environment this project was built in, so the CNN path in
`src/model.py` (and CNN-based scanning) was not executed here; the moment `tensorflow` is
installed, re-running `python src/main.py` will train and save the CNN automatically, and
`src/app.py` will automatically use it for scanning instead — no code changes required.
