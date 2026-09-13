# Fix Summary — Round 2

This round fixes a **real crash bug** left over from round 1, and rebuilds
`app.py` around the requested 8-stage pipeline with selectable experiments
and explicit action buttons everywhere.

## 1. Crash bug: `src/data.py`

`torchvision.datasets.GTSRB` does **not** define `.classes`, `.targets`, or
`._labels`. It only stores raw `(path, label)` tuples in `._samples`.

The previous "fixed" version still did:

```python
targets = np.asarray(full_train_eval._labels if hasattr(full_train_eval, "_labels") else full_train_eval.targets)
...
len(test_ds.classes)
```

Both `hasattr(..., "_labels")` and `.targets` are false/missing, and
`.classes` doesn't exist either — so `make_dataloaders()` raised an
`AttributeError` the instant it ran, which is why training never worked no
matter what the app.py code looked like. This is fixed with a
`_labels_of()` helper that reads `._samples` correctly, and `NUM_CLASSES`
is now a constant (GTSRB always has 43 classes) instead of a nonexistent
attribute lookup.

## 2. Engine now supports live progress callbacks

`src/engine.py::fit()` takes an optional `on_epoch_end(epoch, epochs, row)`
callback so the UI can show a live progress bar / metrics per epoch instead
of freezing until training finishes.

## 3. Experiments are selectable, not fixed

`src/experiments.py::run_all()` now accepts `experiment_names=[...]` to run
**only** the experiments you pick, and merges new results into
`outputs/results.csv` instead of overwriting previously-trained
experiments. The Model Training page exposes this as checkboxes.

## 4. New `src/inference.py`

Central place to discover trained experiments (`list_trained_experiments`),
load any one of them by name (`load_experiment_model` — reads the saved
`model` field so it loads the right architecture, CNN or ResNet18), and run
predictions (`predict_image`) with human-readable GTSRB class names.

## 5. Rebuilt `app.py` — 8-stage pipeline, not 4 pages

Old app: Home / Prediction / Experiment Results / Training Graphs, with
model selection hardcoded to "whichever output folder sorts last" and
prediction firing automatically on upload.

New app follows the exact pipeline you asked for, one page each:

```
Data Collection → Preprocessing → Augmentation → Model Training →
Validation → Testing → Prediction → Experiment Comparison
```

Every stage that does work has a real button, nothing runs implicitly:

| Stage | Button(s) |
|---|---|
| Data Collection | 📥 Download / Verify Dataset, 🔀 Show New Random Samples |
| Preprocessing | 🔧 Preview Preprocessing |
| Augmentation | 🎨 Generate Augmented Previews (with rotation/affine/color-jitter toggles) |
| Model Training | ☑ pick which experiment(s) to run, 🚀 Start Training |
| Validation | ▶️ Run Validation Now |
| Testing | 🧪 Run Test Evaluation Now |
| Prediction | 🔍 Predict (upload does **not** auto-run inference anymore) |
| Experiment Comparison | multiselect + ⬇️ Download results.csv |

Every page that needs a trained model (Validation, Testing, Prediction)
lets you pick **which** trained experiment to use from a dropdown, instead
of silently loading the most recent one.

## What you still need to do

This app trains real PyTorch models on the ~300MB GTSRB dataset — that part
still needs to run on your machine (with internet access and, ideally, a
GPU). Nothing in this fix removes that requirement; it fixes the crash and
rebuilds the UI around it.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then walk the sidebar top to bottom: Data Collection → ... → Experiment Comparison.
