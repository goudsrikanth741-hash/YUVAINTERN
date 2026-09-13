from pathlib import Path
import json
import pandas as pd
import torch

from .data import make_dataloaders
from .models import build_model
from .engine import fit, run_epoch
from .metrics import compute_metrics
from .utils import save_json
from .visualization import save_training_curves, save_confusion_matrix, save_roc_curve, save_comparison


def design_summary(cfg):
    return {
        "independent_variables": ["model_architecture", "augmentation", "transfer_learning", "label_smoothing"],
        "dependent_variables": ["accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_ovr_roc_auc", "test_loss", "training_time_seconds"],
        "control_variables": ["GTSRB dataset", "43 classes", "64x64 input", "Adam optimizer", "learning rate", "weight decay", "epoch budget", "validation split", "random seed", "test preprocessing", "evaluation code"],
        "hypotheses": {
            "baseline_cnn": "Reference performance establishes the minimum benchmark.",
            "cnn_augmentation": "Augmentation should improve robustness and generalization.",
            "resnet18_transfer": "Pretrained ResNet18 should outperform the small CNN.",
            "resnet18_label_smoothing": "Label smoothing may improve calibration and macro metrics by reducing overconfidence."
        },
        "evaluation": "Best validation-accuracy checkpoint is evaluated on the untouched GTSRB test split once per experiment."
    }


def run_all(cfg, experiment_names=None, progress_cb=None):
    """Run the experiments in `cfg["experiments"]`.

    Parameters
    ----------
    cfg: dict
        Loaded experiment configuration (see `src.config.load_config`).
    experiment_names: list[str] | None
        If given, only run the experiments whose keys appear in this list
        (in the given order) instead of every experiment in the config.
        This is what lets the Streamlit "Model Training" page let the user
        pick which experiments to run, rather than always running all of
        them.
    progress_cb: callable | None
        Optional callback `progress_cb(exp_name, exp_index, exp_total, epoch, epoch_total, row)`
        invoked after every training epoch, so a UI can render a live
        progress bar / metrics table.
    """
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    save_json(design_summary(cfg), out / "experiment_design.json")

    all_experiments = cfg["experiments"]
    if experiment_names:
        missing = [n for n in experiment_names if n not in all_experiments]
        if missing:
            raise ValueError(f"Unknown experiment name(s): {missing}. Available: {list(all_experiments)}")
        selected = {name: all_experiments[name] for name in experiment_names}
    else:
        selected = all_experiments

    rows = []
    exp_total = len(selected)
    for exp_index, (name, exp) in enumerate(selected.items(), 1):
        print(f"\n=== {name} ===")
        exp_dir = out / name; exp_dir.mkdir(parents=True, exist_ok=True)
        train_loader, val_loader, test_loader, nclasses = make_dataloaders(
            cfg["data_dir"], cfg["image_size"], cfg["batch_size"], cfg["num_workers"], cfg["val_split"], exp["augmentation"], cfg["seed"]
        )
        model = build_model(exp["model"], nclasses, exp["pretrained"]).to(cfg["device"])
        ckpt = exp_dir / "best_model.pt"

        def _on_epoch_end(epoch, epoch_total, row, _name=name, _idx=exp_index):
            if progress_cb is not None:
                progress_cb(_name, _idx, exp_total, epoch, epoch_total, row)

        history, train_time = fit(
            model, train_loader, val_loader, cfg["device"], cfg["epochs"], cfg["learning_rate"],
            cfg["weight_decay"], exp["label_smoothing"], cfg["patience"], ckpt,
            on_epoch_end=_on_epoch_end,
        )
        pd.DataFrame(history).to_csv(exp_dir / "history.csv", index=False)
        save_training_curves(history, exp_dir / "training_curves.png")
        state = torch.load(ckpt, map_location=cfg["device"])
        model.load_state_dict(state["model_state"])
        criterion = torch.nn.CrossEntropyLoss()
        test_loss, y, pred, probs = run_epoch(model, test_loader, criterion, cfg["device"], train=False)
        m = compute_metrics(y, pred, probs, nclasses)
        save_confusion_matrix(m.pop("confusion_matrix"), exp_dir / "confusion_matrix.png")
        save_roc_curve(y, probs, exp_dir / "roc_auc.png", nclasses)
        pd.DataFrame({"true_label": y, "predicted_label": pred, "confidence": probs.max(1)}).to_csv(exp_dir / "test_predictions.csv", index=False)
        result = {
            "experiment": name, "model": exp["model"], "augmentation": exp["augmentation"],
            "pretrained": exp["pretrained"], "label_smoothing": exp["label_smoothing"],
            **m, "test_loss": test_loss, "training_time_seconds": train_time,
            "best_val_accuracy": state.get("best_val_accuracy", None),
        }
        save_json(result, exp_dir / "metrics.json")
        rows.append(result)

    df = pd.DataFrame(rows)

    # Merge with any previously-run experiments so re-running a subset
    # doesn't wipe out results for experiments that were trained earlier.
    results_csv = out / "results.csv"
    if results_csv.exists() and not df.empty:
        try:
            prev = pd.read_csv(results_csv)
            prev = prev[~prev["experiment"].isin(df["experiment"])]
            df = pd.concat([prev, df], ignore_index=True)
        except Exception:
            pass

    if not df.empty:
        df.to_csv(out / "results.csv", index=False)
        df.to_csv(out / "experiment_comparison.csv", index=False)
        save_comparison(df, out / "experiment_comparison.png")
        predictions = {}
        for _, r in df.iterrows():
            predictions[r["experiment"]] = {
                "expected_rank": "Higher validation/test metrics are expected for transfer learning, with augmentation helping generalization.",
                "interpretation": "Actual measured values are stored in results.csv and metrics.json."
            }
        save_json(predictions, out / "outcomes_prediction.json")
        print("\nCompleted. Results:")
        print(df[["experiment", "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro_ovr"]].to_string(index=False))
    return df
