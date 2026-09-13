import time
import numpy as np
import torch
import torch.nn as nn


def run_epoch(model, loader, criterion, device, train=True, optimizer=None):
    model.train(train)
    total_loss, n = 0.0, 0
    ys, preds, probs = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train and optimizer is None:
            raise ValueError("Optimizer is required when train=True")
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
        p = torch.softmax(logits, dim=1)
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
        ys.append(y.detach().cpu().numpy())
        preds.append(p.argmax(1).detach().cpu().numpy())
        probs.append(p.detach().cpu().numpy())
    return total_loss / max(n, 1), np.concatenate(ys), np.concatenate(preds), np.concatenate(probs)


def fit(model, train_loader, val_loader, device, epochs, lr, weight_decay, label_smoothing, patience,
        checkpoint_path, on_epoch_end=None):
    """Train `model` with early stopping on validation accuracy.

    `on_epoch_end`, if given, is called as `on_epoch_end(epoch, epochs, row)`
    after every epoch, where `row` is the dict appended to `history`. This
    lets callers (e.g. the Streamlit Model Training page) drive a live
    progress bar / metrics table without touching the training loop itself.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = -1.0
    wait = 0
    history = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, _, _, _ = run_epoch(model, train_loader, criterion, device, train=True, optimizer=optimizer)
        val_loss, vy, vp, _ = run_epoch(model, val_loader, criterion, device, train=False, optimizer=None)
        val_acc = float((vy == vp).mean())
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc}
        history.append(row)
        improved = val_acc > best_val
        if improved:
            best_val = val_acc
            wait = 0
            torch.save({"model_state": model.state_dict(), "best_val_accuracy": best_val}, checkpoint_path)
        else:
            wait += 1
        if on_epoch_end is not None:
            on_epoch_end(epoch, epochs, {**row, "best_val_accuracy": best_val, "improved": improved})
        if wait >= patience:
            break
    return history, time.time() - start
