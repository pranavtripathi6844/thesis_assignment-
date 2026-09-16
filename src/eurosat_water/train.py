"""Train the tiny CNN (early stopping on val) and fit the linear probe head."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from eurosat_water.config import PROBE_C_GRID, SEED, TINY_LR, TINY_MAX_EPOCHS, TINY_PATIENCE
from eurosat_water.data import SplitData, to_reflectance
from eurosat_water.evaluate import Result, best_threshold, score
from eurosat_water.features import select_channels
from eurosat_water.models import TinyCNN, count_params, extract_probe_features


def seed_all(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _pos_weight(y: np.ndarray, device: torch.device) -> torch.Tensor:
    n_pos = max(float(y.sum()), 1.0)
    n_neg = float(len(y) - y.sum())
    return torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)


def train_tiny_cnn(
    train: SplitData,
    val: SplitData,
    test: SplitData,
    keep: tuple[str, ...],
    name: str,
    device: torch.device,
    seed: int = SEED,
) -> tuple[Result, dict]:
    seed_all(seed)
    xtr = to_reflectance(select_channels(train.x, train.bands, keep))
    xva = to_reflectance(select_channels(val.x, val.bands, keep))
    xte = to_reflectance(select_channels(test.x, test.bands, keep))
    ytr = torch.from_numpy(train.y.astype(np.float32))
    yva = torch.from_numpy(val.y.astype(np.float32))

    model = TinyCNN(in_ch=xtr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=TINY_LR, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(train.y, device))
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=16, shuffle=True)

    best_state, best_val, stale = None, float("inf"), 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, TINY_MAX_EPOCHS + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(yb)
        train_loss = total / len(ytr)
        model.eval()
        with torch.no_grad():
            val_loss = float(crit(model(xva.to(device)), yva.to(device)).item())
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= TINY_PATIENCE and epoch >= 10:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits = model(xva.to(device)).cpu().numpy()
        test_logits = model(xte.to(device)).cpu().numpy()
    val_probs = 1 / (1 + np.exp(-val_logits))
    test_probs = 1 / (1 + np.exp(-test_logits))
    t, _ = best_threshold(val_probs, val.y)
    extra = {
        "params": count_params(model),
        "epochs_ran": len(history["train_loss"]),
        "best_val_loss": best_val,
        "history": history,
        "bands": list(keep),
        "seed": seed,
    }
    return score(name, "tiny_cnn", test_probs, test.y, threshold=t, extra=extra), extra


def fit_linear_probe(
    model,
    weight_bands: tuple[str, ...],
    train: SplitData,
    val: SplitData,
    test: SplitData,
    keep: tuple[str, ...],
    name: str,
    device: torch.device,
) -> Result:
    ftr = extract_probe_features(model, train.x, train.bands, keep, weight_bands, device).numpy()
    fva = extract_probe_features(model, val.x, val.bands, keep, weight_bands, device).numpy()
    fte = extract_probe_features(model, test.x, test.bands, keep, weight_bands, device).numpy()

    best_clf, best_f1, best_t, best_C = None, -1.0, 0.5, 1.0
    for C in PROBE_C_GRID:
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        C=C,
                        class_weight="balanced",
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=SEED,
                    ),
                ),
            ]
        )
        clf.fit(ftr, train.y)
        probs = clf.predict_proba(fva)[:, 1]
        t, f1 = best_threshold(probs, val.y)
        if f1 > best_f1:
            best_clf, best_f1, best_t, best_C = clf, f1, t, C
    assert best_clf is not None
    probs = best_clf.predict_proba(fte)[:, 1]
    return score(
        name,
        "probe",
        probs,
        test.y,
        threshold=best_t,
        extra={
            "C": best_C,
            "bands": list(keep),
            "feat_dim": int(ftr.shape[1]),
            "trainable_params": int(ftr.shape[1]) + 1,
        },
    )
