"""Majority class, index thresholds, logistic regression, small random forest."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eurosat_water.config import PROBE_C_GRID, SEED
from eurosat_water.evaluate import Result, best_threshold, score
from eurosat_water.features import combined_feature_matrix, image_index_score, index_feature_matrix


def majority(y_train: np.ndarray, y_test: np.ndarray) -> Result:
    # Always predict the majority class of the training set (non-water).
    maj = int(np.bincount(y_train).argmax())
    probs = np.full(len(y_test), 0.01 if maj == 0 else 0.99, dtype=np.float64)
    return score("Majority class", "baseline", probs, y_test, threshold=0.5, extra={"majority": maj})


def _tune_threshold_on_scores(train_s, y_train, val_s, y_val) -> float:
    # Pool train+val for a more stable cut, then lock it before test.
    s = np.concatenate([train_s, val_s])
    y = np.concatenate([y_train, y_val])
    # Map unbounded index scores to a 0-1-ish range for the grid search helper.
    lo, hi = np.quantile(s, [0.02, 0.98])
    scaled = np.clip((s - lo) / (hi - lo + 1e-8), 0, 1)
    t_scaled, _ = best_threshold(scaled, y)
    return float(lo + t_scaled * (hi - lo))


def index_threshold(train, val, test, which: str) -> Result:
    tr = image_index_score(train, which)
    va = image_index_score(val, which)
    te = image_index_score(test, which)
    cut = _tune_threshold_on_scores(tr, train.y, va, val.y)
    # Convert to a monotone probability-like score for PR/ROC.
    all_s = np.concatenate([tr, va, te])
    lo, hi = np.quantile(all_s, [0.02, 0.98])
    probs = np.clip((te - lo) / (hi - lo + 1e-8), 0, 1)
    # Threshold in the same scaled space.
    t_scaled = float(np.clip((cut - lo) / (hi - lo + 1e-8), 0, 1))
    name = which.upper() + " threshold"
    return score(name, "physics", probs, test.y, threshold=t_scaled, extra={"raw_cut": cut})


def _fit_linear(X_train, y_train, X_val, y_val) -> tuple[Pipeline, float]:
    best = None
    best_f1 = -1.0
    best_t = 0.5
    for C in PROBE_C_GRID:
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        C=C,
                        class_weight="balanced",
                        max_iter=1000,
                        solver="lbfgs",
                        random_state=SEED,
                    ),
                ),
            ]
        )
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_val)[:, 1]
        t, f1 = best_threshold(probs, y_val)
        if f1 > best_f1:
            best, best_f1, best_t = clf, f1, t
    assert best is not None
    return best, best_t


def logistic_on_features(train, val, test, which: str) -> Result:
    if which == "indices":
        Xtr, names = index_feature_matrix(train)
        Xva, _ = index_feature_matrix(val)
        Xte, _ = index_feature_matrix(test)
        label = "Logistic (NDWI/MNDWI stats)"
    else:
        Xtr, names = combined_feature_matrix(train)
        Xva, _ = combined_feature_matrix(val)
        Xte, _ = combined_feature_matrix(test)
        label = "Logistic (indices + band means)"
    clf, t = _fit_linear(Xtr, train.y, Xva, val.y)
    probs = clf.predict_proba(Xte)[:, 1]
    return score(label, "physics", probs, test.y, threshold=t, extra={"features": names, "C": clf.named_steps["lr"].C})


def random_forest_indices(train, val, test) -> Result:
    Xtr, names = index_feature_matrix(train)
    Xva, _ = index_feature_matrix(val)
    Xte, _ = index_feature_matrix(test)
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=1,
    )
    clf.fit(Xtr, train.y)
    val_p = clf.predict_proba(Xva)[:, 1]
    t, _ = best_threshold(val_p, val.y)
    probs = clf.predict_proba(Xte)[:, 1]
    return score(
        "Random forest (NDWI/MNDWI stats)",
        "physics",
        probs,
        test.y,
        threshold=t,
        extra={"features": names, "importances": clf.feature_importances_.tolist()},
    )
