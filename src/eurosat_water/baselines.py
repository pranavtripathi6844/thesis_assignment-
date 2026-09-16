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


def index_threshold(train, val, test, which: str) -> Result:
    """Tile-mean NDWI/MNDWI. Cut from val only; ranking metrics use raw scores."""
    del train  # score is unsupervised; only the cut is supervised, and only on val
    va = image_index_score(val, which)
    te = image_index_score(test, which)
    cut, _ = best_threshold(va, val.y)
    name = which.upper() + " threshold"
    return score(
        name,
        "physics",
        te,
        test.y,
        threshold=cut,
        extra={"raw_cut": cut, "threshold_source": "val", "score": "raw_tile_mean_index"},
    )


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
