"""Metrics, threshold selection on val, Wilson intervals, McNemar."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
)


@dataclass
class Result:
    name: str
    family: str
    probs: np.ndarray
    pred: np.ndarray
    y_true: np.ndarray
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    cm: list
    acc_ci: tuple[float, float]
    recall_ci: tuple[float, float]
    extra: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        d = asdict(self)
        d.pop("probs")
        d.pop("pred")
        d.pop("y_true")
        d.pop("extra")
        return d


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    z2 = z * z
    den = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / den
    margin = z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n) / den
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def best_threshold(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Choose a cut from *validation* scores only. Never look at test.

    Candidates: unique scores, plus one value below the min (all-positive)
    and one above the max (all-negative). Predict positive iff score >= t.

    Tie-break: highest F1, then the *highest* threshold (fewer water calls).
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if scores.size == 0:
        return 0.5, 0.0
    uniq = np.unique(scores)
    span = float(uniq[-1] - uniq[0])
    eps = 1e-12 if span == 0.0 else span * 1e-9
    candidates = np.concatenate(([uniq[0] - eps], uniq, [uniq[-1] + eps]))
    best_t = float(candidates[-1])
    best_f1 = -1.0
    for t in candidates:
        f1 = float(f1_score(y, (scores >= t).astype(int), zero_division=0))
        t = float(t)
        if f1 > best_f1 + 1e-15 or (abs(f1 - best_f1) <= 1e-15 and t > best_t):
            best_f1, best_t = f1, t
    return best_t, best_f1


def _safe_roc_auc(y, probs) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, probs))


def score(name: str, family: str, probs: np.ndarray, y: np.ndarray, threshold: float, extra: dict | None = None) -> Result:
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    pred = (probs >= threshold).astype(np.int64)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    n = len(y)
    n_pos = int(y.sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    correct = int((pred == y).sum())
    return Result(
        name=name,
        family=family,
        probs=probs,
        pred=pred,
        y_true=y,
        threshold=float(threshold),
        accuracy=float(accuracy_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        roc_auc=_safe_roc_auc(y, probs),
        pr_auc=float(average_precision_score(y, probs)) if n_pos else float("nan"),
        cm=cm.tolist(),
        acc_ci=wilson_interval(correct, n),
        recall_ci=wilson_interval(tp, n_pos) if n_pos else (0.0, 0.0),
        extra=extra or {},
    )


def mcnemar_exact(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    a_err = np.asarray(pred_a).astype(int) != y
    b_err = np.asarray(pred_b).astype(int) != y
    n10 = int(np.sum(a_err & ~b_err))
    n01 = int(np.sum(~a_err & b_err))
    n = n10 + n01
    if n == 0:
        p = 1.0
    else:
        from scipy.stats import binomtest

        p = float(binomtest(n10, n, 0.5, alternative="two-sided").pvalue)
    return {"n10_a_wrong_b_right": n10, "n01_a_right_b_wrong": n01, "p_value": p}


def roc_points(y, probs):
    fpr, tpr, _ = roc_curve(y, probs)
    return fpr.tolist(), tpr.tolist()


def pr_points(y, probs):
    p, r, _ = precision_recall_curve(y, probs)
    return p.tolist(), r.tolist()
