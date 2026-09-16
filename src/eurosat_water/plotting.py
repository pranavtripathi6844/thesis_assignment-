"""Figures and tables for the thesis report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from eurosat_water.bands import find_band
from eurosat_water.config import RESULTS_DIR
from eurosat_water.data import SplitData, to_reflectance
from eurosat_water.evaluate import Result, pr_points, roc_points
from eurosat_water.features import water_indices

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "figure.dpi": 140,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)

FAMILY_COLOR = {
    "baseline": "#7F7F7F",
    "physics": "#2CA02C",
    "tiny_cnn": "#FF7F0E",
    "probe": "#1F77B4",
}


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _stretch(img: np.ndarray) -> np.ndarray:
    """Per-channel 2–98% stretch. Near-constant channels (water in SWIR) stay dark instead of becoming noise."""
    out = np.zeros_like(img, dtype=np.float64)
    for c in range(img.shape[-1]):
        ch = img[..., c]
        lo, hi = np.percentile(ch, [2, 98])
        if hi - lo < 0.02:
            out[..., c] = np.clip(ch / 0.35, 0, 1)
        else:
            out[..., c] = np.clip((ch - lo) / (hi - lo + 1e-8), 0, 1)
    return out


def rgb_image(x: torch.Tensor, bands: tuple[str, ...]) -> np.ndarray:
    xr = to_reflectance(x).numpy()
    rgb = np.stack(
        [
            xr[find_band(bands, "B04")],
            xr[find_band(bands, "B03")],
            xr[find_band(bands, "B02")],
        ],
        axis=-1,
    )
    return _stretch(rgb)


def swir_nir_g(x: torch.Tensor, bands: tuple[str, ...]) -> np.ndarray:
    xr = to_reflectance(x).numpy()
    fc = np.stack(
        [
            xr[find_band(bands, "B11")],
            xr[find_band(bands, "B08")],
            xr[find_band(bands, "B03")],
        ],
        axis=-1,
    )
    return _stretch(fc)


def pick_examples(split: SplitData, n_water: int = 2, n_other: int = 2) -> list[int]:
    """Prefer different original classes, not the first tiles on disk."""
    water, other = [], []
    seen_w, seen_o = set(), set()
    rng = np.random.RandomState(0)
    order = rng.permutation(len(split))
    for i in order:
        cid = int(split.y_class[i])
        if split.y[i] == 1 and cid not in seen_w and len(water) < n_water:
            water.append(int(i))
            seen_w.add(cid)
        elif split.y[i] == 0 and cid not in seen_o and len(other) < n_other:
            other.append(int(i))
            seen_o.add(cid)
        if len(water) >= n_water and len(other) >= n_other:
            break
    return water + other


def plot_examples(split: SplitData, path: Path) -> Path:
    idxs = pick_examples(split)
    fig, axes = plt.subplots(2, len(idxs), figsize=(3.2 * len(idxs), 6.2))
    for j, i in enumerate(idxs):
        tag = "WATER" if split.y[i] == 1 else "not water"
        cls = split.class_names[int(split.y_class[i])]
        axes[0, j].imshow(rgb_image(split.x[i], split.bands))
        axes[0, j].set_title(f"{tag}\n{cls}", fontsize=10)
        axes[0, j].axis("off")
        axes[1, j].imshow(swir_nir_g(split.x[i], split.bands))
        axes[1, j].axis("off")
    axes[0, 0].set_ylabel("True colour\n(B04-B03-B02)", fontsize=9)
    axes[1, 0].set_ylabel("Water composite\n(SWIR1-NIR-G)", fontsize=9)
    fig.suptitle("Train examples — 2–98% stretch (not raw reflectance)", fontsize=12, y=0.98)
    fig.tight_layout()
    return _save(fig, path)


def plot_band_boxplots(split: SplitData, path: Path) -> Path:
    xr = to_reflectance(split.x).numpy()
    water = split.y == 1
    fig, ax = plt.subplots(figsize=(12, 4.8))
    data, labels, colors = [], [], []
    for i, name in enumerate(split.bands):
        w = xr[water, i].mean(axis=(1, 2))
        nw = xr[~water, i].mean(axis=(1, 2))
        data.extend([nw, w])
        labels.extend([f"{name}\nland", f"{name}\nwater"])
        colors.extend(["#C4C4C4", "#4C72B0"])
    try:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False, widths=0.7)
    except TypeError:
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False, widths=0.7)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    ax.set_ylabel("mean TOA reflectance")
    ax.set_title("Train split: per-band mean reflectance, water vs non-water")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, path)


def plot_index_hist(split: SplitData, path: Path) -> Path:
    idx = water_indices(split)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, key, title in zip(axes, ("ndwi", "mndwi"), ("NDWI (B03, B08)", "MNDWI (B03, B11)")):
        vals = idx[key].mean(axis=(1, 2))
        ax.hist(vals[split.y == 0], bins=12, alpha=0.7, label="non-water", color="#C4C4C4")
        ax.hist(vals[split.y == 1], bins=12, alpha=0.8, label="water", color="#4C72B0")
        ax.set_title(title)
        ax.set_xlabel("tile-mean index")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("count (train)")
    fig.suptitle("Physics signal on the training split", y=1.02)
    fig.tight_layout()
    return _save(fig, path)


def plot_pipeline(path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12.5, 3.2))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    boxes = [
        (0.2, "1. 100 chips\nEuroSAT100", "#E8E8E8"),
        (2.4, "2. Physics\nNDWI / MNDWI", "#C6E7C6"),
        (4.6, "3. Tiny CNN\n~25k params", "#FFE0C2"),
        (6.8, "4. Frozen probe\nResNet-18 MoCo", "#CDE4F5"),
        (9.0, "5. Compare\nF1, PR-AUC, CI", "#F5E6C8"),
        (11.0, "6. Errors &\nlimits of n=20", "#F2D0D0"),
    ]
    for x, text, c in boxes:
        ax.add_patch(plt.Rectangle((x, 0.7), 1.9, 1.8, facecolor=c, edgecolor="#333", lw=1.2, zorder=2))
        ax.text(x + 0.95, 1.6, text, ha="center", va="center", fontsize=8, zorder=3)
        if x < 10:
            ax.annotate("", xy=(x + 2.15, 1.6), xytext=(x + 1.95, 1.6), arrowprops=dict(arrowstyle="->", color="#333"))
    ax.set_title("Study flow: physics first, then capacity-controlled learners, then a frozen pretrained probe", pad=8)
    return _save(fig, path)


def plot_metrics_bar(results: list[Result], path: Path, title: str) -> Path:
    keys = ["accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(keys))
    n = len(results)
    w = 0.8 / max(n, 1)
    for i, r in enumerate(results):
        vals = [getattr(r, k) for k in keys]
        bars = ax.bar(x + (i - n / 2) * w + w / 2, vals, w, label=r.name, color=FAMILY_COLOR.get(r.family, "#333"))
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.axhline(0.8, color="grey", ls="--", lw=0.8, alpha=0.8)
    ax.text(len(keys) - 0.4, 0.82, "majority acc ≈ 0.80", fontsize=7, color="grey")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, path)


def plot_pr_roc(results: list[Result], path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for r in results:
        fpr, tpr = roc_points(r.y_true, r.probs)
        p, rec = pr_points(r.y_true, r.probs)
        c = FAMILY_COLOR.get(r.family, "#333")
        axes[0].plot(fpr, tpr, color=c, label=f"{r.name} ({r.roc_auc:.2f})")
        axes[1].plot(rec, p, color=c, label=f"{r.name} ({r.pr_auc:.2f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title("ROC")
    prev = float(np.mean(results[0].y_true))
    axes[1].axhline(prev, color="grey", ls="--", lw=0.8)
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("Precision–recall")
    axes[0].legend(fontsize=6, loc="lower right")
    axes[1].legend(fontsize=6, loc="lower left")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, path)


def plot_confusion(results: list[Result], path: Path) -> Path:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.3))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = np.asarray(r.cm)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(int(cm.max()), 1))
        for a in range(2):
            for b in range(2):
                ax.text(b, a, int(cm[a, b]), ha="center", va="center", fontsize=13)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["no water", "water"], fontsize=8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["no water", "water"], fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{r.name}\nF1 {r.f1:.2f}", fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


def plot_cv_bars(cv_table: dict[str, dict[str, float]], path: Path) -> Path:
    names = list(cv_table.keys())
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(names))
    for offset, metric, color in ((-0.2, "f1", "#1F77B4"), (0.2, "pr_auc", "#FF7F0E")):
        means = [cv_table[n][f"{metric}_mean"] for n in names]
        stds = [cv_table[n][f"{metric}_std"] for n in names]
        ax.bar(x + offset, means, 0.35, yerr=stds, capsize=4, label=metric.upper() if metric != "pr_auc" else "PR-AUC", color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("5-fold mean ± std")
    ax.set_title("Stability on the same 100 chips (stratified 5-fold CV)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, path)


def plot_curves(history: dict, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, path)


def plot_errors(split: SplitData, result: Result, path: Path) -> Path:
    wrong = np.where(result.pred != result.y_true)[0]
    if len(wrong) == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, f"{result.name}: no errors on the 20-image test set", ha="center", va="center")
        return _save(fig, path)
    k = min(len(wrong), 8)
    fig, axes = plt.subplots(2, k, figsize=(2.6 * k, 5.4))
    if k == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for j in range(k):
        i = int(wrong[j])
        true = "water" if result.y_true[i] else "no water"
        pred = "water" if result.pred[i] else "no water"
        cls = split.class_names[int(split.y_class[i])]
        axes[0, j].imshow(rgb_image(split.x[i], split.bands))
        axes[0, j].set_title(f"true {true} / pred {pred}\n{cls}\np={result.probs[i]:.2f}", fontsize=8)
        axes[0, j].axis("off")
        axes[1, j].imshow(swir_nir_g(split.x[i], split.bands))
        axes[1, j].axis("off")
    axes[0, 0].set_ylabel("RGB")
    axes[1, 0].set_ylabel("SWIR-NIR-G")
    fig.suptitle(f"Misclassified test chips — {result.name}", y=0.98)
    fig.tight_layout()
    return _save(fig, path)


def write_metrics_markdown(results: list[Result], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["name", "family", "accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc", "threshold"]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for r in results:
        row = r.as_row()
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n")
    return path


def write_csv(results: list[Result], path: Path) -> Path:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.as_row() for r in results]
    if not rows:
        return path
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return path
