#!/usr/bin/env python3
"""Run the full EuroSAT100 water-scene study and write thesis figures/tables.

Does not download extra labelled images. Official 60/20/20 is the headline;
5-fold CV reuses the same 100 chips as a stability check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eurosat_water.bands import resolve_keep
from eurosat_water.baselines import (
    index_threshold,
    logistic_on_features,
    majority,
    random_forest_indices,
)
from eurosat_water.config import N_CV_FOLDS, RESULTS_DIR, SEED
from eurosat_water.data import band_mean_table, load_all, stack_splits, subset, summarise
from eurosat_water.evaluate import mcnemar_exact
from eurosat_water.models import get_device, load_frozen_resnet18
from eurosat_water.plotting import (
    plot_band_boxplots,
    plot_confusion,
    plot_cv_bars,
    plot_curves,
    plot_errors,
    plot_examples,
    plot_index_hist,
    plot_metrics_bar,
    plot_pipeline,
    plot_pr_roc,
    write_csv,
    write_metrics_markdown,
)
from eurosat_water.train import fit_linear_probe, train_tiny_cnn


def _dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def _result_record(r) -> dict:
    d = r.as_row()
    d["probs"] = r.probs.tolist()
    d["pred"] = r.pred.tolist()
    d["acc_ci"] = list(r.acc_ci)
    d["recall_ci"] = list(r.recall_ci)
    extra = dict(r.extra)
    extra.pop("history", None)
    extra.pop("importances", None)
    d["extra"] = extra
    return d


def random_keep_sets(all_bands: tuple[str, ...], k: int = 4, draws: int = 3, seed: int = 0):
    rng = np.random.RandomState(seed)
    out = []
    for i in range(draws):
        chosen = tuple(rng.choice(np.array(all_bands), size=k, replace=False).tolist())
        out.append((f"Random {k} bands #{i + 1}", chosen))
    return out


def run_official(splits, device):
    train, val, test = splits["train"], splits["val"], splits["test"]
    results = []

    print("\n== Official split: physics and simple classifiers ==")
    results.append(majority(train.y, test.y))
    results.append(index_threshold(train, val, test, "ndwi"))
    results.append(index_threshold(train, val, test, "mndwi"))
    results.append(logistic_on_features(train, val, test, "indices"))
    results.append(logistic_on_features(train, val, test, "combined"))
    results.append(random_forest_indices(train, val, test))
    for r in results:
        print(f"  {r.name:40s}  F1={r.f1:.3f}  PR-AUC={r.pr_auc:.3f}  acc={r.accuracy:.3f}")

    print("\n== Official split: tiny CNN ==")
    cnn_histories = {}
    for label, key in (("Tiny CNN (water bands)", "Water bands"), ("Tiny CNN (all bands)", "All bands")):
        keep = resolve_keep(key, train.bands)
        r, extra = train_tiny_cnn(train, val, test, keep, label, device)
        results.append(r)
        cnn_histories[label] = extra["history"]
        print(f"  {r.name:40s}  F1={r.f1:.3f}  PR-AUC={r.pr_auc:.3f}  params={extra['params']}")

    print("\n== Official split: frozen ResNet-18 linear probe (zero-mask) ==")
    model, _, weight_bands = load_frozen_resnet18(device)
    print(f"  pretrained channels: {weight_bands}")
    probe_specs = [
        ("Probe RGB", resolve_keep("RGB", train.bands)),
        ("Probe RGB+NIR", resolve_keep("RGB+NIR", train.bands)),
        ("Probe water bands", resolve_keep("Water bands", train.bands)),
        ("Probe all bands", resolve_keep("All bands", train.bands)),
    ]
    probe_specs.extend(random_keep_sets(train.bands))
    for name, keep in probe_specs:
        r = fit_linear_probe(model, weight_bands, train, val, test, keep, name, device)
        results.append(r)
        print(f"  {r.name:40s}  F1={r.f1:.3f}  PR-AUC={r.pr_auc:.3f}  bands={keep}")

    return results, cnn_histories


def _inner_val_indices(idx: np.ndarray, y: np.ndarray, seed: int):
    if len(idx) < 8:
        return idx, idx
    tr, va = train_test_split(idx, test_size=0.25, stratify=y[idx], random_state=seed)
    return tr, va


def run_cv(full, device):
    print("\n== Supplementary: stratified 5-fold CV on all 100 chips ==")
    y = full.y
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
    model, _, weight_bands = load_frozen_resnet18(device)
    keep_water = resolve_keep("Water bands", full.bands)
    keep_all = resolve_keep("All bands", full.bands)

    fold_scores = {
        "Majority": {"f1": [], "pr_auc": []},
        "Logistic indices": {"f1": [], "pr_auc": []},
        "Logistic + means": {"f1": [], "pr_auc": []},
        "Tiny CNN (water)": {"f1": [], "pr_auc": []},
        "Probe water": {"f1": [], "pr_auc": []},
        "Probe all": {"f1": [], "pr_auc": []},
    }

    def _add(name, result):
        fold_scores[name]["f1"].append(result.f1)
        fold_scores[name]["pr_auc"].append(result.pr_auc)

    for fold, (tr_all, te) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        print(f"  fold {fold}/{N_CV_FOLDS}")
        tr, va = _inner_val_indices(tr_all, y, seed=SEED + fold)
        train, val, test = subset(full, tr), subset(full, va), subset(full, te)

        _add("Majority", majority(train.y, test.y))
        _add("Logistic indices", logistic_on_features(train, val, test, "indices"))
        _add("Logistic + means", logistic_on_features(train, val, test, "combined"))
        r_cnn, _ = train_tiny_cnn(train, val, test, keep_water, "cnn", device, seed=SEED + fold)
        _add("Tiny CNN (water)", r_cnn)
        _add("Probe water", fit_linear_probe(model, weight_bands, train, val, test, keep_water, "pw", device))
        _add("Probe all", fit_linear_probe(model, weight_bands, train, val, test, keep_all, "pa", device))

    table = {}
    for name, vals in fold_scores.items():
        f1 = np.asarray(vals["f1"], dtype=float)
        pr = np.asarray(vals["pr_auc"], dtype=float)
        table[name] = {
            "f1_mean": float(f1.mean()),
            "f1_std": float(f1.std(ddof=1) if len(f1) > 1 else 0.0),
            "pr_auc_mean": float(pr.mean()),
            "pr_auc_std": float(pr.std(ddof=1) if len(pr) > 1 else 0.0),
            "f1_folds": [float(v) for v in f1],
            "pr_auc_folds": [float(v) for v in pr],
        }
        print(f"  {name:22s}  F1 {table[name]['f1_mean']:.3f} ± {table[name]['f1_std']:.3f}")
    return table


def pairwise_tests(results) -> dict:
    by_name = {r.name: r for r in results}
    names = [
        "Logistic (NDWI/MNDWI stats)",
        "Tiny CNN (water bands)",
        "Probe water bands",
        "Probe all bands",
        "Majority class",
    ]
    present = [n for n in names if n in by_name]
    out = {}
    for i, a in enumerate(present):
        for b in present[i + 1 :]:
            key = f"{a} vs {b}"
            out[key] = mcnemar_exact(by_name[a].y_true, by_name[a].pred, by_name[b].pred)
    return out


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"device: {device}")

    print("Loading EuroSAT100 (100 images, 13 bands)…")
    splits = load_all()
    summary = summarise(splits)
    print(json.dumps(summary, indent=2))
    _dump_json(summary, RESULTS_DIR / "split_summary.json")
    _dump_json(band_mean_table(splits["train"]), RESULTS_DIR / "band_means_train.json")

    plot_pipeline(RESULTS_DIR / "fig_pipeline.png")
    plot_examples(splits["train"], RESULTS_DIR / "fig_examples.png")
    plot_band_boxplots(splits["train"], RESULTS_DIR / "fig_band_boxplots.png")
    plot_index_hist(splits["train"], RESULTS_DIR / "fig_index_hist.png")

    results, cnn_histories = run_official(splits, device)
    write_csv(results, RESULTS_DIR / "official_metrics.csv")
    write_metrics_markdown(results, RESULTS_DIR / "official_metrics.md")
    _dump_json([_result_record(r) for r in results], RESULTS_DIR / "official_metrics.json")

    # Headline comparison: one model from each family (not every random-4 arm).
    headline_names = {
        "Majority class",
        "NDWI threshold",
        "MNDWI threshold",
        "Logistic (NDWI/MNDWI stats)",
        "Logistic (indices + band means)",
        "Random forest (NDWI/MNDWI stats)",
        "Tiny CNN (water bands)",
        "Tiny CNN (all bands)",
        "Probe RGB",
        "Probe RGB+NIR",
        "Probe water bands",
        "Probe all bands",
    }
    headline = [r for r in results if r.name in headline_names]
    plot_metrics_bar(
        headline,
        RESULTS_DIR / "fig_metrics_official.png",
        "Official 20-image test set — do not read 1.00 as solved (4 water chips)",
    )
    plot_pr_roc(headline, RESULTS_DIR / "fig_pr_roc.png")

    conf_pick = [
        r
        for r in results
        if r.name
        in {
            "Majority class",
            "Logistic (NDWI/MNDWI stats)",
            "Tiny CNN (water bands)",
            "Probe water bands",
            "Probe all bands",
        }
    ]
    plot_confusion(conf_pick, RESULTS_DIR / "fig_confusion.png")

    probe_arms = [r for r in results if r.family == "probe"]
    plot_metrics_bar(
        probe_arms,
        RESULTS_DIR / "fig_band_ablation_probe.png",
        "Frozen probe, identical stem: which bands are visible (unused channels zeroed)",
    )

    for name, hist in cnn_histories.items():
        slug = "water" if "water" in name.lower() else "all"
        plot_curves(hist, RESULTS_DIR / f"fig_tiny_cnn_curves_{slug}.png", name)

    # Error gallery for the physics model and the all-band probe.
    for want in ("Logistic (NDWI/MNDWI stats)", "Tiny CNN (water bands)", "Probe all bands"):
        r = next((x for x in results if x.name == want), None)
        if r is not None:
            slug = want.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            plot_errors(splits["test"], r, RESULTS_DIR / f"fig_errors_{slug}.png")

    tests = pairwise_tests(results)
    _dump_json(tests, RESULTS_DIR / "mcnemar.json")
    print("\nMcNemar (official test):")
    for k, v in tests.items():
        print(f"  {k}: p={v['p_value']:.3f}  (n10={v['n10_a_wrong_b_right']}, n01={v['n01_a_right_b_wrong']})")

    full = stack_splits(splits, ("train", "val", "test"))
    cv_table = run_cv(full, device)
    _dump_json(cv_table, RESULTS_DIR / "cv_f1.json")
    plot_cv_bars(cv_table, RESULTS_DIR / "fig_cv_stability.png")

    print(f"\nWrote figures and tables to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
