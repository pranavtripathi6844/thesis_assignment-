#!/usr/bin/env python3
"""Dataset inventory + EDA figures (train split only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eurosat_water.config import RESULTS_DIR
from eurosat_water.data import band_mean_table, load_all, summarise
from eurosat_water.plotting import plot_band_boxplots, plot_examples, plot_index_hist, plot_pipeline


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    splits = load_all()
    summary = summarise(splits)
    print(json.dumps(summary, indent=2))
    print("class names:", splits["train"].class_names)
    print("bands:", splits["train"].bands)
    print("train band means:", json.dumps(band_mean_table(splits["train"]), indent=2))
    plot_pipeline(RESULTS_DIR / "fig_pipeline.png")
    plot_examples(splits["train"], RESULTS_DIR / "fig_examples.png")
    plot_band_boxplots(splits["train"], RESULTS_DIR / "fig_band_boxplots.png")
    plot_index_hist(splits["train"], RESULTS_DIR / "fig_index_hist.png")
    print(f"saved EDA figures in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
