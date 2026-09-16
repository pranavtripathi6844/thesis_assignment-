# Water-scene classification on 100 Sentinel-2 chips

Thesis study on **EuroSAT100** (exactly 100 images). The goal is not a high accuracy number. The goal is a **pipeline that is honest on 100 labels and still the right pipeline when more images arrive**.

This repository is the thesis method only. It is not a fork of the fine-tuned ResNet-18 demo that reports 100% on 20 test chips.

## What this project does

This is **binary scene classification**, not water **segmentation**. Each 64×64 chip has one label: **water scene** if EuroSAT’s dominant class is `River` or `SeaLake`, otherwise **non-water**. The model answers “is this tile a water-class scene?”, not “which pixels are water?”. A highway or field that contains a river can still be a true negative. Mixed pixels at 10 m GSD are expected.

Official split only: **60 / 20 / 20** (12 / 4 / 4 water). No extra **labelled** EuroSAT images, no synthetic chips, no geometric augmentation. Sentinel-2 MoCo weights are allowed as an unlabelled pretrained extractor.

Three complementary models, in this order:

1. **Physics** — NDWI / MNDWI (mean, max, wet-pixel fraction) + logistic / small random forest. Also a majority-class floor (~0.80 accuracy).
2. **Tiny CNN** from scratch (~25k parameters, dropout, early stopping on **val**).
3. **Frozen Sentinel-2 ResNet-18** (MoCo / SSL4EO) as a feature extractor + logistic head (~512 trained parameters). Unused bands are **zero-masked** so every probe arm has the same stem.

Band arms for the probe: RGB, RGB+NIR, water bands (B03/B08/B11/B12), all 13, and three random 4-band controls.

Headline metrics: **F1** and **PR-AUC**, with Wilson intervals on the 20-image test set. Stratified **5-fold CV on the same 100 chips** is supplementary, not extra data.

## Setup

Python 3.11 recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# inventory + EDA figures
python scripts/explore.py

# full study (official split + 5-fold CV + all report figures)
python scripts/run_all.py

# unit tests that do not download data
pytest
```

Outputs go to `results/`: PNG figures, `official_metrics.md`, CSV/JSON, McNemar tests, CV table.

EuroSAT100 downloads into `data/` on first run (gitignored).

## Report

See [`report/THESIS_OUTLINE.md`](report/THESIS_OUTLINE.md) for the 23-page structure (question → physics → tiny net → frozen probe → comparison → limits). Fill numbers from `results/` after `run_all.py`.

## Citations

- Helber et al., EuroSAT (2019)
- Stewart et al., TorchGeo (2022)
- Wang et al., SSL4EO-S12 (2023) for the MoCo weights
