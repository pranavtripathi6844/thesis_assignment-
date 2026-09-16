# 23-page thesis report — structure

Use this as the skeleton. After `python scripts/run_all.py`, paste numbers from `results/official_metrics.md` and embed the PNGs listed below. Write in the order of the experiment, not as a leaderboard.

**Narrative:** we were given 100 chips and told not to grow the labelled set. A perfect score on 20 test images (4 water) is **weak evidence of generalisation**, not proof of overfitting or of a solved task — our own spectral logistic can also hit 100% on that split. We start from physics as the **reference**, then a capacity-reduced CNN, then a frozen pretrained probe, and we report uncertainty instead of a winner.

---

## Page plan

| Pages | Section | Figures / tables |
|---|---|---|
| 1 | Title, abstract, keywords | — |
| 2–3 | 1. Introduction and task | Table 1: constraints |
| 4 | 2. Related work (short) | — |
| 5–7 | 3. Data | Fig 1 examples, Fig 2 boxplots, Fig 3 NDWI hist, Table 2 splits, Table 3 band means |
| 8 | 4. Design overview | Fig 4 pipeline |
| 9–10 | 5. Physics baselines | Table 4 majority / NDWI / MNDWI / logistic / RF |
| 11–12 | 6. Tiny CNN | Fig 5 train/val curves, Table 5 param count |
| 13–14 | 7. Frozen ResNet-18 probe | Table 6 band arms, Fig 6 ablation |
| 15–17 | 8. Official-split results | Fig 7 metric bars, Fig 8 PR/ROC, Fig 9 confusion, Table 7 with CIs |
| 18–19 | 9. Stability and errors | Fig 10 CV, Fig 11 error gallery, Table 8 McNemar |
| 20–21 | 10. Discussion (why not 100%; how this scales) | Table 9 original repo vs this work |
| 22 | 11. Conclusion | — |
| 23 | References | Helber 2019, TorchGeo, SSL4EO-S12 |

If a section runs short, expand qualitative error analysis and the 10 m GSD / mixed-pixel discussion — not extra experiments.

---

## Section notes (what to argue)

### Abstract (½ page)
EuroSAT100, 20 water / 80 non-water, official 60/20/20. We compare a spectral-index classifier, a ~25k-parameter CNN, and a frozen Sentinel-2 ResNet-18 linear probe. We do not claim a ranking from 4 test positives. The contribution is a small-n protocol that remains valid if the labelled set grows.

### 1. Introduction
- Why water scenes in Sentinel-2 (flood, mapping, NDWI literature).
- Constraint: **exactly 100 labelled chips**, no extra labelled EuroSAT images, accuracy is not the grade. Unlabelled Sentinel-2 pretraining (MoCo) is allowed.
- **Scene classification ≠ segmentation.** Each chip has one label: water scene iff the dominant EuroSAT class is River or SeaLake. The model does not output a water mask, flood extent, or shoreline. A highway/field chip that contains a river can be a **true negative**. A River chip still includes banks and vegetation. Mixed pixels at 10 m / 64×64 are expected, not labelling bugs.
- A 20/20 test score is weak evidence; it needs CV, a physics reference, and leakage checks. Do not write “100% means overfitting.”

### 2. Related work (keep short)
- EuroSAT land-cover classification (Helber).
- Spectral indices NDWI (McFeeters), MNDWI (Xu).
- Self-supervised Sentinel-2 models (SSL4EO-S12 / MoCo) evaluated by linear probe.
- TorchGeo EuroSAT100 is documented as a **tutorial** set, not a benchmark.

### 3. Data
- 13 bands, 64×64, 10 m GSD, L1C TOA (DN ≈ reflectance × 10000).
- Table 2 from `results/split_summary.json`.
- Fig 1: RGB vs SWIR-NIR-G, **different classes**, 2–98% stretch (caption: stretch, not raw).
- Fig 2: water absorbs NIR/SWIR — this is the hypothesis in a plot.
- Fig 3: tile-mean NDWI/MNDWI histograms on **train only**.
- Spatial caveat: a River chip may be a few dark pixels; SeaLake is almost all water; some non-water chips still contain water.

### 4. Design overview
Embed `fig_pipeline.png`. One study, three models of increasing complexity. Same splits, same labels, no augmentation.

### 5. Physics
- Majority class: accuracy floor ≈ 0.80. Anything below this is worse than “always say no water.”
- NDWI and MNDWI: **raw** tile-mean scores; cut chosen on **val only**; PR/ROC from raw scores (no test-set scaling).
- Logistic / RF on index stats (mean, max, wet-pixel fractions); `C` and threshold on **val**.
- This logistic model is the **reference**. Neural nets have to beat it on CV, not on one 20-image split.

### 6. Tiny CNN
- 16-32-64 conv, GAP, dropout 0.4, ~25k weights vs 60 images.
- BCE with `pos_weight`, Adam, early stopping on val loss, threshold on val F1.
- Native 64×64, selected bands (water set and all 13).
- Show train vs val curves: if val rises while train falls, that is overfitting — discuss it, do not hide it.

### 7. Frozen probe
- 11 million weights are **not trained**. Only a logistic head on 512-D features.
- Pretrained on unlabelled Sentinel-2 (SSL4EO MoCo), **not** on extra EuroSAT labels.
- Zero-mask ablation: same stem, unused channels set to 0 after reflectance scaling.
- Arms: RGB, RGB+NIR, water bands, all 13, random-4 × 3 (control: “any 4 bands” vs NIR/SWIR).
- Preprocessing: DN/10000, Resize 256, CenterCrop 224. No train-set z-score (B10 would explode).

### 8. Results
- Lead with F1 and PR-AUC, not accuracy. Thresholds come from unique **val** scores (tie-break: higher cut).
- Print Wilson CIs on accuracy and recall. With 4 positives, recall CI is huge — that is the point.
- Ranking (AP) and thresholded F1 are different questions when val has 4 positives.
- Random-4 vs water-4: if random-4 is close, you cannot claim a unique NIR/SWIR effect on n=20.
- Do **not** write “our model achieves state of the art” or “100% is overfitting.”

### 9. Stability and errors
- 5-fold CV **reuses the 100 chips**. Label it supplementary.
- McNemar on the same 20 test images; expect p > 0.05. Write that pairwise differences are not significant.
- Error gallery: which River tiles fail (narrow rivers), which non-water tiles fire (shadows, wet soil).

### 10. Discussion
Comparison table:

| | Reference repo | This study |
|---|---|---|
| Trainable params | ~11M | 1 cut / ~25k / ~512 |
| Val split | unused | early stop + threshold |
| Physics baseline | none | NDWI, MNDWI, logistic, RF |
| Band ablation | 3/4/13 channels, different stems | same stem, zero-mask + random-4 |
| Perfect test score | 20/20 | not the objective |
| When data grows | must be redesigned | unfreeze last blocks, same code |

Scaling paragraph: (1) indices stay, (2) linear spectral model stays, (3) frozen probe stays, (4) only then fine-tune. Today we stop at (3).

Limitations: n=20 test, scene-level labels, L1C not surface reflectance, Europe only, no season hold-out.

### 11. Conclusion
A conservative small-n protocol for water-scene classification. Physics first. Capacity matched to n. Pretrained weights used as a frozen sensor-specific extractor. Uncertainty reported. Ready to absorb more labels without changing the question.

---

## Figure checklist (from `results/` after `run_all.py`)

- `fig_pipeline.png`
- `fig_examples.png`
- `fig_band_boxplots.png`
- `fig_index_hist.png`
- `fig_tiny_cnn_curves_water.png` / `_all.png`
- `fig_metrics_official.png`
- `fig_pr_roc.png`
- `fig_confusion.png`
- `fig_band_ablation_probe.png`
- `fig_cv_stability.png`
- `fig_errors_*.png`

Tables: `official_metrics.md`, `split_summary.json`, `band_means_train.json`, `cv_f1.json`, `mcnemar.json`.
