# Water-scene classification under a 100-label budget

## 1. Problem

The practical question is simple: given a small Sentinel-2 chip, is the scene a **water scene** or not? We work with EuroSAT100, 100 labelled 64×64 images with 13 spectral bands. We map the original ten land-cover classes to a binary label: **water** if the official class is River or SeaLake (20 chips), **non-water** otherwise (80 chips). The official split is 60 / 20 / 20 images, which leaves only 12 water chips for training and **four** in each of validation and test.

This is **scene classification**, not water **segmentation**. Each tile receives one label, namely the dominant land-cover class. A highway or crop field that happens to contain a river can still be a true negative. A River tile still contains banks and vegetation. At 10 m ground sampling distance a 64×64 chip covers only 640 m × 640 m, so mixed pixels are expected. We therefore do not claim flood extent, shoreline, or per-pixel water fraction. We ask only whether a cheap, honest classifier can recover the water-scene label when labelled data are scarce.

Two properties of the problem drive the design. First, water has a well-known spectral signature: it reflects in the visible green and absorbs in the near-infrared and short-wave infrared. Second, 80% of the chips are non-water, so a classifier that always answers “no water” already reaches **80% accuracy**. Any method we propose has to beat that trivial floor, and accuracy alone cannot be the headline number.

## 2. A three-stage approach

We did not start from a large neural network. We built three models of increasing complexity, each with a different scientific role, and we compared them on the **same** 100 chips. The point is not to declare a winner from four test positives. The point is to see whether learned spatial features or pretrained representations add anything reliable beyond physics.

### Stage 1 — Physics (the reference)

Water indices exist precisely for this job. We compute NDWI from green and NIR (B03, B08) and MNDWI from green and SWIR1 (B03, B11). For each tile we summarise the index map by its mean, its maximum, and the fraction of pixels above two wetness cuts. That is eight numbers. A regularised logistic regression, and a very small random forest, are then trained on those features. We also report a one-parameter rule: threshold the tile-mean index, with the cut chosen on the validation set only.

**Motivation.** With 60 training images the safest model is the one with the fewest free parameters and the strongest prior. The logistic index model has on the order of ten coefficients. If this already works, a neural network has to *beat it*, not merely look more sophisticated. We treat this stage as the **reference**, not as a warm-up to be discarded.

### Stage 2 — A tiny CNN from scratch

The spectral summaries ignore spatial layout: a narrow river and a lake can have similar mean NDWI but different shapes. We therefore train a small convolutional network (three conv–BN–ReLU blocks, global average pooling, dropout, a single logit) on either the four water-related bands or all 13 bands. It has about 24–25 thousand trainable weights, class-weighted binary cross-entropy, early stopping on validation loss, and no augmentation.

**Motivation.** This is the “can we learn from pixels with a budget matched to *n*?” experiment. It is deliberately much smaller than a ResNet. We do not claim that 25k parameters are *proven* to match 60 images; we only claim that this is a capacity-reduced spatial learner, and that its instability, if any, is a result to report.

### Stage 3 — Frozen pretrained ResNet-18

Sentinel-2 already has public self-supervised weights (MoCo on SSL4EO-S12). We load a ResNet-18, **freeze the backbone**, extract a 512-dimensional embedding, and fit a regularised logistic head. Unused bands are zeroed so that every band subset uses the same stem. We compare RGB, RGB+NIR, the water-band subset, all 13 bands, and a few random four-band controls.

**Motivation.** Fine-tuning eleven million weights on 60 labels is not a reasonable estimator. A frozen extractor reuses unlabelled pretraining without adding labelled EuroSAT images. This stage asks whether a sensor-specific representation helps when we are only allowed to train the last linear layer.

The three stages are **separate models**, not one stacked system. At inference we do not vote or cascade them. We compare them.

## 3. Protocol

Hyperparameters and decision thresholds are chosen on **validation** only. The test set is touched once. Thresholds are taken from unique validation scores (including the all-positive and all-negative extremes); ties keep the higher cut, i.e. fewer water calls. Index rules use **raw** NDWI/MNDWI, not a transform that sees the test distribution. Because a single 20-image test set is dominated by four water chips, we also run stratified **five-fold cross-validation on the same 100 images** as a stability check, not as extra data.

## 4. Why these metrics (and why not accuracy alone)

The class prior is 0.80 non-water. Accuracy of 0.80 is therefore the majority-class baseline, not evidence of skill. We keep accuracy as a sanity check and put the weight on metrics that treat water as the positive class.

**Accuracy.** *Pro:* easy to explain; useful once compared with the 0.80 floor (anything below that is worse than “always say no water”). *Con:* dominated by the negative class; a model that never finds water can still look strong.

**Precision.** *Pro:* answers “when we call water, how often is it a water scene?”; penalises false alarms. *Con:* can be high if the model is overly conservative and misses most water chips.

**Recall (sensitivity).** *Pro:* answers “how many of the true water scenes do we catch?”; the quantity that matters if missing water is costly. *Con:* with only four test positives, one miss moves recall by 0.25. A Wilson 95% interval on 3/4 is roughly 0.30–0.95, which is almost uninformative as a point claim.

**F1.** *Pro:* harmonic mean of precision and recall; a single thresholded summary that does not ignore the minority class. *Con:* depends on the chosen cut. With four validation positives that cut is noisy, so official-split F1 can look perfect while the same model is unstable under resampling.

**ROC-AUC.** *Pro:* ranking quality, threshold-free; chance is 0.5. *Con:* with 80% negatives it can look optimistic: a model can rank well and still be unusable at an operating point. It does not describe the error we would actually ship.

**PR-AUC (average precision).** *Pro:* ranking quality with respect to the positive class; chance equals the positive rate (0.20 here), so it is harder to inflate than ROC-AUC. *Con:* still threshold-free, so it does not tell us whether the validation-selected cut is any good. In this study several models reach AP = 1.0 on the official test set while F1 is only 0.86: they **order** the twenty chips correctly but the **cut** chosen on validation misses one water scene. That distinction is part of the result.

We also report confusion counts, Wilson intervals on accuracy and recall, and McNemar tests between models on the same twenty test chips. The intervals and tests are there to stop us from over-interpreting 20/20 or 19/20.

## 5. What the experiments showed

On the official test set, majority accuracy is 0.80 with F1 = 0. A raw NDWI threshold already reaches F1 ≈ 0.86. Logistic regression on the eight index statistics reaches **AP = 1.0** and **F1 = 0.86**: perfect ranking, one threshold error. The tiny CNN on water bands can still look perfect on this particular split (F1 = 1.0), as can the frozen probe with all bands. RGB-only embeddings do not (F1 = 0.33). Pairwise McNemar tests are not significant; with four positives they cannot be.

Cross-validation on the same 100 chips is the more honest summary. Logistic index features give F1 **0.80 ± 0.17**. The frozen probe is in the same band (water bands **0.83 ± 0.13**, all bands **0.77 ± 0.21**). The tiny CNN is worse and much noisier (**0.63 ± 0.34**). Neural complexity did not buy stability.

## 6. Reading

We approached the problem as a small-sample remote-sensing question, not as a contest to maximise accuracy. Physics first, because the spectrum of water is known; a small CNN next, to test spatial learning under a tight parameter budget; a frozen Sentinel-2 ResNet last, to use pretraining without fine-tuning millions of weights on 60 labels. Accuracy is reported against an 0.80 floor. F1 and PR-AUC are the headline metrics because they respect the minority class; we keep them separate because one is a cut and the other is a ranking. The evidence we have is that **spectral features are a competitive reference**, that a tiny CNN is not shown to be reliable at this *n*, and that four test positives cannot support a claim of superiority. If more labels arrive, the same ladder still applies: keep the index model, keep the frozen probe, and only then unfreeze the backbone.
