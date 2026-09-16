| name | family | accuracy | precision | recall | f1 | pr_auc | roc_auc | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Majority class | baseline | 0.800 | 0.000 | 0.000 | 0.000 | 0.200 | 0.500 | 0.500 |
| NDWI threshold | physics | 0.950 | 1.000 | 0.750 | 0.857 | 0.893 | 0.953 | 0.406 |
| MNDWI threshold | physics | 0.900 | 0.750 | 0.750 | 0.750 | 0.850 | 0.906 | 0.266 |
| Logistic (NDWI/MNDWI stats) | physics | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.445 |
| Logistic (indices + band means) | physics | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.482 |
| Random forest (NDWI/MNDWI stats) | physics | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.629 |
| Tiny CNN (water bands) | tiny_cnn | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.776 |
| Tiny CNN (all bands) | tiny_cnn | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.270 |
| Probe RGB | probe | 0.800 | 0.500 | 0.250 | 0.333 | 0.650 | 0.875 | 0.537 |
| Probe RGB+NIR | probe | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.298 |
| Probe water bands | probe | 0.950 | 1.000 | 0.750 | 0.857 | 1.000 | 1.000 | 0.335 |
| Probe all bands | probe | 0.950 | 0.800 | 1.000 | 0.889 | 1.000 | 1.000 | 0.188 |
| Random 4 bands #1 | probe | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.353 |
| Random 4 bands #2 | probe | 0.950 | 1.000 | 0.750 | 0.857 | 1.000 | 1.000 | 0.656 |
| Random 4 bands #3 | probe | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.454 |
