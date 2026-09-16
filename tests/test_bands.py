import numpy as np
import pytest

from eurosat_water.bands import canonical, find_band, keep_mask, resolve_keep
from eurosat_water.evaluate import best_threshold, wilson_interval


def test_canonical_aliases():
    assert canonical("B01") == "B1"
    assert canonical("B1") == "B1"
    assert canonical("B08") == "B8"
    assert canonical("B8A") == "B8A"
    assert canonical("B08A") == "B8A"
    assert canonical("B10") == "B10"
    assert canonical("B11") == "B11"


def test_find_band_independent_of_spelling():
    bands = ("B01", "B02", "B03", "B04", "B08", "B8A", "B11")
    assert find_band(bands, "B03") == 2
    assert find_band(bands, "B8") == 4
    assert find_band(bands, "B08A") == 5


def test_resolve_keep_uses_dataset_spelling():
    all_bands = ("B01", "B02", "B03", "B04", "B08", "B11", "B12")
    keep = resolve_keep("Water bands", all_bands)
    assert keep == ("B03", "B08", "B11", "B12")


def test_keep_mask_by_canonical_name():
    weight = ("B1", "B2", "B3", "B4", "B8", "B11", "B12")
    mask = keep_mask(weight, ("B03", "B08", "B11", "B12"))
    assert mask == [False, False, True, False, True, True, True]


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(4, 4)
    assert 0 <= lo <= 1 and 0 <= hi <= 1
    assert hi >= lo
    # 4/4 is not a 100% CI down to 0
    assert lo > 0.4


def test_resolve_keep_unknown_name_raises():
    all_bands = ("B01", "B02", "B03", "B04")
    with pytest.raises(KeyError, match="Unknown band set"):
        resolve_keep("not-a-set", all_bands)


def test_best_threshold_uses_validation_scores_not_fixed_grid():
    # Grid 0.05–0.95 would miss the separating cut at 0.025.
    scores = np.array([0.01, 0.02, 0.03, 0.04])
    y = np.array([0, 0, 1, 1])
    t, f1 = best_threshold(scores, y)
    pred = (scores >= t).astype(int)
    assert f1 == 1.0
    assert np.array_equal(pred, y)
    assert 0.02 < t <= 0.03


def test_best_threshold_tie_breaks_to_higher_cut():
    # No positives: every cut has F1=0; pick the highest (all-negative).
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    y = np.array([0, 0, 0, 0])
    t, f1 = best_threshold(scores, y)
    assert f1 == 0.0
    assert t > 0.4
    assert np.all(scores < t)
