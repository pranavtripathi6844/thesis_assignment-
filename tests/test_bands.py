import numpy as np

from eurosat_water.bands import canonical, find_band, keep_mask, resolve_keep
from eurosat_water.evaluate import wilson_interval


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
