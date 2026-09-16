"""Physics features: NDWI, MNDWI, and per-tile band statistics."""

from __future__ import annotations

import numpy as np
import torch

from eurosat_water.bands import find_band
from eurosat_water.data import SplitData, to_reflectance


def _index(g: np.ndarray, other: np.ndarray) -> np.ndarray:
    return (g - other) / (g + other + 1e-8)


def water_indices(split: SplitData) -> dict[str, np.ndarray]:
    """Pixel-wise NDWI (green/NIR) and MNDWI (green/SWIR1), shape (N, H, W)."""
    x = to_reflectance(split.x).numpy()
    g = x[:, find_band(split.bands, "B03")]
    nir = x[:, find_band(split.bands, "B08")]
    swir = x[:, find_band(split.bands, "B11")]
    return {"ndwi": _index(g, nir), "mndwi": _index(g, swir)}


def index_feature_matrix(split: SplitData) -> tuple[np.ndarray, list[str]]:
    idx = water_indices(split)
    cols = []
    names = []
    for key, arr in idx.items():
        cols.append(arr.mean(axis=(1, 2)))
        names.append(f"{key}_mean")
        cols.append(arr.max(axis=(1, 2)))
        names.append(f"{key}_max")
        cols.append((arr > 0.0).mean(axis=(1, 2)))
        names.append(f"{key}_frac_gt0")
        cols.append((arr > 0.3).mean(axis=(1, 2)))
        names.append(f"{key}_frac_gt03")
    return np.stack(cols, axis=1).astype(np.float64), names


def band_mean_matrix(split: SplitData) -> tuple[np.ndarray, list[str]]:
    x = to_reflectance(split.x).numpy()
    means = x.mean(axis=(2, 3)).astype(np.float64)
    names = [f"mean_{b}" for b in split.bands]
    return means, names


def combined_feature_matrix(split: SplitData) -> tuple[np.ndarray, list[str]]:
    a, na = index_feature_matrix(split)
    b, nb = band_mean_matrix(split)
    return np.concatenate([a, b], axis=1), na + nb


def image_index_score(split: SplitData, which: str = "mndwi") -> np.ndarray:
    """One scalar per tile: mean index. Used as a threshold baseline."""
    return water_indices(split)[which].mean(axis=(1, 2))


def select_channels(x: torch.Tensor, bands: tuple[str, ...], keep: tuple[str, ...]) -> torch.Tensor:
    from eurosat_water.bands import find_band as _find

    idx = [_find(bands, b) for b in keep]
    return x[:, idx]
