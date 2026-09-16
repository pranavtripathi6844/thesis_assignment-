"""Load EuroSAT100 by band name and relabel River / SeaLake as water scenes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torchgeo.datasets import EuroSAT100

from eurosat_water.bands import canonical
from eurosat_water.config import DATA_DIR, REFLECTANCE_SCALE


def _is_water(class_name: str) -> bool:
    token = (
        class_name.lower()
        .replace(" ", "")
        .replace("&", "")
        .replace("_", "")
        .replace("-", "")
    )
    return token in {"river", "sealake", "seaandlake"}


@dataclass
class SplitData:
    x: torch.Tensor  # (N, C, H, W) float, original DNs
    y: np.ndarray  # binary
    y_class: np.ndarray  # original 10-class ids
    class_names: tuple[str, ...]
    bands: tuple[str, ...]
    split: str

    def __len__(self) -> int:
        return int(self.y.shape[0])

    @property
    def n_water(self) -> int:
        return int(self.y.sum())


def all_band_names() -> tuple[str, ...]:
    names = getattr(EuroSAT100, "all_band_names", None)
    if names:
        return tuple(names)
    return (
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B10",
        "B11",
        "B12",
    )


def load_split(split: str, root: str | None = None, download: bool = True) -> SplitData:
    bands = all_band_names()
    ds = EuroSAT100(
        root=str(root or DATA_DIR),
        split=split,
        bands=bands,
        download=download,
    )
    images = []
    labels = []
    classes = []
    class_names = tuple(ds.classes)
    water_ids = {i for i, c in enumerate(class_names) if _is_water(c)}
    for i in range(len(ds)):
        sample = ds[i]
        images.append(sample["image"].float())
        cid = int(sample["label"])
        classes.append(cid)
        labels.append(1.0 if cid in water_ids else 0.0)
    x = torch.stack(images, dim=0)
    resolved = tuple(getattr(ds, "bands", bands))
    return SplitData(
        x=x,
        y=np.asarray(labels, dtype=np.int64),
        y_class=np.asarray(classes, dtype=np.int64),
        class_names=class_names,
        bands=tuple(resolved),
        split=split,
    )


def load_all(root: str | None = None) -> dict[str, SplitData]:
    return {split: load_split(split, root=root) for split in ("train", "val", "test")}


def subset(split: SplitData, idx) -> SplitData:
    idx = np.asarray(idx)
    return SplitData(
        x=split.x[idx],
        y=split.y[idx],
        y_class=split.y_class[idx],
        class_names=split.class_names,
        bands=split.bands,
        split=f"{split.split}[{len(idx)}]",
    )


def stack_splits(splits: dict[str, SplitData], names: tuple[str, ...]) -> SplitData:
    first = splits[names[0]]
    x = torch.cat([splits[n].x for n in names], dim=0)
    y = np.concatenate([splits[n].y for n in names], axis=0)
    y_class = np.concatenate([splits[n].y_class for n in names], axis=0)
    return SplitData(
        x=x,
        y=y,
        y_class=y_class,
        class_names=first.class_names,
        bands=first.bands,
        split="+".join(names),
    )


def to_reflectance(x: torch.Tensor) -> torch.Tensor:
    """Map DNs to [0, ~1] TOA reflectance. Detect if already scaled."""
    if float(x.max()) > 1.5:
        return x / REFLECTANCE_SCALE
    return x


def summarise(splits: dict[str, SplitData]) -> dict:
    rows = {}
    for name, sp in splits.items():
        rows[name] = {
            "n": len(sp),
            "n_water": sp.n_water,
            "n_nonwater": len(sp) - sp.n_water,
            "positive_rate": float(sp.y.mean()),
            "shape": tuple(sp.x.shape),
            "dtype": str(sp.x.dtype),
            "min": float(sp.x.min()),
            "max": float(sp.x.max()),
            "bands": list(sp.bands),
        }
    return rows


def band_mean_table(split: SplitData) -> dict[str, dict[str, float]]:
    """Per-band mean reflectance, water vs non-water (train only)."""
    x = to_reflectance(split.x).numpy()
    water = split.y == 1
    out = {}
    for i, name in enumerate(split.bands):
        w = x[water, i].mean(axis=(1, 2))
        nw = x[~water, i].mean(axis=(1, 2))
        out[name] = {
            "canonical": canonical(name),
            "water_mean": float(w.mean()) if len(w) else float("nan"),
            "nonwater_mean": float(nw.mean()) if len(nw) else float("nan"),
            "water_std": float(w.std()) if len(w) else float("nan"),
            "nonwater_std": float(nw.std()) if len(nw) else float("nan"),
        }
    return out
