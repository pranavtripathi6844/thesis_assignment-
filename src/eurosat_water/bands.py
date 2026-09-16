"""Band-set definitions. Always refer to Sentinel-2 bands by name, never by index."""

from __future__ import annotations

BAND_SETS: dict[str, tuple[str, ...] | None] = {
    "RGB": ("B04", "B03", "B02"),
    "RGB+NIR": ("B04", "B03", "B02", "B08"),
    "Water bands": ("B03", "B08", "B11", "B12"),
    "All bands": None,  # filled from the dataset at runtime
}

WATER_INDEX_BANDS = ("B03", "B08", "B11")


def canonical(name: str) -> str:
    """Map B01/B1, B08A/B8A, etc. onto a single token."""
    n = str(name).strip().upper().replace("_", "")
    if n in {"B8A", "B08A"}:
        return "B8A"
    if n.startswith("B") and len(n) == 3 and n[1] == "0":
        return "B" + n[2]
    return n


def find_band(bands: tuple[str, ...] | list[str], wanted: str) -> int:
    target = canonical(wanted)
    for i, name in enumerate(bands):
        if canonical(name) == target:
            return i
    raise KeyError(f"Band {wanted!r} not in {tuple(bands)}")


def resolve_keep(band_set: str | tuple[str, ...], all_bands: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(band_set, tuple):
        wanted = band_set
    elif band_set not in BAND_SETS:
        raise KeyError(f"Unknown band set {band_set!r}. Known: {tuple(BAND_SETS)}")
    elif BAND_SETS[band_set] is None:
        return tuple(all_bands)
    else:
        wanted = BAND_SETS[band_set]
    return tuple(all_bands[find_band(all_bands, b)] for b in wanted)


def weight_band_names(meta_bands) -> tuple[str, ...]:
    return tuple(canonical(b) for b in meta_bands)


def align_indices(data_bands: tuple[str, ...], weight_bands: tuple[str, ...]) -> list[int]:
    """For each pretrained channel, the matching dataset channel index."""
    return [find_band(data_bands, b) for b in weight_bands]


def keep_mask(weight_bands: tuple[str, ...], keep: tuple[str, ...]) -> list[bool]:
    keep_c = {canonical(b) for b in keep}
    return [canonical(b) in keep_c for b in weight_bands]
