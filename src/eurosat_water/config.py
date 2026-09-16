"""Shared constants. The study never leaves EuroSAT100 (100 chips)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
REFERENCE_DIR = ROOT / "reference" / "original-eurosat-water"

SEED = 42
N_CV_FOLDS = 5
TINY_MAX_EPOCHS = 40
TINY_PATIENCE = 8
TINY_LR = 1e-3
TINY_DROPOUT = 0.4
PROBE_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)

# EuroSAT100 is L1C TOA reflectance stored as DN ≈ reflectance × 10000.
REFLECTANCE_SCALE = 10000.0
