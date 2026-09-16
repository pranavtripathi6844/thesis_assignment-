"""Tiny CNN (from scratch) and frozen Sentinel-2 ResNet-18 linear probe."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from eurosat_water.bands import align_indices, keep_mask, weight_band_names
from eurosat_water.config import REFLECTANCE_SCALE, TINY_DROPOUT


class TinyCNN(nn.Module):
    """~25k params at 13 input channels. Capacity is the point."""

    def __init__(self, in_ch: int, dropout: float = TINY_DROPOUT):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x).squeeze(1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_frozen_resnet18(device: torch.device):
    from torchgeo.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.SENTINEL2_ALL_MOCO
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    raw = weights.meta.get("bands") or weights.meta.get("in_chans")
    if raw is None or isinstance(raw, int):
        # SSL4EO-S12 default order
        wb = (
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B8A",
            "B9",
            "B10",
            "B11",
            "B12",
        )
    else:
        wb = tuple(raw)
    return model, weights, weight_band_names(wb)


def to_reflectance_tensor(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if float(x.max()) > 1.5:
        return x / REFLECTANCE_SCALE
    return x


def zero_mask_to_weight_order(
    x: torch.Tensor,
    data_bands: tuple[str, ...],
    keep: tuple[str, ...],
    weight_bands: tuple[str, ...],
) -> torch.Tensor:
    """Reorder to the pretrained stem and zero unused channels (already reflectance)."""
    x = to_reflectance_tensor(x)
    idx = align_indices(data_bands, weight_bands)
    aligned = x[:, idx].clone()
    mask = keep_mask(weight_bands, keep)
    for i, on in enumerate(mask):
        if not on:
            aligned[:, i] = 0
    return aligned


@torch.no_grad()
def extract_probe_features(
    model: nn.Module,
    x: torch.Tensor,
    data_bands: tuple[str, ...],
    keep: tuple[str, ...],
    weight_bands: tuple[str, ...],
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    """MoCo stem expects DN/10000 = reflectance. Then Resize(256) + CenterCrop(224)."""
    masked = zero_mask_to_weight_order(x, data_bands, keep, weight_bands)
    masked = F.interpolate(masked, size=256, mode="bilinear", align_corners=False)
    off = (256 - 224) // 2
    masked = masked[:, :, off : off + 224, off : off + 224]
    feats = []
    model.eval()
    for i in range(0, len(masked), batch_size):
        batch = masked[i : i + batch_size].to(device)
        feats.append(model(batch).detach().cpu())
    return torch.cat(feats, dim=0)
