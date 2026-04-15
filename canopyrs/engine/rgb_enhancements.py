"""
RGB enhancement utilities for multi-method detection ensemble.

All single-image enhancement functions follow the convention:
  - Input:  float32 array shaped (H, W, 3) with NaN for nodata pixels
  - Output: uint8 array shaped (H, W, 3), nodata pixels filled with 0

``enhance_cir()`` additionally accepts a (H, W) NIR band.

These functions are used by RgbEnhancerComponent to produce per-method
tile sets that feed the multi-enhancement WBF detection ensemble.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from skimage import exposure


# ---------------------------------------------------------------------------
# Module-level defaults (Altum-PT → visible RGB, from M3M reference stats)
# ---------------------------------------------------------------------------

DEFAULT_TARGET_MEANS: np.ndarray = np.array([73.1, 88.2, 44.0], dtype=np.float64)
DEFAULT_TARGET_STDS: np.ndarray  = np.array([55.2, 58.2, 37.1], dtype=np.float64)

# Canonical method identifiers
ENHANCEMENT_METHODS: List[str] = [
    "linear",
    "gamma",
    "white_balance",
    "gamma_wb",
    "scurve",
    "cir",
    "blend_wb_sc",
    "stat_wt",
]

# Composite methods must be computed after the listed source base images
COMPOSITE_SOURCES: Dict[str, List[str]] = {
    "blend_wb_sc": ["gamma_wb", "scurve"],   # 60 % gamma_wb + 40 % scurve
    "stat_wt":     ["linear", "gamma", "white_balance", "gamma_wb", "scurve"],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Clip float 0–1 to uint8 0–255, filling NaN with 0."""
    out = np.nan_to_num(arr, nan=0.0)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def _per_band_pct_stretch(img: np.ndarray, p_lo: int = 2, p_hi: int = 98) -> np.ndarray:
    """Per-band percentile stretch to 0–1 float32."""
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[-1]):
        band = img[..., c]
        valid = band[np.isfinite(band)]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, p_lo), np.percentile(valid, p_hi)
        out[..., c] = (band - lo) / (hi - lo + 1e-9)
    return np.clip(out, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Named enhancement functions
# ---------------------------------------------------------------------------

def enhance_linear(img: np.ndarray) -> np.ndarray:
    """Linear per-band percentile stretch."""
    return _to_uint8(_per_band_pct_stretch(img))


def enhance_gamma(img: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Linear stretch followed by gamma correction (brightens midtones)."""
    stretched = _per_band_pct_stretch(img)
    corrected = np.power(np.clip(stretched, 0, 1), 1.0 / gamma)
    return _to_uint8(corrected)


def enhance_white_balance(
    img: np.ndarray,
    target_means: np.ndarray = DEFAULT_TARGET_MEANS,
    target_stds: np.ndarray = DEFAULT_TARGET_STDS,
) -> np.ndarray:
    """White-balance the stretched image to match per-band target statistics."""
    stretched = _per_band_pct_stretch(img) * 255.0
    out = np.zeros_like(stretched)
    for c in range(3):
        band = stretched[..., c]
        valid = band[np.isfinite(band)]
        if valid.size == 0:
            continue
        src_mean = float(np.mean(valid))
        src_std  = float(np.std(valid))
        out[..., c] = (band - src_mean) * (target_stds[c] / (src_std + 1e-9)) + target_means[c]
    return np.clip(np.nan_to_num(out, nan=0.0), 0, 255).astype(np.uint8)


def enhance_gamma_wb(
    img: np.ndarray,
    gamma: float = 2.2,
    target_means: np.ndarray = DEFAULT_TARGET_MEANS,
    target_stds: np.ndarray = DEFAULT_TARGET_STDS,
) -> np.ndarray:
    """Gamma correction followed by white balance."""
    gamma_img = np.power(np.clip(_per_band_pct_stretch(img), 0, 1), 1.0 / gamma)
    return enhance_white_balance(gamma_img, target_means, target_stds)


def enhance_scurve(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Sigmoidal (S-curve) contrast: lift shadows, compress highlights."""
    stretched = _per_band_pct_stretch(img)
    out = np.zeros_like(stretched)
    for c in range(3):
        out[..., c] = exposure.adjust_sigmoid(
            np.clip(stretched[..., c], 0, 1),
            cutoff=0.5,
            gain=8 * strength + 2,
        )
    return _to_uint8(out)


def enhance_cir(img: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """False-colour CIR composite (NIR→R, Red→G, Green→B).

    Args:
        img: Float32 (H, W, 3) RGB image with NaN for nodata.
        nir: Float32 (H, W) NIR band with NaN for nodata.
    """
    cir = np.stack([nir, img[..., 0], img[..., 1]], axis=-1)
    return _to_uint8(_per_band_pct_stretch(cir))


def enhance_blend(
    img_a: np.ndarray,
    img_b: np.ndarray,
    alpha: float = 0.6,
) -> np.ndarray:
    """Pixel-wise weighted average of two uint8 enhanced images.

    Default ``alpha=0.6`` gives 60 % *img_a* + 40 % *img_b*
    (gamma_wb as the colour-accurate base, scurve for local contrast).
    """
    return np.clip(
        alpha * img_a.astype(np.float32) + (1.0 - alpha) * img_b.astype(np.float32),
        0, 255,
    ).astype(np.uint8)


def stat_weighted_blend(
    images_dict: Dict[str, np.ndarray],
    target_means: np.ndarray = DEFAULT_TARGET_MEANS,
    target_stds: np.ndarray  = DEFAULT_TARGET_STDS,
) -> np.ndarray:
    """Blend multiple uint8 images weighted by similarity to target statistics.

    Each image receives weight ``1 / distance(stats, target_stats)``.  The
    image whose channel distribution is closest to the target training data
    automatically receives the highest weight.

    Args:
        images_dict:  Mapping of method name → uint8 (H, W, 3) image.
        target_means: Target per-channel mean pixel values (0–255 scale).
        target_stds:  Target per-channel std pixel values (0–255 scale).

    Returns:
        Uint8 (H, W, 3) blended image.
    """
    weights: Dict[str, float] = {}
    for name, img in images_dict.items():
        flat = img.reshape(-1, 3).astype(np.float64)
        valid = flat[(flat > 0).all(axis=1)]
        if valid.size == 0:
            weights[name] = 1e-9
            continue
        means = valid.mean(axis=0)
        stds  = valid.std(axis=0)
        dist = (
            np.sum(((means - target_means) / (target_means + 1e-9)) ** 2)
            + np.sum(((stds - target_stds)  / (target_stds  + 1e-9)) ** 2)
        )
        weights[name] = 1.0 / (dist + 1e-6)

    total   = sum(weights.values())
    blended = np.zeros_like(next(iter(images_dict.values())), dtype=np.float32)
    for name, img in images_dict.items():
        blended += (weights[name] / total) * img.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Batch apply: compute all requested methods for a single image
# ---------------------------------------------------------------------------

def compute_enhancements(
    img: np.ndarray,
    methods: List[str],
    nir: Optional[np.ndarray] = None,
    target_means: np.ndarray = DEFAULT_TARGET_MEANS,
    target_stds: np.ndarray  = DEFAULT_TARGET_STDS,
) -> Dict[str, np.ndarray]:
    """Compute all requested enhancements for a single image.

    Handles inter-method dependencies (composites: ``blend_wb_sc``,
    ``stat_wt``) by always computing base sources first.

    Args:
        img:          Float32 (H, W, 3) source image with NaN for nodata.
        methods:      List of method names to compute.
        nir:          Float32 (H, W) NIR band — required only for ``"cir"``.
        target_means: White-balance target channel means (0–255 scale).
        target_stds:  White-balance target channel stds  (0–255 scale).

    Returns:
        Dict mapping every requested method name to its uint8 (H, W, 3) result.
    """
    # Determine which base methods must be computed
    need_base: set = set()
    for m in methods:
        if m in COMPOSITE_SOURCES:
            need_base.update(COMPOSITE_SOURCES[m])
        elif m not in COMPOSITE_SOURCES:
            need_base.add(m)

    # Compute base methods
    computed: Dict[str, np.ndarray] = {}
    for m in need_base:
        if m == "linear":
            computed[m] = enhance_linear(img)
        elif m == "gamma":
            computed[m] = enhance_gamma(img)
        elif m == "white_balance":
            computed[m] = enhance_white_balance(img, target_means, target_stds)
        elif m == "gamma_wb":
            computed[m] = enhance_gamma_wb(img, target_means=target_means, target_stds=target_stds)
        elif m == "scurve":
            computed[m] = enhance_scurve(img)
        elif m == "cir":
            if nir is None:
                raise ValueError("NIR band required for 'cir' enhancement.")
            computed[m] = enhance_cir(img, nir)
        else:
            raise ValueError(f"Unknown base enhancement method: '{m}'.")

    # Compute composite methods
    for m in methods:
        if m == "blend_wb_sc":
            computed[m] = enhance_blend(computed["gamma_wb"], computed["scurve"], alpha=0.6)
        elif m == "stat_wt":
            sw_inputs = {k: computed[k] for k in COMPOSITE_SOURCES["stat_wt"]}
            computed[m] = stat_weighted_blend(sw_inputs, target_means, target_stds)

    # Return only the requested set
    return {m: computed[m] for m in methods if m in computed}
