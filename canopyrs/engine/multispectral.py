"""
Multispectral utility functions for CanopyRS – Path A (Early Resampling).

These utilities are used by the SAM segmenter wrappers to:
1. Compute Vegetation Indices (VI) from a multi-band MS tile.
2. Convert a single-band VI array into a 3-channel 8-bit grayscale image
   that can be fed directly into SAM as an alternative input to RGB.

Usage example inside a SAM wrapper::

    from canopyrs.engine.multispectral import calculate_vi, vi_to_sam_input

    vi = calculate_vi(
        ms_data,
        index_type="ndvi",
        nir_band_idx=4,
        red_band_idx=2,
    )
    sam_input = vi_to_sam_input(vi)   # shape (H, W, 3), dtype uint8
"""

from typing import Optional

import numpy as np


def calculate_vi(
    ms_data: np.ndarray,
    index_type: str = "ndvi",
    nir_band_idx: int = 4,
    red_band_idx: int = 2,
    green_band_idx: int = 1,
    blue_band_idx: int = 0,
    red_edge_band_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Calculate a Vegetation Index from a multi-band numpy array.

    Args:
        ms_data: Multi-band raster array with shape ``(C, H, W)`` or ``(H, W)``
                 for a pre-selected single band.  Values can be any numeric type
                 (the function casts to float32 internally).
        index_type: Which index to compute.  Supported values:

            * ``"ndvi"``   – Normalised Difference Vegetation Index
              ``(NIR - Red) / (NIR + Red)``
            * ``"nir"``    – Raw NIR band, normalised to [0, 1]
            * ``"pri"``    – Photochemical Reflectance Index
              ``(Green - Blue) / (Green + Blue)``
            * ``"ndre"``   – Normalised Difference Red-Edge Index
              ``(NIR - RedEdge) / (NIR + RedEdge)``
              (requires ``red_edge_band_idx`` to be set)
            * ``"evi"``    – Enhanced Vegetation Index
              ``2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)``
            * ``"gndvi"``  – Green Normalised Difference Vegetation Index
              ``(NIR - Green) / (NIR + Green)``
              Sensitive to chlorophyll content in mid-to-late season.

        nir_band_idx:       Zero-based band index for Near-Infrared (default 4).
        red_band_idx:       Zero-based band index for Red (default 2).
        green_band_idx:     Zero-based band index for Green (default 1).
        blue_band_idx:      Zero-based band index for Blue (default 0).
        red_edge_band_idx:  Zero-based band index for Red-Edge (used by NDRE).
                            If ``None`` and index_type is "ndre", falls back to
                            ``nir_band_idx``.

    Returns:
        Single-band float32 array of shape ``(H, W)`` whose values are in the
        range appropriate for the chosen index (e.g. [-1, 1] for NDVI).
        ``NaN`` and ``Inf`` values are replaced with 0 before returning.

    Raises:
        ValueError: If the requested index type is not supported.
        IndexError: If a required band index exceeds the number of bands.
    """
    if ms_data.ndim == 2:
        # Already a single band – just normalise and return
        vi = ms_data.astype(np.float32)
        vi = np.nan_to_num(vi, nan=0.0, posinf=0.0, neginf=0.0)
        return vi

    n_bands = ms_data.shape[0]

    def _band(idx: int, name: str) -> np.ndarray:
        if idx >= n_bands:
            raise IndexError(
                f"Band index {idx} for '{name}' exceeds the number of bands "
                f"in the MS tile ({n_bands}).  Adjust the band index in "
                f"SegmenterConfig."
            )
        return ms_data[idx].astype(np.float32)

    eps = 1e-8  # avoid division by zero

    index_type_lower = index_type.lower()

    if index_type_lower == "ndvi":
        nir = _band(nir_band_idx, "NIR")
        red = _band(red_band_idx, "Red")
        denom = nir + red
        with np.errstate(divide='ignore', invalid='ignore'):
            vi = np.where(np.abs(denom) > eps, (nir - red) / denom, 0.0)

    elif index_type_lower == "nir":
        vi = _band(nir_band_idx, "NIR")

    elif index_type_lower == "pri":
        green = _band(green_band_idx, "Green")
        blue = _band(blue_band_idx, "Blue")
        denom = green + blue
        with np.errstate(divide='ignore', invalid='ignore'):
            vi = np.where(np.abs(denom) > eps, (green - blue) / denom, 0.0)

    elif index_type_lower == "ndre":
        re_idx = red_edge_band_idx if red_edge_band_idx is not None else nir_band_idx
        nir = _band(nir_band_idx, "NIR")
        re = _band(re_idx, "RedEdge")
        denom = nir + re
        with np.errstate(divide='ignore', invalid='ignore'):
            vi = np.where(np.abs(denom) > eps, (nir - re) / denom, 0.0)

    elif index_type_lower == "evi":
        nir = _band(nir_band_idx, "NIR")
        red = _band(red_band_idx, "Red")
        blue = _band(blue_band_idx, "Blue")
        denom = nir + 6.0 * red - 7.5 * blue + 1.0
        with np.errstate(divide='ignore', invalid='ignore'):
            vi = np.where(np.abs(denom) > eps, 2.5 * (nir - red) / denom, 0.0)

    elif index_type_lower == "gndvi":
        nir = _band(nir_band_idx, "NIR")
        green = _band(green_band_idx, "Green")
        denom = nir + green
        with np.errstate(divide='ignore', invalid='ignore'):
            vi = np.where(np.abs(denom) > eps, (nir - green) / denom, 0.0)

    elif index_type_lower == "ndwi":
        green = _band(green_band_idx, "Green")
        nir = _band(nir_band_idx, "NIR")
        denom = green + nir
        with np.errstate(divide='ignore', invalid='ignore'):
            vi = np.where(np.abs(denom) > eps, (green - nir) / denom, 0.0)

    else:
        raise ValueError(
            f"Unsupported index_type '{index_type}'.  "
            f"Supported types: 'ndvi', 'nir', 'pri', 'ndre', 'evi', 'gndvi', 'ndwi'."
        )

    vi = np.nan_to_num(vi.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return vi


def vi_to_sam_input(vi_array: np.ndarray) -> np.ndarray:
    """
    Convert a single-channel Vegetation Index array to a 3-channel 8-bit
    grayscale image suitable for SAM inference.

    Following the methodology described in:
        Hartke et al. (2025), "Segment Anything Model for tree crown delineation
        using near-infrared imagery", J. For. Res.,
        https://doi.org/10.1007/s11676-025-01929-5

    The single VI band is:
    1. Normalised to [0, 1] using its own min/max.
    2. Scaled to [0, 255] and cast to ``uint8``.
    3. Duplicated across three channels to form a ``(H, W, 3)`` grayscale-RGB
       image that can be directly consumed by SAM's processor.

    Args:
        vi_array: Single-channel VI array of shape ``(H, W)``, any float dtype.

    Returns:
        ``uint8`` numpy array of shape ``(H, W, 3)`` with values in [0, 255].
    """
    vi = vi_array.astype(np.float32)

    vi_min = float(np.nanmin(vi))
    vi_max = float(np.nanmax(vi))

    if vi_max > vi_min:
        vi_norm = (vi - vi_min) / (vi_max - vi_min)
    else:
        vi_norm = np.zeros_like(vi)

    vi_8bit = np.clip(vi_norm * 255.0, 0, 255).astype(np.uint8)

    # Stack into 3-channel grayscale image: (H, W, 3)
    sam_input = np.stack([vi_8bit, vi_8bit, vi_8bit], axis=-1)

    return sam_input


def select_best_masks(
    rgb_masks: np.ndarray,
    rgb_scores: np.ndarray,
    ms_masks: np.ndarray,
    ms_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each predicted instance, select the mask with the higher IoU score
    between the RGB-derived and the MS-derived predictions.

    Args:
        rgb_masks:  ``(N, H, W)`` uint8 array of masks from RGB inference.
        rgb_scores: ``(N,)`` float32 array of IoU scores from RGB inference.
        ms_masks:   ``(N, H, W)`` uint8 array of masks from MS inference.
        ms_scores:  ``(N,)`` float32 array of IoU scores from MS inference.

    Returns:
        Tuple of ``(selected_masks, selected_scores)`` with the same shapes
        as the inputs but containing the per-instance best prediction.
    """
    assert rgb_masks.shape == ms_masks.shape, (
        f"Mask shape mismatch: rgb {rgb_masks.shape} vs ms {ms_masks.shape}"
    )
    # Ensure scores are 1-D (N,) — SAM3 may return (N, 1) raw logits
    rgb_scores = rgb_scores.reshape(-1)
    ms_scores = ms_scores.reshape(-1)
    assert rgb_scores.shape == ms_scores.shape

    # Choose MS mask where its score is strictly higher than RGB score
    use_ms = ms_scores > rgb_scores  # (N,) bool

    selected_masks = np.where(use_ms[:, np.newaxis, np.newaxis], ms_masks, rgb_masks)
    selected_scores = np.where(use_ms, ms_scores, rgb_scores)

    return selected_masks, selected_scores
