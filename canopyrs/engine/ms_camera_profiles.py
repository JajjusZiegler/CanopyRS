"""
Multispectral camera band-layout profiles for CanopyRS.

Each profile defines the zero-based band indices for standard spectral channels
and metadata (wavelengths, number of bands, supported vegetation indices) for a
particular camera model.  Profiles are used by :class:`SegmenterConfig` to
auto-populate the ``ms_*_band_idx`` fields when ``ms_camera`` is specified,
removing the need for users to look up band indices manually.

Supported camera models
-----------------------
* ``"micasense_altum"``        / ``"altum"``
* ``"micasense_rededge_mx"``   / ``"rededge_mx"``
* ``"micasense_rededge_mx_dual"`` / ``"mx_dual"`` / ``"rededge_mx_dual"``
* ``"micasense_altum_pt"``     / ``"altum_pt"``
* ``"dji_p4_multispectral"``   / ``"p4_multispectral"``

Usage in a YAML pipeline config::

    segmenter:
      model: sam3
      ms_camera: mx_dual        # ← pick a camera preset
      ms_index_type: ndvi       # ← pick a VI

Band indices can still be overridden per-field after selecting a profile::

    segmenter:
      ms_camera: mx_dual
      ms_nir_band_idx: 9        # use NIR from Camera-2 instead of Camera-1
"""

from typing import Dict, Any, List, Optional


# ---------------------------------------------------------------------------
# Camera profile schema (dict keys)
# ---------------------------------------------------------------------------
# Each profile is a plain dict with the following keys:
#   description     – human-readable label
#   n_bands         – total number of spectral bands
#   ms_blue_band_idx        – zero-based index for the Blue channel
#   ms_green_band_idx       – zero-based index for the Green channel
#   ms_red_band_idx         – zero-based index for the Red channel
#   ms_red_edge_band_idx    – zero-based index for Red-Edge (or None)
#   ms_nir_band_idx         – zero-based index for NIR
#   supported_vis   – list of VI names that make sense for this camera
#   wavelengths_nm  – informational dict of {label: wavelength_nm}
# ---------------------------------------------------------------------------

MS_CAMERA_PROFILES: Dict[str, Dict[str, Any]] = {

    # ------------------------------------------------------------------
    # MicaSense Altum  (5-band)
    # Blue 475 | Green 560 | Red 668 | Red-Edge 717 | NIR 840
    # ------------------------------------------------------------------
    "micasense_altum": {
        "description": (
            "MicaSense Altum (5-band): Blue, Green, Red, Red-Edge, NIR. "
            "Also includes a LWIR thermal band that is typically stored as a "
            "separate file and is NOT counted here."
        ),
        "n_bands": 5,
        "ms_blue_band_idx": 0,
        "ms_green_band_idx": 1,
        "ms_red_band_idx": 2,
        "ms_red_edge_band_idx": 3,
        "ms_nir_band_idx": 4,
        "supported_vis": ["ndvi", "nir", "pri", "ndre", "evi"],
        "wavelengths_nm": {
            "blue": 475,
            "green": 560,
            "red": 668,
            "red_edge": 717,
            "nir": 840,
        },
    },

    # ------------------------------------------------------------------
    # MicaSense RedEdge-MX  (5-band, same spectral layout as Altum)
    # Blue 475 | Green 560 | Red 668 | Red-Edge 717 | NIR 840
    # ------------------------------------------------------------------
    "micasense_rededge_mx": {
        "description": (
            "MicaSense RedEdge-MX (5-band): Blue, Green, Red, Red-Edge, NIR. "
            "Spectral bands identical to the Altum."
        ),
        "n_bands": 5,
        "ms_blue_band_idx": 0,
        "ms_green_band_idx": 1,
        "ms_red_band_idx": 2,
        "ms_red_edge_band_idx": 3,
        "ms_nir_band_idx": 4,
        "supported_vis": ["ndvi", "nir", "pri", "ndre", "evi"],
        "wavelengths_nm": {
            "blue": 475,
            "green": 560,
            "red": 668,
            "red_edge": 717,
            "nir": 840,
        },
    },

    # ------------------------------------------------------------------
    # MicaSense RedEdge-MX Dual  (10-band: two RedEdge-MX heads)
    #
    # Camera-1 (bands 0–4):
    #   Blue 475 | Green 560 | Red 668 | Red-Edge 717 | NIR 840
    # Camera-2 (bands 5–9):
    #   Coastal-Blue 444 | Green 531 | Red 650 | Red-Edge 705 | NIR 740
    #
    # Default primary indices use Camera-1 (broadest NDVI support).
    # Override ms_red_edge_band_idx / ms_nir_band_idx to use Camera-2
    # bands 8 / 9 for narrow-band indices.
    # ------------------------------------------------------------------
    "micasense_rededge_mx_dual": {
        "description": (
            "MicaSense RedEdge-MX Dual (10-band): two RedEdge-MX sensor heads. "
            "Camera-1 bands 0-4 (Blue 475, Green 560, Red 668, RedEdge 717, NIR 840); "
            "Camera-2 bands 5-9 (CoastalBlue 444, Green 531, Red 650, RedEdge 705, NIR 740). "
            "Default primary indices use Camera-1.  Override individual band "
            "indices to select Camera-2 alternatives."
        ),
        "n_bands": 10,
        "ms_blue_band_idx": 0,        # Blue 475 nm  (cam1)
        "ms_green_band_idx": 1,       # Green 560 nm (cam1)
        "ms_red_band_idx": 2,         # Red 668 nm   (cam1)
        "ms_red_edge_band_idx": 3,    # RedEdge 717 nm (cam1); alt: 8 (705 nm, cam2)
        "ms_nir_band_idx": 4,         # NIR 840 nm   (cam1); alt: 9 (740 nm, cam2)
        "supported_vis": ["ndvi", "nir", "pri", "ndre", "evi"],
        "wavelengths_nm": {
            # Camera 1
            "blue_cam1": 475,
            "green_cam1": 560,
            "red_cam1": 668,
            "red_edge_cam1": 717,
            "nir_cam1": 840,
            # Camera 2
            "coastal_blue_cam2": 444,
            "green_cam2": 531,
            "red_cam2": 650,
            "red_edge_cam2": 705,
            "nir_cam2": 740,
        },
    },

    # ------------------------------------------------------------------
    # MicaSense Altum-PT  (5 MS bands + panchromatic — 6 bands total)
    # MS bands: Blue 475 | Green 560 | Red 668 | Red-Edge 717 | NIR 840
    # Band 5: Panchromatic (grey, ~400-700 nm) — excluded from VI defaults
    # ------------------------------------------------------------------
    "micasense_altum_pt": {
        "description": (
            "MicaSense Altum-PT (6-band): same 5 MS bands as the Altum plus a "
            "panchromatic band (band 5, ~400–700 nm).  VI calculations use "
            "bands 0–4 by default."
        ),
        "n_bands": 6,
        "ms_blue_band_idx": 0,
        "ms_green_band_idx": 1,
        "ms_red_band_idx": 2,
        "ms_red_edge_band_idx": 3,
        "ms_nir_band_idx": 4,
        "supported_vis": ["ndvi", "nir", "pri", "ndre", "evi"],
        "wavelengths_nm": {
            "blue": 475,
            "green": 560,
            "red": 668,
            "red_edge": 717,
            "nir": 840,
            "panchromatic": 630,  # approximate centre of broadband response
        },
    },

    # ------------------------------------------------------------------
    # DJI P4 Multispectral  (5-band)
    # Blue 450 | Green 560 | Red 650 | Red-Edge 730 | NIR 840
    # (plus an integrated RGB sensor whose bands are stored separately)
    # ------------------------------------------------------------------
    "dji_p4_multispectral": {
        "description": (
            "DJI Phantom 4 Multispectral (5-band): Blue 450, Green 560, "
            "Red 650, Red-Edge 730, NIR 840.  Band ordering as exported by "
            "DJI Terra / Pix4Dfields."
        ),
        "n_bands": 5,
        "ms_blue_band_idx": 0,
        "ms_green_band_idx": 1,
        "ms_red_band_idx": 2,
        "ms_red_edge_band_idx": 3,
        "ms_nir_band_idx": 4,
        "supported_vis": ["ndvi", "nir", "pri", "ndre", "evi"],
        "wavelengths_nm": {
            "blue": 450,
            "green": 560,
            "red": 650,
            "red_edge": 730,
            "nir": 840,
        },
    },
}

# ---------------------------------------------------------------------------
# Convenience aliases (short names)
# ---------------------------------------------------------------------------
MS_CAMERA_PROFILES["altum"] = MS_CAMERA_PROFILES["micasense_altum"]
MS_CAMERA_PROFILES["rededge_mx"] = MS_CAMERA_PROFILES["micasense_rededge_mx"]
MS_CAMERA_PROFILES["mx_dual"] = MS_CAMERA_PROFILES["micasense_rededge_mx_dual"]
MS_CAMERA_PROFILES["rededge_mx_dual"] = MS_CAMERA_PROFILES["micasense_rededge_mx_dual"]
MS_CAMERA_PROFILES["altum_pt"] = MS_CAMERA_PROFILES["micasense_altum_pt"]
MS_CAMERA_PROFILES["p4_multispectral"] = MS_CAMERA_PROFILES["dji_p4_multispectral"]

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

# Band-index field names that a profile can supply
_BAND_INDEX_FIELDS: List[str] = [
    "ms_blue_band_idx",
    "ms_green_band_idx",
    "ms_red_band_idx",
    "ms_red_edge_band_idx",
    "ms_nir_band_idx",
]


def get_profile(camera_name: str) -> Dict[str, Any]:
    """
    Return the camera profile dict for *camera_name* (case-insensitive).

    Args:
        camera_name: One of the supported camera names or aliases.

    Returns:
        Profile dict (see module-level ``MS_CAMERA_PROFILES``).

    Raises:
        ValueError: If the camera name is not recognised.
    """
    key = camera_name.lower().strip()
    if key not in MS_CAMERA_PROFILES:
        known = sorted(list_cameras() + [
            "altum", "rededge_mx", "mx_dual", "rededge_mx_dual",
            "altum_pt", "p4_multispectral",
        ])
        raise ValueError(
            f"Unknown ms_camera '{camera_name}'.  "
            f"Supported cameras: {known}."
        )
    return MS_CAMERA_PROFILES[key]


def apply_profile_to_config_data(
    data: Dict[str, Any],
    camera_name: str,
) -> Dict[str, Any]:
    """
    Inject band-index values from a camera profile into a raw config dict.

    Only band-index fields that are **not already present** in *data* are
    injected.  This preserves explicit overrides supplied by the user.

    Args:
        data:         Raw config dict (e.g. from ``model_validator(mode='before')``).
        camera_name:  Camera profile key (see :func:`get_profile`).

    Returns:
        Updated *data* dict (mutated in-place and returned).
    """
    profile = get_profile(camera_name)
    for field in _BAND_INDEX_FIELDS:
        if field not in data:
            data[field] = profile[field]
    return data


def list_cameras() -> List[str]:
    """
    Return a sorted list of canonical (non-alias) camera names.

    Returns:
        List of canonical camera name strings.
    """
    canonical = [
        "micasense_altum",
        "micasense_rededge_mx",
        "micasense_rededge_mx_dual",
        "micasense_altum_pt",
        "dji_p4_multispectral",
    ]
    return canonical


def list_supported_vis(camera_name: str) -> List[str]:
    """
    Return the list of recommended vegetation indices for a camera model.

    Args:
        camera_name: Camera profile key.

    Returns:
        List of VI name strings (e.g. ``["ndvi", "nir", "pri", "ndre", "evi"]``).
    """
    return get_profile(camera_name)["supported_vis"]
