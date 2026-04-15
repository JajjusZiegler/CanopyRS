from typing import List

from canopyrs.engine.config_parsers.base import BaseConfig
from canopyrs.engine.rgb_enhancements import ENHANCEMENT_METHODS


class RgbEnhancerConfig(BaseConfig):
    """Configuration for the RGB enhancer component.

    Each method listed in ``detection_methods`` is applied to every tile and
    the resulting tile sets are stored in ``data_state.enhanced_tiles_paths``
    so that the detector can run on each method's tiles and WBF-fuse the results.

    Composite methods (``"blend_wb_sc"``, ``"stat_wt"``) are automatically
    computed from their required base enhancements — no need to list those
    base methods separately unless you also want them written to disk.
    """

    # Enhancement methods applied to tiles for detection inference
    detection_methods: List[str] = [
        "gamma_wb",
        "blend_wb_sc",
        "stat_wt",
        "scurve",
        "white_balance",
    ]

    # White-balance target statistics (Altum-PT → visible RGB, 0–255 scale)
    # These are the per-channel mean and std of the RGB-calibrated training imagery
    # to which the Altum-PT data is aligned during white-balance enhancement.
    target_means: List[float] = [73.1, 88.2, 44.0]
    target_stds:  List[float] = [55.2, 58.2, 37.1]

    # NIR band index (0-based) used for CIR when CIR appears in detection_methods.
    # The MS tiles from ``data_state.ms_tiles_path`` are used as the NIR source.
    # Altum-PT Metashape 7-band export: band 5 → 0-based index 4.
    nir_band_index: int = 4

    def model_post_init(self, __context) -> None:
        for m in self.detection_methods:
            if m not in ENHANCEMENT_METHODS:
                raise ValueError(
                    f"Unknown enhancement method '{m}'. "
                    f"Valid methods: {ENHANCEMENT_METHODS}"
                )
