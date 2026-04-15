"""
RgbEnhancerComponent: applies configurable RGB enhancements to image tiles.

The component reads tiles from ``data_state.tiles_path``, applies every method
listed in ``config.detection_methods``, and writes each enhanced set to::

    <output_path>/enhanced_tiles/<method_name>/<tile_filename>

After completion ``data_state.enhanced_tiles_paths`` is a ``dict`` mapping
each method name to the string path of its enhanced tile directory.  The
``DetectorComponent`` reads this field and runs object detection on each
tile set, WBF-fusing the per-method results into a single set of detections.

Tile I/O uses rasterio; per-tile enhancement is parallelised with a
``ThreadPoolExecutor`` (CPU-bound work, GIL released inside NumPy/skimage).
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
import rasterio

from canopyrs.engine.constants import StateKey
from canopyrs.engine.components.base import (
    BaseComponent,
    ComponentResult,
    validate_requirements,
)
from canopyrs.engine.config_parsers.rgb_enhancer import RgbEnhancerConfig
from canopyrs.engine.data_state import DataState
from canopyrs.engine.rgb_enhancements import compute_enhancements


class RgbEnhancerComponent(BaseComponent):
    """
    Applies a set of RGB enhancements to each image tile.

    Requirements:
        - tiles_path: Directory containing 3-band RGB tiles (any numeric dtype)

    Produces:
        - enhanced_tiles_paths: Dict[method_name, str] — one directory per method
    """

    name = "rgb_enhancer"

    BASE_REQUIRES_STATE = {StateKey.TILES_PATH}
    BASE_REQUIRES_COLUMNS: Set[str] = set()

    BASE_PRODUCES_STATE = {StateKey.ENHANCED_TILES_PATHS}
    BASE_PRODUCES_COLUMNS: Set[str] = set()

    BASE_STATE_HINTS = {
        StateKey.TILES_PATH: (
            "RgbEnhancerComponent needs tiles. Add a tilerizer before rgb_enhancer."
        ),
    }

    def __init__(
        self,
        config: RgbEnhancerConfig,
        parent_output_path: str = None,
        component_id: int = None,
        max_workers: int = 4,
    ):
        super().__init__(config, parent_output_path, component_id)
        self.max_workers = max_workers

        self.requires_state   = set(self.BASE_REQUIRES_STATE)
        self.requires_columns = set(self.BASE_REQUIRES_COLUMNS)
        self.produces_state   = set(self.BASE_PRODUCES_STATE)
        self.produces_columns = set(self.BASE_PRODUCES_COLUMNS)
        self.state_hints      = dict(self.BASE_STATE_HINTS)
        self.column_hints     = {}

    @validate_requirements
    def __call__(self, data_state: DataState) -> ComponentResult:
        """
        Enhance all tiles with each configured method and write them to disk.

        Within ``<output_path>/enhanced_tiles/`` a sub-directory per method is
        created.  Every tile is written as a uint8 GeoTIFF preserving the
        source tile's spatial metadata (CRS, transform, bounds).
        """
        tiles_path = Path(data_state.tiles_path)
        tile_files = sorted(tiles_path.glob("*.tif"))
        if not tile_files:
            tile_files = sorted(tiles_path.rglob("*.tif"))

        if not tile_files:
            print(f"RgbEnhancerComponent: no tiles found in '{tiles_path}'. Skipping.")
            return ComponentResult(
                state_updates={StateKey.ENHANCED_TILES_PATHS: {}},
            )

        methods      = self.config.detection_methods
        target_means = np.array(self.config.target_means, dtype=np.float64)
        target_stds  = np.array(self.config.target_stds,  dtype=np.float64)

        # Resolve optional NIR source for CIR (from ms_tiles_path)
        ms_tiles_path: Optional[Path] = None
        if "cir" in methods and data_state.ms_tiles_path:
            ms_tiles_path = Path(data_state.ms_tiles_path)

        # Create per-method output directories
        base_out = self.output_path / "enhanced_tiles"
        method_dirs: Dict[str, Path] = {}
        for m in methods:
            d = base_out / m
            d.mkdir(parents=True, exist_ok=True)
            method_dirs[m] = d

        def _process_tile(tile_file: Path) -> None:
            img, nodata_mask, profile = _read_rgb_tile(tile_file)

            nir: Optional[np.ndarray] = None
            if ms_tiles_path is not None:
                ms_tile = ms_tiles_path / tile_file.name
                if ms_tile.exists():
                    nir = _read_nir_band(ms_tile, self.config.nir_band_index)

            enhanced = compute_enhancements(
                img=img,
                methods=methods,
                nir=nir,
                target_means=target_means,
                target_stds=target_stds,
            )

            out_profile = profile.copy()
            out_profile.update(
                dtype="uint8",
                count=3,
                nodata=0,
                compress="deflate",
                predictor=2,
                photometric="RGB",
            )
            for m, enh_img in enhanced.items():
                if nodata_mask is not None:
                    enh_img[nodata_mask] = 0
                _write_rgb_tile(enh_img, method_dirs[m] / tile_file.name, out_profile)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_process_tile, tf): tf for tf in tile_files}
            for future in concurrent.futures.as_completed(futures):
                exc = future.exception()
                if exc:
                    tf = futures[future]
                    print(
                        f"RgbEnhancerComponent: error enhancing '{tf.name}': {exc!r}"
                    )

        enhanced_tiles_paths = {m: str(d) for m, d in method_dirs.items()}
        print(
            f"RgbEnhancerComponent: enhanced {len(tile_files)} tiles "
            f"× {len(methods)} methods."
        )

        return ComponentResult(
            state_updates={StateKey.ENHANCED_TILES_PATHS: enhanced_tiles_paths},
        )


# ---------------------------------------------------------------------------
# Tile I/O helpers
# ---------------------------------------------------------------------------

def _read_rgb_tile(tile_path: Path):
    """Read a 3-band tile as float32 (H, W, 3) with NaN for nodata pixels.

    Returns:
        img:          float32 (H, W, 3), nodata set to NaN
        nodata_mask:  bool (H, W), True for pixels that had nodata in all bands
        profile:      rasterio profile dict of the source tile
    """
    with rasterio.open(tile_path) as src:
        data    = src.read([1, 2, 3]).astype(np.float32)   # (3, H, W)
        nodata  = src.nodata
        profile = src.profile.copy()

    img = np.transpose(data, (1, 2, 0))  # (H, W, 3)

    if nodata is not None:
        nodata_mask = np.all(data == nodata, axis=0)   # (H, W)
    else:
        # Treat all-zero pixels as nodata (common for uint16 tiles without explicit nodata)
        nodata_mask = np.all(data == 0, axis=0)

    img[nodata_mask] = np.nan
    return img, nodata_mask, profile


def _read_nir_band(ms_tile_path: Path, band_index: int) -> np.ndarray:
    """Read a single NIR band as float32 (H, W) with NaN for nodata."""
    with rasterio.open(ms_tile_path) as src:
        band   = src.read(band_index + 1).astype(np.float32)   # rasterio is 1-based
        nodata = src.nodata
    if nodata is not None:
        band[band == nodata] = np.nan
    return band


def _write_rgb_tile(img: np.ndarray, out_path: Path, profile: dict) -> None:
    """Write a uint8 (H, W, 3) image as a GeoTIFF."""
    data = np.transpose(img, (2, 0, 1))   # (3, H, W)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)
