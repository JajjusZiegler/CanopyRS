"""
Canopy Height Model (CHM) creator for Swiss forest inventories.

This standalone tool creates a CHM by subtracting a swissALTI3D DTM from an
input DSM, both in EPSG:2056 (LV95, the Swiss national coordinate reference
system).

Workflow
--------
1. Read DSM extent (must be EPSG:2056 or be reprojectable to it).
2. Query the Swisstopo STAC API for swissALTI3D tiles covering the DSM extent.
3. Download those tiles (1 m or 0.5 m resolution GeoTIFFs).
4. Merge all downloaded tiles into a single mosaic.
5. Resample the mosaic to match the DSM's exact grid (bicubic by default).
6. (Optional) apply a constant vertical datum correction offset to the DTM.
7. Compute CHM = DSM − DTM, clamp negative values to 0.
8. Write the CHM as a GeoTIFF.

Datum note
----------
swissALTI3D heights are orthometric heights in the LHN95 vertical datum.
Photogrammetric DSMs exported from Metashape / Pix4D / similar software in
EPSG:2056 should also carry LHN95 heights, so no correction is needed by
default.  If your DSM uses a different height reference (e.g. WGS84
ellipsoidal heights), pass ``datum_correction_m`` with the systematic offset
that must be **added** to the DTM to bring it into the same datum as the DSM
(i.e. ``dtm_corrected = dtm + datum_correction_m``).

Usage
-----
Command-line::

    python -m canopyrs.tools.chm_creator \\
        --dsm /path/to/dsm.tif \\
        --output /path/to/chm.tif \\
        [--dtm-resolution 0.5] \\
        [--resampling bicubic] \\
        [--datum-correction 0.0] \\
        [--min-tree-height 2.0] \\
        [--tmp-dir /tmp/chm_tiles] \\
        [--keep-tmp]

Python API::

    from canopyrs.tools.chm_creator import ChmCreator

    creator = ChmCreator(
        dsm_path="/path/to/dsm.tif",
        output_path="/path/to/chm.tif",
    )
    chm_path = creator.run()
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAC_BASE_URL = "https://data.geo.admin.ch/api/stac/v0.9"
SWISSALTI3D_COLLECTION = "ch.swisstopo.swissalti3d"

# The Swisstopo STAC API uses WGS84 (EPSG:4326) for bbox queries.
STAC_BBOX_CRS = "EPSG:4326"

# swissALTI3D is only available in EPSG:2056.
SWISSALTI3D_CRS = "EPSG:2056"

# Default preferred tile resolution in metres (also accepted: "2").
DEFAULT_DTM_RESOLUTION = "0.5"

# Maximum number of STAC items returned per query (API limit is 100).
STAC_MAX_ITEMS = 100


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ChmCreator:
    """
    Standalone Canopy Height Model creator.

    Attributes:
        dsm_path:            Path to the input DSM GeoTIFF.
        output_path:         Destination path for the CHM GeoTIFF.
        dtm_resolution:      Resolution of the swissALTI3D tiles to download:
                             ``"0.5"`` (50 cm, default) or ``"2"`` (2 m).
        resampling:          Resampling algorithm for up/down-sampling the DTM
                             to match the DSM grid.  One of the :mod:`rasterio`
                             :class:`~rasterio.enums.Resampling` names:
                             ``"bicubic"`` (default), ``"bilinear"``,
                             ``"nearest"``, ``"lanczos"``.
        datum_correction_m:  Constant vertical offset **added to the DTM**
                             (in metres) to correct for datum mismatches.
                             ``0.0`` (default) means no correction.
        min_tree_height:     Minimum tree height (metres) to keep in the CHM.
                             Pixels below this threshold are set to 0.  Useful
                             for removing low-lying vegetation.  Default: 2.0.
        tmp_dir:             Directory for intermediate files.  Created and
                             deleted automatically when ``keep_tmp=False``.
        keep_tmp:            Keep intermediate tile downloads after the run.
    """

    def __init__(
        self,
        dsm_path: str | Path,
        output_path: str | Path,
        dtm_resolution: str = DEFAULT_DTM_RESOLUTION,
        resampling: str = "bicubic",
        datum_correction_m: float = 0.0,
        min_tree_height: float = 2.0,
        tmp_dir: Optional[str | Path] = None,
        keep_tmp: bool = False,
    ) -> None:
        self.dsm_path = Path(dsm_path)
        self.output_path = Path(output_path)
        self.dtm_resolution = dtm_resolution
        self.resampling = resampling
        self.datum_correction_m = datum_correction_m
        self.min_tree_height = min_tree_height
        self.tmp_dir = Path(tmp_dir) if tmp_dir else None
        self.keep_tmp = keep_tmp

        if not self.dsm_path.exists():
            raise FileNotFoundError(f"DSM not found: {self.dsm_path}")
        if self.dtm_resolution not in ("0.5", "2"):
            raise ValueError(
                f"dtm_resolution must be '0.5' or '2', got '{self.dtm_resolution}'"
            )

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def run(self) -> Path:
        """
        Execute the full CHM creation workflow.

        Returns:
            Path to the written CHM GeoTIFF.
        """
        import rasterio

        # Set up temp directory
        _own_tmp = self.tmp_dir is None
        if _own_tmp:
            _tmp = tempfile.mkdtemp(prefix="canopyrs_chm_")
            work_dir = Path(_tmp)
        else:
            work_dir = self.tmp_dir
            work_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info("CHM Creator – DSM: %s", self.dsm_path)

            # 1. Get DSM bounds in WGS84 for STAC query and in native CRS
            bounds_wgs84, dsm_profile = self._read_dsm_info()
            logger.info(
                "DSM bounds (WGS84): W=%.6f S=%.6f E=%.6f N=%.6f",
                *bounds_wgs84,
            )

            # 2. Download swissALTI3D tiles
            tile_paths = self._download_tiles(bounds_wgs84, work_dir)
            if not tile_paths:
                raise RuntimeError(
                    "No swissALTI3D tiles were downloaded.  "
                    "Check that the DSM extent overlaps with Switzerland (EPSG:2056)."
                )
            logger.info("Downloaded %d DTM tile(s)", len(tile_paths))

            # 3. Merge tiles
            merged_dtm, merged_meta = self._merge_tiles(tile_paths)
            logger.info(
                "Merged DTM shape: %s  dtype: %s",
                merged_dtm.shape,
                merged_dtm.dtype,
            )

            # 4. Resample merged DTM to match DSM grid (extent + pixel size)
            dtm_resampled = self._resample_to_dsm(merged_dtm, merged_meta, dsm_profile)
            logger.info("Resampled DTM to DSM grid")

            # 5. Optional datum correction
            if self.datum_correction_m != 0.0:
                logger.info(
                    "Applying datum correction: %.3f m added to DTM",
                    self.datum_correction_m,
                )
                dtm_resampled = dtm_resampled + float(self.datum_correction_m)

            # 6. Compute CHM
            dsm_data = self._read_dsm_data()
            chm = self._compute_chm(dsm_data, dtm_resampled)
            logger.info(
                "CHM computed  min=%.1f  max=%.1f  mean=%.1f",
                float(np.nanmin(chm)),
                float(np.nanmax(chm)),
                float(np.nanmean(chm[chm > 0])) if np.any(chm > 0) else 0.0,
            )

            # 7. Write output
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            chm_meta = {**dsm_profile, "dtype": "float32", "count": 1, "nodata": -9999.0}
            nodata_mask = ~np.isfinite(dsm_data[0])
            chm[0][nodata_mask] = -9999.0

            with rasterio.open(self.output_path, "w", **chm_meta) as dst:
                dst.write(chm)
            logger.info("CHM written to %s", self.output_path)

        finally:
            if _own_tmp and not self.keep_tmp:
                shutil.rmtree(work_dir, ignore_errors=True)

        return self.output_path

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _read_dsm_info(self) -> Tuple[Tuple[float, float, float, float], dict]:
        """
        Read the DSM profile and compute its bounding box in WGS84.

        Returns:
            Tuple of ``(bbox_wgs84, dsm_profile)`` where ``bbox_wgs84`` is
            ``(west, south, east, north)`` in EPSG:4326 and ``dsm_profile``
            is the rasterio dataset profile.
        """
        import rasterio
        from pyproj import Transformer

        with rasterio.open(self.dsm_path) as src:
            profile = src.profile.copy()
            bounds = src.bounds
            crs = src.crs

        if crs is None:
            raise ValueError(
                f"DSM has no CRS: {self.dsm_path}.  "
                f"Please assign EPSG:2056 (or any reprojectable CRS) first."
            )

        epsg = crs.to_epsg()
        if epsg == 4326:
            west, south, east, north = (
                bounds.left, bounds.bottom, bounds.right, bounds.top
            )
        else:
            t = Transformer.from_crs(crs, STAC_BBOX_CRS, always_xy=True)
            west, south = t.transform(bounds.left, bounds.bottom)
            east, north = t.transform(bounds.right, bounds.top)

        return (west, south, east, north), profile

    def _read_dsm_data(self) -> np.ndarray:
        """Read the DSM band(s) as a float32 numpy array (C, H, W)."""
        import rasterio

        with rasterio.open(self.dsm_path) as src:
            data = src.read().astype(np.float32)
            nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        return data

    def _download_tiles(
        self,
        bbox_wgs84: Tuple[float, float, float, float],
        work_dir: Path,
    ) -> List[Path]:
        """
        Query the Swisstopo STAC API and download matching swissALTI3D tiles.

        Args:
            bbox_wgs84: ``(west, south, east, north)`` in EPSG:4326.
            work_dir:   Directory to save downloaded tiles.

        Returns:
            List of paths to downloaded GeoTIFF tiles.
        """
        import requests

        west, south, east, north = bbox_wgs84
        url = f"{STAC_BASE_URL}/collections/{SWISSALTI3D_COLLECTION}/items"
        params = {
            "bbox": f"{west:.6f},{south:.6f},{east:.6f},{north:.6f}",
            "limit": STAC_MAX_ITEMS,
        }

        logger.info("Querying STAC API: %s?bbox=%s", url, params["bbox"])
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        features = response.json().get("features", [])
        logger.info("STAC returned %d item(s)", len(features))

        tile_paths: List[Path] = []
        for feature in features:
            download_url = _extract_asset_url(feature, self.dtm_resolution)
            if download_url is None:
                logger.warning(
                    "No suitable asset found for item '%s' at resolution '%s' — skipping",
                    feature.get("id", "?"),
                    self.dtm_resolution,
                )
                continue

            filename = Path(download_url).name
            dest = work_dir / filename
            if dest.exists():
                logger.debug("Tile already cached: %s", filename)
                tile_paths.append(dest)
                continue

            logger.info("Downloading %s …", filename)
            tile_response = requests.get(download_url, timeout=120, stream=True)
            tile_response.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in tile_response.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
            tile_paths.append(dest)

        return tile_paths

    def _merge_tiles(
        self, tile_paths: List[Path]
    ) -> Tuple[np.ndarray, dict]:
        """
        Merge multiple DTM tiles into a single mosaic.

        Args:
            tile_paths: List of GeoTIFF tile paths.

        Returns:
            Tuple of ``(merged_data, merged_meta)`` where ``merged_data``
            has shape ``(1, H, W)`` (float32) and ``merged_meta`` is the
            rasterio profile for the mosaic.
        """
        import rasterio
        from rasterio.merge import merge as rio_merge

        datasets = [rasterio.open(p) for p in tile_paths]
        try:
            mosaic, transform = rio_merge(datasets)
        finally:
            for ds in datasets:
                ds.close()

        with rasterio.open(tile_paths[0]) as ref:
            meta = ref.profile.copy()

        meta.update(
            {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "count": 1,
                "dtype": "float32",
            }
        )
        # Convert to float32; replace any fill/nodata with NaN
        mosaic_f = mosaic[0].astype(np.float32)
        if ref.nodata is not None:
            mosaic_f[mosaic_f == ref.nodata] = np.nan

        return mosaic_f[np.newaxis, :, :], meta

    def _resample_to_dsm(
        self,
        dtm_data: np.ndarray,
        dtm_meta: dict,
        dsm_profile: dict,
    ) -> np.ndarray:
        """
        Reproject and resample the DTM mosaic to exactly match the DSM grid.

        Args:
            dtm_data:    ``(1, H_dtm, W_dtm)`` float32 array.
            dtm_meta:    rasterio profile for the DTM mosaic.
            dsm_profile: rasterio profile for the DSM.

        Returns:
            ``(1, H_dsm, W_dsm)`` float32 array aligned to the DSM grid.
        """
        import rasterio
        from rasterio.enums import Resampling as RioResampling
        from rasterio.warp import reproject

        resampling_enum = _parse_resampling(self.resampling)

        dst_data = np.full(
            (1, dsm_profile["height"], dsm_profile["width"]),
            fill_value=np.nan,
            dtype=np.float32,
        )

        reproject(
            source=dtm_data,
            destination=dst_data,
            src_transform=dtm_meta["transform"],
            src_crs=dtm_meta.get("crs", rasterio.crs.CRS.from_epsg(2056)),
            dst_transform=dsm_profile["transform"],
            dst_crs=dsm_profile["crs"],
            resampling=resampling_enum,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

        return dst_data

    @staticmethod
    def _compute_chm(
        dsm_data: np.ndarray,
        dtm_data: np.ndarray,
        min_tree_height: float = 0.0,
    ) -> np.ndarray:
        """
        Compute CHM = DSM − DTM, clamping pixels below ``min_tree_height`` to 0.

        Args:
            dsm_data:        ``(1, H, W)`` float32 DSM array (NaN = nodata).
            dtm_data:        ``(1, H, W)`` float32 DTM array (NaN = nodata).
            min_tree_height: Pixels with CHM < min_tree_height are set to 0.

        Returns:
            ``(1, H, W)`` float32 CHM array.
        """
        chm = dsm_data - dtm_data
        # Clamp: negative values and values below min height → 0
        chm = np.where(np.isfinite(chm) & (chm >= min_tree_height), chm, 0.0)
        return chm.astype(np.float32)

    def _compute_chm_with_threshold(
        self,
        dsm_data: np.ndarray,
        dtm_data: np.ndarray,
    ) -> np.ndarray:
        """Compute CHM applying the configured ``min_tree_height`` threshold."""
        return self._compute_chm(dsm_data, dtm_data, self.min_tree_height)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_asset_url(feature: dict, resolution: str) -> Optional[str]:
    """
    Extract the download URL for the requested resolution from a STAC feature.

    The Swisstopo STAC v0.9 API names assets with keys that embed the
    resolution, e.g. ``"2056_5728_0.5"`` or ``"2056_5728_2"``.

    Args:
        feature:    GeoJSON Feature dict from the STAC API response.
        resolution: Desired resolution string (``"0.5"`` or ``"2"``).

    Returns:
        Download URL string, or ``None`` if no matching asset was found.
    """
    assets: dict = feature.get("assets", {})

    # Primary strategy: look for an asset key that contains the resolution
    # string and points to a GeoTIFF.
    for key, asset in assets.items():
        href: str = asset.get("href", "")
        if resolution in key and href.lower().endswith(".tif"):
            return href

    # Fallback: any GeoTIFF asset containing the resolution in its URL.
    for key, asset in assets.items():
        href = asset.get("href", "")
        if resolution in href and href.lower().endswith(".tif"):
            return href

    # Last resort: first available GeoTIFF asset.
    for key, asset in assets.items():
        href = asset.get("href", "")
        if href.lower().endswith(".tif"):
            return href

    return None


def _parse_resampling(name: str):
    """Convert a resampling name string to a rasterio Resampling enum value."""
    from rasterio.enums import Resampling

    _MAP = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "bicubic": Resampling.cubic,        # rasterio calls it "cubic"
        "cubic": Resampling.cubic,
        "cubic_spline": Resampling.cubic_spline,
        "lanczos": Resampling.lanczos,
        "average": Resampling.average,
        "mode": Resampling.mode,
    }
    key = name.lower().strip()
    if key not in _MAP:
        raise ValueError(
            f"Unknown resampling method '{name}'.  "
            f"Supported: {sorted(_MAP.keys())}"
        )
    return _MAP[key]


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m canopyrs.tools.chm_creator",
        description=(
            "Create a Canopy Height Model (CHM) by subtracting a swissALTI3D DTM "
            "from a DSM.  The DSM must be in EPSG:2056 (Swiss LV95)."
        ),
    )
    p.add_argument(
        "--dsm", "-i",
        required=True,
        metavar="PATH",
        help="Path to the input DSM GeoTIFF (EPSG:2056).",
    )
    p.add_argument(
        "--output", "-o",
        required=True,
        metavar="PATH",
        help="Destination path for the output CHM GeoTIFF.",
    )
    p.add_argument(
        "--dtm-resolution",
        default=DEFAULT_DTM_RESOLUTION,
        choices=["0.5", "2"],
        metavar="RES",
        help=(
            "Resolution of the swissALTI3D tiles to download: "
            "'0.5' (50 cm, default) or '2' (2 m)."
        ),
    )
    p.add_argument(
        "--resampling",
        default="bicubic",
        metavar="METHOD",
        help=(
            "Resampling method for matching DTM to DSM grid.  "
            "Choices: nearest, bilinear, bicubic (default), lanczos, average."
        ),
    )
    p.add_argument(
        "--datum-correction",
        type=float,
        default=0.0,
        metavar="METRES",
        help=(
            "Constant offset (metres) added to the DTM to correct for "
            "vertical datum mismatches.  0.0 = no correction (default)."
        ),
    )
    p.add_argument(
        "--min-tree-height",
        type=float,
        default=2.0,
        metavar="METRES",
        help=(
            "Minimum tree height (metres).  CHM pixels below this value are "
            "set to 0 to filter out low vegetation.  Default: 2.0 m."
        ),
    )
    p.add_argument(
        "--tmp-dir",
        default=None,
        metavar="PATH",
        help="Directory for intermediate downloaded tiles.  Auto-created if omitted.",
    )
    p.add_argument(
        "--keep-tmp",
        action="store_true",
        default=False,
        help="Do not delete intermediate downloaded tiles after the run.",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )
    return p


def main(argv=None) -> None:
    """Entry-point for ``python -m canopyrs.tools.chm_creator``."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s – %(message)s",
    )

    creator = ChmCreator(
        dsm_path=args.dsm,
        output_path=args.output,
        dtm_resolution=args.dtm_resolution,
        resampling=args.resampling,
        datum_correction_m=args.datum_correction,
        min_tree_height=args.min_tree_height,
        tmp_dir=args.tmp_dir,
        keep_tmp=args.keep_tmp,
    )
    chm_path = creator.run()
    print(f"CHM written to: {chm_path}")


if __name__ == "__main__":
    main()
