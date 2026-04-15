"""
CanopyRS output visualizer.

Overlays detector bounding boxes and final segmentation masks on the source
raster.  Works with the standard pipeline output folder structure produced by
``infer.py``.

Usage
-----
    # Basic — source raster auto-detected from tile paths in the gpkg
    python tools/visualize_output.py -o /tmp/canopyrs_sam3_test

    # With explicit source raster (faster, recommended for large outputs)
    python tools/visualize_output.py \\
        -o /tmp/canopyrs_sam3_test \\
        -i /path/to/orthomosaic.tif

    # Save PNG only, no interactive window
    python tools/visualize_output.py -o /tmp/canopyrs_sam3_test --no-show

    # Control figure resolution
    python tools/visualize_output.py -o /tmp/canopyrs_sam3_test --dpi 150

Output
------
A PNG file is saved next to the output folder as ``<folder_name>_viz.png``.
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

# Auto-relaunch inside canopyrs_env if dependencies are missing.
try:
    import geopandas  # noqa: F401
except ImportError:
    import os, subprocess
    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if env != "canopyrs_env":
        print("geopandas not found — re-running inside canopyrs_env …", flush=True)
        result = subprocess.run(
            ["conda", "run", "-n", "canopyrs_env", "python"] + sys.argv,
            check=False,
        )
        sys.exit(result.returncode)
    else:
        raise

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import Normalize
from rasterio.enums import ColorInterp
from rasterio.plot import reshape_as_image
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds
from shapely.geometry import box


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_gpkg(folder: Path, pattern: str) -> Optional[Path]:
    """Return the first .gpkg matching *pattern* inside *folder*."""
    hits = sorted(folder.glob(pattern))
    return hits[0] if hits else None


def _find_output_folder(output_dir: Path):
    """
    Locate the final mask gpkg and the detector-box gpkg inside an infer
    output directory.

    Returns (final_gpkg, detector_gpkg, det_folder_name)
    """
    # Final masks live at the root of the output folder
    final = _find_gpkg(output_dir, "*inferfinal.gpkg")
    if final is None:
        # Fall back: last aggregator folder
        agg_dirs = sorted(output_dir.glob("[0-9]*aggregator*"))
        if agg_dirs:
            final = _find_gpkg(agg_dirs[-1], "*infer.gpkg")
            if final is None:
                final = _find_gpkg(agg_dirs[-1], "*.gpkg")

    # Detector boxes: first aggregator folder (after detector)
    det_box = None
    agg_dirs = sorted(output_dir.glob("[0-9]*aggregator*"))
    if agg_dirs:
        det_box = _find_gpkg(agg_dirs[0], "*infer.gpkg")
        if det_box is None:
            det_box = _find_gpkg(agg_dirs[0], "*.gpkg")

    return final, det_box


def _auto_detect_raster(det_gdf: gpd.GeoDataFrame) -> Optional[Path]:
    """Try to recover the source raster path from tile_path column."""
    if det_gdf is None or "tile_path" not in det_gdf.columns:
        return None
    tile = Path(det_gdf["tile_path"].iloc[0])
    # Tile path is typically: <output>/<step>/tiles/infer/<name>/<tile>.tif
    # Walk up to find the original raster in the tilerizer config
    tilerizer_config = tile
    for _ in range(6):
        tilerizer_config = tilerizer_config.parent
        cfg = tilerizer_config / "tilerizer_config.yaml"
        if cfg.exists():
            import yaml
            with cfg.open() as f:
                data = yaml.safe_load(f)
            rp = data.get("raster_path") or data.get("imagery_path")
            if rp and Path(rp).exists():
                return Path(rp)
    return None


def _load_raster_preview(
    raster_path: Path,
    target_pixels: int = 4096,
    bounds_utm=None,
    crs=None,
) -> tuple:
    """
    Read a downsampled RGB preview from *raster_path*.

    Returns (rgb_uint8_HWC, extent_in_raster_crs, raster_crs).
    extent = (left, right, bottom, top) for imshow.
    """
    with rasterio.open(raster_path) as src:
        # Determine which bands are RGB
        ci = src.colorinterp
        rgb_indices = None
        if len(ci) >= 3:
            if ci[:3] == (ColorInterp.red, ColorInterp.green, ColorInterp.blue):
                rgb_indices = [1, 2, 3]
            else:
                # MS raster: use first 3 bands (Blue/Green/Red for Altum-PT)
                # Map to display-friendly order: Altum B0=Blue, B1=Green, B2=Red → swap to R,G,B
                rgb_indices = [3, 2, 1]  # Red=band3, Green=band2, Blue=band1 → display RGB

        scale = min(target_pixels / src.width, target_pixels / src.height)
        out_w = max(1, int(src.width * scale))
        out_h = max(1, int(src.height * scale))

        data = src.read(
            indexes=rgb_indices or [1, 2, 3],
            out_shape=(3, out_h, out_w),
            resampling=rasterio.enums.Resampling.average,
        )

        nodata = src.nodata
        raster_crs = src.crs

        # Compute full raster bounds in its own CRS
        left, bottom, right, top = src.bounds

    # Replace nodata with 0
    if nodata is not None:
        data = np.where(data == nodata, 0, data)

    # Normalise to uint8
    data = data.astype(np.float32)
    for i in range(3):
        band = data[i]
        valid = band[band > 0]
        if len(valid) == 0:
            continue
        p2, p98 = np.percentile(valid, [2, 98])
        if p98 > p2:
            band = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
        else:
            band = np.clip(band, 0, 255)
        data[i] = band

    rgb = reshape_as_image(data).astype(np.uint8)
    extent = (left, right, bottom, top)
    return rgb, extent, raster_crs


def _reproject_gdf(gdf: gpd.GeoDataFrame, target_crs) -> gpd.GeoDataFrame:
    if gdf.crs is None or gdf.crs == target_crs:
        return gdf
    return gdf.to_crs(target_crs)


# ---------------------------------------------------------------------------
# Main visualizer
# ---------------------------------------------------------------------------

def visualize(
    output_dir: Path,
    raster_path: Optional[Path] = None,
    save_png: bool = True,
    show: bool = True,
    dpi: int = 150,
    score_col: str = "detector_score",
):
    output_dir = Path(output_dir)
    final_gpkg, det_gpkg = _find_output_folder(output_dir)

    if final_gpkg is None:
        sys.exit(f"ERROR: could not find final mask gpkg in {output_dir}")

    print(f"Final masks : {final_gpkg}")
    print(f"Detector boxes: {det_gpkg or 'not found'}")

    masks_gdf = gpd.read_file(final_gpkg)
    det_gdf = gpd.read_file(det_gpkg) if det_gpkg else None

    # Auto-detect raster
    if raster_path is None:
        raster_path = _auto_detect_raster(det_gdf or masks_gdf)
        if raster_path:
            print(f"Source raster : {raster_path}")
        else:
            print("Source raster : not found (background will be blank)")

    # ----- figure setup -----
    fig, ax = plt.subplots(figsize=(14, 14), dpi=dpi)
    ax.set_aspect("equal")
    ax.set_title(output_dir.name, fontsize=10, pad=6)

    viz_crs = masks_gdf.crs

    # ----- raster background -----
    if raster_path is not None and Path(raster_path).exists():
        rgb, extent, raster_crs = _load_raster_preview(raster_path)
        # Reproject extent corners if CRS differs
        if raster_crs and raster_crs != viz_crs:
            from rasterio.warp import transform_bounds
            l, b, r, t = transform_bounds(raster_crs, viz_crs, extent[0], extent[2], extent[1], extent[3])
            extent = (l, r, b, t)
        ax.imshow(rgb, extent=extent, origin="upper", interpolation="bilinear", zorder=0)
    else:
        ax.set_facecolor("#1a1a2e")

    # ----- detector bounding boxes -----
    if det_gdf is not None and len(det_gdf) > 0:
        det_gdf = _reproject_gdf(det_gdf, viz_crs)
        # Draw boxes only (no fill)
        det_gdf.plot(
            ax=ax,
            facecolor="none",
            edgecolor="#FFD700",
            linewidth=0.6,
            alpha=0.85,
            zorder=2,
        )

    # ----- final segmentation masks -----
    if len(masks_gdf) > 0:
        # Colour by aggregator_score or fall-back to segmenter_score
        score_col_use = "aggregator_score" if "aggregator_score" in masks_gdf.columns else \
                        "segmenter_score"  if "segmenter_score"  in masks_gdf.columns else None

        if score_col_use:
            norm = Normalize(
                vmin=masks_gdf[score_col_use].quantile(0.05),
                vmax=masks_gdf[score_col_use].quantile(0.95),
            )
            masks_gdf.plot(
                ax=ax,
                column=score_col_use,
                cmap="YlGn",
                norm=norm,
                edgecolor="#00CC44",
                linewidth=0.4,
                alpha=0.55,
                zorder=3,
            )
            sm = plt.cm.ScalarMappable(cmap="YlGn", norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.01,
                         label=score_col_use.replace("_", " ").title())
        else:
            masks_gdf.plot(
                ax=ax,
                facecolor="#00CC44",
                edgecolor="#009933",
                linewidth=0.4,
                alpha=0.5,
                zorder=3,
            )

    # ----- legend -----
    legend_handles = [
        mpatches.Patch(facecolor="none", edgecolor="#FFD700", linewidth=1.2,
                       label=f"Detector boxes  (n={len(det_gdf) if det_gdf is not None else 0})"),
        mpatches.Patch(facecolor="#00CC44", edgecolor="#009933", alpha=0.6,
                       label=f"Crown masks  (n={len(masks_gdf)})"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
              framealpha=0.7, facecolor="#111111", labelcolor="white")

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.tick_params(labelsize=7)

    plt.tight_layout()

    if save_png:
        out_png = output_dir.parent / f"{output_dir.name}_viz.png"
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {out_png}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize CanopyRS inference output (detector boxes + crown masks)."
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Path to the infer.py output folder.",
    )
    parser.add_argument(
        "-i", "--imagery", default=None,
        help="Path to the source raster.  Auto-detected if omitted.",
    )
    parser.add_argument(
        "--no-show", action="store_true", default=False,
        help="Do not open an interactive window; only save PNG.",
    )
    parser.add_argument(
        "--no-save", action="store_true", default=False,
        help="Do not save PNG.",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Figure DPI (default: 150).",
    )
    args = parser.parse_args()

    visualize(
        output_dir=Path(args.output),
        raster_path=Path(args.imagery) if args.imagery else None,
        save_png=not args.no_save,
        show=not args.no_show,
        dpi=args.dpi,
    )
