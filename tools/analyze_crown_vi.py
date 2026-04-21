import argparse
import sys
import time
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling
import matplotlib.pyplot as plt

try:
    import psutil
    import os
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable
    tqdm.write = print

# Ensure repository root is on path
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from canopyrs.engine.multispectral import calculate_vi
from canopyrs.engine.ms_camera_profiles import get_profile

warnings.filterwarnings("ignore", category=UserWarning)

def load_and_clip_ortho(tif_path, aoi_gdf=None, max_pix=None, bands=[1, 2, 3, 4, 5], nodata_sentinel=-32767):
    """
    Loads an orthomosaic, optionally clips to an AOI extent, and optionally downsamples.
    """
    with rasterio.open(tif_path) as src:
        crs = src.crs
        raw_nodata = src.nodata

        # Downsample if requested
        if max_pix and max(src.width, src.height) > max_pix:
            scale = max_pix / max(src.width, src.height)
            new_w = max(1, int(src.width * scale))
            new_h = max(1, int(src.height * scale))
            
            # Read from file resampled directly to use overviews!
            data = src.read(
                indexes=bands,
                out_shape=(len(bands), new_h, new_w),
                resampling=Resampling.average
            )
            win_tf = src.transform * src.transform.scale(src.width / new_w, src.height / new_h)
            
            # If we also need to clip to AOI, mask the *downsampled* dataset in memory
            if aoi_gdf is not None:
                with rasterio.MemoryFile() as memfile:
                    prof = src.profile.copy()
                    prof.update({"height": new_h, "width": new_w, "transform": win_tf, "count": len(bands)})
                    with memfile.open(**prof) as mem_src:
                        mem_src.write(data)
                        if crs is not None and aoi_gdf.crs != crs:
                            aoi_gdf = aoi_gdf.to_crs(crs)
                        shapes = [geom for geom in aoi_gdf.geometry if geom.is_valid]
                        out_image, out_transform = mask(mem_src, shapes, crop=True, indexes=None)
                        data, win_tf = out_image, out_transform
                
        else:
            if aoi_gdf is not None:
                if crs is not None and aoi_gdf.crs != crs:
                    aoi_gdf = aoi_gdf.to_crs(crs)
                shapes = [geom for geom in aoi_gdf.geometry if geom.is_valid]
                out_image, out_transform = mask(src, shapes, crop=True, indexes=bands)
                data, win_tf = out_image, out_transform
            else:
                data = src.read(indexes=bands)
                win_tf = src.transform

    # Mask nodata & normalize
    nodata_mask = np.any(data == nodata_sentinel, axis=0)
    if raw_nodata is not None:
        nodata_mask |= np.any(data == int(raw_nodata), axis=0)

    data = data.astype(np.float32)
    if data[~np.broadcast_to(nodata_mask, data.shape)].max() > 1.5:
        data = data / 32768.0
    data = np.clip(data, 0.0, 1.0)

    return data, win_tf, crs, nodata_mask


def extract_crown_stats(vi_maps, transform, crs_obj, crowns_gdf):
    from rasterio.features import geometry_mask
    from shapely.geometry import mapping
    
    if crowns_gdf.crs and crs_obj and crowns_gdf.crs != crs_obj:
        crowns_gdf = crowns_gdf.to_crs(crs_obj)
        
    results = []
    # Grab first key to get spatial dimensions
    first_vi = list(vi_maps.values())[0]
    H, W = first_vi.shape
    
    # Use tqdm only if there are enough crowns
    iterator = crowns_gdf.iterrows()
    if len(crowns_gdf) > 10:
        iterator = tqdm(crowns_gdf.iterrows(), total=len(crowns_gdf), desc="Extracting crown stats", leave=False)
        
    for idx, row in iterator:
        geom = row.geometry
        if geom is None or geom.is_empty or not geom.is_valid:
            continue
        try:
            geom_mask = geometry_mask([mapping(geom)], out_shape=(H, W), transform=transform, invert=True)
        except Exception:
            continue
            
        crown_res = {"crown_id": idx}
        valid_px_found = False
        for vi_name, vi_arr in vi_maps.items():
            px = vi_arr[geom_mask & ~np.isnan(vi_arr)]
            if px.size >= 3:
                crown_res[f"{vi_name}_median"] = float(np.median(px))
                valid_px_found = True
                
        if valid_px_found:
            results.append(crown_res)
            
    return results


def plot_temporal_trends(df, out_dir):
    """
    Plots the temporal trends per crown for each VI.
    Input df should have: crown_id, date, <vi>_median columns...
    """
    dates = sorted(df['date'].unique())
    vi_cols = [c for c in df.columns if c.endswith('_median')]
    
    fig, axes = plt.subplots(1, len(vi_cols), figsize=(6 * len(vi_cols), 5), constrained_layout=True)
    if len(vi_cols) == 1:
        axes = [axes]
        
    x_vals = list(range(len(dates)))
    
    for ax, vi_col in zip(axes, vi_cols):
        vi_name = vi_col.replace('_median', '').upper()
        piv = df.pivot_table(index='crown_id', columns='date', values=vi_col)[dates].dropna()
        
        for _, rv in piv.iterrows():
            ax.plot(x_vals, rv.values, color="grey", alpha=0.2, linewidth=0.8)
            
        med = piv.median()
        ax.plot(x_vals, med.values, color="darkgreen", linewidth=2.5, marker="o", label="Site Median")
        
        q25, q75 = piv.quantile(0.25), piv.quantile(0.75)
        ax.fill_between(x_vals, q25.values, q75.values, color="green", alpha=0.15, label="IQR")
        
        ax.set_title(f"{vi_name} Temporal Trends", fontsize=11)
        ax.set_xticks(x_vals)
        ax.set_xticklabels(dates, rotation=30, ha="right")
        ax.set_ylabel(f"Median {vi_name}")
        ax.legend()
        ax.grid(alpha=0.3)
        
    out_path = Path(out_dir) / "crown_vi_timeseries.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved temporal trends to {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Process multispectral datasets to extract tree crown VI metrics.")
    parser.add_argument("--orthos-dir", type=str, required=True, help="Directory containing the orthomosaics. Expects subfolders or files by date.")
    parser.add_argument("--crowns", type=str, required=True, help="Path to crown polygons shapefile/gpkg.")
    parser.add_argument("--aoi", type=str, default=None, help="Path to AOI perimeter to clip processing.")
    parser.add_argument("--out-dir", type=str, default="output", help="Directory to save outputs.")
    parser.add_argument("--max-pix", type=int, default=None, help="Downsample orthos to this maximum size on the long edge.")
    parser.add_argument("--indices", type=str, nargs="+", default=["ndvi", "ndre", "gndvi"], help="VIs to compute")
    parser.add_argument("--camera", type=str, default="altum_pt", help="Camera profile name (e.g. altum_pt)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    crowns_gdf = gpd.read_file(args.crowns)
    aoi_gdf = gpd.read_file(args.aoi) if args.aoi else None

    profile = get_profile(args.camera)
    band_indices = {
        "nir": profile["ms_nir_band_idx"], "red": profile["ms_red_band_idx"],
        "green": profile["ms_green_band_idx"], "blue": profile["ms_blue_band_idx"],
        "red_edge": profile["ms_red_edge_band_idx"]
    }

    # Gather TIFs -- assuming date is inferrable or they are ordered
    print(f"Searching for '*_Ortho.tif' in {args.orthos_dir}...", flush=True)
    all_tifs = sorted(Path(args.orthos_dir).rglob("*_Ortho.tif"))
    if not all_tifs:
        print(f"No '*_Ortho.tif' found in {args.orthos_dir}", flush=True)
        return
    print(f"Found {len(all_tifs)} datasets. Starting processing...", flush=True)

    all_results = []
    
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())

    # We'll use a main progress bar over the dates
    for tif_path in tqdm(all_tifs, desc="Processing datasets", position=0):
        t0 = time.time()
        # Extract date from folder or filename (assuming YYYYMMDD prefix)
        date_str = tif_path.parent.parent.name.split('_')[0] if len(tif_path.parent.parent.name) > 8 else tif_path.stem.split('_')[0]
        
        if HAS_PSUTIL:
            mem_mb = process.memory_info().rss / (1024 ** 2)
            tqdm.write(f"\n[{date_str}] -> Loading {tif_path.name} | Mem: {mem_mb:.1f} MB")
        else:
            tqdm.write(f"\n[{date_str}] -> Loading {tif_path.name}")
        
        data, transform, crs, nodata_mask = load_and_clip_ortho(
            tif_path=str(tif_path),
            aoi_gdf=aoi_gdf,
            max_pix=args.max_pix
        )
        
        t1 = time.time()
        tqdm.write(f"[{date_str}] -> Read & clipped in {t1 - t0:.1f}s | Shape: {data.shape}")
        
        vi_maps = {}
        for vi in args.indices:
            vi_arr = calculate_vi(
                data, index_type=vi, 
                nir_band_idx=band_indices["nir"], red_band_idx=band_indices["red"],
                green_band_idx=band_indices["green"], blue_band_idx=band_indices["blue"],
                red_edge_band_idx=band_indices["red_edge"]
            )
            vi_arr[nodata_mask] = np.nan
            vi_maps[vi] = vi_arr
            
        t2 = time.time()
        tqdm.write(f"[{date_str}] -> Computed VI maps ({', '.join(vi_maps.keys())}) in {t2 - t1:.1f}s")
            
        crown_stats = extract_crown_stats(vi_maps, transform, crs, crowns_gdf)
        t3 = time.time()
        tqdm.write(f"[{date_str}] -> Extracted stats for {len(crown_stats)} valid crowns in {t3 - t2:.1f}s")
        
        for stat in crown_stats:
            stat["date"] = date_str
        all_results.extend(crown_stats)
        
        if HAS_PSUTIL:
            mem_mb = process.memory_info().rss / (1024 ** 2)
            tqdm.write(f"[{date_str}] -> Finished. Mem: {mem_mb:.1f} MB")

    df = pd.DataFrame(all_results)
    if not df.empty:
        csv_path = out_dir / "crown_vi_statistics.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved stats to {csv_path}")
        
        # Plot
        plot_temporal_trends(df, out_dir)
    else:
        print("No crown stats could be extracted.")

if __name__ == "__main__":
    main()
