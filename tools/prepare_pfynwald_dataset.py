#!/usr/bin/env python3
"""
Prepares the Pfynwald 20240823 annotated dataset for fine-tuning.

Steps:
  1. Loads crown annotations and AOI from GeoPackage
  2. Splits the AOI into train (80%) / valid (20%) bands (N→S)
  3. Tilerizes the RGB orthomosaic at 1 cm / 4000 px tiles with 50% overlap
  4. Outputs COCO JSON + tiles to OUTPUT_ROOT/pfynwald_20240823/

Usage:
    conda run -n canopyrs_env python tools/prepare_pfynwald_dataset.py
    conda run -n canopyrs_env python tools/prepare_pfynwald_dataset.py --output-root /tmp/canopyrs_training
"""

import argparse
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

# ── paths (edit here or override via CLI) ──────────────────────────────────
ANNOTATION_PATH = Path(
    "/mnt/m/working_package_2/2024_dronecampaign/02_processing"
    "/metashape_projects/Upscale_Metashapeprojects/Pfynwald"
    "/20240823/annotation/20240803_pfynwald.gpkg"
)
AOI_PATH = Path(
    "/mnt/m/working_package_2/2024_dronecampaign/02_processing"
    "/metashape_projects/Upscale_Metashapeprojects/Pfynwald"
    "/20240823/annotation/20240803_pfynwald_aoi.gpkg"
)
RGB_ORTHO_PATH = Path(
    "/mnt/m/working_package_2/2024_dronecampaign/02_processing"
    "/metashape_projects/Upscale_Metashapeprojects/Pfynwald"
    "/20240823/exports/20240823_Pfynwald_rgb_model_ortho_smooth_100.tif"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/m/working_package_2/2024_dronecampaign/02_processing"
    "/metashape_projects/Upscale_Metashapeprojects/Pfynwald"
    "/20240823/CanopyRS/training_data"
)

DATASET_NAME   = "pfynwald_20240823"
GROUND_RES     = 0.01   # 1 cm — matches preset_seg_multi_NQOS_selvamask_SAM3_FT_quality_1cm
TILE_SIZE      = 4000   # pixels — 40 m at 1 cm
TILE_OVERLAP   = 0.5
TRAIN_FRACTION = 0.8    # north fraction → train; remainder → valid
# ──────────────────────────────────────────────────────────────────────────


def build_aoi_splits(aoi_gdf: gpd.GeoDataFrame, train_fraction: float):
    """
    Splits the AOI polygon into a train band (northern fraction) and a
    valid band (southern fraction) by a horizontal cut at the given fraction
    of the total height.
    """
    union = aoi_gdf.union_all()
    minx, miny, maxx, maxy = union.bounds
    split_y = miny + (maxy - miny) * (1.0 - train_fraction)  # keep top part for train

    train_box  = box(minx, split_y, maxx, maxy)
    valid_box  = box(minx, miny,    maxx, split_y)

    train_geom = union.intersection(train_box)
    valid_geom = union.intersection(valid_box)

    train_gdf = gpd.GeoDataFrame(geometry=[train_geom], crs=aoi_gdf.crs)
    valid_gdf = gpd.GeoDataFrame(geometry=[valid_geom], crs=aoi_gdf.crs)

    return train_gdf, valid_gdf


def main(output_root: Path):
    from geodataset.aoi import AOIFromPackageConfig
    from canopyrs.data.detection.tilerize import tilerize_with_overlap

    output_path = output_root / DATASET_NAME
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading annotations from: {ANNOTATION_PATH}")
    crowns_gdf = gpd.read_file(ANNOTATION_PATH)
    print(f"  → {len(crowns_gdf)} crowns, CRS: {crowns_gdf.crs}")

    print(f"Loading AOI from: {AOI_PATH}")
    aoi_gdf = gpd.read_file(AOI_PATH).to_crs(crowns_gdf.crs)
    print(f"  → AOI area: {aoi_gdf.area.sum():.0f} m²")

    print(f"Splitting AOI: {TRAIN_FRACTION:.0%} train / {1 - TRAIN_FRACTION:.0%} valid (N→S band split)")
    train_gdf, valid_gdf = build_aoi_splits(aoi_gdf, TRAIN_FRACTION)

    aois_config = AOIFromPackageConfig(
        aois={
            "train": train_gdf,
            "valid": valid_gdf,
        }
    )

    print(f"\nTilerizing RGB orthomosaic:")
    print(f"  raster  : {RGB_ORTHO_PATH}")
    print(f"  labels  : {ANNOTATION_PATH}")
    print(f"  output  : {output_path}")
    print(f"  tiles   : {TILE_SIZE}px @ {GROUND_RES*100:.0f}cm = {TILE_SIZE*GROUND_RES:.0f}m, {TILE_OVERLAP:.0%} overlap")

    coco_outputs, tiles_path = tilerize_with_overlap(
        raster_path=RGB_ORTHO_PATH,
        labels=ANNOTATION_PATH,             # pass path — GeoDataFrame causes ambiguous truth value in LabeledRasterTilerizer
        main_label_category_column_name=None,   # single class: tree
        coco_categories_list=[{"id": 1, "name": "tree", "supercategory": ""}],
        aois_config=aois_config,
        output_path=output_path,
        ground_resolution=GROUND_RES,
        scale_factor=None,
        tile_size=TILE_SIZE,
        tile_overlap=TILE_OVERLAP,
    )

    print("\nDone!")
    print(f"  tiles path : {tiles_path}")
    for fold, path in coco_outputs.items():
        print(f"  {fold:6s} COCO : {path}")

    print(f"""
─────────────────────────────────────────────────────────────────
Next steps
─────────────────────────────────────────────────────────────────
1. Fine-tune the detector:

   conda run -n canopyrs_env python train.py \\
     -m detector \\
     -c canopyrs/config/detectors/dino_swinL_pfynwald_FT.yaml

2. After detector training finishes, update the segmenter config
   (canopyrs/config/segmenters/sam3_pfynwald_FT.yaml) with the
   path to the saved detector checkpoint, then fine-tune SAM3:

   conda run -n canopyrs_env python train.py \\
     -m segmenter \\
     -c canopyrs/config/segmenters/sam3_pfynwald_FT.yaml
─────────────────────────────────────────────────────────────────
  data_root_path   : {output_root}
  dataset_name     : {DATASET_NAME}
─────────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Pfynwald 20240823 training dataset")
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help=f"Root folder for tilerized output (default: {DEFAULT_OUTPUT_ROOT})"
    )
    args = parser.parse_args()
    main(args.output_root)
