"""
TilerizerComponent with simplified architecture.

Single __call__() method returns ComponentResult.
Pipeline handles state updates. Tilerizer handles its own file I/O internally.

Multispectral support (Path A – Early Resampling)
--------------------------------------------------
When ``data_state.ms_imagery_path`` is set, the tilerizer additionally creates
resampled multispectral (MS) tiles that match every RGB tile in spatial extent
**and** pixel dimensions.  Each MS tile is saved to::

    <output_path>/ms_tiles/<same_filename_as_rgb_tile>

The geographic bounding box of the RGB tile is re-used unchanged, so no
coordinate transformation is needed downstream.  Resampling uses bilinear
interpolation which is appropriate for continuous spectral data such as
reflectance bands.  The resulting ``ms_tiles/`` directory path is stored on
``data_state.ms_tiles_path`` so that subsequent SAM segmenters can load the
co-registered MS tiles for dual-stream inference.
"""

from pathlib import Path
from typing import Set, Optional, List

import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

from geodataset.aoi import AOIConfig
from geodataset.tilerize import RasterTilerizer, LabeledRasterTilerizer, RasterPolygonTilerizer

from canopyrs.engine.constants import Col, StateKey, INFER_AOI_NAME
from canopyrs.engine.components.base import BaseComponent, ComponentResult, validate_requirements
from canopyrs.engine.config_parsers.tilerizer import TilerizerConfig
from canopyrs.engine.data_state import DataState


class TilerizerComponent(BaseComponent):
    """
    Creates image tiles from raster imagery.

    Tile types:
        - 'tile': Unlabeled regular grid tiles (for inference)
        - 'tile_labeled': Labeled regular grid tiles (for training data)
        - 'polygon': Per-polygon tiles (for classifier input)

    Requirements vary by tile_type:
        - 'tile': imagery_path only
        - 'tile_labeled': imagery_path + infer_gdf
        - 'polygon': imagery_path + infer_gdf

    Produces:
        - tiles_path  (always)
        - ms_tiles_path (when data_state.ms_imagery_path is set)
        - infer_coco_path (only for 'tile_labeled' and 'polygon')
    """

    name = 'tilerizer'

    BASE_REQUIRES_STATE = {StateKey.IMAGERY_PATH}
    BASE_REQUIRES_COLUMNS: Set[str] = set()

    BASE_PRODUCES_STATE = {StateKey.TILES_PATH}
    BASE_PRODUCES_COLUMNS: Set[str] = set()

    BASE_STATE_HINTS = {
        StateKey.IMAGERY_PATH: "Tilerizer needs an imagery_path to the raster file.",
        StateKey.INFER_GDF: "This tile_type requires a GeoDataFrame with labels/polygons.",
    }

    BASE_COLUMN_HINTS = {
        Col.GEOMETRY: "GeoDataFrame must have a 'geometry' column.",
    }

    def __init__(
        self,
        config: TilerizerConfig,
        parent_output_path: str = None,
        component_id: int = None,
        infer_aois_config: Optional[AOIConfig] = None
    ):
        super().__init__(config, parent_output_path, component_id)
        self.infer_aois_config = infer_aois_config

        # Validate tile_type
        if config.tile_type not in ['tile', 'tile_labeled', 'polygon']:
            raise ValueError(
                f"Invalid tile_type: '{config.tile_type}'. "
                f"Must be 'tile', 'tile_labeled', or 'polygon'."
            )

        # Set base requirements
        self.requires_state = set(self.BASE_REQUIRES_STATE)
        self.requires_columns = set(self.BASE_REQUIRES_COLUMNS)
        self.produces_state = set(self.BASE_PRODUCES_STATE)
        self.produces_columns = set(self.BASE_PRODUCES_COLUMNS)

        # Set hints
        self.state_hints = dict(self.BASE_STATE_HINTS)
        self.column_hints = dict(self.BASE_COLUMN_HINTS)

        # Tile-type-specific requirements
        if config.tile_type == 'tile':
            # Unlabeled tiles - no additional requirements or produces
            pass

        elif config.tile_type == 'tile_labeled':
            # Labeled tiles - requires infer_gdf, produces COCO
            self.requires_state.add(StateKey.INFER_GDF)
            self.requires_columns.add(Col.GEOMETRY)
            self.produces_state.add(StateKey.INFER_COCO_PATH)
            self.state_hints[StateKey.INFER_GDF] = (
                f"tile_type='tile_labeled' requires infer_gdf with labels. "
                f"Use tile_type='tile' for unlabeled tiles."
            )

        elif config.tile_type == 'polygon':
            # Polygon tiles - requires infer_gdf, produces COCO
            self.requires_state.add(StateKey.INFER_GDF)
            self.requires_columns.add(Col.GEOMETRY)
            self.produces_state.add(StateKey.INFER_COCO_PATH)
            self.state_hints[StateKey.INFER_GDF] = (
                f"tile_type='polygon' requires infer_gdf with polygons."
            )

    @classmethod
    def run_standalone(
        cls,
        config: TilerizerConfig,
        imagery_path: str,
        output_path: str,
        infer_gdf: gpd.GeoDataFrame = None,
        infer_aois_config: Optional[AOIConfig] = None,
        ms_imagery_path: Optional[str] = None,
    ) -> 'DataState':
        """
        Run tilerizer standalone on raster imagery.

        Args:
            config: Tilerizer configuration (tile_type determines requirements)
            imagery_path: Path to the raster file
            output_path: Where to save outputs
            infer_gdf: GeoDataFrame with labels/polygons
                        (required for tile_type='tile_labeled' or 'polygon')
            infer_aois_config: Area of Interest configuration (optional)
            ms_imagery_path: Optional path to a co-registered multispectral
                             orthomosaic.  When provided, resampled MS tiles are
                             created alongside the RGB tiles.

        Returns:
            DataState with tiling results (access .tiles_path and
            .ms_tiles_path for the tile directories)

        Example:
            result = TilerizerComponent.run_standalone(
                config=TilerizerConfig(tile_type='tile', tile_size=512, ...),
                imagery_path='./rgb.tif',
                output_path='./output',
                ms_imagery_path='./multispectral.tif',
            )
            print(result.tiles_path)
            print(result.ms_tiles_path)  # set when ms_imagery_path is given
        """
        from canopyrs.engine.pipeline import run_component
        return run_component(
            component=cls(config, infer_aois_config=infer_aois_config),
            output_path=output_path,
            imagery_path=imagery_path,
            infer_gdf=infer_gdf,
            ms_imagery_path=ms_imagery_path,
        )

    @validate_requirements
    def __call__(self, data_state: DataState) -> ComponentResult:
        """
        Create tiles from raster imagery.

        Returns ComponentResult - Pipeline handles state updates.
        Tilerizer handles its own file I/O internally via geodataset.
        """
        self._check_crs_match(data_state)

        # Handle config columns
        columns_to_pass = data_state.infer_gdf_columns_to_pass.copy()
        if self.config.other_labels_attributes_column_names:
            columns_to_pass.update(self.config.other_labels_attributes_column_names)

        columns_to_pass = [col for col in columns_to_pass if col not in {Col.GEOMETRY, Col.TILE_PATH}]  # already taken care of by COCO format

        # Process based on tile_type
        if self.config.tile_type == 'tile':
            if data_state.infer_gdf is not None:
                raise ValueError(
                    "infer_gdf provided but tile_type='tile' creates unlabeled tiles. "
                    "Use tile_type='tile_labeled' or 'polygon' if labels are needed for subsequent components, like a prompted Segmenter."
                )
            # Unlabeled tiles only
            tiles_path, infer_coco_path = self._process_unlabeled_tiles(data_state)

        elif self.config.tile_type == 'tile_labeled':
            # Labeled regular grid tiles
            tiles_path, infer_coco_path = self._process_labeled_tiles(
                data_state, columns_to_pass
            )

        elif self.config.tile_type == 'polygon':
            # Polygon tiles
            tiles_path, infer_coco_path = self._process_polygon_tiles(
                data_state, columns_to_pass
            )

        else:
            raise ValueError(f"Invalid tile_type: {self.config.tile_type}")

        # Create resampled MS tiles when a multispectral orthomosaic is provided
        ms_tiles_path = None
        if data_state.ms_imagery_path:
            ms_tiles_path = self._create_ms_tiles(
                rgb_tiles_path=tiles_path,
                ms_imagery_path=data_state.ms_imagery_path,
                output_path=self.output_path,
            )
            print(
                f"TilerizerComponent: Created {len(list(ms_tiles_path.glob('*.tif')))} "
                f"resampled MS tiles in '{ms_tiles_path}'."
            )

        # Save config
        if self.output_path:
            self.config.to_yaml(self.output_path / "tilerizer_config.yaml")

        # Register the COCO file that geodataset already wrote (if any)
        output_files = {}
        if infer_coco_path is not None:
            output_files['coco'] = infer_coco_path

        state_updates = {
            StateKey.TILES_PATH: tiles_path,
            StateKey.INFER_COCO_PATH: infer_coco_path,
        }
        if ms_tiles_path is not None:
            state_updates[StateKey.MS_TILES_PATH] = str(ms_tiles_path)

        return ComponentResult(
            gdf=None,  # Tilerizer doesn't modify the GDF
            produced_columns=columns_to_pass,
            objects_are_new=False,
            state_updates=state_updates,
            save_gpkg=False,
            save_coco=False,  # COCO handled internally by tilerizer
            output_files=output_files,
        )

    # ------------------------------------------------------------------
    # MS tile creation (Path A – Early Resampling)
    # ------------------------------------------------------------------

    def _create_ms_tiles(
        self,
        rgb_tiles_path: Path,
        ms_imagery_path: str,
        output_path: Path,
    ) -> Path:
        """
        Create resampled multispectral tiles that match RGB tiles 1-to-1.

        For every RGB tile in ``rgb_tiles_path`` the method:
        1. Opens the tile and reads its geographic bounding box and pixel
           dimensions.
        2. Opens the MS orthomosaic and extracts the same geographic region
           using bilinear resampling scaled to the RGB tile's pixel dimensions.
        3. Saves the resampled MS tile to ``output_path/ms_tiles/`` using the
           **same filename** as the RGB tile so they can be matched trivially.

        Args:
            rgb_tiles_path: Directory (or Path) containing the RGB tile files.
            ms_imagery_path: Path to the co-registered MS orthomosaic.
            output_path: Component output directory.  The ``ms_tiles/``
                         sub-directory is created inside it.

        Returns:
            Path to the ``ms_tiles/`` directory.
        """
        rgb_tiles_path = Path(rgb_tiles_path)
        ms_tiles_path = output_path / "ms_tiles"
        ms_tiles_path.mkdir(parents=True, exist_ok=True)

        rgb_tile_files = sorted(rgb_tiles_path.glob("*.tif"))
        if not rgb_tile_files:
            # Some tilerizers may use sub-folders (e.g. LabeledRasterTilerizer)
            rgb_tile_files = sorted(rgb_tiles_path.rglob("*.tif"))

        if not rgb_tile_files:
            print(
                f"TilerizerComponent: No RGB tiles found in '{rgb_tiles_path}'. "
                f"Skipping MS tile creation."
            )
            return ms_tiles_path

        with rasterio.open(ms_imagery_path) as src_ms:
            ms_crs = src_ms.crs
            ms_transform = src_ms.transform
            ms_count = src_ms.count
            ms_dtype = src_ms.dtypes[0]
            ms_nodata = src_ms.nodata

            for rgb_tile_file in rgb_tile_files:
                try:
                    with rasterio.open(rgb_tile_file) as src_rgb:
                        bounds = src_rgb.bounds
                        rgb_height = src_rgb.height
                        rgb_width = src_rgb.width
                        rgb_transform = src_rgb.transform
                        rgb_crs = src_rgb.crs

                    # Build the window into the MS raster corresponding to the
                    # same geographic bounding box as the RGB tile.
                    window_ms = from_bounds(
                        bounds.left,
                        bounds.bottom,
                        bounds.right,
                        bounds.top,
                        transform=ms_transform,
                    )

                    # Read the MS data and resample it on-the-fly to match the
                    # RGB tile's pixel dimensions.  Using boundless=True ensures
                    # that border tiles (partially outside the MS extent) are
                    # filled with zeros instead of raising an error.
                    ms_data = src_ms.read(
                        window=window_ms,
                        out_shape=(ms_count, rgb_height, rgb_width),
                        resampling=Resampling.bilinear,
                        boundless=True,
                        fill_value=0,
                    )

                    # The output metadata mirrors the RGB tile (same CRS,
                    # transform, dimensions) but with the MS band count/dtype.
                    ms_meta = {
                        "driver": "GTiff",
                        "dtype": ms_dtype,
                        "width": rgb_width,
                        "height": rgb_height,
                        "count": ms_count,
                        "crs": rgb_crs if rgb_crs else ms_crs,
                        "transform": rgb_transform,
                    }
                    if ms_nodata is not None:
                        ms_meta["nodata"] = ms_nodata

                    # Preserve the relative sub-path so nested tile structures
                    # (e.g. from LabeledRasterTilerizer) are mirrored.
                    relative_path = rgb_tile_file.relative_to(rgb_tiles_path)
                    ms_tile_path = ms_tiles_path / relative_path
                    ms_tile_path.parent.mkdir(parents=True, exist_ok=True)

                    with rasterio.open(ms_tile_path, "w", **ms_meta) as dst:
                        dst.write(ms_data)

                except Exception as exc:
                    print(
                        f"TilerizerComponent: Warning – could not create MS tile "
                        f"for '{rgb_tile_file.name}': {exc}"
                    )

        return ms_tiles_path

    # ------------------------------------------------------------------
    # Standard tiling helpers (unchanged from upstream CanopyRS)
    # ------------------------------------------------------------------

    def _process_labeled_tiles(self, data_state: DataState, columns_to_pass: Set[str]):
        """Process labeled regular grid tiles (tile_type='tile_labeled')."""
        tilerizer = LabeledRasterTilerizer(
            raster_path=self._effective_imagery_path(data_state.imagery_path),
            labels_path=None,
            labels_gdf=data_state.infer_gdf,
            output_path=self.output_path,
            tile_size=self.config.tile_size,
            tile_overlap=self.config.tile_overlap,
            aois_config=self.infer_aois_config,
            scale_factor=self.config.scale_factor,
            ground_resolution=self.config.ground_resolution,
            ignore_black_white_alpha_tiles_threshold=self.config.ignore_black_white_alpha_tiles_threshold,
            min_intersection_ratio=self.config.min_intersection_ratio,
            ignore_tiles_without_labels=self.config.ignore_tiles_without_labels,
            main_label_category_column_name=self.config.main_label_category_column_name,
            other_labels_attributes_column_names=list(columns_to_pass),
        )
        coco_paths = tilerizer.generate_coco_dataset()
        return tilerizer.tiles_path, coco_paths.get(INFER_AOI_NAME)

    def _effective_imagery_path(self, imagery_path: str) -> str:
        """
        Return a preprocessed raster path when rgb_band_indices is set.

        Creates a contrast-stretched uint8 3-band GeoTIFF by:
         1. Selecting rgb_band_indices (1-based) from the source raster.
         2. Applying a per-band 2–98th percentile stretch → 0–255 uint8.

        This ensures the detector receives properly-exposed RGB tiles
        regardless of whether the source uses float32 reflectance (0.0–1.0)
        or raw DN (0–65535).
        """
        if not self.config.rgb_band_indices:
            return imagery_path

        import rasterio
        import numpy as np
        from rasterio.enums import Resampling

        out_path = str(self.output_path / "_rgb_stretched.tif")

        with rasterio.open(imagery_path) as src:
            nodata = src.nodata
            meta = src.meta.copy()
            meta.update(
                count=len(self.config.rgb_band_indices),
                dtype="uint8",
                nodata=0,
            )

            # Sample at reduced resolution to compute percentiles cheaply
            sample_scale = min(1.0, 4096 / max(src.width, src.height))
            sw = max(1, int(src.width * sample_scale))
            sh = max(1, int(src.height * sample_scale))
            sample = src.read(
                indexes=self.config.rgb_band_indices,
                out_shape=(len(self.config.rgb_band_indices), sh, sw),
                resampling=Resampling.average,
            ).astype(np.float32)

            # Compute per-band 2–98th percentile stretch params
            p_low = np.zeros(len(self.config.rgb_band_indices), dtype=np.float32)
            p_high = np.zeros(len(self.config.rgb_band_indices), dtype=np.float32)
            for i in range(len(self.config.rgb_band_indices)):
                band = sample[i]
                valid = band[np.isfinite(band)]
                if nodata is not None:
                    valid = valid[valid != nodata]
                if len(valid) == 0:
                    p_low[i], p_high[i] = 0.0, 1.0
                else:
                    p_low[i] = np.percentile(valid, 2)
                    p_high[i] = np.percentile(valid, 98)

            print(
                f"TilerizerComponent: band reorder + stretch "
                f"(bands {self.config.rgb_band_indices}, 2–98th pct) → {out_path}"
            )

            with rasterio.open(out_path, "w", **meta) as dst:
                # Read and write in row-blocks to keep memory bounded
                for ji, window in src.block_windows(1):
                    data = src.read(
                        indexes=self.config.rgb_band_indices,
                        window=window,
                    ).astype(np.float32)
                    out = np.zeros_like(data, dtype=np.uint8)
                    for i in range(data.shape[0]):
                        lo, hi = p_low[i], p_high[i]
                        band = data[i]
                        if hi > lo:
                            band = (band - lo) / (hi - lo) * 255.0
                        else:
                            band = band * 255.0
                        out[i] = np.clip(band, 0, 255).astype(np.uint8)
                        if nodata is not None:
                            out[i][data[i] == nodata] = 0
                    dst.write(out, window=window)

        return out_path

    def _process_unlabeled_tiles(self, data_state: DataState):
        """Process unlabeled tiles (tile_type='tile' without infer_gdf)."""
        tilerizer = RasterTilerizer(
            raster_path=self._effective_imagery_path(data_state.imagery_path),
            output_path=self.output_path,
            tile_size=self.config.tile_size,
            tile_overlap=self.config.tile_overlap,
            aois_config=self.infer_aois_config,
            scale_factor=self.config.scale_factor,
            ground_resolution=self.config.ground_resolution,
            ignore_black_white_alpha_tiles_threshold=self.config.ignore_black_white_alpha_tiles_threshold,
        )
        tilerizer.generate_tiles()
        return tilerizer.tiles_path, None

    def _process_polygon_tiles(self, data_state: DataState, columns_to_pass: Set[str]):
        """Process polygon tiles (tile_type='polygon')."""
        tilerizer = RasterPolygonTilerizer(
            raster_path=data_state.imagery_path,
            output_path=self.output_path,
            labels_path=None,
            labels_gdf=data_state.infer_gdf,
            tile_size=self.config.tile_size,
            use_variable_tile_size=self.config.use_variable_tile_size,
            variable_tile_size_pixel_buffer=self.config.variable_tile_size_pixel_buffer,
            aois_config=self.infer_aois_config,
            scale_factor=self.config.scale_factor,
            ground_resolution=self.config.ground_resolution,
            main_label_category_column_name=self.config.main_label_category_column_name,
            other_labels_attributes_column_names=list(columns_to_pass),
            coco_n_workers=self.config.coco_n_workers,
        )
        coco_paths = tilerizer.generate_coco_dataset()
        return tilerizer.tiles_folder_path, coco_paths.get(INFER_AOI_NAME)

    def _check_crs_match(self, data_state: DataState):
        """Check if the CRS of the raster and GeoDataFrame match."""
        if data_state.infer_gdf is None:
            return

        try:
            with rasterio.open(data_state.imagery_path) as src:
                raster_crs = src.crs
        except Exception as e:
            raise RuntimeError(f"Failed to open raster: {e}")

        gdf_crs = data_state.infer_gdf.crs

        if raster_crs is not None and gdf_crs is None:
            raise ValueError("Raster has CRS but infer_gdf does not.")
        elif raster_crs is None and gdf_crs is not None:
            raise ValueError("Raster has no CRS but infer_gdf does.")
