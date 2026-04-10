from typing import Optional, Union

from pydantic import Field

from canopyrs.engine.config_parsers.base import BaseConfig


class InferIOConfig(BaseConfig):
    input_imagery: Optional[str]
    output_folder: str

    tiles_path: Optional[str] = None

    # Optional secondary multispectral orthomosaic (Path A – Early Resampling).
    # When provided, the tilerizer will resample the MS image to match each
    # RGB tile's spatial extent and pixel dimensions, saving the result in an
    # ``ms_tiles/`` sub-directory alongside the regular tiles.  Downstream SAM
    # segmenters can then use the resampled MS tiles to compute vegetation
    # indices and run dual-stream inference, selecting the best mask per
    # detected crown based on IoU score.
    #
    # The MS raster must be co-registered to ``input_imagery``.  It may have a
    # different spatial resolution (e.g. 5 cm/px vs. 1 cm/px for RGB).
    multispectral_imagery: Optional[str] = None

    input_gpkg: Optional[str] = None
    input_coco: Optional[str] = None
    infer_gdf_columns_to_pass: Optional[list[str]] = None

    aoi_config: str = 'generate'
    aoi_type: Optional[str] = 'band'
    aoi: Union[str, dict] = Field(default_factory=lambda: {
            'percentage': 1.0,
            'position': 1
    })
