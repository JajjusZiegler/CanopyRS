from pydantic import Field
from typing import List, Optional

from canopyrs.engine.config_parsers.base import BaseConfig


class TilerizerConfig(BaseConfig):
    tile_type: str = 'tile'
    tile_size: int = 1024
    tile_overlap: float = 0.5
    ground_resolution: Optional[float] = None
    scale_factor: Optional[float] = None
    use_variable_tile_size: bool = False
    variable_tile_size_pixel_buffer: int = 5
    ignore_black_white_alpha_tiles_threshold: float = 0.75
    coco_n_workers: int = 5
    output_dtype: str = 'uint8'  # expected by most models

    # Optional 1-based band indices to select/reorder before tiling.
    # Use this for multispectral rasters where the first 3 bands are not R,G,B.
    # Example for MicaSense Altum-PT (Blue=1,Green=2,Red=3,RE=4,NIR=5,Thermal=6,Alpha=7):
    #   rgb_band_indices: [3, 2, 1]  → writes Red, Green, Blue tiles
    # When None, all bands are written as-is.
    rgb_band_indices: Optional[List[int]] = None

    ignore_tiles_without_labels: bool = True    # impacts inference and evaluation!
    min_intersection_ratio: float = 0.4     # impacts evaluation

    main_label_category_column_name: Optional[str] = None
    other_labels_attributes_column_names: list = Field(default_factory=list)
