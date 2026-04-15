"""
DetectorComponent with simplified architecture.

Single __call__() method returns flattened GDF.
Pipeline handles merging, object_id assignment, and I/O.

Multi-enhancement WBF mode
--------------------------
When ``data_state.enhanced_tiles_paths`` is populated (by RgbEnhancerComponent),
the detector runs on each set of enhanced tiles and Weighted Box Fusion (WBF)
is applied per tile to consolidate detections from all enhancement methods.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import geopandas as gpd
from shapely.geometry import box as shapely_box

from geodataset.dataset import UnlabeledRasterDataset

from canopyrs.engine.constants import Col, StateKey
from canopyrs.engine.components.base import BaseComponent, ComponentResult, validate_requirements
from canopyrs.engine.config_parsers import DetectorConfig
from canopyrs.engine.data_state import DataState
from canopyrs.engine.models.registry import DETECTOR_REGISTRY
from canopyrs.engine.models.utils import collate_fn_images

# IoU threshold used when WBF-fusing detections from multiple enhancement methods.
_WBF_IOU_THRESH = 0.4


class DetectorComponent(BaseComponent):
    """
    Runs object detection on image tiles.

    Requirements:
        - tiles_path: Directory containing tiles to process

    Produces:
        - infer_gdf: GeoDataFrame with detected bounding boxes
        - Columns: geometry, object_id, tile_path, detector_score, detector_class
    """

    name = 'detector'

    BASE_REQUIRES_STATE = {StateKey.TILES_PATH}
    BASE_REQUIRES_COLUMNS: Set[str] = set()

    BASE_PRODUCES_STATE = {StateKey.INFER_GDF, StateKey.INFER_COCO_PATH}
    BASE_PRODUCES_COLUMNS = {Col.GEOMETRY, Col.OBJECT_ID, Col.TILE_PATH, Col.DETECTOR_SCORE, Col.DETECTOR_CLASS}

    BASE_STATE_HINTS = {
        StateKey.TILES_PATH: (
            "Detector needs tiles to process. Add a tilerizer before detector."
        ),
    }

    BASE_COLUMN_HINTS: dict = {}

    def __init__(
        self,
        config: DetectorConfig,
        parent_output_path: str = None,
        component_id: int = None
    ):
        super().__init__(config, parent_output_path, component_id)

        # Store model class (instantiate in __call__ to avoid loading during validation)
        if config.model not in DETECTOR_REGISTRY:
            raise ValueError(f'Invalid detector model: {config.model}')
        self._model_class = DETECTOR_REGISTRY.get(config.model)

        # Set requirements
        self.requires_state = set(self.BASE_REQUIRES_STATE)
        self.requires_columns = set(self.BASE_REQUIRES_COLUMNS)
        self.produces_state = set(self.BASE_PRODUCES_STATE)
        self.produces_columns = set(self.BASE_PRODUCES_COLUMNS)

        # Set hints
        self.state_hints = dict(self.BASE_STATE_HINTS)
        self.column_hints = dict(self.BASE_COLUMN_HINTS)

    @classmethod
    def run_standalone(
        cls,
        config: DetectorConfig,
        tiles_path: str,
        output_path: str,
    ) -> 'DataState':
        """
        Run detector standalone on pre-tiled imagery.

        Args:
            config: Detector configuration
            tiles_path: Path to directory containing tiles
            output_path: Where to save outputs

        Returns:
            DataState with detection results (access .infer_gdf for the GeoDataFrame)

        Example:
            result = DetectorComponent.run_standalone(
                config=DetectorConfig(model='faster_rcnn_detectron2', ...),
                tiles_path='./tiles',
                output_path='./output',
            )
            print(result.infer_gdf)
        """
        from canopyrs.engine.pipeline import run_component
        return run_component(
            component=cls(config),
            output_path=output_path,
            tiles_path=tiles_path,
        )

    @validate_requirements
    def __call__(self, data_state: DataState) -> ComponentResult:
        """
        Run object detection on tiles.

        If ``data_state.enhanced_tiles_paths`` is set, detection is run on
        each enhanced tile set and results are fused with WBF before building
        the output GeoDataFrame.  Otherwise, the standard single-run path on
        ``data_state.tiles_path`` is used.
        """
        if data_state.enhanced_tiles_paths:
            return self._call_enhanced(data_state)
        return self._call_standard(data_state)

    # ------------------------------------------------------------------
    # Standard (single tile set) detection
    # ------------------------------------------------------------------

    def _call_standard(self, data_state: DataState) -> ComponentResult:
        """Run detection on ``data_state.tiles_path`` — original behaviour."""
        detector = self._model_class(self.config)
        infer_ds = UnlabeledRasterDataset(
            fold=None,
            root_path=data_state.tiles_path,
            transform=None,
        )
        tiles_paths, boxes, boxes_scores, classes = detector.infer(infer_ds, collate_fn_images)
        gdf = self._build_gdf(tiles_paths, boxes, boxes_scores, classes)
        print(f"DetectorComponent: {len(gdf)} detections.")
        return self._as_result(gdf)

    # ------------------------------------------------------------------
    # Multi-method (enhanced tiles + WBF) detection
    # ------------------------------------------------------------------

    def _call_enhanced(self, data_state: DataState) -> ComponentResult:
        """Run detection on each enhanced tile set, then WBF-fuse per tile."""
        detector = self._model_class(self.config)

        # Accumulate per-tile-filename detections across all enhancement methods
        # tile_filename → list of (shapely_geom, score, cls_id)
        tile_dets: Dict[str, List[Tuple]] = {}

        for method_name, method_path in data_state.enhanced_tiles_paths.items():
            infer_ds = UnlabeledRasterDataset(
                fold=None,
                root_path=method_path,
                transform=None,
            )
            m_tile_paths, m_boxes, m_scores, m_classes = detector.infer(
                infer_ds, collate_fn_images
            )
            for i, tile_path in enumerate(m_tile_paths):
                key = Path(tile_path).name
                if key not in tile_dets:
                    tile_dets[key] = []
                for geom, score, cls in zip(m_boxes[i], m_scores[i], m_classes[i]):
                    tile_dets[key].append((geom, float(score), int(cls)))

        # WBF-fuse per tile; use original tile paths for downstream aggregation
        original_tiles_path = Path(data_state.tiles_path)
        # Build a filename → full path lookup to handle AOI subdirectories
        # (e.g. tiles_path = .../tiles/ but actual files are in .../tiles/infer/)
        orig_path_lookup: Dict[str, str] = {
            p.name: str(p)
            for p in original_tiles_path.rglob("*.tif")
        }
        rows = []
        unique_id = 0
        for tile_filename, dets in tile_dets.items():
            if not dets:
                continue
            geoms   = [d[0] for d in dets]
            scores  = [d[1] for d in dets]
            fused   = _wbf_geographic(geoms, scores, iou_thresh=_WBF_IOU_THRESH)
            orig_path = orig_path_lookup.get(tile_filename, str(original_tiles_path / tile_filename))
            for fused_geom, fused_score in fused:
                rows.append({
                    Col.GEOMETRY:       fused_geom,
                    Col.TILE_PATH:      orig_path,
                    Col.DETECTOR_SCORE: fused_score,
                    Col.DETECTOR_CLASS: 0,
                    Col.OBJECT_ID:      unique_id,
                })
                unique_id += 1

        if rows:
            gdf = gpd.GeoDataFrame(rows, geometry=Col.GEOMETRY, crs=None)
            gdf[Col.GEOMETRY] = gdf[Col.GEOMETRY].buffer(0)
            gdf = gdf[gdf.is_valid & ~gdf.is_empty]
        else:
            gdf = gpd.GeoDataFrame(
                columns=list(self.produces_columns), crs=None
            )

        n_methods = len(data_state.enhanced_tiles_paths)
        print(
            f"DetectorComponent (enhanced × {n_methods} methods, WBF): "
            f"{len(gdf)} detections."
        )
        return self._as_result(gdf)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _build_gdf(self, tiles_paths, boxes, boxes_scores, classes):
        rows = []
        unique_id = 0
        for i, tile_path in enumerate(tiles_paths):
            for geom, score, cls in zip(boxes[i], boxes_scores[i], classes[i]):
                rows.append({
                    Col.GEOMETRY:       geom,
                    Col.TILE_PATH:      str(tile_path),
                    Col.DETECTOR_SCORE: score,
                    Col.DETECTOR_CLASS: cls,
                    Col.OBJECT_ID:      unique_id,
                })
                unique_id += 1
        if not rows:
            return gpd.GeoDataFrame(columns=list(self.produces_columns), crs=None)
        gdf = gpd.GeoDataFrame(rows, geometry=Col.GEOMETRY, crs=None)
        gdf[Col.GEOMETRY] = gdf[Col.GEOMETRY].buffer(0)
        return gdf[gdf.is_valid & ~gdf.is_empty]

    @staticmethod
    def _as_result(gdf: gpd.GeoDataFrame) -> ComponentResult:
        return ComponentResult(
            gdf=gdf,
            produced_columns=DetectorComponent.BASE_PRODUCES_COLUMNS,
            objects_are_new=True,
            save_gpkg=True,
            gpkg_name_suffix="notaggregated",
            save_coco=True,
            coco_scores_column=Col.DETECTOR_SCORE,
            coco_categories_column=Col.DETECTOR_CLASS,
        )


# ---------------------------------------------------------------------------
# Weighted Box Fusion (WBF) in geographic coordinates
# ---------------------------------------------------------------------------

def _wbf_geographic(
    geoms: List,
    scores: List[float],
    iou_thresh: float = 0.4,
) -> List[Tuple]:
    """Weighted Box Fusion for boxes expressed in geographic coordinates.

    All overlapping boxes that share IoU ≥ ``iou_thresh`` are grouped into a
    cluster.  Each cluster is replaced by a single box whose coordinates are
    the score-weighted average of the cluster members, with the cluster's
    maximum score as the fused confidence.

    Args:
        geoms:      List of Shapely geometries (bounding boxes assumed).
        scores:     Confidence scores, same length as *geoms*.
        iou_thresh: IoU threshold for clustering.

    Returns:
        List of ``(shapely.geometry, float_score)`` tuples.
    """
    import torch
    from torchvision.ops import box_iou

    if not geoms:
        return []

    bboxes = torch.tensor(
        [[g.bounds[0], g.bounds[1], g.bounds[2], g.bounds[3]] for g in geoms],
        dtype=torch.float32,
    )
    score_t = torch.tensor(scores, dtype=torch.float32)

    # Sort by descending score so that the highest-confidence box anchors each cluster
    order   = score_t.argsort(descending=True)
    bboxes  = bboxes[order]
    score_t = score_t[order]

    used   = torch.zeros(len(bboxes), dtype=torch.bool)
    result = []

    for i in range(len(bboxes)):
        if used[i]:
            continue
        iou_row = box_iou(bboxes[i:i+1], bboxes)[0]   # shape (N,)
        cluster = (iou_row >= iou_thresh) & (~used)
        cluster[i] = True   # always include the anchor

        cb  = bboxes[cluster]    # (K, 4)
        cs  = score_t[cluster]   # (K,)
        w   = cs / cs.sum()      # normalised weights
        fb  = (w.unsqueeze(1) * cb).sum(dim=0)   # (4,) fused box

        result.append((
            shapely_box(fb[0].item(), fb[1].item(), fb[2].item(), fb[3].item()),
            float(cs.max()),
        ))
        used[cluster] = True

    return result
