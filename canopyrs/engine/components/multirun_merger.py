"""
MultiRunMergerComponent: consolidates multiple GeoDataFrames of tree crown polygons
produced by independent segmentation pipeline runs into a single consensus GDF.

Typical usage: run the CanopyRS segmentation pipeline N times, each time using a
different pseudo-RGB spectral combination derived from a multispectral image (e.g.
NIR+Red+Green, Red+Green+Blue, NDVI-grayscale, CIR).  Each run stores its output
GeoDataFrame in ``data_state.infer_gdf``.  Collect those GDFs into a list and assign
the list to ``data_state.multirun_gdfs``, then run this component.

Example::

    # Run pipeline 3 times with different pseudo-RGB band combinations
    gdfs = []
    for band_combo in [('nir', 'red', 'green'), ('red', 'green', 'blue'), ('ndvi_grey',)]:
        result = run_pipeline(ms_tif, band_combo)  # returns DataState
        gdfs.append(result.infer_gdf)

    # Merge them
    config = MultiRunMergerConfig(
        iou_threshold=0.3,
        min_observations=2,
        geometry_method='majority_union',
        majority_threshold=0.4,
        score_aggregation='mean',
    )
    merged = MultiRunMergerComponent.run_standalone(config, gdfs, './output')
    print(merged.infer_gdf[['geometry', 'segmenter_score', 'observation_count', 'run_agreement']])
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import List, Optional, Set

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union

from canopyrs.engine.constants import Col, StateKey
from canopyrs.engine.components.base import (
    BaseComponent,
    ComponentResult,
    validate_requirements,
)
from canopyrs.engine.config_parsers.multirun_merger import MultiRunMergerConfig
from canopyrs.engine.data_state import DataState


# =============================================================================
# Union-Find (Disjoint Set)
# =============================================================================

class _UnionFind:
    """Minimal union-find / disjoint-set data structure."""

    def __init__(self, n: int):
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path compression
            x = self._parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1


# =============================================================================
# IoU helper
# =============================================================================

def _iou(geom_a, geom_b) -> float:
    """Return the intersection-over-union of two Shapely geometries."""
    try:
        inter = geom_a.intersection(geom_b).area
        if inter == 0.0:
            return 0.0
        union = geom_a.union(geom_b).area
        if union == 0.0:
            return 0.0
        return inter / union
    except Exception:
        return 0.0


# =============================================================================
# MultiRunMergerComponent
# =============================================================================

class MultiRunMergerComponent(BaseComponent):
    """
    Consolidates multiple GeoDataFrames of tree crown polygons from independent
    segmentation pipeline runs into a single consensus GDF.

    Each input GDF (one per pipeline run) must contain at minimum:
    ``geometry``, ``canopyrs_object_id``, and ``tile_path``.  An optional
    ``segmenter_score`` (or ``aggregator_score``) column is used for scoring.

    The algorithm:
    1. Tag each GDF with a ``run_index`` (0, 1, 2, …).
    2. Concatenate all tagged GDFs into a flat GDF.
    3. Reproject to EPSG:3857 for area/IoU calculations when the CRS is geographic.
    4. Build an STRtree spatial index.
    5. Build an IoU overlap graph: polygons from *different* runs with
       IoU >= ``config.iou_threshold`` get an edge.
    6. Find connected components with union-find (no networkx dependency).
    7. For each component apply ``min_observations`` filter, compute consensus
       geometry (``geometry_method``), consensus score (``score_aggregation``),
       and optional metadata columns.
    8. Assign new ``canopyrs_object_id`` values.
    9. Reproject back to the original CRS.

    Inputs (from DataState):
        ``multirun_gdfs``: ``List[gpd.GeoDataFrame]`` — one GDF per pipeline run.
        Falls back to ``infer_gdf`` (passthrough) when ``multirun_gdfs`` is None.

    Outputs (to DataState):
        ``infer_gdf``: merged consensus GDF with optional metadata columns.
    """

    name = "multirun_merger"

    BASE_REQUIRES_STATE: Set[str] = {StateKey.MULTIRUN_GDFS}
    BASE_REQUIRES_COLUMNS: Set[str] = set()

    BASE_PRODUCES_STATE: Set[str] = {StateKey.INFER_GDF}
    BASE_PRODUCES_COLUMNS: Set[str] = {
        Col.GEOMETRY,
        Col.OBJECT_ID,
        Col.TILE_PATH,
        Col.SEGMENTER_SCORE,
    }

    BASE_STATE_HINTS = {
        StateKey.MULTIRUN_GDFS: (
            "MultiRunMergerComponent needs a list of GDFs. "
            "Set data_state.multirun_gdfs before running this component."
        ),
    }

    def __init__(
        self,
        config: MultiRunMergerConfig,
        parent_output_path: str = None,
        component_id: int = None,
    ):
        super().__init__(config, parent_output_path, component_id)

        self.requires_state   = set(self.BASE_REQUIRES_STATE)
        self.requires_columns = set(self.BASE_REQUIRES_COLUMNS)
        self.produces_state   = set(self.BASE_PRODUCES_STATE)
        self.produces_columns = set(self.BASE_PRODUCES_COLUMNS)
        self.state_hints      = dict(self.BASE_STATE_HINTS)
        self.column_hints     = {}

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    @validate_requirements
    def __call__(self, data_state: DataState) -> ComponentResult:
        """
        Merge all GDFs in ``data_state.multirun_gdfs`` into one consensus GDF.

        If ``multirun_gdfs`` contains only one GDF (or is None while ``infer_gdf``
        is set) the component acts as a passthrough with a warning.
        """
        gdfs: Optional[List[gpd.GeoDataFrame]] = data_state.multirun_gdfs

        # Passthrough / degenerate cases
        if not gdfs:
            if data_state.infer_gdf is not None:
                warnings.warn(
                    "MultiRunMergerComponent: multirun_gdfs is empty/None but "
                    "infer_gdf is set — passing infer_gdf through unchanged.",
                    stacklevel=2,
                )
                return ComponentResult(
                    gdf=data_state.infer_gdf,
                    objects_are_new=True,
                    save_gpkg=True,
                    gpkg_name_suffix="multirun_merged",
                )
            return ComponentResult(
                gdf=gpd.GeoDataFrame(),
                objects_are_new=True,
                save_gpkg=False,
            )

        if len(gdfs) == 1:
            warnings.warn(
                "MultiRunMergerComponent: only one GDF provided in multirun_gdfs — "
                "returning it unchanged.",
                stacklevel=2,
            )
            return ComponentResult(
                gdf=gdfs[0],
                objects_are_new=True,
                save_gpkg=True,
                gpkg_name_suffix="multirun_merged",
            )

        merged_gdf = _merge_gdfs(gdfs, self.config)

        return ComponentResult(
            gdf=merged_gdf,
            objects_are_new=True,
            save_gpkg=True,
            gpkg_name_suffix="multirun_merged",
        )

    @classmethod
    def run_standalone(
        cls,
        config: MultiRunMergerConfig,
        gdfs: List[gpd.GeoDataFrame],
        output_path: str,
        product_name: str = "multirun_merged",
    ) -> DataState:
        """
        Merge a list of GDFs without a full pipeline setup.

        Args:
            config:       MultiRunMergerConfig instance.
            gdfs:         List of GeoDataFrames to merge (one per pipeline run).
            output_path:  Directory where the output GeoPackage will be written.
            product_name: Name prefix for the output files.

        Returns:
            DataState with ``infer_gdf`` set to the merged GDF.
        """
        from canopyrs.engine.data_state import DataState

        data_state = DataState(
            parent_output_path=output_path,
            product_name=product_name,
            multirun_gdfs=gdfs,
        )

        component = cls(config, output_path, component_id=0)

        # Bypass the validate_requirements decorator (multirun_gdfs is set directly)
        merged_gdf = _merge_gdfs(gdfs, config) if len(gdfs) > 1 else (gdfs[0] if gdfs else gpd.GeoDataFrame())

        data_state.infer_gdf = merged_gdf
        return data_state


# =============================================================================
# Core merge logic (module-level for testability)
# =============================================================================

def _merge_gdfs(
    gdfs: List[gpd.GeoDataFrame],
    config: MultiRunMergerConfig,
) -> gpd.GeoDataFrame:
    """
    Main merge implementation.  Called both by ``__call__`` and ``run_standalone``.
    """
    # --- 1. Tag each GDF with run_index and concatenate ----------------------
    tagged: List[gpd.GeoDataFrame] = []
    for run_idx, gdf in enumerate(gdfs):
        g = gdf.copy()
        g["_run_index"] = run_idx
        tagged.append(g)

    flat = gpd.GeoDataFrame(pd.concat(tagged, ignore_index=True))

    # Preserve original CRS from first GDF
    original_crs = gdfs[0].crs

    # --- 2. Determine score column -------------------------------------------
    score_col = _resolve_score_column(flat)

    # --- 3. Reproject to metric CRS for area/IoU calculations ----------------
    needs_reproject = (original_crs is not None) and original_crs.is_geographic
    if needs_reproject:
        flat = flat.to_crs(epsg=3857)

    # --- 4. Build STRtree spatial index --------------------------------------
    from shapely.strtree import STRtree

    geometries = list(flat.geometry)
    run_indices = flat["_run_index"].tolist()
    n = len(flat)

    tree = STRtree(geometries)

    # --- 5 & 6. Build IoU overlap graph with union-find ----------------------
    uf = _UnionFind(n)

    for i, geom_i in enumerate(geometries):
        run_i = run_indices[i]
        # Query by bounding box
        candidate_indices = tree.query(geom_i)
        for j in candidate_indices:
            j = int(j)
            if j <= i:
                continue
            if run_indices[j] == run_i:
                # Only cluster polygons from *different* runs
                continue
            iou_val = _iou(geom_i, geometries[j])
            if iou_val >= config.iou_threshold:
                uf.union(i, j)

    # Group rows by connected component
    from collections import defaultdict
    clusters: dict = defaultdict(list)
    for idx in range(n):
        root = uf.find(idx)
        clusters[root].append(idx)

    n_runs_total = len(gdfs)

    # --- 7. Process each cluster ---------------------------------------------
    result_rows = []

    for root, members in clusters.items():
        cluster_df = flat.iloc[members].copy()
        distinct_runs = sorted(cluster_df["_run_index"].unique().tolist())
        obs_count = len(distinct_runs)

        # Apply min_observations filter
        if obs_count < config.min_observations:
            continue

        cluster_geoms = list(cluster_df.geometry)

        # Compute consensus geometry
        consensus_geom = _compute_geometry(
            cluster_geoms=cluster_geoms,
            cluster_df=cluster_df,
            score_col=score_col,
            geometry_method=config.geometry_method,
            majority_threshold=config.majority_threshold,
            n_runs_total=n_runs_total,
        )
        if consensus_geom is None or consensus_geom.is_empty:
            continue

        # Compute consensus score
        consensus_score = _compute_score(
            cluster_df=cluster_df,
            score_col=score_col,
            score_aggregation=config.score_aggregation,
            obs_count=obs_count,
        )

        # Pick a representative tile_path (from highest-score row if possible)
        tile_path = _pick_tile_path(cluster_df, score_col)

        row: dict = {
            Col.GEOMETRY: consensus_geom,
            Col.TILE_PATH: tile_path,
            Col.SEGMENTER_SCORE: consensus_score,
        }

        if config.add_observation_count:
            row["observation_count"] = obs_count

        if config.add_run_agreement:
            row["run_agreement"] = _compute_run_agreement(cluster_geoms)

        if config.add_source_runs:
            row["source_runs"] = str(distinct_runs)

        result_rows.append(row)

    if not result_rows:
        out_gdf = gpd.GeoDataFrame(columns=[Col.GEOMETRY, Col.OBJECT_ID, Col.TILE_PATH, Col.SEGMENTER_SCORE])
        if original_crs is not None:
            out_gdf = out_gdf.set_crs(original_crs)
        return out_gdf

    # --- 8. Assign new canopyrs_object_id ------------------------------------
    out_gdf = gpd.GeoDataFrame(result_rows, geometry=Col.GEOMETRY)
    out_gdf[Col.OBJECT_ID] = range(len(out_gdf))

    # --- 9. Reproject back to original CRS -----------------------------------
    if needs_reproject and original_crs is not None:
        out_gdf = out_gdf.set_crs(epsg=3857)
        out_gdf = out_gdf.to_crs(original_crs)
    elif original_crs is not None:
        out_gdf = out_gdf.set_crs(original_crs)

    return out_gdf


# =============================================================================
# Helper functions
# =============================================================================

def _resolve_score_column(flat: gpd.GeoDataFrame) -> Optional[str]:
    """Return the score column name present in the flat GDF, or None."""
    if Col.SEGMENTER_SCORE in flat.columns and flat[Col.SEGMENTER_SCORE].notna().any():
        return Col.SEGMENTER_SCORE
    if Col.AGGREGATOR_SCORE in flat.columns and flat[Col.AGGREGATOR_SCORE].notna().any():
        return Col.AGGREGATOR_SCORE
    return None


def _compute_geometry(
    cluster_geoms: list,
    cluster_df: gpd.GeoDataFrame,
    score_col: Optional[str],
    geometry_method: str,
    majority_threshold: float,
    n_runs_total: int,
) -> object:
    """Compute the consensus geometry for a cluster."""

    if geometry_method == 'best_score':
        if score_col and score_col in cluster_df.columns:
            best_idx = cluster_df[score_col].idxmax()
            return cluster_df.loc[best_idx, Col.GEOMETRY]
        # Fallback: largest polygon
        return max(cluster_geoms, key=lambda g: g.area)

    all_union = unary_union(cluster_geoms)

    if geometry_method == 'intersection_union_blend':
        intersection = cluster_geoms[0]
        for g in cluster_geoms[1:]:
            intersection = intersection.intersection(g)
            if intersection.is_empty:
                break
        if intersection.is_empty:
            return all_union
        # Buffer the intersection slightly toward the union.  The factor 0.1
        # gives a ~10 % "reach" relative to the sqrt of the gap area, which
        # produces a gentle smoothing without over-inflating small polygons.
        _BLEND_BUFFER_FACTOR = 0.1
        blend_buffer = ((all_union.area - intersection.area) ** 0.5) * _BLEND_BUFFER_FACTOR
        blended = intersection.buffer(blend_buffer)
        return blended

    # 'majority_union' (default)
    n_runs_in_cluster = len(cluster_df["_run_index"].unique())
    min_votes = math.ceil(n_runs_in_cluster * majority_threshold)
    min_votes = max(min_votes, 1)

    # Count how many runs contributed each polygon and keep majority
    majority_geoms = []
    for geom, run_count in _group_by_run(cluster_df):
        if run_count >= min_votes:
            majority_geoms.append(geom)

    if not majority_geoms:
        return all_union

    return unary_union(majority_geoms)


def _group_by_run(cluster_df: gpd.GeoDataFrame):
    """
    Yield (geometry, run_count) for each polygon in the cluster.

    Since a single run may contribute multiple polygons to the cluster
    (unlikely but possible if the same run produced overlapping detections),
    we weight by run occurrence.
    """
    run_counts = cluster_df["_run_index"].value_counts()
    for _, row in cluster_df.iterrows():
        yield row[Col.GEOMETRY], run_counts[row["_run_index"]]


def _compute_score(
    cluster_df: gpd.GeoDataFrame,
    score_col: Optional[str],
    score_aggregation: str,
    obs_count: int,
) -> float:
    """Compute the consensus score for a cluster."""
    if score_col is None or score_col not in cluster_df.columns:
        # No score column available: use 1/obs_count as a proxy so that trees
        # seen by more runs (higher confidence) get a lower raw score but the
        # caller can interpret it as an inverse-frequency weight.  In practice
        # a real score column is almost always present.
        return 1.0 / obs_count

    scores = cluster_df[score_col].dropna()
    if scores.empty:
        return 1.0 / obs_count

    if score_aggregation == 'max':
        return float(scores.max())
    if score_aggregation == 'weighted_mean':
        areas = cluster_df.loc[scores.index, Col.GEOMETRY].area
        total_area = areas.sum()
        if total_area == 0:
            return float(scores.mean())
        return float((scores * areas).sum() / total_area)
    # 'mean' (default)
    return float(scores.mean())


def _pick_tile_path(cluster_df: gpd.GeoDataFrame, score_col: Optional[str]) -> Optional[str]:
    """Pick a representative tile_path from the cluster."""
    if Col.TILE_PATH not in cluster_df.columns:
        return None
    if score_col and score_col in cluster_df.columns:
        valid = cluster_df[cluster_df[score_col].notna()]
        if not valid.empty:
            return valid.loc[valid[score_col].idxmax(), Col.TILE_PATH]
    return cluster_df[Col.TILE_PATH].iloc[0]


def _compute_run_agreement(cluster_geoms: list) -> float:
    """
    Compute run agreement as intersection.area / union.area for the cluster.
    Returns 0.0 if union area is 0 or intersection is empty.
    """
    if len(cluster_geoms) == 1:
        return 1.0
    try:
        all_union = unary_union(cluster_geoms)
        intersection = cluster_geoms[0]
        for g in cluster_geoms[1:]:
            intersection = intersection.intersection(g)
            if intersection.is_empty:
                return 0.0
        union_area = all_union.area
        if union_area == 0:
            return 0.0
        return float(intersection.area / union_area)
    except Exception:
        return 0.0
