"""
Unit tests for MultiRunMergerComponent and its helpers.

Tests cover:
- _UnionFind data structure
- _iou helper function
- _compute_run_agreement helper
- _merge_gdfs core logic (geometry methods, score aggregation,
  min_observations filter, IoU threshold, metadata columns, CRS handling)
- MultiRunMergerComponent.__call__ (passthrough and merge paths)

No heavy dependencies (no torch / cv2 / SAM models) are required.
"""

import warnings
from typing import List

import geopandas as gpd
import pytest
from shapely.geometry import box

from canopyrs.engine.config_parsers.multirun_merger import MultiRunMergerConfig
from canopyrs.engine.components.multirun_merger import (
    MultiRunMergerComponent,
    _UnionFind,
    _compute_run_agreement,
    _iou,
    _merge_gdfs,
)
from canopyrs.engine.constants import Col
from canopyrs.engine.data_state import DataState

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Use a projected (metric) CRS so _merge_gdfs skips the geographic reprojection
# branch and IoU values are pure Cartesian — easy to reason about in tests.
_METRIC_CRS = "EPSG:32632"
_GEO_CRS    = "EPSG:4326"


def _make_gdf(
    boxes: List[tuple],
    scores: List[float] = None,
    tile_path: str = "/tile.tif",
    crs: str = _METRIC_CRS,
) -> gpd.GeoDataFrame:
    """Build a minimal segmenter-output GDF from a list of (minx,miny,maxx,maxy)."""
    geoms = [box(*b) for b in boxes]
    n = len(geoms)
    data: dict = {
        Col.GEOMETRY: geoms,
        Col.OBJECT_ID: list(range(n)),
        Col.TILE_PATH: [tile_path] * n,
    }
    if scores is not None:
        data[Col.SEGMENTER_SCORE] = scores
    return gpd.GeoDataFrame(data, geometry=Col.GEOMETRY, crs=crs)


def _cfg(**kwargs) -> MultiRunMergerConfig:
    return MultiRunMergerConfig(**kwargs)


def _component(**cfg_kwargs) -> MultiRunMergerComponent:
    config = _cfg(**cfg_kwargs)
    return MultiRunMergerComponent(config, parent_output_path="/tmp", component_id=0)


def _state(gdfs: list) -> DataState:
    ds = DataState(parent_output_path="/tmp", product_name="test")
    ds.multirun_gdfs = gdfs
    return ds


# =============================================================================
# _UnionFind
# =============================================================================

class TestUnionFind:
    def test_each_element_is_own_root_initially(self):
        uf = _UnionFind(5)
        for i in range(5):
            assert uf.find(i) == i

    def test_union_two_elements_same_root(self):
        uf = _UnionFind(5)
        uf.union(0, 1)
        assert uf.find(0) == uf.find(1)

    def test_union_is_transitive(self):
        uf = _UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.find(0) == uf.find(2)

    def test_separate_components_do_not_merge(self):
        uf = _UnionFind(6)
        uf.union(0, 1)
        uf.union(2, 3)
        # 0-1 form one component, 2-3 another, 4-5 untouched
        assert uf.find(0) == uf.find(1)
        assert uf.find(2) == uf.find(3)
        assert uf.find(0) != uf.find(2)
        # 4 and 5 are their own roots (separate from 0-1 and 2-3)
        assert uf.find(4) != uf.find(0)
        assert uf.find(4) != uf.find(2)
        assert uf.find(4) != uf.find(5)

    def test_union_same_element_no_op(self):
        uf = _UnionFind(3)
        uf.union(1, 1)
        assert uf.find(1) == 1

    def test_all_elements_merged(self):
        uf = _UnionFind(4)
        for i in range(3):
            uf.union(i, i + 1)
        root = uf.find(0)
        for i in range(1, 4):
            assert uf.find(i) == root


# =============================================================================
# _iou
# =============================================================================

class TestIou:
    def test_identical_boxes_returns_1(self):
        b = box(0, 0, 10, 10)
        assert _iou(b, b) == pytest.approx(1.0)

    def test_no_overlap_returns_0(self):
        a = box(0, 0, 5, 5)
        b = box(10, 10, 15, 15)
        assert _iou(a, b) == pytest.approx(0.0)

    def test_half_overlap(self):
        # a: (0,0)→(10,10), b: (5,0)→(15,10)
        # intersection = 50, union = 150  →  IoU = 1/3
        a = box(0, 0, 10, 10)
        b = box(5, 0, 15, 10)
        assert _iou(a, b) == pytest.approx(1 / 3, rel=1e-5)

    def test_contained_box(self):
        # b completely inside a  →  IoU = area(b) / area(a) = 36/100
        a = box(0, 0, 10, 10)
        b = box(2, 2, 8, 8)
        assert _iou(a, b) == pytest.approx(36 / 100, rel=1e-5)

    def test_slight_overlap(self):
        # a: (0,0,10,10), b: (9,0,19,10)
        # intersection = 1×10 = 10, union = 10*10 + 10*10 - 10 = 190  →  IoU ≈ 0.0526
        a = box(0, 0, 10, 10)
        b = box(9, 0, 19, 10)
        assert _iou(a, b) == pytest.approx(10 / 190, rel=1e-4)


# =============================================================================
# _compute_run_agreement
# =============================================================================

class TestComputeRunAgreement:
    def test_single_geometry_returns_1(self):
        assert _compute_run_agreement([box(0, 0, 10, 10)]) == pytest.approx(1.0)

    def test_identical_geometries_returns_1(self):
        b = box(0, 0, 10, 10)
        assert _compute_run_agreement([b, b]) == pytest.approx(1.0)

    def test_no_overlap_returns_0(self):
        a = box(0, 0, 5, 5)
        b = box(10, 10, 15, 15)
        assert _compute_run_agreement([a, b]) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = box(0, 0, 10, 10)   # area 100
        b = box(5, 0, 15, 10)   # area 100, overlap 50
        # intersection = 50, union = 150
        val = _compute_run_agreement([a, b])
        assert val == pytest.approx(50 / 150, rel=1e-4)

    def test_three_identical_geometries(self):
        b = box(0, 0, 10, 10)
        assert _compute_run_agreement([b, b, b]) == pytest.approx(1.0)


# =============================================================================
# _merge_gdfs — edge cases & filtering
# =============================================================================

class TestMergeEdgeCases:
    def test_no_overlap_min_obs_1_keeps_all(self):
        """Two runs with completely separate trees → both kept (min_obs=1)."""
        gdf1 = _make_gdf([(0, 0, 10, 10)], scores=[0.9])
        gdf2 = _make_gdf([(100, 100, 110, 110)], scores=[0.8])
        result = _merge_gdfs([gdf1, gdf2], _cfg(iou_threshold=0.3, min_observations=1))
        assert len(result) == 2

    def test_no_overlap_min_obs_2_filters_all(self):
        """Two runs with completely separate trees → all filtered (min_obs=2)."""
        gdf1 = _make_gdf([(0, 0, 10, 10)], scores=[0.9])
        gdf2 = _make_gdf([(100, 100, 110, 110)], scores=[0.8])
        result = _merge_gdfs([gdf1, gdf2], _cfg(iou_threshold=0.3, min_observations=2))
        assert len(result) == 0

    def test_empty_result_has_correct_columns(self):
        """Empty output GDF should still have the standard columns."""
        gdf1 = _make_gdf([(0, 0, 10, 10)], scores=[0.9])
        gdf2 = _make_gdf([(100, 100, 110, 110)], scores=[0.8])
        result = _merge_gdfs([gdf1, gdf2], _cfg(iou_threshold=0.3, min_observations=2))
        assert Col.GEOMETRY in result.columns
        assert Col.OBJECT_ID in result.columns


# =============================================================================
# _merge_gdfs — IoU threshold and merging
# =============================================================================

class TestMergeIoU:
    def test_identical_trees_two_runs_merged_to_one(self):
        b = (0, 0, 10, 10)
        gdf1 = _make_gdf([b], scores=[0.9])
        gdf2 = _make_gdf([b], scores=[0.8])
        result = _merge_gdfs([gdf1, gdf2], _cfg(iou_threshold=0.3))
        assert len(result) == 1

    def test_below_threshold_stays_separate(self):
        # IoU ≈ 10/190 ≈ 0.053, below default 0.3 → not merged
        gdf1 = _make_gdf([(0, 0, 10, 10)], scores=[0.9])
        gdf2 = _make_gdf([(9, 0, 19, 10)], scores=[0.8])
        result = _merge_gdfs([gdf1, gdf2], _cfg(iou_threshold=0.3, min_observations=1))
        assert len(result) == 2

    def test_above_threshold_merged(self):
        # (0,0,10,10) vs (2,0,12,10): intersection=80, union=120, IoU≈0.667 > 0.3
        gdf1 = _make_gdf([(0, 0, 10, 10)], scores=[0.9])
        gdf2 = _make_gdf([(2, 0, 12, 10)], scores=[0.8])
        result = _merge_gdfs([gdf1, gdf2], _cfg(iou_threshold=0.3))
        assert len(result) == 1

    def test_two_trees_each_matched_across_runs(self):
        """Two run-pairs, both overlapping → 2 merged output polygons."""
        gdf1 = _make_gdf([(0, 0, 10, 10), (100, 0, 110, 10)], scores=[0.9, 0.8])
        gdf2 = _make_gdf([(0, 0, 10, 10), (100, 0, 110, 10)], scores=[0.7, 0.6])
        result = _merge_gdfs([gdf1, gdf2], _cfg(iou_threshold=0.3))
        assert len(result) == 2

    def test_same_run_polygons_not_merged(self):
        """Overlapping polygons from the *same* run must NOT be merged together."""
        # Both polygons in the same GDF overlap with IoU=0.667;
        # the second GDF has a non-overlapping box at (100,0,110,10).
        # Because same-run polygons are skipped in the IoU graph, the two
        # boxes from run 0 remain in separate clusters.
        gdf1 = _make_gdf([(0, 0, 10, 10), (2, 0, 12, 10)], scores=[0.9, 0.8])
        gdf2 = _make_gdf([(100, 0, 110, 10)], scores=[0.5])
        result = _merge_gdfs([gdf1, gdf2], _cfg(iou_threshold=0.3, min_observations=1))
        # With min_obs=1 both polygons from gdf1 survive as separate singletons (1 run each)
        # plus the singleton from gdf2 → 3 output polygons
        assert len(result) == 3

    def test_three_runs_all_overlap_single_output(self):
        b = (0, 0, 10, 10)
        gdfs = [_make_gdf([b], scores=[0.9 - i * 0.1]) for i in range(3)]
        result = _merge_gdfs(gdfs, _cfg(iou_threshold=0.3))
        assert len(result) == 1

    def test_three_runs_observation_count_correct(self):
        b = (0, 0, 10, 10)
        gdfs = [_make_gdf([b], scores=[0.9 - i * 0.1]) for i in range(3)]
        result = _merge_gdfs(gdfs, _cfg(iou_threshold=0.3, add_observation_count=True))
        assert int(result["observation_count"].iloc[0]) == 3

    def test_three_runs_partial_agreement_min_obs_2(self):
        """Tree A detected by 3 runs; Tree B only by run 0 → min_obs=2 keeps only A."""
        b_a = (0, 0, 10, 10)
        b_b = (100, 0, 110, 10)
        gdf0 = _make_gdf([b_a, b_b], scores=[0.9, 0.8])
        gdf1 = _make_gdf([b_a], scores=[0.7])
        gdf2 = _make_gdf([b_a], scores=[0.6])
        result = _merge_gdfs([gdf0, gdf1, gdf2], _cfg(iou_threshold=0.3, min_observations=2))
        assert len(result) == 1


# =============================================================================
# _merge_gdfs — geometry methods
# =============================================================================

class TestGeometryMethods:
    """Each geometry method should return a valid, non-empty polygon."""

    _BOX_A = (0, 0, 10, 10)
    _BOX_B = (2, 0, 12, 10)   # IoU ≈ 0.667 with _BOX_A

    def _merge(self, method: str) -> gpd.GeoDataFrame:
        gdf1 = _make_gdf([self._BOX_A], scores=[0.9])
        gdf2 = _make_gdf([self._BOX_B], scores=[0.5])
        return _merge_gdfs([gdf1, gdf2], _cfg(geometry_method=method, iou_threshold=0.3))

    def test_majority_union_valid(self):
        result = self._merge("majority_union")
        assert len(result) == 1
        assert not result.geometry.iloc[0].is_empty

    def test_intersection_union_blend_valid(self):
        result = self._merge("intersection_union_blend")
        assert len(result) == 1
        assert not result.geometry.iloc[0].is_empty

    def test_best_score_picks_highest_score_polygon(self):
        # Score 0.9 belongs to _BOX_A = (0,0,10,10)
        result = self._merge("best_score")
        assert len(result) == 1
        assert result.geometry.iloc[0].bounds == pytest.approx(self._BOX_A, rel=1e-5)

    def test_best_score_does_not_pick_lower_score_polygon(self):
        # Should NOT be _BOX_B
        result = self._merge("best_score")
        assert result.geometry.iloc[0].bounds != pytest.approx(self._BOX_B, rel=1e-5)

    def test_intersection_union_blend_smaller_than_union(self):
        """Blended geometry should not be larger than the plain union."""
        gdf1 = _make_gdf([self._BOX_A], scores=[0.9])
        gdf2 = _make_gdf([self._BOX_B], scores=[0.5])
        all_union = box(*self._BOX_A).union(box(*self._BOX_B))
        result = self._merge("intersection_union_blend")
        # Blended area must be <= union area (with some tolerance for buffer)
        assert result.geometry.iloc[0].area <= all_union.area * 1.1


# =============================================================================
# _merge_gdfs — score aggregation
# =============================================================================

class TestScoreAggregation:
    """All tests merge two identical boxes to isolate score logic."""

    _BOX = (0, 0, 10, 10)

    def _score(self, score_a: float, score_b: float, method: str) -> float:
        gdf1 = _make_gdf([self._BOX], scores=[score_a])
        gdf2 = _make_gdf([self._BOX], scores=[score_b])
        result = _merge_gdfs([gdf1, gdf2], _cfg(score_aggregation=method, iou_threshold=0.3))
        return float(result[Col.SEGMENTER_SCORE].iloc[0])

    def test_mean(self):
        assert self._score(0.8, 0.6, "mean") == pytest.approx(0.7, rel=1e-5)

    def test_max(self):
        assert self._score(0.8, 0.6, "max") == pytest.approx(0.8, rel=1e-5)

    def test_max_picks_larger(self):
        assert self._score(0.2, 0.95, "max") == pytest.approx(0.95, rel=1e-5)

    def test_weighted_mean_equal_areas_equals_mean(self):
        # Identical boxes → equal areas → weighted_mean == arithmetic mean
        assert self._score(0.8, 0.6, "weighted_mean") == pytest.approx(0.7, rel=1e-4)

    def test_weighted_mean_unequal_areas(self):
        """Weighted mean with different-sized boxes from two runs."""
        # box_a area=100, box_b area=4  →  weighted = (0.9*100 + 0.5*4) / 104 ≈ 0.8846
        box_a = (0, 0, 10, 10)   # area 100
        box_b = (0, 0, 2, 2)     # area 4    →  IoU = 4/100 = 0.04 < 0.3 threshold
        # With default iou_threshold=0.3 they are *not* merged.
        # Lower threshold to 0.01 to force merge.
        gdf1 = _make_gdf([box_a], scores=[0.9])
        gdf2 = _make_gdf([box_b], scores=[0.5])
        result = _merge_gdfs([gdf1, gdf2], _cfg(score_aggregation="weighted_mean", iou_threshold=0.01))
        assert len(result) == 1
        score = float(result[Col.SEGMENTER_SCORE].iloc[0])
        expected = (0.9 * 100 + 0.5 * 4) / (100 + 4)
        assert score == pytest.approx(expected, rel=1e-3)

    def test_no_score_column_returns_float(self):
        """Without a score column the aggregator should still return a valid float."""
        gdf1 = _make_gdf([(0, 0, 10, 10)])  # no scores=
        gdf2 = _make_gdf([(0, 0, 10, 10)])
        result = _merge_gdfs([gdf1, gdf2], _cfg(iou_threshold=0.3))
        assert len(result) == 1
        # Score fallback: 1/obs_count = 1/2 = 0.5
        assert float(result[Col.SEGMENTER_SCORE].iloc[0]) == pytest.approx(0.5, rel=1e-5)


# =============================================================================
# _merge_gdfs — metadata columns
# =============================================================================

class TestMetadataColumns:
    _BOX = (0, 0, 10, 10)

    def _merge(self, **kwargs) -> gpd.GeoDataFrame:
        gdf1 = _make_gdf([self._BOX], scores=[0.8])
        gdf2 = _make_gdf([self._BOX], scores=[0.9])
        return _merge_gdfs([gdf1, gdf2], _cfg(iou_threshold=0.3, **kwargs))

    # observation_count
    def test_observation_count_present_by_default(self):
        result = self._merge()
        assert "observation_count" in result.columns

    def test_observation_count_value(self):
        result = self._merge(add_observation_count=True)
        assert int(result["observation_count"].iloc[0]) == 2

    def test_observation_count_absent_when_disabled(self):
        result = self._merge(add_observation_count=False)
        assert "observation_count" not in result.columns

    # run_agreement
    def test_run_agreement_present_by_default(self):
        result = self._merge()
        assert "run_agreement" in result.columns

    def test_run_agreement_between_0_and_1(self):
        result = self._merge(add_run_agreement=True)
        val = float(result["run_agreement"].iloc[0])
        assert 0.0 <= val <= 1.0

    def test_run_agreement_identical_boxes_is_1(self):
        result = self._merge(add_run_agreement=True)
        assert float(result["run_agreement"].iloc[0]) == pytest.approx(1.0, rel=1e-5)

    def test_run_agreement_absent_when_disabled(self):
        result = self._merge(add_run_agreement=False)
        assert "run_agreement" not in result.columns

    # source_runs
    def test_source_runs_present_by_default(self):
        result = self._merge()
        assert "source_runs" in result.columns

    def test_source_runs_absent_when_disabled(self):
        result = self._merge(add_source_runs=False)
        assert "source_runs" not in result.columns

    def test_source_runs_contains_both_run_indices(self):
        result = self._merge(add_source_runs=True)
        source_runs_str = str(result["source_runs"].iloc[0])
        assert "0" in source_runs_str
        assert "1" in source_runs_str

    # canopyrs_object_id
    def test_object_id_column_present(self):
        result = self._merge()
        assert Col.OBJECT_ID in result.columns

    def test_object_ids_are_unique_multi_output(self):
        gdf1 = _make_gdf([(0, 0, 10, 10), (100, 0, 110, 10)], scores=[0.8, 0.7])
        gdf2 = _make_gdf([(0, 0, 10, 10), (100, 0, 110, 10)], scores=[0.9, 0.6])
        result = _merge_gdfs([gdf1, gdf2], _cfg(iou_threshold=0.3))
        assert result[Col.OBJECT_ID].nunique() == len(result)

    # tile_path
    def test_tile_path_column_present(self):
        result = self._merge()
        assert Col.TILE_PATH in result.columns

    def test_tile_path_nonempty(self):
        result = self._merge()
        assert result[Col.TILE_PATH].iloc[0] is not None


# =============================================================================
# _merge_gdfs — CRS handling
# =============================================================================

class TestCrsPreservation:
    def test_projected_crs_preserved(self):
        gdf1 = _make_gdf([(0, 0, 10, 10)], scores=[0.9], crs="EPSG:32632")
        gdf2 = _make_gdf([(0, 0, 10, 10)], scores=[0.8], crs="EPSG:32632")
        result = _merge_gdfs([gdf1, gdf2], _cfg(iou_threshold=0.3))
        assert result.crs is not None
        assert result.crs.to_epsg() == 32632

    def test_geographic_crs_preserved(self):
        # Small lat/lon box near Zurich; reprojection to 3857 and back is transparent.
        gdf1 = _make_gdf([(8.0, 47.0, 8.001, 47.001)], scores=[0.9], crs="EPSG:4326")
        gdf2 = _make_gdf([(8.0, 47.0, 8.001, 47.001)], scores=[0.8], crs="EPSG:4326")
        result = _merge_gdfs([gdf1, gdf2], _cfg(iou_threshold=0.3))
        assert result.crs is not None
        assert result.crs.to_epsg() == 4326

    def test_geographic_non_overlap_crs_preserved(self):
        gdf1 = _make_gdf([(8.0, 47.0, 8.001, 47.001)], scores=[0.9], crs="EPSG:4326")
        gdf2 = _make_gdf([(9.0, 48.0, 9.001, 48.001)], scores=[0.8], crs="EPSG:4326")
        result = _merge_gdfs([gdf1, gdf2], _cfg(iou_threshold=0.3, min_observations=1))
        if len(result) > 0:
            assert result.crs.to_epsg() == 4326


# =============================================================================
# MultiRunMergerComponent.__call__ (component-level)
# =============================================================================

class TestMultiRunMergerComponentCall:
    def test_single_gdf_warns_and_passthrough(self):
        gdf = _make_gdf([(0, 0, 10, 10)], scores=[0.9])
        comp = _component(iou_threshold=0.3)
        ds = _state([gdf])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = comp(ds)

        assert any("only one GDF" in str(warning.message) for warning in w)
        assert len(result.gdf) == 1

    def test_two_gdfs_merged(self):
        gdf1 = _make_gdf([(0, 0, 10, 10)], scores=[0.9])
        gdf2 = _make_gdf([(0, 0, 10, 10)], scores=[0.8])
        comp = _component(iou_threshold=0.3)
        result = comp(_state([gdf1, gdf2]))
        assert len(result.gdf) == 1

    def test_result_sets_correct_gpkg_suffix(self):
        gdf1 = _make_gdf([(0, 0, 10, 10)], scores=[0.9])
        gdf2 = _make_gdf([(0, 0, 10, 10)], scores=[0.8])
        comp = _component(iou_threshold=0.3)
        result = comp(_state([gdf1, gdf2]))
        assert result.gpkg_name_suffix == "multirun_merged"

    def test_result_objects_are_new(self):
        gdf1 = _make_gdf([(0, 0, 10, 10)], scores=[0.9])
        gdf2 = _make_gdf([(0, 0, 10, 10)], scores=[0.8])
        comp = _component(iou_threshold=0.3)
        result = comp(_state([gdf1, gdf2]))
        assert result.objects_are_new is True

    def test_missing_multirun_gdfs_raises(self):
        """validate_requirements should raise when multirun_gdfs is None."""
        from canopyrs.engine.components.base import ComponentValidationError
        comp = _component(iou_threshold=0.3)
        ds = DataState(parent_output_path="/tmp", product_name="test")
        ds.multirun_gdfs = None   # not set → validator should raise
        with pytest.raises(ComponentValidationError):
            comp(ds)

    def test_non_overlapping_two_trees_each_run_min_obs_1(self):
        """Four separate trees across two runs all survive with min_obs=1."""
        gdf1 = _make_gdf([(0, 0, 10, 10), (50, 0, 60, 10)], scores=[0.9, 0.8])
        gdf2 = _make_gdf([(200, 0, 210, 10), (300, 0, 310, 10)], scores=[0.7, 0.6])
        comp = _component(iou_threshold=0.3, min_observations=1)
        result = comp(_state([gdf1, gdf2]))
        assert len(result.gdf) == 4

    def test_three_gdfs_all_overlap_single_output(self):
        b = (0, 0, 10, 10)
        gdfs = [_make_gdf([b], scores=[0.9 - i * 0.1]) for i in range(3)]
        comp = _component(iou_threshold=0.3)
        result = comp(_state(gdfs))
        assert len(result.gdf) == 1

    def test_output_observation_count_from_call(self):
        b = (0, 0, 10, 10)
        gdfs = [_make_gdf([b], scores=[0.9 - i * 0.1]) for i in range(3)]
        comp = _component(iou_threshold=0.3, add_observation_count=True)
        result = comp(_state(gdfs))
        assert int(result.gdf["observation_count"].iloc[0]) == 3
