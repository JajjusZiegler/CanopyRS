"""
Unit tests for canopyrs/engine/multispectral.py

These tests cover the pure-Python utilities for vegetation index calculation
and SAM-format conversion.  They do not require SAM or any
geospatial libraries to be installed.
"""

import numpy as np
import pytest

from canopyrs.engine.multispectral import calculate_vi, vi_to_sam_input, select_best_masks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ms_tile(H=64, W=64, n_bands=5, seed=42):
    """Create a synthetic MS tile (C, H, W) with values in [0, 1]."""
    rng = np.random.default_rng(seed)
    return rng.random((n_bands, H, W), dtype=np.float32)


# ---------------------------------------------------------------------------
# calculate_vi
# ---------------------------------------------------------------------------

class TestCalculateVi:
    def test_ndvi_output_shape(self):
        ms = make_ms_tile()
        vi = calculate_vi(ms, index_type="ndvi")
        assert vi.shape == (64, 64)

    def test_ndvi_range(self):
        ms = make_ms_tile()
        vi = calculate_vi(ms, index_type="ndvi")
        assert vi.min() >= -1.0 - 1e-5
        assert vi.max() <= 1.0 + 1e-5

    def test_ndvi_dtype(self):
        ms = make_ms_tile()
        vi = calculate_vi(ms, index_type="ndvi")
        assert vi.dtype == np.float32

    def test_nir_output(self):
        ms = make_ms_tile()
        vi = calculate_vi(ms, index_type="nir", nir_band_idx=4)
        # NIR index just returns the NIR band
        np.testing.assert_array_equal(vi, ms[4].astype(np.float32))

    def test_pri_output_shape(self):
        ms = make_ms_tile()
        vi = calculate_vi(ms, index_type="pri")
        assert vi.shape == (64, 64)

    def test_evi_output_shape(self):
        ms = make_ms_tile()
        vi = calculate_vi(ms, index_type="evi")
        assert vi.shape == (64, 64)

    def test_ndre_output_shape(self):
        ms = make_ms_tile()
        vi = calculate_vi(ms, index_type="ndre", nir_band_idx=4, red_edge_band_idx=3)
        assert vi.shape == (64, 64)

    def test_no_nan_in_output(self):
        ms = make_ms_tile()
        for index_type in ("ndvi", "nir", "pri", "evi", "ndre"):
            vi = calculate_vi(ms, index_type=index_type, red_edge_band_idx=3)
            assert not np.any(np.isnan(vi)), f"NaN found for index_type={index_type}"

    def test_no_inf_in_output(self):
        # Zero denominator should be handled gracefully
        ms = np.zeros((5, 32, 32), dtype=np.float32)
        vi = calculate_vi(ms, index_type="ndvi")
        assert not np.any(np.isinf(vi))
        assert not np.any(np.isnan(vi))

    def test_single_band_input_passthrough(self):
        """A 2-D (H, W) input should be returned as-is (after NaN cleanup)."""
        single = np.random.default_rng(0).random((64, 64), dtype=np.float32)
        vi = calculate_vi(single)
        assert vi.shape == (64, 64)
        assert vi.dtype == np.float32

    def test_unsupported_index_type_raises(self):
        ms = make_ms_tile()
        with pytest.raises(ValueError, match="Unsupported"):
            calculate_vi(ms, index_type="unknown_index")

    def test_band_index_out_of_range_raises(self):
        ms = make_ms_tile(n_bands=3)
        with pytest.raises(IndexError):
            calculate_vi(ms, index_type="ndvi", nir_band_idx=10)

    def test_custom_band_indices(self):
        ms = make_ms_tile(n_bands=6)
        vi = calculate_vi(ms, index_type="ndvi", nir_band_idx=5, red_band_idx=0)
        assert vi.shape == (64, 64)


# ---------------------------------------------------------------------------
# vi_to_sam_input
# ---------------------------------------------------------------------------

class TestViToSamInput:
    def test_output_shape(self):
        vi = np.random.default_rng(0).random((64, 64), dtype=np.float32)
        sam = vi_to_sam_input(vi)
        assert sam.shape == (64, 64, 3)

    def test_output_dtype(self):
        vi = np.random.default_rng(0).random((64, 64), dtype=np.float32)
        sam = vi_to_sam_input(vi)
        assert sam.dtype == np.uint8

    def test_output_range(self):
        vi = np.random.default_rng(0).random((64, 64), dtype=np.float32) * 2 - 1
        sam = vi_to_sam_input(vi)
        assert int(sam.min()) >= 0
        assert int(sam.max()) <= 255

    def test_three_channels_identical(self):
        vi = np.random.default_rng(0).random((64, 64), dtype=np.float32)
        sam = vi_to_sam_input(vi)
        np.testing.assert_array_equal(sam[:, :, 0], sam[:, :, 1])
        np.testing.assert_array_equal(sam[:, :, 0], sam[:, :, 2])

    def test_constant_vi_returns_zeros(self):
        """When VI is constant, normalisation gives a zero array."""
        vi = np.ones((64, 64), dtype=np.float32) * 0.5
        sam = vi_to_sam_input(vi)
        assert sam.sum() == 0

    def test_ndvi_pipeline(self):
        """End-to-end: MS → NDVI → SAM input."""
        ms = make_ms_tile()
        vi = calculate_vi(ms, index_type="ndvi")
        sam = vi_to_sam_input(vi)
        assert sam.shape == (64, 64, 3)
        assert sam.dtype == np.uint8


# ---------------------------------------------------------------------------
# select_best_masks
# ---------------------------------------------------------------------------

class TestSelectBestMasks:
    def _make_masks_and_scores(self, N=4, H=64, W=64, seed=None):
        rng = np.random.default_rng(seed)
        masks = rng.integers(0, 2, size=(N, H, W), dtype=np.uint8)
        scores = rng.random(N).astype(np.float32)
        return masks, scores

    def test_output_shapes(self):
        rgb_m, rgb_s = self._make_masks_and_scores(seed=1)
        ms_m, ms_s = self._make_masks_and_scores(seed=2)
        sel_m, sel_s = select_best_masks(rgb_m, rgb_s, ms_m, ms_s)
        assert sel_m.shape == rgb_m.shape
        assert sel_s.shape == rgb_s.shape

    def test_selects_higher_score(self):
        N, H, W = 3, 8, 8
        rgb_masks = np.zeros((N, H, W), dtype=np.uint8)
        ms_masks = np.ones((N, H, W), dtype=np.uint8)
        rgb_scores = np.array([0.8, 0.5, 0.9], dtype=np.float32)
        ms_scores  = np.array([0.6, 0.9, 0.7], dtype=np.float32)

        sel_m, sel_s = select_best_masks(rgb_masks, rgb_scores, ms_masks, ms_scores)

        # Index 0: RGB wins (0.8 > 0.6)  → all zeros
        np.testing.assert_array_equal(sel_m[0], rgb_masks[0])
        # Index 1: MS wins (0.9 > 0.5)   → all ones
        np.testing.assert_array_equal(sel_m[1], ms_masks[1])
        # Index 2: RGB wins (0.9 > 0.7)  → all zeros
        np.testing.assert_array_equal(sel_m[2], rgb_masks[2])

        np.testing.assert_array_equal(sel_s, [0.8, 0.9, 0.9])

    def test_shape_mismatch_raises(self):
        rgb_m = np.zeros((3, 8, 8), dtype=np.uint8)
        ms_m = np.ones((4, 8, 8), dtype=np.uint8)  # different N
        rgb_s = np.ones(3, dtype=np.float32)
        ms_s = np.ones(4, dtype=np.float32)
        with pytest.raises(AssertionError):
            select_best_masks(rgb_m, rgb_s, ms_m, ms_s)

    def test_all_rgb_wins_when_ms_scores_lower(self):
        N, H, W = 5, 16, 16
        rgb_masks = np.ones((N, H, W), dtype=np.uint8)
        ms_masks = np.zeros((N, H, W), dtype=np.uint8)
        rgb_scores = np.ones(N, dtype=np.float32) * 0.9
        ms_scores = np.ones(N, dtype=np.float32) * 0.5

        sel_m, sel_s = select_best_masks(rgb_masks, rgb_scores, ms_masks, ms_scores)
        np.testing.assert_array_equal(sel_m, rgb_masks)
        np.testing.assert_array_equal(sel_s, rgb_scores)
