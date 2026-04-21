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

        np.testing.assert_allclose(sel_s, [0.8, 0.9, 0.9], rtol=1e-6)

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


# ---------------------------------------------------------------------------
# ms_camera_profiles
# ---------------------------------------------------------------------------

from canopyrs.engine.ms_camera_profiles import (
    get_profile,
    apply_profile_to_config_data,
    list_cameras,
    list_supported_vis,
    MS_CAMERA_PROFILES,
)


class TestMsCameraProfiles:
    # -- get_profile ----------------------------------------------------------

    def test_get_profile_canonical_name(self):
        p = get_profile("micasense_altum")
        assert p["n_bands"] == 5
        assert p["ms_nir_band_idx"] == 4

    def test_get_profile_short_alias(self):
        p = get_profile("altum")
        assert p["n_bands"] == 5

    def test_get_profile_mx_dual_alias(self):
        p = get_profile("mx_dual")
        assert p["n_bands"] == 10
        assert p["ms_nir_band_idx"] == 4  # Camera-1 NIR

    def test_get_profile_case_insensitive(self):
        p = get_profile("MX_Dual")
        assert p["n_bands"] == 10

    def test_get_profile_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown ms_camera"):
            get_profile("my_imaginary_camera")

    # -- mx_dual band layout --------------------------------------------------

    def test_mx_dual_has_10_bands(self):
        p = get_profile("mx_dual")
        assert p["n_bands"] == 10

    def test_mx_dual_cam2_wavelengths_present(self):
        p = get_profile("mx_dual")
        assert "coastal_blue_cam2" in p["wavelengths_nm"]
        assert "nir_cam2" in p["wavelengths_nm"]

    def test_mx_dual_default_indices_use_cam1(self):
        """Default primary indices must be in 0–4 (Camera-1 range)."""
        p = get_profile("mx_dual")
        assert p["ms_blue_band_idx"] <= 4
        assert p["ms_nir_band_idx"] <= 4

    # -- apply_profile_to_config_data -----------------------------------------

    def test_apply_profile_injects_missing_fields(self):
        data = {"ms_camera": "altum", "ms_index_type": "ndvi"}
        apply_profile_to_config_data(data, "altum")
        assert data["ms_nir_band_idx"] == 4
        assert data["ms_blue_band_idx"] == 0

    def test_apply_profile_respects_explicit_override(self):
        """Explicit ms_nir_band_idx must NOT be overwritten by profile."""
        data = {"ms_camera": "mx_dual", "ms_nir_band_idx": 9}
        apply_profile_to_config_data(data, "mx_dual")
        assert data["ms_nir_band_idx"] == 9  # override preserved
        assert data["ms_blue_band_idx"] == 0   # profile-injected

    def test_apply_profile_mx_dual_with_cam2_override(self):
        data = {"ms_red_edge_band_idx": 8, "ms_nir_band_idx": 9}
        apply_profile_to_config_data(data, "mx_dual")
        # Overrides preserved
        assert data["ms_red_edge_band_idx"] == 8
        assert data["ms_nir_band_idx"] == 9
        # Profile-injected
        assert data["ms_red_band_idx"] == 2

    # -- list helpers ---------------------------------------------------------

    def test_list_cameras_returns_canonical_names(self):
        cameras = list_cameras()
        assert "micasense_altum" in cameras
        assert "micasense_rededge_mx_dual" in cameras

    def test_list_supported_vis(self):
        vis = list_supported_vis("mx_dual")
        assert "ndvi" in vis
        assert "ndre" in vis

    # -- SegmenterConfig integration -----------------------------------------

    def test_segmenter_config_ms_camera_auto_populates_bands(self):
        """ms_camera='mx_dual' must set band indices from the MX Dual profile."""
        from canopyrs.engine.config_parsers.segmenter import SegmenterConfig
        cfg = SegmenterConfig(ms_camera="mx_dual", ms_index_type="ndvi")
        assert cfg.ms_camera == "mx_dual"
        assert cfg.ms_nir_band_idx == 4   # Camera-1 NIR from profile
        assert cfg.ms_blue_band_idx == 0

    def test_segmenter_config_explicit_override_wins(self):
        """Explicit band index must take precedence over camera profile."""
        from canopyrs.engine.config_parsers.segmenter import SegmenterConfig
        cfg = SegmenterConfig(ms_camera="mx_dual", ms_nir_band_idx=9)
        assert cfg.ms_nir_band_idx == 9   # user override
        assert cfg.ms_blue_band_idx == 0   # from profile

    def test_segmenter_config_no_camera_uses_defaults(self):
        """Without ms_camera the original Altum defaults must be preserved."""
        from canopyrs.engine.config_parsers.segmenter import SegmenterConfig
        cfg = SegmenterConfig()
        assert cfg.ms_nir_band_idx == 4
        assert cfg.ms_red_band_idx == 2
        assert cfg.ms_camera is None

    def test_segmenter_config_altum_same_as_defaults(self):
        """Setting ms_camera='altum' must give the same indices as the defaults."""
        from canopyrs.engine.config_parsers.segmenter import SegmenterConfig
        default_cfg = SegmenterConfig()
        altum_cfg = SegmenterConfig(ms_camera="altum")
        for field in [
            "ms_blue_band_idx",
            "ms_green_band_idx",
            "ms_red_band_idx",
            "ms_red_edge_band_idx",
            "ms_nir_band_idx",
        ]:
            assert getattr(altum_cfg, field) == getattr(default_cfg, field), (
                f"Field {field} mismatch: altum={getattr(altum_cfg, field)}, "
                f"default={getattr(default_cfg, field)}"
            )

    def test_calculate_vi_with_mx_dual_10band(self):
        """calculate_vi must work correctly using band indices from MX Dual profile."""
        p = get_profile("mx_dual")
        ms = np.random.default_rng(7).random((10, 64, 64), dtype=np.float32)
        vi = calculate_vi(
            ms,
            index_type="ndvi",
            nir_band_idx=p["ms_nir_band_idx"],
            red_band_idx=p["ms_red_band_idx"],
        )
        assert vi.shape == (64, 64)
        assert vi.dtype == np.float32
        assert vi.min() >= -1.0 - 1e-5
        assert vi.max() <= 1.0 + 1e-5

    def test_calculate_vi_mx_dual_cam2_nir(self):
        """calculate_vi with Camera-2 NIR override (band 9) must succeed."""
        ms = np.random.default_rng(8).random((10, 32, 32), dtype=np.float32)
        vi = calculate_vi(ms, index_type="ndvi", nir_band_idx=9, red_band_idx=7)
        assert vi.shape == (32, 32)
        assert not np.any(np.isnan(vi))

    # -- AltumPT profile tests ------------------------------------------------

    def test_altum_pt_profile_has_7_bands(self):
        """AltumPT Metashape export profile must declare 7 bands (not 6)."""
        p = get_profile("altum_pt")
        assert p["n_bands"] == 7, (
            f"altum_pt should have 7 bands (Blue, Green, Red, RE, NIR, Thermal, Alpha) "
            f"but n_bands={p['n_bands']}"
        )

    def test_altum_pt_canonical_alias_identical(self):
        """'altum_pt' alias must resolve to the same profile as 'micasense_altum_pt'."""
        assert get_profile("altum_pt") is get_profile("micasense_altum_pt")

    def test_altum_pt_vi_bands_in_range_0_to_4(self):
        """All VI band indices for AltumPT must be in bands 0–4 (the MS bands)."""
        p = get_profile("altum_pt")
        for field in [
            "ms_blue_band_idx",
            "ms_green_band_idx",
            "ms_red_band_idx",
            "ms_red_edge_band_idx",
            "ms_nir_band_idx",
        ]:
            assert 0 <= p[field] <= 4, (
                f"{field}={p[field]} is outside the MS reflectance band range 0–4 "
                f"for AltumPT.  Thermal (5) and Alpha (6) must not be used for VIs."
            )

    def test_altum_pt_ndvi_on_7band_tile(self):
        """NDVI must compute correctly on a synthetic 7-band AltumPT tile."""
        p = get_profile("altum_pt")
        ms = np.random.default_rng(9).random((7, 32, 32), dtype=np.float32)
        vi = calculate_vi(
            ms,
            index_type="ndvi",
            nir_band_idx=p["ms_nir_band_idx"],
            red_band_idx=p["ms_red_band_idx"],
        )
        assert vi.shape == (32, 32)
        assert vi.dtype == np.float32
        assert not np.any(np.isnan(vi))
        assert vi.min() >= -1.0 - 1e-5
        assert vi.max() <= 1.0 + 1e-5

    def test_segmenter_config_altum_pt_auto_populates(self):
        """ms_camera='altum_pt' must set band indices 0–4 from the AltumPT profile."""
        from canopyrs.engine.config_parsers.segmenter import SegmenterConfig
        cfg = SegmenterConfig(ms_camera="altum_pt", ms_index_type="ndvi")
        assert cfg.ms_camera == "altum_pt"
        assert cfg.ms_blue_band_idx == 0
        assert cfg.ms_green_band_idx == 1
        assert cfg.ms_red_band_idx == 2
        assert cfg.ms_red_edge_band_idx == 3
        assert cfg.ms_nir_band_idx == 4

    def test_segmenter_config_altum_pt_does_not_break_mx_dual(self):
        """Adding altum_pt must not affect the mx_dual SegmenterConfig defaults."""
        from canopyrs.engine.config_parsers.segmenter import SegmenterConfig
        cfg = SegmenterConfig(ms_camera="mx_dual", ms_index_type="ndvi")
        assert cfg.ms_nir_band_idx == 4   # Camera-1 NIR (unchanged)
        assert cfg.ms_blue_band_idx == 0
        assert cfg.ms_red_band_idx == 2

    def test_segmenter_config_rgb_only_no_ms_camera_unaffected(self):
        """RGB-only config (ms_camera=None, ms_index_type=None) must be unaffected
        by the addition of altum_pt to the profile registry."""
        from canopyrs.engine.config_parsers.segmenter import SegmenterConfig
        cfg = SegmenterConfig()   # pure RGB defaults
        assert cfg.ms_camera is None
        assert cfg.ms_index_type is None
        assert cfg.ms_nir_band_idx == 4
        assert cfg.ms_red_band_idx == 2
        assert cfg.ms_blue_band_idx == 0

    # -- MXdual all VIs -------------------------------------------------------

    def test_mx_dual_all_vis_with_cam1_bands(self):
        """All supported VIs must compute without error on a 10-band MX Dual tile
        using Camera-1 default band indices."""
        p = get_profile("mx_dual")
        ms = np.random.default_rng(10).random((10, 32, 32), dtype=np.float32)
        for vi_type in p["supported_vis"]:
            vi = calculate_vi(
                ms,
                index_type=vi_type,
                nir_band_idx=p["ms_nir_band_idx"],
                red_band_idx=p["ms_red_band_idx"],
                green_band_idx=p["ms_green_band_idx"],
                blue_band_idx=p["ms_blue_band_idx"],
                red_edge_band_idx=p["ms_red_edge_band_idx"],
            )
            assert vi.shape == (32, 32), f"Shape wrong for VI '{vi_type}'"
            assert not np.any(np.isnan(vi)), f"NaN found for VI '{vi_type}'"

