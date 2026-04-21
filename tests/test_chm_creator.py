"""
Unit tests for canopyrs/tools/chm_creator.py

All network I/O is mocked so the tests run entirely offline.  Raster I/O
uses rasterio and temporary files created in pytest's ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _RASTERIO_AVAILABLE, reason="rasterio not installed"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_geotiff(
    path: Path,
    data: np.ndarray,
    crs_epsg: int = 2056,
    west: float = 2600000.0,
    south: float = 1200000.0,
    east: float = 2600100.0,
    north: float = 1200100.0,
    nodata: float = -9999.0,
) -> Path:
    """Write a single-band float32 GeoTIFF and return its path."""
    h, w = data.shape[-2], data.shape[-1]
    transform = from_bounds(west, south, east, north, w, h)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(crs_epsg),
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data.astype(np.float32).reshape(1, h, w))
    return path


# ---------------------------------------------------------------------------
# _extract_asset_url
# ---------------------------------------------------------------------------

from canopyrs.tools.chm_creator import _extract_asset_url, _parse_resampling


class TestExtractAssetUrl:
    def _feature(self, assets: dict) -> dict:
        return {"id": "test_item", "assets": assets}

    def test_key_matches_resolution_and_tif(self):
        feature = self._feature({
            "2056_5728_0.5": {"href": "https://example.com/tile_0.5m.tif"},
        })
        url = _extract_asset_url(feature, "0.5")
        assert url == "https://example.com/tile_0.5m.tif"

    def test_resolution_in_href_fallback(self):
        feature = self._feature({
            "some_asset": {"href": "https://example.com/tile_res0.5.tif"},
        })
        url = _extract_asset_url(feature, "0.5")
        assert url == "https://example.com/tile_res0.5.tif"

    def test_last_resort_any_tif(self):
        feature = self._feature({
            "dtm": {"href": "https://example.com/tile.tif"},
        })
        url = _extract_asset_url(feature, "0.5")
        assert url == "https://example.com/tile.tif"

    def test_no_tif_returns_none(self):
        feature = self._feature({
            "thumbnail": {"href": "https://example.com/thumb.png"},
        })
        assert _extract_asset_url(feature, "0.5") is None

    def test_empty_assets_returns_none(self):
        feature = self._feature({})
        assert _extract_asset_url(feature, "0.5") is None

    def test_prefers_resolution_match_over_fallback(self):
        feature = self._feature({
            "2056_5728_2": {"href": "https://example.com/tile_2m.tif"},
            "2056_5728_0.5": {"href": "https://example.com/tile_0.5m.tif"},
        })
        url = _extract_asset_url(feature, "0.5")
        assert url == "https://example.com/tile_0.5m.tif"


# ---------------------------------------------------------------------------
# _parse_resampling
# ---------------------------------------------------------------------------

class TestParseResampling:
    def test_bicubic_maps_to_cubic(self):
        from rasterio.enums import Resampling
        assert _parse_resampling("bicubic") == Resampling.cubic

    def test_bilinear(self):
        from rasterio.enums import Resampling
        assert _parse_resampling("bilinear") == Resampling.bilinear

    def test_nearest(self):
        from rasterio.enums import Resampling
        assert _parse_resampling("nearest") == Resampling.nearest

    def test_lanczos(self):
        from rasterio.enums import Resampling
        assert _parse_resampling("lanczos") == Resampling.lanczos

    def test_case_insensitive(self):
        from rasterio.enums import Resampling
        assert _parse_resampling("Bicubic") == Resampling.cubic

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown resampling"):
            _parse_resampling("magic_resample")


# ---------------------------------------------------------------------------
# ChmCreator._compute_chm (static, no I/O)
# ---------------------------------------------------------------------------

from canopyrs.tools.chm_creator import ChmCreator


class TestComputeChm:
    def test_basic_subtraction(self):
        dsm = np.array([[[10.0, 8.0], [6.0, 5.0]]], dtype=np.float32)
        dtm = np.array([[[3.0, 3.0], [3.0, 3.0]]], dtype=np.float32)
        chm = ChmCreator._compute_chm(dsm, dtm)
        expected = np.array([[[7.0, 5.0], [3.0, 2.0]]], dtype=np.float32)
        np.testing.assert_allclose(chm, expected)

    def test_negative_values_clamped_to_zero(self):
        dsm = np.array([[[2.0, 1.0]]], dtype=np.float32)
        dtm = np.array([[[5.0, 5.0]]], dtype=np.float32)
        chm = ChmCreator._compute_chm(dsm, dtm)
        assert (chm == 0.0).all(), "Negative CHM values must be clamped to 0"

    def test_min_height_threshold(self):
        dsm = np.array([[[10.0, 1.5, 5.0]]], dtype=np.float32)
        dtm = np.array([[[0.0,  0.0, 0.0]]], dtype=np.float32)
        chm = ChmCreator._compute_chm(dsm, dtm, min_tree_height=2.0)
        # 10 m → kept; 1.5 m → below threshold → 0; 5 m → kept
        np.testing.assert_allclose(chm[0, 0], [10.0, 0.0, 5.0])

    def test_nan_dsm_propagates_as_zero(self):
        dsm = np.array([[[np.nan, 5.0]]], dtype=np.float32)
        dtm = np.array([[[1.0,   1.0]]], dtype=np.float32)
        chm = ChmCreator._compute_chm(dsm, dtm)
        # NaN diff → not finite → 0
        assert chm[0, 0, 0] == 0.0
        assert chm[0, 0, 1] == 4.0

    def test_output_dtype_float32(self):
        dsm = np.ones((1, 4, 4), dtype=np.float32) * 5.0
        dtm = np.ones((1, 4, 4), dtype=np.float32) * 2.0
        chm = ChmCreator._compute_chm(dsm, dtm)
        assert chm.dtype == np.float32

    def test_output_shape_preserved(self):
        dsm = np.ones((1, 16, 32), dtype=np.float32) * 10.0
        dtm = np.ones((1, 16, 32), dtype=np.float32) * 4.0
        chm = ChmCreator._compute_chm(dsm, dtm)
        assert chm.shape == (1, 16, 32)


# ---------------------------------------------------------------------------
# ChmCreator – constructor validation
# ---------------------------------------------------------------------------

class TestChmCreatorInit:
    def test_missing_dsm_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ChmCreator(
                dsm_path=tmp_path / "nonexistent.tif",
                output_path=tmp_path / "chm.tif",
            )

    def test_invalid_resolution_raises(self, tmp_path):
        dsm = tmp_path / "dsm.tif"
        _make_geotiff(dsm, np.ones((4, 4), dtype=np.float32) * 500.0)
        with pytest.raises(ValueError, match="dtm_resolution"):
            ChmCreator(dsm_path=dsm, output_path=tmp_path / "chm.tif", dtm_resolution="3")

    def test_valid_init(self, tmp_path):
        dsm = tmp_path / "dsm.tif"
        _make_geotiff(dsm, np.ones((4, 4), dtype=np.float32) * 500.0)
        creator = ChmCreator(dsm_path=dsm, output_path=tmp_path / "chm.tif")
        assert creator.dsm_path == dsm
        assert creator.dtm_resolution == "0.5"
        assert creator.datum_correction_m == 0.0
        assert creator.min_tree_height == 2.0


# ---------------------------------------------------------------------------
# ChmCreator._read_dsm_info
# ---------------------------------------------------------------------------

class TestReadDsmInfo:
    def test_bounds_in_wgs84(self, tmp_path):
        dsm = tmp_path / "dsm.tif"
        # Small 4×4 tile in EPSG:2056 around Zurich
        _make_geotiff(
            dsm,
            np.ones((4, 4), dtype=np.float32) * 500.0,
            crs_epsg=2056,
            west=2683000.0,
            south=1247000.0,
            east=2683100.0,
            north=1247100.0,
        )
        creator = ChmCreator(dsm_path=dsm, output_path=tmp_path / "chm.tif")
        bbox_wgs84, profile = creator._read_dsm_info()
        west, south, east, north = bbox_wgs84
        # Rough check: Zurich area is roughly 8–9°E, 47–48°N
        assert 6.0 < west < 11.0, f"west={west} out of expected range for CH"
        assert 45.0 < south < 48.0, f"south={south} out of expected range for CH"
        assert west < east
        assert south < north

    def test_profile_returned(self, tmp_path):
        dsm = tmp_path / "dsm.tif"
        _make_geotiff(dsm, np.ones((4, 4), dtype=np.float32) * 500.0)
        creator = ChmCreator(dsm_path=dsm, output_path=tmp_path / "chm.tif")
        _, profile = creator._read_dsm_info()
        assert "width" in profile
        assert "height" in profile
        assert "transform" in profile

    def test_no_crs_raises(self, tmp_path):
        dsm = tmp_path / "dsm_nocrs.tif"
        # Write raster without CRS
        with rasterio.open(
            dsm, "w", driver="GTiff", height=4, width=4,
            count=1, dtype="float32"
        ) as dst:
            dst.write(np.ones((1, 4, 4), dtype=np.float32))
        creator = ChmCreator(dsm_path=dsm, output_path=tmp_path / "chm.tif")
        with pytest.raises(ValueError, match="no CRS"):
            creator._read_dsm_info()


# ---------------------------------------------------------------------------
# ChmCreator._merge_tiles
# ---------------------------------------------------------------------------

class TestMergeTiles:
    def test_single_tile_passthrough(self, tmp_path):
        tile = tmp_path / "tile.tif"
        data = np.arange(16, dtype=np.float32).reshape(4, 4) + 100.0
        _make_geotiff(tile, data)
        creator = ChmCreator(dsm_path=tile, output_path=tmp_path / "chm.tif")
        merged, meta = creator._merge_tiles([tile])
        assert merged.shape == (1, 4, 4)
        assert merged.dtype == np.float32

    def test_two_adjacent_tiles_merged(self, tmp_path):
        # Two horizontally adjacent 4×4 tiles
        tile1 = tmp_path / "tile1.tif"
        tile2 = tmp_path / "tile2.tif"
        _make_geotiff(
            tile1, np.ones((4, 4), dtype=np.float32) * 10.0,
            west=2600000.0, east=2600040.0,
            south=1200000.0, north=1200040.0,
        )
        _make_geotiff(
            tile2, np.ones((4, 4), dtype=np.float32) * 20.0,
            west=2600040.0, east=2600080.0,
            south=1200000.0, north=1200040.0,
        )
        creator = ChmCreator(dsm_path=tile1, output_path=tmp_path / "chm.tif")
        merged, meta = creator._merge_tiles([tile1, tile2])
        # Merged extent should be wider
        assert merged.shape[2] == 8, f"Expected width 8, got {merged.shape[2]}"
        assert merged.shape[1] == 4


# ---------------------------------------------------------------------------
# ChmCreator._resample_to_dsm
# ---------------------------------------------------------------------------

class TestResampleToDsm:
    def test_output_matches_dsm_shape(self, tmp_path):
        dsm_path = tmp_path / "dsm.tif"
        dtm_src = tmp_path / "dtm_src.tif"
        _make_geotiff(
            dsm_path, np.ones((8, 8), dtype=np.float32) * 500.0,
            west=2600000.0, east=2600080.0,
            south=1200000.0, north=1200080.0,
        )
        _make_geotiff(
            dtm_src, np.ones((4, 4), dtype=np.float32) * 450.0,
            west=2600000.0, east=2600080.0,
            south=1200000.0, north=1200080.0,
        )
        creator = ChmCreator(dsm_path=dsm_path, output_path=tmp_path / "chm.tif")
        _, dsm_profile = creator._read_dsm_info()
        dtm_data, dtm_meta = creator._merge_tiles([dtm_src])
        resampled = creator._resample_to_dsm(dtm_data, dtm_meta, dsm_profile)
        assert resampled.shape == (1, 8, 8), (
            f"Resampled DTM shape {resampled.shape} does not match DSM (1,8,8)"
        )
        assert resampled.dtype == np.float32


# ---------------------------------------------------------------------------
# ChmCreator.run – full end-to-end with mocked download
# ---------------------------------------------------------------------------

class TestChmCreatorRun:
    """End-to-end tests with the Swisstopo network call replaced by a mock."""

    def _make_dtm_tile(self, tmp_path: Path, filename: str, value: float = 450.0) -> Path:
        tile = tmp_path / filename
        _make_geotiff(
            tile, np.ones((8, 8), dtype=np.float32) * value,
            west=2600000.0, east=2600080.0,
            south=1200000.0, north=1200080.0,
        )
        return tile

    def test_run_produces_output_file(self, tmp_path):
        dsm = tmp_path / "dsm.tif"
        _make_geotiff(
            dsm, np.ones((8, 8), dtype=np.float32) * 500.0,
            west=2600000.0, east=2600080.0,
            south=1200000.0, north=1200080.0,
        )
        dtm_tile = self._make_dtm_tile(tmp_path / "tiles", "dtm_tile.tif")

        chm_out = tmp_path / "chm.tif"
        creator = ChmCreator(
            dsm_path=dsm,
            output_path=chm_out,
            tmp_dir=tmp_path / "tiles",
            keep_tmp=True,
        )

        # Patch network calls
        stac_response = MagicMock()
        stac_response.raise_for_status = MagicMock()
        stac_response.json.return_value = {
            "features": [
                {
                    "id": "swissalti3d_tile_1",
                    "assets": {
                        "2056_5728_0.5": {"href": dtm_tile.as_uri()},
                    },
                }
            ]
        }

        tile_response = MagicMock()
        tile_response.raise_for_status = MagicMock()
        with open(dtm_tile, "rb") as fh:
            tile_bytes = fh.read()
        tile_response.iter_content = lambda chunk_size: [tile_bytes]

        def _fake_get(url, **kwargs):
            if "stac" in url:
                return stac_response
            return tile_response

        with patch("canopyrs.tools.chm_creator.requests.get", side_effect=_fake_get):
            # Also patch the file-download path to use the already-downloaded tile
            with patch.object(creator, "_download_tiles", return_value=[dtm_tile]):
                result = creator.run()

        assert result == chm_out
        assert chm_out.exists(), "CHM output file was not created"

    def test_run_chm_values_correct(self, tmp_path):
        """CHM = DSM(500) − DTM(450) = 50 m everywhere."""
        dsm = tmp_path / "dsm.tif"
        _make_geotiff(
            dsm, np.ones((8, 8), dtype=np.float32) * 500.0,
            west=2600000.0, east=2600080.0,
            south=1200000.0, north=1200080.0,
        )
        dtm_tile = tmp_path / "dtm_tile.tif"
        _make_geotiff(
            dtm_tile, np.ones((8, 8), dtype=np.float32) * 450.0,
            west=2600000.0, east=2600080.0,
            south=1200000.0, north=1200080.0,
        )

        chm_out = tmp_path / "chm.tif"
        creator = ChmCreator(
            dsm_path=dsm,
            output_path=chm_out,
            min_tree_height=0.0,   # no threshold — keep all positive values
        )

        with patch.object(creator, "_download_tiles", return_value=[dtm_tile]):
            creator.run()

        with rasterio.open(chm_out) as src:
            chm_data = src.read(1)

        # All valid pixels should be ~50 m
        valid = chm_data[chm_data > 0]
        assert len(valid) > 0, "No valid CHM pixels found"
        np.testing.assert_allclose(valid, 50.0, atol=0.5)

    def test_datum_correction_applied(self, tmp_path):
        """datum_correction_m=+10 should reduce CHM by 10 m."""
        dsm = tmp_path / "dsm.tif"
        _make_geotiff(
            dsm, np.ones((8, 8), dtype=np.float32) * 500.0,
            west=2600000.0, east=2600080.0,
            south=1200000.0, north=1200080.0,
        )
        dtm_tile = tmp_path / "dtm_tile.tif"
        _make_geotiff(
            dtm_tile, np.ones((8, 8), dtype=np.float32) * 440.0,
            west=2600000.0, east=2600080.0,
            south=1200000.0, north=1200080.0,
        )

        chm_out = tmp_path / "chm.tif"
        # DTM is 440 + 10 correction = 450 → CHM = 500 - 450 = 50
        creator = ChmCreator(
            dsm_path=dsm,
            output_path=chm_out,
            datum_correction_m=10.0,
            min_tree_height=0.0,
        )

        with patch.object(creator, "_download_tiles", return_value=[dtm_tile]):
            creator.run()

        with rasterio.open(chm_out) as src:
            chm_data = src.read(1)

        valid = chm_data[chm_data > 0]
        assert len(valid) > 0
        np.testing.assert_allclose(valid, 50.0, atol=0.5)

    def test_min_tree_height_filters_low_vegetation(self, tmp_path):
        """Pixels with CHM < min_tree_height must be set to 0."""
        dsm_arr = np.array([[[455.0, 452.0, 460.0]]], dtype=np.float32)
        dsm = tmp_path / "dsm.tif"
        with rasterio.open(
            dsm, "w", driver="GTiff", height=1, width=3, count=1,
            dtype="float32",
            crs=CRS.from_epsg(2056),
            transform=from_bounds(2600000, 1200000, 2600030, 1200010, 3, 1),
        ) as dst:
            dst.write(dsm_arr)

        dtm_tile = tmp_path / "dtm.tif"
        _make_geotiff(
            dtm_tile,
            np.ones((1, 3), dtype=np.float32) * 450.0,
            west=2600000.0, east=2600030.0,
            south=1200000.0, north=1200010.0,
        )

        chm_out = tmp_path / "chm.tif"
        creator = ChmCreator(
            dsm_path=dsm,
            output_path=chm_out,
            min_tree_height=3.0,  # only keep CHM >= 3 m
        )

        with patch.object(creator, "_download_tiles", return_value=[dtm_tile]):
            creator.run()

        with rasterio.open(chm_out) as src:
            chm_data = src.read(1).flatten()

        # CHM values: [5, 2, 10].  min_tree_height=3 → 2 filtered → [5, 0, 10]
        assert chm_data[1] == 0.0, f"Low-vegetation pixel not filtered: {chm_data[1]}"
        assert chm_data[0] > 0.0, f"Valid tree pixel wrongly zeroed: {chm_data[0]}"
        assert chm_data[2] > 0.0, f"Valid tree pixel wrongly zeroed: {chm_data[2]}"

    def test_no_tiles_found_raises(self, tmp_path):
        dsm = tmp_path / "dsm.tif"
        _make_geotiff(dsm, np.ones((4, 4), dtype=np.float32) * 500.0)
        creator = ChmCreator(dsm_path=dsm, output_path=tmp_path / "chm.tif")
        with patch.object(creator, "_download_tiles", return_value=[]):
            with pytest.raises(RuntimeError, match="No swissALTI3D tiles"):
                creator.run()
