import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.geometry import Polygon, box
from pathlib import Path
import sys
import importlib.util
from pathlib import Path

# Ensure repository root is on path to import our script
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

spec = importlib.util.spec_from_file_location("analyze_crown_vi", str(_repo_root / "tools" / "analyze_crown_vi.py"))
analyze_crown_vi = importlib.util.module_from_spec(spec)
sys.modules["analyze_crown_vi"] = analyze_crown_vi
spec.loader.exec_module(analyze_crown_vi)
from analyze_crown_vi import load_and_clip_ortho, extract_crown_stats

@pytest.fixture
def dummy_tif(tmp_path):
    """Creates a local dummy 5-band GeoTIFF."""
    tif_path = tmp_path / "dummy_ortho.tif"
    
    # 5 bands, 100x100 pixels
    data = np.ones((5, 100, 100), dtype=np.int16) * 16384 # ~0.5 reflectance
    data[0, :, :] = 10000 # Dummy difference between bands
    transform = from_origin(100.0, 100.0, 1.0, 1.0)
    
    with rasterio.open(
        tif_path, 'w', driver='GTiff',
        height=100, width=100,
        count=5, dtype=str(data.dtype),
        crs='+proj=utm +zone=32 +datum=WGS84',
        transform=transform,
        nodata=-32767
    ) as dst:
        dst.write(data)
        
    return tif_path

@pytest.fixture
def dummy_aoi():
    """Returns a GeoDataFrame acting as an AOI (clipping to top-left 50x50px)."""
    # Grid goes from x=100 to 200, y=0 to 100
    aoi_geom = box(100.0, 50.0, 150.0, 100.0) 
    return gpd.GeoDataFrame({'geometry': [aoi_geom]}, crs='+proj=utm +zone=32 +datum=WGS84')

@pytest.fixture
def dummy_crowns():
    """Returns a GeoDataFrame acting as tree crowns."""
    # Place a crown inside the AOI
    crown = Polygon([(110, 80), (120, 80), (120, 90), (110, 90)])
    return gpd.GeoDataFrame({'crown_id': [1], 'geometry': [crown]}, crs='+proj=utm +zone=32 +datum=WGS84')


def test_load_and_clip_ortho(dummy_tif, dummy_aoi):
    # Test loading and clipping
    data, win_tf, crs, nodata_mask = load_and_clip_ortho(
        tif_path=str(dummy_tif),
        aoi_gdf=dummy_aoi,
        max_pix=None
    )
    
    # Extent logic: 50x50 pixels out of 100x100
    assert data.shape[1] == 50
    assert data.shape[2] == 50
    
    # Values should be normalized [0, 1]
    assert np.all(data >= 0) and np.all(data <= 1)


def test_downsampling_ortho(dummy_tif):
    # Test downsampling to 20 pixels
    data, win_tf, crs, nodata_mask = load_and_clip_ortho(
        tif_path=str(dummy_tif),
        aoi_gdf=None,
        max_pix=20
    )
    
    assert data.shape[1] == 20
    assert data.shape[2] == 20


def test_extract_crown_stats(dummy_tif, dummy_crowns):
    # Load all data
    data, win_tf, crs, nodata_mask = load_and_clip_ortho(str(dummy_tif))
    
    # Mock some VI maps
    vi_maps = {
        'ndvi': np.ones((100, 100)) * 0.8,
        'ndre': np.ones((100, 100)) * 0.3
    }
    
    stats = extract_crown_stats(vi_maps, win_tf, crs, dummy_crowns)
    
    assert len(stats) == 1
    assert stats[0]['crown_id'] == 0  # index 0 of dataframe
    assert np.isclose(stats[0]['ndvi_median'], 0.8)
    assert np.isclose(stats[0]['ndre_median'], 0.3)
