#!/usr/bin/env bash
# Run with:  bash check_install.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate canopyrs_env

echo "=== Python ==="
python --version
echo "Executable: $(which python)"

echo ""
echo "=== CUDA / nvcc ==="
nvcc --version 2>/dev/null | grep "release" || echo "nvcc not on PATH"
echo "CUDA_HOME=${CUDA_HOME}"

echo ""
echo "=== Key packages ==="
python << 'PYEOF'
results = {}

def chk(name, fn):
    try:
        results[name] = fn()
    except Exception as e:
        results[name] = f"FAILED: {e}"

import sys
print(f"  sys.executable: {sys.executable}\n")

chk("canopyrs",    lambda: __import__("canopyrs") and "OK")
chk("torch",       lambda: (__import__("torch"), f"{__import__('torch').__version__}  cuda_avail={__import__('torch').cuda.is_available()}  torch.version.cuda={__import__('torch').version.cuda}")[1])
chk("torchvision", lambda: __import__("torchvision").__version__)
chk("detectron2",  lambda: __import__("detectron2").__version__)
chk("detrex",      lambda: __import__("detrex") and "OK")

def gdal_ver():
    try:
        import gdal; return gdal.__version__
    except Exception:
        from osgeo import gdal; return gdal.__version__
chk("GDAL",        gdal_ver)

chk("geopandas",   lambda: __import__("geopandas").__version__)
chk("rasterio",    lambda: __import__("rasterio").__version__)
chk("shapely",     lambda: __import__("shapely").__version__)
chk("numpy",       lambda: __import__("numpy").__version__)
chk("transformers",lambda: __import__("transformers").__version__)
chk("sam2",        lambda: __import__("sam2") and "OK")
chk("wandb",       lambda: __import__("wandb").__version__)
chk("tensorboard", lambda: __import__("tensorboard").__version__)

ok  = {k: v for k, v in results.items() if "FAILED" not in str(v)}
bad = {k: v for k, v in results.items() if "FAILED" in str(v)}

for k, v in ok.items():
    print(f"  [OK] {k:<15} {v}")
if bad:
    print("")
    for k, v in bad.items():
        print(f"  [!!] {k:<15} {v}")
else:
    print("\n  All packages imported successfully.")
PYEOF

echo ""
echo "=== Quick canopyrs smoke test ==="
python -c "import canopyrs; print('CanopyRS import OK')"
