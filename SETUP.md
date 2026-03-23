# CanopyRS — Environment Setup Guide (WSL / Linux)

This guide documents the full setup procedure for CanopyRS in a **conda environment on WSL (Ubuntu)**, including known issues and their fixes.

---

## Requirements

| Requirement | Version |
|---|---|
| OS | WSL 2 — Ubuntu 22.04 recommended |
| Python | 3.10 |
| CUDA Driver | Any (13.x tested) |
| CUDA Toolkit | **12.6** (must match PyTorch wheels) |
| conda | Miniconda or Anaconda |

---

## Step 1 — Clone the repository

Clone into the **WSL native filesystem** (`~/projects/`), not `/mnt/c/`. This avoids slow I/O and file permission issues.

```bash
mkdir -p ~/projects
git clone https://github.com/hugobaudchon/CanopyRS.git ~/projects/CanopyRS
cd ~/projects/CanopyRS
git submodule update --init --recursive
```

The `detrex` submodule (detectron2 fork) must be initialised — the folder is empty by default.

---

## Step 2 — Install system build dependencies

`detectron2` compiles C++ extensions and requires native build tools:

```bash
sudo apt-get update
sudo apt-get install -y build-essential ninja-build python3-dev libglib2.0-dev git
```

---

## Step 3 — Create the conda environment

```bash
conda create -n canopyrs_env -c conda-forge python=3.10 mamba
conda activate canopyrs_env
```

---

## Step 4 — Install GDAL via mamba

GDAL must be installed through conda/mamba before pip — the pip wheel requires a matching system binary that mamba handles automatically.

```bash
mamba install gdal=3.6.2 -c conda-forge
```

---

## Step 5 — Install PyTorch (CUDA 12.6)

```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
```

> **Important:** PyTorch wheels are compiled for **CUDA 12.6**. Even if your system has a newer CUDA driver (e.g. 13.x), the toolkit used to build C++ extensions must also be 12.6. See [Troubleshooting](#troubleshooting) below.

---

## Step 6 — Install CanopyRS

```bash
pip install ninja          # speeds up C++ compilation
pip install -e .
```

---

## Step 7 — Install detectron2 and detrex

These must be built from source with `--no-build-isolation`. Use `MAX_JOBS` to prevent WSL from running out of memory during parallel C++ compilation.

```bash
MAX_JOBS=4 pip install --no-build-isolation -e ./detrex/detectron2
MAX_JOBS=4 pip install --no-build-isolation -e ./detrex
```

> **Note:** You may see `sam2 requires iopath>=0.1.10 but iopath 0.1.9 is installed`. This is a known conflict between detectron2 and SAM2 — it can be safely ignored.

---

## Step 8 — Verify the installation

```bash
bash /path/to/CanopyRS/check_install.sh
```

Expected output:

```
=== Python ===
Python 3.10.x
Executable: /home/<user>/anaconda3/envs/canopyrs_env/bin/python

=== CUDA / nvcc ===
Cuda compilation tools, release 12.6, V12.6.85
CUDA_HOME=/usr/local/cuda-12.6

=== Key packages ===
  [OK] canopyrs        OK
  [OK] torch           2.7.1+cu126  cuda_avail=True  torch.version.cuda=12.6
  [OK] torchvision     0.22.1+cu126
  [OK] detectron2      0.6  OK
  [OK] detrex          OK
  [OK] GDAL            3.6.2
  ...
  All packages imported successfully.
```

---

## Step 9 — SAM 3 (optional, gated model)

SAM 3 is hosted by Meta on HuggingFace and requires access approval before use.

**a) Request access**

1. Go to [https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)
2. Click **"Request access"** and accept Meta's licence terms (approval is usually instant)

**b) Run the SAM 3 setup script**

```bash
bash /path/to/CanopyRS/setup_sam3_hf.sh
```

The script will:
- Prompt for your HuggingFace token (input is hidden — do not paste tokens into chat or logs)
- Verify your account has access to `facebook/sam3`
- Pre-download the model weights (~5–10 GB) to `~/.cache/huggingface/hub`

Get or create a token at: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

## Daily Usage

```bash
conda activate canopyrs_env
cd ~/projects/CanopyRS
```

**Detection (quick test with included sample raster):**

```bash
python infer.py -c preset_det_single_S_fasterrcnn_r50 \
    -i assets/20240130_zf2tower_m3m_rgb_test_crop.tif \
    -o /tmp/canopyrs_test
```

**Segmentation with SAM 3:**

```bash
python infer.py -c preset_seg_multi_NQOS_selvamask_SAM3_FT_quality \
    -i assets/20240130_zf2tower_m3m_rgb_test_crop.tif \
    -o /tmp/canopyrs_sam3_test
```

See the [full documentation](https://hugobaudchon.github.io/CanopyRS/) for all available presets and pipeline options.

---

## Helper Scripts

| Script | Purpose |
|---|---|
| `setup_canopyrs_wsl.sh` | Full automated setup from scratch (clone → conda env → all deps) |
| `fix_detectron2_build.sh` | Fix missing build tools if detectron2/detrex fail to compile |
| `fix_cuda_mismatch.sh` | Fix CUDA version mismatch (installs CUDA 12.6 toolkit alongside newer version) |
| `check_install.sh` | Verify all packages are correctly installed |
| `setup_sam3_hf.sh` | Log in to HuggingFace and pre-download SAM 3 weights |

---

## Troubleshooting

### detectron2/detrex fail to build — missing build tools

```
ERROR: Failed building editable for detectron2
```

**Fix:** Install the C++ toolchain and retry:

```bash
sudo apt-get install -y build-essential ninja-build python3-dev
```

Or run: `bash fix_detectron2_build.sh`

---

### CUDA version mismatch

```
RuntimeError: The detected CUDA version (13.x) mismatches the version that was used
to compile PyTorch (12.6).
```

**Cause:** Your system `nvcc` is a newer version than the CUDA 12.6 that PyTorch was compiled against. detectron2's build system checks these match.

**Fix:** Install the CUDA 12.6 toolkit alongside your existing toolkit (no driver changes required — NVIDIA drivers are backward-compatible), then build with `CUDA_HOME` pointing to 12.6:

```bash
bash fix_cuda_mismatch.sh
```

This also installs a conda env activation hook so `nvcc 12.6` is automatically used inside `canopyrs_env` without affecting your global system.

---

### iopath version conflict warning

```
sam2 0.4.1 requires iopath>=0.1.10, but you have iopath 0.1.9
```

**Safe to ignore.** This is a known conflict between detectron2 and SAM2 that does not affect functionality.

---

### HuggingFace login — `huggingface-cli: command not found`

The `huggingface-cli` entry point may not be installed even when `huggingface_hub` is. Use `setup_sam3_hf.sh` which calls the Python API directly instead of the CLI.
