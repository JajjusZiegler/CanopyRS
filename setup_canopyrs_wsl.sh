#!/usr/bin/env bash
# =============================================================================
# CanopyRS — WSL / Linux conda environment setup script
# Usage:  bash setup_canopyrs_wsl.sh
#
# What this script does:
#   1. Clones CanopyRS (+ submodules) into ~/projects/CanopyRS
#   2. Creates a conda env "canopyrs_env" with Python 3.10 + mamba
#   3. Installs GDAL 3.6.2 via mamba (avoids painful binary issues)
#   4. Installs PyTorch 2.7.1 + torchvision 0.22.1 for CUDA 12.6
#   5. Installs CanopyRS (editable) + detrex/detectron2
#   6. Verifies the installation
#
# Requirements (already on your machine):
#   - conda  (from Miniconda / Anaconda)
#   - git
#   - CUDA 12.6 driver  (run `nvcc --version` to confirm)
# =============================================================================

set -e   # abort on first error

# ── colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── config ────────────────────────────────────────────────────────────────────
REPO_URL="https://github.com/hugobaudchon/CanopyRS.git"
TARGET_DIR="$HOME/projects/CanopyRS"
ENV_NAME="canopyrs_env"
PYTHON_VER="3.10"
TORCH_INDEX="https://download.pytorch.org/whl/cu126"

# ── 0. pre-flight checks ─────────────────────────────────────────────────────
info "Running pre-flight checks..."

command -v conda &>/dev/null  || error "conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html"
command -v git   &>/dev/null  || error "git not found. Run: sudo apt install git"

# Source conda so 'conda activate' works in this script
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"

success "All pre-flight checks passed."

# ── 1. clone repository ───────────────────────────────────────────────────────
info "Cloning CanopyRS into $TARGET_DIR ..."
mkdir -p "$HOME/projects"

if [ -d "$TARGET_DIR/.git" ]; then
    warn "Repository already exists at $TARGET_DIR — skipping clone."
    cd "$TARGET_DIR"
else
    git clone "$REPO_URL" "$TARGET_DIR"
    cd "$TARGET_DIR"
fi

info "Initialising git submodules (detrex / detectron2)..."
git submodule update --init --recursive
success "Submodules ready."

# ── 2. create conda environment ───────────────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
    warn "Conda env '${ENV_NAME}' already exists — skipping creation."
else
    info "Creating conda env '${ENV_NAME}' with Python ${PYTHON_VER} + mamba..."
    conda create -y -n "$ENV_NAME" -c conda-forge python="$PYTHON_VER" mamba
    success "Conda env '${ENV_NAME}' created."
fi

conda activate "$ENV_NAME"
success "Activated conda env: $ENV_NAME"

# ── 3. install GDAL via mamba ─────────────────────────────────────────────────
info "Installing GDAL 3.6.2 via mamba (conda-forge)..."
mamba install -y gdal=3.6.2 -c conda-forge
success "GDAL installed."

# ── 4. install PyTorch + torchvision (CUDA 12.6) ─────────────────────────────
info "Installing PyTorch 2.7.1 + torchvision 0.22.1 (cu126)..."
pip install torch==2.7.1 torchvision==0.22.1 --index-url "$TORCH_INDEX"
success "PyTorch installed."

# ── 5. install CanopyRS (editable) ────────────────────────────────────────────
info "Installing CanopyRS in editable mode..."
python -m pip install -e .
success "CanopyRS installed."

# ── 6. install detrex + detectron2 ────────────────────────────────────────────
info "Installing detectron2 and detrex (editable, no build isolation)..."
python -m pip install --no-build-isolation -e ./detrex/detectron2 -e ./detrex
success "detrex + detectron2 installed."

# ── 7. iopath conflict note ───────────────────────────────────────────────────
warn "You may see: 'sam2 requires iopath>=0.1.10 but iopath 0.1.9 is installed'."
warn "This is a known detectron2/SAM2 conflict — it can be safely ignored."

# ── 8. verify ─────────────────────────────────────────────────────────────────
info "Verifying installation..."
python -c "import canopyrs; print('CanopyRS installed successfully')" \
    && success "Installation verified!" \
    || error "Verification failed — check the logs above."

# ── 9. huggingface login reminder ─────────────────────────────────────────────
echo ""
echo -e "${YELLOW}========================================================"
echo "  OPTIONAL: SAM 3 requires HuggingFace access"
echo ""
echo "  1. Go to: https://huggingface.co/facebook/sam3"
echo "  2. Click 'Request access' and accept Meta's licence"
echo "  3. Run:   huggingface-cli login"
echo -e "========================================================${NC}"
echo ""
success "All done!  Activate your env with:   conda activate ${ENV_NAME}"
echo ""
echo -e "  Then test with:"
echo -e "    cd ~/projects/CanopyRS"
echo -e "    python infer.py -c preset_det_single_S_fasterrcnn_r50 \\"
echo -e "        -i assets/20240130_zf2tower_m3m_rgb_test_crop.tif \\"
echo -e "        -o /tmp/canopyrs_test_output"
