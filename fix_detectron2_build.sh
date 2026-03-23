#!/usr/bin/env bash
# =============================================================================
# CanopyRS — Fix detectron2 / detrex build failure
# Usage:  bash fix_detectron2_build.sh
#
# Root cause: WSL Ubuntu often ships without the C++ build toolchain,
# ninja-build, or Python dev headers — all required to compile detectron2.
# =============================================================================

set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

ENV_NAME="canopyrs_env"
REPO_DIR="$HOME/projects/CanopyRS"

# ── 1. Install apt build dependencies ────────────────────────────────────────
info "Installing C++ build toolchain + ninja + Python dev headers..."
sudo apt-get update -qq
sudo apt-get install -y \
    build-essential \
    ninja-build \
    python3-dev \
    libglib2.0-dev \
    git
success "Build dependencies installed."

# ── 2. Check nvcc availability ────────────────────────────────────────────────
info "Checking CUDA toolkit (nvcc)..."
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+")
    success "nvcc found — CUDA $CUDA_VER. detectron2 will be built with GPU support."
else
    warn "nvcc NOT found. detectron2 will be built for CPU only."
    warn "If you need GPU support, install the CUDA 12.6 toolkit first:"
    warn "  https://developer.nvidia.com/cuda-12-6-0-download-archive"
    warn "  (select: Linux → x86_64 → WSL-Ubuntu → 2.0 → deb(local))"
    echo ""
    read -rp "Continue with CPU-only build? [y/N] " choice
    [[ "$choice" =~ ^[Yy]$ ]] || exit 0
fi

# ── 3. Activate conda env ─────────────────────────────────────────────────────
info "Activating conda env '${ENV_NAME}'..."
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ── 4. Install ninja Python binding (speeds up compilation) ──────────────────
info "Installing ninja Python binding..."
pip install ninja
success "ninja installed."

# ── 5. Move to repo ───────────────────────────────────────────────────────────
[ -d "$REPO_DIR" ] || error "Repo not found at $REPO_DIR. Run setup_canopyrs_wsl.sh first."
cd "$REPO_DIR"

# ── 6. Build detectron2 first ─────────────────────────────────────────────────
info "Building detectron2 (this takes 5–15 min on first run)..."
# MAX_JOBS prevents OOM during parallel C++ compilation
MAX_JOBS=4 python -m pip install --no-build-isolation -e ./detrex/detectron2
success "detectron2 installed."

# ── 7. Build detrex ───────────────────────────────────────────────────────────
info "Building detrex..."
MAX_JOBS=4 python -m pip install --no-build-isolation -e ./detrex
success "detrex installed."

# ── 8. Verify ─────────────────────────────────────────────────────────────────
info "Verifying full installation..."
python -c "
import canopyrs
import torch, torchvision
print(f'  canopyrs   : OK')
print(f'  torch      : {torch.__version__}  (CUDA: {torch.cuda.is_available()})')
print(f'  torchvision: {torchvision.__version__}')
try:
    import detectron2
    print(f'  detectron2 : {detectron2.__version__}  OK')
except Exception as e:
    print(f'  detectron2 : FAILED — {e}')
try:
    import detrex
    print(f'  detrex     : OK')
except Exception as e:
    print(f'  detrex     : FAILED — {e}')
"

success "All done! Activate your env any time with:   conda activate ${ENV_NAME}"
