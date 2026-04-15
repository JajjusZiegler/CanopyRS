#!/usr/bin/env bash
# =============================================================================
# CanopyRS — Fix CUDA version mismatch when building detectron2 / detrex
#
# Problem:  nvcc on PATH is 13.1, but torch wheels are compiled for CUDA 12.6.
#           detectron2's build checks these match and raises RuntimeError.
#
# Fix:      Install the CUDA 12.6 *toolkit* (no driver changes needed —
#           the GPU driver is always backward-compatible with older toolkits),
#           then build detectron2/detrex with CUDA_HOME pointing to 12.6.
#
# Usage:    bash fix_cuda_mismatch.sh
# =============================================================================

set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

ENV_NAME="canopyrs_env"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_TARGET="12.6"
CUDA_INSTALL_PATH="/usr/local/cuda-12.6"

# ── 1. Show current state ─────────────────────────────────────────────────────
info "Current nvcc on PATH: $(nvcc --version 2>/dev/null | grep 'release' || echo 'not found')"
info "PyTorch requires CUDA $CUDA_TARGET toolkit (nvcc) to build detectron2."

# ── 2. Check if 12.6 toolkit already installed ───────────────────────────────
if [ -x "$CUDA_INSTALL_PATH/bin/nvcc" ]; then
    success "CUDA 12.6 toolkit already at $CUDA_INSTALL_PATH — skipping install."
else
    info "Installing CUDA 12.6 toolkit (toolkit only — no driver changes)..."

    # Add NVIDIA's CUDA repo for WSL-Ubuntu
    KEYRING_DEB="cuda-keyring_1.1-1_all.deb"
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/$KEYRING_DEB" \
        -O "/tmp/$KEYRING_DEB"
    sudo dpkg -i "/tmp/$KEYRING_DEB"
    sudo apt-get update -qq

    # Install only the compiler toolkit — NOT the driver or full CUDA stack
    sudo apt-get install -y cuda-toolkit-12-6

    success "CUDA 12.6 toolkit installed at $CUDA_INSTALL_PATH."
fi

# ── 3. Verify 12.6 nvcc ───────────────────────────────────────────────────────
"$CUDA_INSTALL_PATH/bin/nvcc" --version | grep "release" \
    || error "CUDA 12.6 nvcc not found after install — check the apt logs above."

# ── 4. Activate conda env ──────────────────────────────────────────────────────
info "Activating conda env '${ENV_NAME}'..."
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

if ! command -v sudo >/dev/null 2>&1; then
    error "sudo is required to install CUDA toolkit packages."
fi

# ── 5. Move to repo ───────────────────────────────────────────────────────────
[ -d "$REPO_DIR" ] || error "Repo not found at $REPO_DIR."
cd "$REPO_DIR"

# ── 6. Build detectron2 with CUDA 12.6 pinned ────────────────────────────────
info "Building detectron2 with CUDA 12.6 toolkit (5–15 min)..."

# Override PATH so the 12.6 nvcc is found first, and set CUDA_HOME explicitly.
# MAX_JOBS=4 prevents WSL from OOMing during parallel C++ compilation.
CUDA_HOME="$CUDA_INSTALL_PATH" \
PATH="$CUDA_INSTALL_PATH/bin:$PATH" \
MAX_JOBS=4 \
    python -m pip install ninja \
    || error "Failed to install ninja in ${ENV_NAME}."

CUDA_HOME="$CUDA_INSTALL_PATH" \
PATH="$CUDA_INSTALL_PATH/bin:$PATH" \
MAX_JOBS=4 \
    python -m pip install --no-build-isolation -e ./detrex/detectron2

success "detectron2 installed."

# ── 7. Build detrex ───────────────────────────────────────────────────────────
info "Building detrex..."
CUDA_HOME="$CUDA_INSTALL_PATH" \
PATH="$CUDA_INSTALL_PATH/bin:$PATH" \
MAX_JOBS=4 \
    python -m pip install --no-build-isolation -e ./detrex

success "detrex installed."

# ── 8. Add CUDA 12.6 to conda env activation (persistent) ────────────────────
# So future pip installs and scripts in this env always find the right nvcc.
info "Pinning CUDA 12.6 paths into conda env activation scripts..."
ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
DEACTIVATE_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"
mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

cat > "$ACTIVATE_DIR/cuda126.sh" <<'EOF'
export _OLD_CUDA_HOME="$CUDA_HOME"
export _OLD_PATH="$PATH"
export CUDA_HOME="/usr/local/cuda-12.6"
export PATH="/usr/local/cuda-12.6/bin:$PATH"
EOF

cat > "$DEACTIVATE_DIR/cuda126.sh" <<'EOF'
export CUDA_HOME="$_OLD_CUDA_HOME"
export PATH="$_OLD_PATH"
unset _OLD_CUDA_HOME _OLD_PATH
EOF

success "Done — conda env will now automatically use nvcc 12.6 when activated."

# ── 9. Full verification ──────────────────────────────────────────────────────
info "Verifying installation..."
python -c "
import canopyrs, torch, torchvision
print(f'  canopyrs   : OK')
print(f'  torch      : {torch.__version__}  (CUDA available: {torch.cuda.is_available()})')
print(f'  torchvision: {torchvision.__version__}')
try:
    import detectron2; print(f'  detectron2 : {detectron2.__version__}  OK')
except Exception as e: print(f'  detectron2 : FAILED — {e}')
try:
    import detrex; print(f'  detrex     : OK')
except Exception as e: print(f'  detrex     : FAILED — {e}')
"

success "All done!"
echo ""
echo -e "  Activate anytime with:  ${CYAN}conda activate ${ENV_NAME}${NC}"
echo -e "  nvcc 12.6 will be set automatically inside the env."
