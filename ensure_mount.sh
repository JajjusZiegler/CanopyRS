#!/usr/bin/env bash
# =============================================================================
# CanopyRS — WSL startup check
# Verifies conda env is active and M: drive is mounted.
# Called automatically on terminal open via ~/.bashrc or VS Code tasks.
#
# Usage:  bash ensure_mount.sh
# =============================================================================

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}   $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }

ERRORS=0

# ── 1. Conda env ──────────────────────────────────────────────────────────────
echo -e "${CYAN}── Conda environment ──────────────────────────────────────────${NC}"
if [[ "$CONDA_DEFAULT_ENV" == "canopyrs_env" ]]; then
    ok "canopyrs_env is active"
else
    warn "canopyrs_env is NOT active (current: ${CONDA_DEFAULT_ENV:-none})"
    echo "  Run:  conda activate canopyrs_env"
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null
    conda activate canopyrs_env 2>/dev/null && ok "Activated canopyrs_env" || { fail "Could not activate canopyrs_env"; ((ERRORS++)); }
fi

# ── 2. M: drive ───────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}── Drive mounts ────────────────────────────────────────────────${NC}"

MOUNT_POINT="/mnt/m"
ORTHO_PATH="${MOUNT_POINT}/working_package_2/2024_dronecampaign/02_processing/metashape_projects/Upscale_Metashapeprojects"
TARGET_TIF="${ORTHO_PATH}/Pfynwald/20250704/exports/20250704_Pfynwald_rgb_model_ortho_smooth_100.tif"

if mountpoint -q "$MOUNT_POINT" 2>/dev/null || [ -d "$MOUNT_POINT" ]; then
    ok "M: drive mounted at $MOUNT_POINT"
else
    warn "M: drive not mounted — attempting to mount..."
    sudo mkdir -p "$MOUNT_POINT"
    sudo mount -t drvfs M: "$MOUNT_POINT" -o metadata,uid=1000,gid=1000 2>/dev/null \
        && ok "M: drive mounted successfully" \
        || { fail "Could not mount M: — is the drive connected?"; ((ERRORS++)); }
fi

# Check the orthomosaic directory is reachable
if [ -d "$ORTHO_PATH" ]; then
    ok "Orthomosaic directory accessible"
else
    warn "Orthomosaic directory not found: $ORTHO_PATH"
    ((ERRORS++))
fi

# Check the specific Pfynwald TIF
if [ -f "$TARGET_TIF" ]; then
    SIZE=$(du -h "$TARGET_TIF" | cut -f1)
    ok "Pfynwald TIF found ($SIZE): $TARGET_TIF"
else
    warn "Pfynwald TIF not found: $TARGET_TIF"
    echo "  (This is OK if you haven't transferred the file yet)"
fi

# ── 3. CUDA ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}── CUDA ────────────────────────────────────────────────────────${NC}"
if [ -x "/usr/local/cuda-12.6/bin/nvcc" ]; then
    ok "CUDA 12.6 nvcc found"
else
    warn "CUDA 12.6 nvcc not found at /usr/local/cuda-12.6/bin/nvcc"
fi

CUDA_OK=$(python -c "import torch; print('yes' if torch.cuda.is_available() else 'no')" 2>/dev/null)
if [ "$CUDA_OK" == "yes" ]; then
    GPU=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    ok "GPU available: $GPU"
else
    warn "torch.cuda.is_available() = False"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
if [ "$ERRORS" -eq 0 ]; then
    echo -e "${GREEN}All checks passed. Ready to run inference.${NC}"
    echo ""
    echo "  Example — SAM3 quality on Pfynwald ortho:"
    echo "  python infer.py \\"
    echo "      -c preset_seg_multi_NQOS_selvamask_SAM3_FT_quality \\"
    echo "      -i '${TARGET_TIF}' \\"
    echo "      -o /tmp/canopyrs_pfynwald_sam3"
else
    echo -e "${RED}${ERRORS} issue(s) found — see warnings above.${NC}"
    exit 1
fi
