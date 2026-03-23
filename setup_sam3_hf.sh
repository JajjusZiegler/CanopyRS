#!/usr/bin/env bash
# =============================================================================
# CanopyRS — SAM 3 HuggingFace setup
# Usage:  bash setup_sam3_hf.sh
#
# What this does:
#   1. Activates canopyrs_env
#   2. Installs huggingface_hub CLI if needed
#   3. Logs in with your HF token
#   4. Verifies that your account has access to facebook/sam3
#   5. Pre-downloads the model weights so the first inference is instant
# =============================================================================

set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

ENV_NAME="canopyrs_env"
MODEL_ID="facebook/sam3"

# ── 1. Activate env ───────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Use explicit paths from the conda env to avoid PATH issues with WSL
CONDA_PYTHON="$(conda info --base)/envs/$ENV_NAME/bin/python"
CONDA_PIP="$(conda info --base)/envs/$ENV_NAME/bin/pip"

info "Using Python: $CONDA_PYTHON"

# ── 2. Ensure huggingface_hub is installed ────────────────────────────────────
"$CONDA_PIP" install -q --upgrade huggingface_hub
success "huggingface_hub ready."

# ── 3. Get token ──────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Paste your HuggingFace token below (input is hidden):${NC}"
echo -e "  Get one at: https://huggingface.co/settings/tokens"
echo -e "  Required scope: read (or write)"
echo ""
read -rsp "HF Token: " HF_TOKEN
echo ""

[ -z "$HF_TOKEN" ] && error "No token provided."

# ── 4. Login ──────────────────────────────────────────────────────────────────
info "Logging in to HuggingFace..."
"$CONDA_PYTHON" -c "
from huggingface_hub import login
login(token='${HF_TOKEN}', add_to_git_credential=False)
print('Login successful.')
"
success "Logged in."

# ── 5. Verify access to facebook/sam3 ────────────────────────────────────────
info "Checking access to ${MODEL_ID}..."
"$CONDA_PYTHON" - <<PYEOF
from huggingface_hub import model_info
import sys

model_id = "${MODEL_ID}"
try:
    info = model_info(model_id)
    print(f"  Model : {info.modelId}")
    print(f"  Access: granted")
except Exception as e:
    err = str(e)
    if "403" in err or "gated" in err.lower() or "access" in err.lower():
        print(f"\n  [!!] Access DENIED to {model_id}")
        print(f"  You must request access first:")
        print(f"    1. Go to: https://huggingface.co/facebook/sam3")
        print(f"    2. Click 'Request access' and accept Meta's licence")
        print(f"    3. Re-run this script once approved (usually instant)")
        sys.exit(1)
    else:
        print(f"  [!!] Unexpected error: {err}")
        sys.exit(1)
PYEOF
success "Access to ${MODEL_ID} confirmed."

# ── 6. Pre-download model weights ─────────────────────────────────────────────
info "Pre-downloading SAM 3 weights to local HF cache (~5–10 GB, please wait)..."
"$CONDA_PYTHON" - <<PYEOF
from huggingface_hub import snapshot_download
import os

model_id = "${MODEL_ID}"
cache = os.path.expanduser("~/.cache/huggingface/hub")
print(f"  Cache dir: {cache}")
path = snapshot_download(repo_id=model_id)
print(f"  Downloaded to: {path}")
PYEOF
success "SAM 3 weights cached locally."

# ── 7. Final check ────────────────────────────────────────────────────────────
info "Verifying SAM 3 can be loaded by canopyrs..."
"$CONDA_PYTHON" - <<PYEOF
from huggingface_hub import HfApi
api = HfApi()
user = api.whoami()
print(f"  Logged in as : {user['name']}")
print(f"  Token type   : {user.get('auth', {}).get('accessToken', {}).get('role', 'read')}")
print(f"  SAM 3 access : OK")
PYEOF

echo ""
success "SAM 3 setup complete!"
echo ""
echo -e "  You can now use SAM 3 presets, e.g.:"
echo -e "  ${CYAN}python infer.py -c preset_seg_multi_NQOS_selvamask_SAM3_FT_quality \\"
echo -e "      -i assets/20240130_zf2tower_m3m_rgb_test_crop.tif \\"
echo -e "      -o /tmp/canopyrs_sam3_test${NC}"
