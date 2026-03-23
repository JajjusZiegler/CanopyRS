#!/usr/bin/env bash
# =============================================================================
# CanopyRS — Batch inference on a site folder
#
# Scans a site directory for date subfolders, finds the RGB orthomosaic in
# each exports/ subfolder, and runs CanopyRS inference. Output is written to
# a canopyRS/ directory at the same level as exports/.
#
# Usage:
#   bash batch_infer.sh [OPTIONS] <SITE_FOLDER>
#
# Examples:
#   # SAM3 quality on all Marteloskop dates
#   bash batch_infer.sh /mnt/m/working_package_2/.../Marteloskop
#
#   # SAM3 fast preset, skip already-processed dates
#   bash batch_infer.sh -c preset_seg_multi_NQOS_selvamask_SAM3_FT_fast \
#       /mnt/m/working_package_2/.../Marteloskop
#
#   # Dry-run to preview what would be processed
#   bash batch_infer.sh --dry-run /mnt/m/working_package_2/.../Marteloskop
#
# Options:
#   -c <config>     CanopyRS pipeline config name (default: SAM3 quality)
#   -p <pattern>    Glob pattern to match the TIF file (default: *_rgb_model_ortho_smooth_100.tif)
#   -o <subdir>     Output subfolder name under each date dir (default: canopyRS)
#   --aoi <path>    Path to an AOI GeoPackage (.gpkg) — inference is restricted to
#                   this polygon, excluding border artefacts of the orthomosaic
#   --dry-run       Print what would be processed without running inference
#   --skip-existing  Skip dates that already have a completed *_inferfinal.gpkg
#   --force         Re-run even if output already exists (overrides --skip-existing)
#   --skip-errors   Continue batch even if a site fails
# =============================================================================

set -uo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
CONFIG="preset_seg_multi_NQOS_selvamask_SAM3_FT_quality"
TIF_PATTERN="*_rgb_model_ortho_smooth_100.tif"
OUTPUT_SUBDIR="canopyRS"
AOI_PATH=""
DRY_RUN=false
FORCE=false
SKIP_EXISTING=true
SKIP_ERRORS=false
LOCAL_CACHE=false
LOCAL_CACHE_DIR="$HOME/data/_canopyrs_cache"
ENV_NAME="canopyrs_env"
REPO_DIR="$HOME/projects/CanopyRS"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[SKIP]${NC}  $*"; }
fail()    { echo -e "${RED}[FAIL]${NC}  $*"; }
header()  { echo -e "\n${BOLD}${CYAN}$*${NC}"; }

# ── Argument parsing ──────────────────────────────────────────────────────────
SITE_FOLDER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c) CONFIG="$2"; shift 2 ;;
        -p) TIF_PATTERN="$2"; shift 2 ;;
        -o) OUTPUT_SUBDIR="$2"; shift 2 ;;
        --aoi) AOI_PATH="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --skip-existing) SKIP_EXISTING=true; shift ;;
        --force) FORCE=true; SKIP_EXISTING=false; shift ;;
        --skip-errors) SKIP_ERRORS=true; shift ;;
        --no-skip) SKIP_EXISTING=false; shift ;;
        --local-cache) LOCAL_CACHE=true; shift ;;
        --cache-dir) LOCAL_CACHE_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,30p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *) SITE_FOLDER="$1"; shift ;;
    esac
done

[ -z "$SITE_FOLDER" ] && { echo "Usage: bash batch_infer.sh [OPTIONS] <SITE_FOLDER>"; exit 1; }
[ -d "$SITE_FOLDER" ] || { echo "Site folder not found: $SITE_FOLDER"; exit 1; }

# ── Resolve Python path ───────────────────────────────────────────────────────
# If canopyrs_env is already active, use it directly.
# Otherwise find it under the base conda installation.
if [[ "$CONDA_DEFAULT_ENV" == "$ENV_NAME" ]] && command -v python &>/dev/null; then
    PYTHON="$(command -v python)"
else
    # Strip any active env suffix to get the real base path
    CONDA_BASE="${CONDA_PREFIX_1:-${CONDA_PREFIX%/envs/*}}"
    # Fallback: common install locations
    if [ -z "$CONDA_BASE" ]; then
        for candidate in "$HOME/anaconda3" "$HOME/miniconda3" "/opt/conda"; do
            if [ -x "$candidate/envs/$ENV_NAME/bin/python" ]; then
                CONDA_BASE="$candidate"; break
            fi
        done
    fi
    PYTHON="$CONDA_BASE/envs/$ENV_NAME/bin/python"
fi
[ -x "$PYTHON" ] || { echo "Python not found at $PYTHON — is $ENV_NAME installed?"; exit 1; }
info "Using Python: $PYTHON"

# ── Discover date folders ─────────────────────────────────────────────────────
SITE_NAME=$(basename "$SITE_FOLDER")
header "Batch inference — $SITE_NAME"
info "Config   : $CONFIG"
info "Pattern  : $TIF_PATTERN"
info "Output   : <date>/$OUTPUT_SUBDIR/"
[ "$DRY_RUN" = true ]      && warn "DRY-RUN mode — no inference will be run"
[ "$SKIP_EXISTING" = true ] && info "Skip existing: ON (use --no-skip or --force to re-run)"
[ "$LOCAL_CACHE" = true ]   && info "Local cache: $LOCAL_CACHE_DIR (TIFs copied before inference)"
if [ -n "$AOI_PATH" ]; then
    [ -f "$AOI_PATH" ] || { echo "AOI file not found: $AOI_PATH"; exit 1; }
    info "AOI      : $AOI_PATH (edge artefacts will be excluded)"
fi
echo ""

# Collect date-like subdirectories (e.g. 20250606, 20240130)
mapfile -t DATE_DIRS < <(find "$SITE_FOLDER" -maxdepth 1 -mindepth 1 -type d \
    | grep -E '/[0-9]{8}$' | sort)

if [ ${#DATE_DIRS[@]} -eq 0 ]; then
    echo "No date-format subfolders (YYYYMMDD) found in $SITE_FOLDER"
    exit 1
fi

info "Found ${#DATE_DIRS[@]} date folder(s): $(basename -a "${DATE_DIRS[@]}" | tr '\n' ' ')"
echo ""

# ── Per-site counters ─────────────────────────────────────────────────────────
COUNT_OK=0
COUNT_SKIP=0
COUNT_FAIL=0
COUNT_NOTIF=0
LOG_FILE="/tmp/canopyrs_batch_$(date +%Y%m%d_%H%M%S).log"

echo "Batch log: $LOG_FILE"
echo ""

# ── Main loop ─────────────────────────────────────────────────────────────────
for DATE_DIR in "${DATE_DIRS[@]}"; do
    DATE=$(basename "$DATE_DIR")
    EXPORTS_DIR="$DATE_DIR/exports"
    OUTPUT_DIR="$DATE_DIR/$OUTPUT_SUBDIR"

    echo -e "${BOLD}── $DATE ──────────────────────────────────────────────────────${NC}"

    # Check exports/ exists
    if [ ! -d "$EXPORTS_DIR" ]; then
        warn "No exports/ folder found — skipping"
        echo "[$DATE] SKIP: no exports/ folder" >> "$LOG_FILE"
        COUNT_SKIP=$((COUNT_SKIP + 1)); continue
    fi

    # Find the TIF
    TIF=$(find "$EXPORTS_DIR" -maxdepth 1 -name "$TIF_PATTERN" | head -1)
    if [ -z "$TIF" ]; then
        warn "No TIF matching '$TIF_PATTERN' in exports/ — skipping"
        echo "[$DATE] SKIP: no matching TIF" >> "$LOG_FILE"
        COUNT_NOTIF=$((COUNT_NOTIF + 1)); continue
    fi
    TIF_NAME=$(basename "$TIF")
    TIF_SIZE=$(du -h "$TIF" 2>/dev/null | cut -f1)
    info "TIF: $TIF_NAME ($TIF_SIZE)"

    # Check if already done — look for the actual CanopyRS output (*_inferfinal.gpkg)
    EXISTING_RESULT=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*_inferfinal.gpkg" 2>/dev/null | head -1)
    if [ -n "$EXISTING_RESULT" ] && [ "$SKIP_EXISTING" = true ] && [ "$FORCE" = false ]; then
        warn "Already processed ($(basename "$EXISTING_RESULT")) — skipping. Use --force to re-run."
        echo "[$DATE] SKIP: already done" >> "$LOG_FILE"
        COUNT_SKIP=$((COUNT_SKIP + 1)); continue
    fi

    # Dry-run stops here
    if [ "$DRY_RUN" = true ]; then
        info "DRY-RUN: would run → $TIF → $OUTPUT_DIR"
        echo "[$DATE] DRY-RUN: $TIF_NAME" >> "$LOG_FILE"
        continue
    fi

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Optionally copy TIF to local NVMe cache before inference
    # (dramatically faster for network drives — avoids repeated remote reads during tiling)
    INFER_TIF="$TIF"
    LOCAL_TIF=""
    if [ "$LOCAL_CACHE" = true ] && [ "$DRY_RUN" = false ]; then
        mkdir -p "$LOCAL_CACHE_DIR"
        LOCAL_TIF="$LOCAL_CACHE_DIR/$TIF_NAME"
        if [ ! -f "$LOCAL_TIF" ]; then
            info "Copying TIF to local cache ($TIF_SIZE)..."
            COPY_START=$(date +%s)
            cp "$TIF" "$LOCAL_TIF"
            COPY_END=$(date +%s)
            COPY_TIME=$(( COPY_END - COPY_START ))
            success "Copied in ${COPY_TIME}s → $LOCAL_TIF"
        else
            info "Using cached TIF: $LOCAL_TIF"
        fi
        INFER_TIF="$LOCAL_TIF"
    fi

    # Run inference
    info "Running inference → output: $OUTPUT_DIR"
    START_TIME=$(date +%s)

    # Build AOI argument (optional)
    AOI_ARGS=()
    [ -n "$AOI_PATH" ] && AOI_ARGS=(-aoi "$AOI_PATH")

    if "$PYTHON" "$REPO_DIR/infer.py" \
            -c "$CONFIG" \
            -i "$INFER_TIF" \
            -o "$OUTPUT_DIR" \
            "${AOI_ARGS[@]}" \
            2>&1 | tee -a "$LOG_FILE"; then
        END_TIME=$(date +%s)
        ELAPSED=$(( END_TIME - START_TIME ))
        ELAPSED_FMT=$(printf '%dm%02ds' $((ELAPSED/60)) $((ELAPSED%60)))
        success "Done in $ELAPSED_FMT → $OUTPUT_DIR/results.gpkg"
        echo "[$DATE] OK: $TIF_NAME → $OUTPUT_DIR ($ELAPSED_FMT)" >> "$LOG_FILE"
        COUNT_OK=$((COUNT_OK + 1))

        # Clean up local cache after successful inference
        if [ "$LOCAL_CACHE" = true ] && [ -n "$LOCAL_TIF" ] && [ -f "$LOCAL_TIF" ]; then
            rm -f "$LOCAL_TIF"
            info "Removed local cache copy."
        fi
    else
        fail "Inference failed for $DATE"
        echo "[$DATE] FAIL: $TIF_NAME" >> "$LOG_FILE"
        COUNT_FAIL=$((COUNT_FAIL + 1))
        # Clean up cache on failure too
        if [ "$LOCAL_CACHE" = true ] && [ -n "$LOCAL_TIF" ] && [ -f "$LOCAL_TIF" ]; then
            rm -f "$LOCAL_TIF"
        fi
        if [ "$SKIP_ERRORS" = false ]; then
            echo ""
            echo -e "${RED}Stopping batch due to error. Use --skip-errors to continue on failure.${NC}"
            break
        fi
    fi
    echo ""
done

# ── Summary ───────────────────────────────────────────────────────────────────
header "Batch complete — $SITE_NAME"
echo -e "  ${GREEN}Processed : $COUNT_OK${NC}"
echo -e "  ${YELLOW}Skipped   : $COUNT_SKIP${NC}"
[ $COUNT_NOTIF -gt 0 ] && echo -e "  ${YELLOW}No TIF    : $COUNT_NOTIF${NC}"
[ $COUNT_FAIL  -gt 0 ] && echo -e "  ${RED}Failed    : $COUNT_FAIL${NC}"
echo ""
echo "Full log: $LOG_FILE"

# Exit with error code if any failed
[ $COUNT_FAIL -gt 0 ] && exit 1 || exit 0
