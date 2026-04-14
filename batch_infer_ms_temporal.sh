#!/usr/bin/env bash
# =============================================================================
# CanopyRS — Multi-temporal Multispectral (MS) batch inference
#
# Scans a site directory for date subfolders, finds the 5-band multispectral
# orthomosaic in each exports/ subfolder, and runs CanopyRS Detectree2 MS
# inference.  An AOI GeoPackage is strongly recommended to exclude edge
# artefacts that are common in drone orthomosaics.
#
# Usage:
#   bash batch_infer_ms_temporal.sh [OPTIONS] <SITE_FOLDER>
#
# Examples:
#   # Run single date (20250908) for Marteloskop with AOI clipping
#   bash batch_infer_ms_temporal.sh \
#       -d 20250908 \
#       --aoi "/mnt/m/working_package_2/2024_dronecampaign/02_processing/metashape_projects/Upscale_Metashapeprojects/Marteloskop/AOI_Marteloskop.gpkg" \
#       "/mnt/m/working_package_2/2024_dronecampaign/02_processing/metashape_projects/Upscale_Metashapeprojects/Marteloskop"
#
#   # Run all available dates with local NVMe caching (faster on network drives)
#   bash batch_infer_ms_temporal.sh \
#       --local-cache \
#       --aoi "/mnt/m/.../AOI_Marteloskop.gpkg" \
#       "/mnt/m/.../Marteloskop"
#
#   # Dry-run to preview which dates would be processed
#   bash batch_infer_ms_temporal.sh --dry-run "/mnt/m/.../Marteloskop"
#
# Options:
#   -c <config>       Pipeline config name
#                     (default: preset_seg_standalone_detectree2_ms)
#   -p <pattern>      Glob pattern to match the MS TIF
#                     (default: *_multispec_ortho_100cm.tif)
#   -o <subdir>       Output subfolder under each date dir (default: canopyRS_ms)
#   --aoi <path>      Path to site AOI GeoPackage — strongly recommended to
#                     exclude orthomosaic edge artefacts.
#   -d <YYYYMMDD>     Process only this single date folder (e.g. -d 20250908)
#   --dry-run         Print what would be processed without running inference
#   --force           Re-run even if output already exists
#   --skip-errors     Continue batch even if a single date fails
#   --local-cache     Copy TIF to local NVMe before inference (faster on
#                     network/SMB drives)
#   --cache-dir <p>   Local cache directory (default: ~/data/_canopyrs_cache)
# =============================================================================

set -uo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
CONFIG="preset_seg_standalone_detectree2_ms"
TIF_PATTERN="*_multispec_ortho_100cm.tif"
OUTPUT_SUBDIR="canopyRS_ms"
AOI_PATH=""
DATE_FILTER=""
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
        -c)            CONFIG="$2";            shift 2 ;;
        -p)            TIF_PATTERN="$2";       shift 2 ;;
        -o)            OUTPUT_SUBDIR="$2";     shift 2 ;;
        --aoi)         AOI_PATH="$2";          shift 2 ;;
        -d)            DATE_FILTER="$2";       shift 2 ;;
        --dry-run)     DRY_RUN=true;           shift ;;
        --force)       FORCE=true; SKIP_EXISTING=false; shift ;;
        --skip-errors) SKIP_ERRORS=true;       shift ;;
        --local-cache) LOCAL_CACHE=true;       shift ;;
        --cache-dir)   LOCAL_CACHE_DIR="$2";   shift 2 ;;
        -h|--help)
            sed -n '3,45p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *)  SITE_FOLDER="$1"; shift ;;
    esac
done

[ -z "$SITE_FOLDER" ] && { echo "Usage: bash batch_infer_ms_temporal.sh [OPTIONS] <SITE_FOLDER>"; exit 1; }
[ -d "$SITE_FOLDER" ] || { echo "Site folder not found: $SITE_FOLDER"; exit 1; }

if [ -n "$AOI_PATH" ]; then
    [ -f "$AOI_PATH" ] || { echo "AOI file not found: $AOI_PATH"; exit 1; }
fi

# ── Resolve Python path ───────────────────────────────────────────────────────
if [[ "$CONDA_DEFAULT_ENV" == "$ENV_NAME" ]] && command -v python &>/dev/null; then
    PYTHON="$(command -v python)"
else
    CONDA_BASE="${CONDA_PREFIX_1:-${CONDA_PREFIX%/envs/*}}"
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

# ── Discover date folders ─────────────────────────────────────────────────────
SITE_NAME=$(basename "$SITE_FOLDER")
header "Multi-temporal MS inference — $SITE_NAME"
info "Config   : $CONFIG"
info "Pattern  : $TIF_PATTERN"
info "Output   : <date>/$OUTPUT_SUBDIR/"
info "Python   : $PYTHON"
[ "$DRY_RUN" = true ]      && warn "DRY-RUN mode — no inference will be run"
[ "$SKIP_EXISTING" = true ] && info "Skip existing: ON (use --force to re-run)"
[ "$LOCAL_CACHE" = true ]   && info "Local cache  : $LOCAL_CACHE_DIR"
[ -n "$DATE_FILTER" ]       && info "Date filter  : $DATE_FILTER (single date only)"
if [ -n "$AOI_PATH" ]; then
    info "AOI      : $AOI_PATH"
else
    warn "No AOI provided — edge artefacts will be included. Use --aoi to restrict inference area."
fi
echo ""

mapfile -t DATE_DIRS < <(find "$SITE_FOLDER" -maxdepth 1 -mindepth 1 -type d \
    | grep -E '/[0-9]{8}$' | sort)

if [ ${#DATE_DIRS[@]} -eq 0 ]; then
    echo "No date-format subfolders (YYYYMMDD) found in $SITE_FOLDER"
    exit 1
fi

info "Found ${#DATE_DIRS[@]} date folder(s): $(basename -a "${DATE_DIRS[@]}" | tr '\n' ' ')"
echo ""

# ── Counters & log ────────────────────────────────────────────────────────────
COUNT_OK=0; COUNT_SKIP=0; COUNT_FAIL=0; COUNT_NOTIF=0
LOG_FILE="/tmp/canopyrs_ms_$(date +%Y%m%d_%H%M%S).log"
echo "Batch log: $LOG_FILE"
echo ""

# ── Main loop ─────────────────────────────────────────────────────────────────
for DATE_DIR in "${DATE_DIRS[@]}"; do
    DATE=$(basename "$DATE_DIR")
    if [ -n "$DATE_FILTER" ] && [ "$DATE" != "$DATE_FILTER" ]; then
        continue
    fi
    EXPORTS_DIR="$DATE_DIR/exports"
    OUTPUT_DIR="$DATE_DIR/$OUTPUT_SUBDIR"

    echo -e "${BOLD}── $DATE ──────────────────────────────────────────────────────${NC}"

    if [ ! -d "$EXPORTS_DIR" ]; then
        warn "No exports/ folder — skipping"
        echo "[$DATE] SKIP: no exports/" >> "$LOG_FILE"
        COUNT_SKIP=$((COUNT_SKIP + 1)); continue
    fi

    TIF=$(find "$EXPORTS_DIR" -maxdepth 1 -name "$TIF_PATTERN" | head -1)
    if [ -z "$TIF" ]; then
        warn "No TIF matching '$TIF_PATTERN' — skipping"
        echo "[$DATE] SKIP: no matching TIF" >> "$LOG_FILE"
        COUNT_NOTIF=$((COUNT_NOTIF + 1)); continue
    fi
    TIF_NAME=$(basename "$TIF")
    TIF_SIZE=$(du -h "$TIF" 2>/dev/null | cut -f1)
    info "TIF: $TIF_NAME ($TIF_SIZE)"

    # Skip if already done
    EXISTING=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*_inferfinal.gpkg" 2>/dev/null | head -1)
    if [ -n "$EXISTING" ] && [ "$SKIP_EXISTING" = true ] && [ "$FORCE" = false ]; then
        warn "Already processed ($(basename "$EXISTING")) — use --force to re-run"
        echo "[$DATE] SKIP: already done" >> "$LOG_FILE"
        COUNT_SKIP=$((COUNT_SKIP + 1)); continue
    fi

    if [ "$DRY_RUN" = true ]; then
        info "DRY-RUN: $TIF_NAME → $OUTPUT_DIR"
        echo "[$DATE] DRY-RUN: $TIF_NAME" >> "$LOG_FILE"
        continue
    fi

    mkdir -p "$OUTPUT_DIR"

    # Optional local cache copy (avoids repeated reads from network/SMB drives)
    INFER_TIF="$TIF"
    LOCAL_TIF=""
    if [ "$LOCAL_CACHE" = true ]; then
        mkdir -p "$LOCAL_CACHE_DIR"
        LOCAL_TIF="$LOCAL_CACHE_DIR/$TIF_NAME"
        if [ ! -f "$LOCAL_TIF" ]; then
            info "Copying to local cache ($TIF_SIZE)..."
            COPY_START=$(date +%s)
            cp "$TIF" "$LOCAL_TIF"
            success "Copied in $(( $(date +%s) - COPY_START ))s"
        else
            info "Using cached TIF: $LOCAL_TIF"
        fi
        INFER_TIF="$LOCAL_TIF"
    fi

    # Build command
    CMD=( "$PYTHON" "$REPO_DIR/infer.py" -c "$CONFIG" -i "$INFER_TIF" -o "$OUTPUT_DIR" )
    [ -n "$AOI_PATH" ] && CMD+=( -aoi "$AOI_PATH" )

    info "Running MS inference → $OUTPUT_DIR"
    START_TIME=$(date +%s)

    if "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
        ELAPSED=$(( $(date +%s) - START_TIME ))
        ELAPSED_FMT=$(printf '%dm%02ds' $((ELAPSED/60)) $((ELAPSED%60)))
        success "Done in $ELAPSED_FMT → $OUTPUT_DIR"
        echo "[$DATE] OK: $TIF_NAME ($ELAPSED_FMT)" >> "$LOG_FILE"
        COUNT_OK=$((COUNT_OK + 1))
    else
        fail "Inference failed for $DATE"
        echo "[$DATE] FAIL: $TIF_NAME" >> "$LOG_FILE"
        COUNT_FAIL=$((COUNT_FAIL + 1))
        if [ "$SKIP_ERRORS" = false ]; then
            echo -e "\n${RED}Stopping. Use --skip-errors to continue on failure.${NC}"
            break
        fi
    fi

    # Clean up local cache
    if [ "$LOCAL_CACHE" = true ] && [ -n "$LOCAL_TIF" ] && [ -f "$LOCAL_TIF" ]; then
        rm -f "$LOCAL_TIF"
    fi

    echo ""
done

# ── Summary ───────────────────────────────────────────────────────────────────
header "Complete — $SITE_NAME"
echo -e "  ${GREEN}Processed : $COUNT_OK${NC}"
echo -e "  ${YELLOW}Skipped   : $COUNT_SKIP${NC}"
[ $COUNT_NOTIF -gt 0 ] && echo -e "  ${YELLOW}No TIF    : $COUNT_NOTIF${NC}"
[ $COUNT_FAIL  -gt 0 ] && echo -e "  ${RED}Failed    : $COUNT_FAIL${NC}"
echo -e "\n  Config used   : $CONFIG"
[ -n "$AOI_PATH" ] && echo -e "  AOI applied   : $AOI_PATH"
echo -e "  Full log      : $LOG_FILE"
echo ""

[ $COUNT_FAIL -gt 0 ] && exit 1 || exit 0
