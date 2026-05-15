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
#   -c <config>         CanopyRS pipeline config name (default: SAM3 quality)
#   -p <pattern>        Glob pattern to match the RGB TIF (default: *_rgb_model_ortho_smooth_100.tif)
#   -o <subdir>         Output subfolder name under each date dir (default: canopyRS)
#                       Ignored when --output-root is set.
#   --output-root <dir> Absolute root directory for all outputs. Each date's results
#                       go to <dir>/<YYYYMMDD>/ instead of <date_dir>/<subdir>/.
#                       Useful for collecting all dates in one place (e.g. for temporal matching).
#   --ms-pattern <pat>  Glob pattern to match the multispectral TIF (e.g. *_multispec_ortho_100cm.tif).
#                       When set, the matching MS file is passed as -ms to the pipeline,
#                       enabling RGB+MS dual-stream inference.
#   --aoi <path>        Path to an AOI GeoPackage (.gpkg) — inference is restricted to
#                       this polygon, excluding border artefacts of the orthomosaic
#   --dry-run           Print what would be processed without running inference
#   --skip-existing     Skip dates that already have a completed *_inferfinal.gpkg
#   --force             Re-run even if output already exists (overrides --skip-existing)
#   --skip-errors       Continue batch even if a site fails
# =============================================================================

set -uo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
CONFIG="preset_seg_multi_NQOS_selvamask_SAM3_FT_quality"
TIF_PATTERN="*_rgb_model_ortho_smooth_100.tif"
OUTPUT_SUBDIR="canopyRS"
OUTPUT_ROOT=""       # when set, overrides OUTPUT_SUBDIR; outputs go to $OUTPUT_ROOT/$DATE/
MS_TIF_PATTERN=""    # when set, finds the MS TIF and passes -ms to infer.py
AOI_PATH=""
DATE_FILTER=""   # when set, only this YYYYMMDD date is processed
DRY_RUN=false
FORCE=false
SKIP_EXISTING=true
SKIP_ERRORS=false
DELETE_TILES=false
LOCAL_CACHE=false
LOCAL_CACHE_DIR="$HOME/data/_canopyrs_cache"
ENV_NAME="canopyrs_env"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --ms-pattern)  MS_TIF_PATTERN="$2"; shift 2 ;;
        --aoi) AOI_PATH="$2"; shift 2 ;;
        -d) DATE_FILTER="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --skip-existing) SKIP_EXISTING=true; shift ;;
        --force) FORCE=true; SKIP_EXISTING=false; shift ;;
        --skip-errors) SKIP_ERRORS=true; shift ;;
        --delete-tiles) DELETE_TILES=true; shift ;;
        --no-skip) SKIP_EXISTING=false; shift ;;
        --local-cache) LOCAL_CACHE=true; shift ;;
        --cache-dir) LOCAL_CACHE_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,35p' "$0" | sed 's/^# \?//'
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
if [ -n "$OUTPUT_ROOT" ]; then
    info "Output   : $OUTPUT_ROOT/<date>/"
else
    info "Output   : <date>/$OUTPUT_SUBDIR/"
fi
[ -n "$MS_TIF_PATTERN" ] && info "MS pattern: $MS_TIF_PATTERN (RGB+MS dual-stream)"
[ "$DRY_RUN" = true ]      && warn "DRY-RUN mode — no inference will be run"
[ "$SKIP_EXISTING" = true ] && info "Skip existing: ON (use --no-skip or --force to re-run)"
[ "$LOCAL_CACHE" = true ]   && info "Local cache: $LOCAL_CACHE_DIR (TIFs copied before inference)"
if [ -n "$AOI_PATH" ]; then
    [ -f "$AOI_PATH" ] || { echo "AOI file not found: $AOI_PATH"; exit 1; }
    info "AOI      : $AOI_PATH (edge artefacts will be excluded)"
fi
if [ -n "$OUTPUT_ROOT" ]; then
    mkdir -p "$OUTPUT_ROOT" || { echo "Cannot create output root: $OUTPUT_ROOT"; exit 1; }
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

# ── Stage detection ───────────────────────────────────────────────────────────
# Inspect an output directory and return which pipeline stage has completed.
#
# Output (newline-separated):
#   DONE                          — *_inferfinal.gpkg found; nothing left to do
#   STAGE2\n<gpkg>\n<tiles_dir>   — aggregated detections + tiles present; resume from segmenter
#   STAGE0\n<tiles_dir>           — tiles present; resume from detector
#   FRESH                         — no intermediate outputs; start from scratch
detect_stage() {
    local out_dir="$1"

    if find "$out_dir" -maxdepth 1 -name '*_inferfinal.gpkg' -quit 2>/dev/null | grep -q .; then
        echo "DONE"; return
    fi

    if [ -d "$out_dir/2_aggregator" ]; then
        local agg_gpkg
        agg_gpkg=$(find "$out_dir/2_aggregator" -maxdepth 1 -name '*.gpkg' \
            ! -name '*notaggregated*' 2>/dev/null | sort | head -1)
        if [ -n "$agg_gpkg" ]; then
            local tiles_dir
            tiles_dir=$(find "$out_dir/0_tilerizer" -maxdepth 2 -type d -name 'tiles' \
                2>/dev/null | head -1)
            if [ -n "$tiles_dir" ] && [ -d "$tiles_dir" ]; then
                local coco_json
                coco_json=$(find "$out_dir/2_aggregator" -maxdepth 1 -name '*.json' \
                    2>/dev/null | head -1)
                [ -z "$coco_json" ] && coco_json=$(find "$out_dir/1_detector" \
                    -maxdepth 1 -name '*.json' 2>/dev/null | head -1)
                if [ -n "$coco_json" ]; then
                    echo "STAGE2"; echo "$agg_gpkg"; echo "$tiles_dir"; echo "$coco_json"; return
                fi
            fi
        fi
    fi

    if [ -d "$out_dir/0_tilerizer" ]; then
        local tiles_dir
        tiles_dir=$(find "$out_dir/0_tilerizer" -maxdepth 2 -type d -name 'tiles' \
            2>/dev/null | head -1)
        if [ -n "$tiles_dir" ] && [ -d "$tiles_dir" ]; then
            echo "STAGE0"; echo "$tiles_dir"; return
        fi
    fi

    echo "FRESH"
}

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
    if [ -n "$DATE_FILTER" ] && [ "$DATE" != "$DATE_FILTER" ]; then
        continue
    fi
    EXPORTS_DIR="$DATE_DIR/exports"
    if [ -n "$OUTPUT_ROOT" ]; then
        OUTPUT_DIR="$OUTPUT_ROOT/$DATE"
    else
        OUTPUT_DIR="$DATE_DIR/$OUTPUT_SUBDIR"
    fi

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

    # Detect pipeline stage (handles skip + resume in one pass)
    mapfile -t STAGE_RESULT < <(detect_stage "$OUTPUT_DIR")
    STAGE="${STAGE_RESULT[0]:-FRESH}"

    if [ "$STAGE" = "DONE" ]; then
        if [ "$SKIP_EXISTING" = true ] && [ "$FORCE" = false ]; then
            warn "Already complete — skipping. Use --force to re-run."
            echo "[$DATE] SKIP: already done" >> "$LOG_FILE"
            COUNT_SKIP=$((COUNT_SKIP + 1)); continue
        fi
    fi

    # Dry-run stops here
    if [ "$DRY_RUN" = true ]; then
        case "$STAGE" in
            DONE)    info "DRY-RUN: already complete → $OUTPUT_DIR" ;;
            STAGE2)  info "DRY-RUN: would resume from stage 2 (aggregated detections) → $OUTPUT_DIR" ;;
            STAGE0)  info "DRY-RUN: would resume from stage 0 (existing tiles) → $OUTPUT_DIR" ;;
            FRESH)   info "DRY-RUN: would run from scratch → $OUTPUT_DIR" ;;
        esac
        echo "[$DATE] DRY-RUN: $TIF_NAME ($STAGE)" >> "$LOG_FILE"
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

    # Build resume args based on detected stage
    RESUME_ARGS=()
    case "$STAGE" in
        STAGE2)
            AGG_GPKG="${STAGE_RESULT[1]}"
            TILES_DIR="${STAGE_RESULT[2]}"
            COCO_JSON="${STAGE_RESULT[3]}"
            RESUME_ARGS=( --resume-from-gpkg "$AGG_GPKG" -t "$TILES_DIR" --input-coco "$COCO_JSON" )
            info "Resuming from stage 2 — aggregated detections: $(basename "$AGG_GPKG")"
            info "  Tiles: $TILES_DIR"
            info "  COCO: $(basename "$COCO_JSON")"
            ;;
        STAGE0)
            TILES_DIR="${STAGE_RESULT[1]}"
            RESUME_ARGS=( -t "$TILES_DIR" )
            info "Resuming from stage 0 — existing tiles: $TILES_DIR"
            ;;
        DONE|FRESH) ;;
    esac

    # Run inference
    info "Running inference → output: $OUTPUT_DIR"
    START_TIME=$(date +%s)

    # Build AOI argument (optional)
    AOI_ARGS=()
    [ -n "$AOI_PATH" ] && AOI_ARGS=(-aoi "$AOI_PATH")

    # Build MS argument (optional — requires --ms-pattern)
    MS_ARGS=()
    if [ -n "$MS_TIF_PATTERN" ]; then
        MS_TIF=$(find "$EXPORTS_DIR" -maxdepth 1 -name "$MS_TIF_PATTERN" | head -1)
        if [ -n "$MS_TIF" ]; then
            MS_ARGS=(-ms "$MS_TIF")
            info "MS TIF: $(basename "$MS_TIF")"
        else
            warn "No MS TIF matching '$MS_TIF_PATTERN' in exports/ — running without MS"
        fi
    fi

    DELETE_TILES_ARGS=()
    [ "$DELETE_TILES" = true ] && DELETE_TILES_ARGS=(--delete-tiles)

    if "$PYTHON" "$REPO_DIR/infer.py" \
            -c "$CONFIG" \
            -i "$INFER_TIF" \
            -o "$OUTPUT_DIR" \
            "${AOI_ARGS[@]}" \
            "${MS_ARGS[@]}" \
            "${DELETE_TILES_ARGS[@]}" \
            "${RESUME_ARGS[@]}" \
            2>&1 | tee -a "$LOG_FILE"; then
        END_TIME=$(date +%s)
        ELAPSED=$(( END_TIME - START_TIME ))
        ELAPSED_FMT=$(printf '%dm%02ds' $((ELAPSED/60)) $((ELAPSED%60)))
        success "Done in $ELAPSED_FMT → $OUTPUT_DIR/results.gpkg"
        echo "[$DATE] OK: $TIF_NAME → $OUTPUT_DIR ($ELAPSED_FMT, resume=$STAGE)" >> "$LOG_FILE"
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
