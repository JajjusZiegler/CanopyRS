#!/usr/bin/env bash
# =============================================================================
# CanopyRS — Multi-temporal RGB batch inference (1cm orthomosaics)
#
# Scans a site directory for date subfolders, finds the 1cm RGB orthomosaic in
# each exports/ subfolder, and runs the high-resolution DINO+SAM3 pipeline.
# An AOI GeoPackage is strongly recommended to exclude edge artefacts that are
# common in drone orthomosaics.
#
# Usage:
#   bash batch_infer_rgb_temporal.sh [OPTIONS] <SITE_FOLDER>
#
# Examples:
#   # Run all dates for Marteloskop with AOI clipping
#   bash batch_infer_rgb_temporal.sh \
#       --aoi /mnt/m/.../Marteloskop/aoi_marteloskop.gpkg \
#       /mnt/m/.../Marteloskop
#
#   # Dry-run to see which dates would be processed
#   bash batch_infer_rgb_temporal.sh --dry-run /mnt/m/.../Pfynwald
#
#   # Use a different TIF pattern (e.g. non-smoothed mosaic)
#   bash batch_infer_rgb_temporal.sh \
#       -p "*_rgb_model_ortho_100.tif" \
#       --aoi /mnt/m/.../Pfynwald/aoi_pfynwald.gpkg \
#       /mnt/m/.../Pfynwald
#
# Options:
#   -c <config>       Pipeline config name
#                     (default: preset_seg_multi_NQOS_selvamask_SAM3_FT_quality_1cm)
#   -p <pattern>      Glob pattern to match the TIF (default: *_rgb_model_ortho_smooth_100.tif)
#   -o <subdir>       Output subfolder name under each date dir (default: canopyRS_rgb)
#   --aoi <path>      Path to site AOI GeoPackage — strongly recommended to exclude
#                     edge artefacts. Inference is restricted to this polygon.#   -d <YYYYMMDD>     Process only this single date folder (e.g. -d 20240815)#   --dry-run         Print what would be processed without running inference
#   --force           Re-run even if output already exists
#   --skip-errors     Continue batch even if a single date fails
#   --local-cache     Copy TIF to local NVMe before inference (faster on network drives)
#   --cache-dir <p>   Local cache directory (default: ~/data/_canopyrs_cache)
# =============================================================================

set -uo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
CONFIG="preset_seg_multi_NQOS_selvamask_SAM3_FT_quality_1cm"
TIF_PATTERN="*_rgb_model_ortho_smooth_100.tif"
OUTPUT_SUBDIR="canopyRS_rgb"
AOI_PATH=""
DATE_FILTER=""
DRY_RUN=false
FORCE=false
SKIP_EXISTING=true
SKIP_ERRORS=false
LOCAL_CACHE=false
LOCAL_CACHE_DIR="$HOME/data/_canopyrs_cache"
DELETE_TILES=false
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
        -c)           CONFIG="$2";            shift 2 ;;
        -p)           TIF_PATTERN="$2";       shift 2 ;;
        -o)           OUTPUT_SUBDIR="$2";     shift 2 ;;
        --aoi)        AOI_PATH="$2";          shift 2 ;;
        -d)           DATE_FILTER="$2";       shift 2 ;;
        --dry-run)    DRY_RUN=true;           shift ;;
        --force)      FORCE=true; SKIP_EXISTING=false; shift ;;
        --skip-errors) SKIP_ERRORS=true;      shift ;;
        --delete-tiles) DELETE_TILES=true;    shift ;;
        --local-cache) LOCAL_CACHE=true;      shift ;;
        --cache-dir)  LOCAL_CACHE_DIR="$2";   shift 2 ;;
        -h|--help)
            sed -n '3,36p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *)  SITE_FOLDER="$1"; shift ;;
    esac
done

[ -z "$SITE_FOLDER" ] && { echo "Usage: bash batch_infer_rgb_temporal.sh [OPTIONS] <SITE_FOLDER>"; exit 1; }
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
header "Multi-temporal RGB inference — $SITE_NAME"
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

# ── Filename sanitizer ───────────────────────────────────────────────────────
# geodataset requires product names matching ^[a-z0-9]+(?:_[a-z0-9]+)*$
# This function converts a filename stem to that format:
#   • transliterates non-ASCII (ü→u, ä→a, ö→o, é→e …) via iconv
#   • lowercases
#   • replaces any remaining non-alphanumeric chars with underscores
#   • collapses repeated underscores
sanitize_stem() {
    echo "$1" \
        | iconv -f UTF-8 -t ASCII//TRANSLIT 2>/dev/null \
        | tr '[:upper:]' '[:lower:]' \
        | sed 's/[^a-z0-9]/_/g; s/__*/_/g; s/^_//; s/_$//'
}

# ── Stage detection ───────────────────────────────────────────────────────────
# Inspect an output directory and return which pipeline stage has completed.
#
# Output (newline-separated):
#   DONE                               — *_inferfinal.gpkg found; nothing left to do
#   STAGE2\n<gpkg>\n<tiles_dir>\n<coco> — aggregated detections + tiles present; resume from segmenter
#   STAGE2_NOTILES\n<gpkg>\n<coco>     — aggregated detections exist but tiles deleted; re-tilerize then resume
#   STAGE0\n<tiles_dir>                — tiles present but no aggregated detections; resume from detector
#   FRESH                              — no intermediate outputs; start from scratch
#
# Stage 2 resumes are only possible when 2_aggregator/*.gpkg (non-notaggregated)
# AND 0_tilerizer/*/tiles/ both exist (tiles must not have been deleted).
detect_stage() {
    local out_dir="$1"

    # Complete: final GeoPackage exists in the output root
    if find "$out_dir" -maxdepth 1 -name '*_inferfinal.gpkg' -quit 2>/dev/null | grep -q .; then
        echo "DONE"; return
    fi

    # Stage 2: aggregated detections + tiles both present
    if [ -d "$out_dir/2_aggregator" ]; then
        local agg_gpkg
        agg_gpkg=$(find "$out_dir/2_aggregator" -maxdepth 1 -name '*.gpkg' \
            ! -name '*notaggregated*' 2>/dev/null | sort | head -1)
        if [ -n "$agg_gpkg" ]; then
            # Prefer COCO JSON from 2_aggregator; fall back to 1_detector
            local coco_json
            coco_json=$(find "$out_dir/2_aggregator" -maxdepth 1 -name '*.json' \
                2>/dev/null | head -1)
            [ -z "$coco_json" ] && coco_json=$(find "$out_dir/1_detector" \
                -maxdepth 1 -name '*.json' 2>/dev/null | head -1)
            if [ -n "$coco_json" ]; then
                local tiles_dir
                tiles_dir=$(find "$out_dir/0_tilerizer" -maxdepth 2 -type d -name 'tiles' \
                    2>/dev/null | head -1)
                if [ -n "$tiles_dir" ] && [ -d "$tiles_dir" ]; then
                    echo "STAGE2"; echo "$agg_gpkg"; echo "$tiles_dir"; echo "$coco_json"; return
                else
                    # Tiles deleted but detections intact — re-tilerize then resume
                    echo "STAGE2_NOTILES"; echo "$agg_gpkg"; echo "$coco_json"; return
                fi
            fi
        fi
    fi

    # Stage 0: tiles present but aggregated detections absent/tiles missing for stage2
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

# ── Counters & log ────────────────────────────────────────────────────────────
COUNT_OK=0; COUNT_SKIP=0; COUNT_FAIL=0; COUNT_NOTIF=0
LOG_FILE="/tmp/canopyrs_rgb_$(date +%Y%m%d_%H%M%S).log"
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

    mkdir -p "$OUTPUT_DIR"

    # Optional local cache copy (avoids repeated reads from network drives)
    INFER_TIF="$TIF"
    LOCAL_TIF=""
    SYMLINK_TIF=""  # symlink used when filename needs sanitizing
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

    # Sanitize TIF filename if it contains chars geodataset rejects
    # (non-ASCII, uppercase, hyphens). Create a symlink with a clean name.
    TIF_STEM="${TIF_NAME%.tif}"
    CLEAN_STEM=$(sanitize_stem "$TIF_STEM")
    if [ "$CLEAN_STEM" != "$TIF_STEM" ]; then
        SYMLINK_DIR="/tmp/canopyrs_symlinks"
        mkdir -p "$SYMLINK_DIR"
        SYMLINK_TIF="$SYMLINK_DIR/${CLEAN_STEM}.tif"
        ln -sf "$INFER_TIF" "$SYMLINK_TIF"
        info "Filename sanitized: '$TIF_STEM' → '$CLEAN_STEM' (symlink)"
        INFER_TIF="$SYMLINK_TIF"
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
        STAGE2_NOTILES)
            AGG_GPKG="${STAGE_RESULT[1]}"
            COCO_JSON="${STAGE_RESULT[2]}"
            RESUME_ARGS=( --resume-from-gpkg "$AGG_GPKG" --input-coco "$COCO_JSON" )
            info "Resuming from stage 2 (tiles deleted — will re-tilerize): $(basename "$AGG_GPKG")"
            info "  COCO: $(basename "$COCO_JSON")"
            ;;
        STAGE0)
            TILES_DIR="${STAGE_RESULT[1]}"
            RESUME_ARGS=( -t "$TILES_DIR" )
            info "Resuming from stage 0 — existing tiles: $TILES_DIR"
            ;;
        DONE|FRESH) ;;
    esac

    # Build command
    CMD=( "$PYTHON" "$REPO_DIR/infer.py" -c "$CONFIG" -i "$INFER_TIF" -o "$OUTPUT_DIR" )
    [ -n "$AOI_PATH" ]     && CMD+=( -aoi "$AOI_PATH" )
    [ "$DELETE_TILES" = true ] && CMD+=( --delete-tiles )
    CMD+=( "${RESUME_ARGS[@]}" )

    info "Running inference → $OUTPUT_DIR"
    START_TIME=$(date +%s)

    if "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
        ELAPSED=$(( $(date +%s) - START_TIME ))
        ELAPSED_FMT=$(printf '%dm%02ds' $((ELAPSED/60)) $((ELAPSED%60)))
        success "Done in $ELAPSED_FMT → $OUTPUT_DIR"
        echo "[$DATE] OK: $TIF_NAME ($ELAPSED_FMT, resume=$STAGE)" >> "$LOG_FILE"
        COUNT_OK=$((COUNT_OK + 1))
    else
        fail "Inference failed for $DATE"
        echo "[$DATE] FAIL: $TIF_NAME (stage=$STAGE)" >> "$LOG_FILE"
        COUNT_FAIL=$((COUNT_FAIL + 1))
        if [ "$SKIP_ERRORS" = false ]; then
            echo -e "\n${RED}Stopping. Use --skip-errors to continue on failure.${NC}"
            break
        fi
    fi

    # Clean up local cache and any sanitizing symlink
    if [ "$LOCAL_CACHE" = true ] && [ -n "$LOCAL_TIF" ] && [ -f "$LOCAL_TIF" ]; then
        rm -f "$LOCAL_TIF"
    fi
    if [ -n "$SYMLINK_TIF" ] && [ -L "$SYMLINK_TIF" ]; then
        rm -f "$SYMLINK_TIF"
        SYMLINK_TIF=""
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
