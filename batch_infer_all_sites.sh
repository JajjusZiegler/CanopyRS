#!/usr/bin/env bash
# =============================================================================
# CanopyRS — Run inference across ALL sites in a project root
#
# Loops over every site folder (e.g. Pfynwald, Davos_LWF, …) inside a given
# project root, calls batch_infer_rgb_temporal.sh for each one, and:
#   • auto-detects a site-level AOI.gpkg if present
#   • embeds the config name in the output subfolder so runs with different
#     settings never overwrite each other
#   • deletes intermediate tiles after each run (--delete-tiles)
#
# Output structure:
#   <date>/CanopyRS/<config_name>/   (e.g. 20240823/CanopyRS/preset_seg_multi_NQOS_selvamask_SAM3_FT_quality_1cm/)
#
# Usage:
#   bash batch_infer_all_sites.sh [OPTIONS] <PROJECT_ROOT>
#
# Examples:
#   # Dry-run — preview all sites and dates without running inference
#   bash batch_infer_all_sites.sh --dry-run \
#       /mnt/m/working_package_2/2024_dronecampaign/02_processing/metashape_projects/Upscale_Metashapeprojects
#
#   # Full run — all sites, auto AOI, delete tiles
#   bash batch_infer_all_sites.sh \
#       /mnt/m/working_package_2/2024_dronecampaign/02_processing/metashape_projects/Upscale_Metashapeprojects
#
#   # Run only specific sites (space-separated)
#   bash batch_infer_all_sites.sh --sites "Pfynwald Davos_LWF" \
#       /mnt/m/working_package_2/2024_dronecampaign/02_processing/metashape_projects/Upscale_Metashapeprojects
#
#   # Different pipeline config
#   bash batch_infer_all_sites.sh \
#       -c preset_seg_multi_NQOS_selvamask_SAM3_FT_quality \
#       /mnt/m/working_package_2/...
#
# Options:
#   -c <config>       CanopyRS pipeline config name
#                     (default: preset_seg_multi_NQOS_selvamask_SAM3_FT_quality_1cm)
#   -p <pattern>      Glob pattern to match the RGB TIF
#                     (default: *_rgb_model_ortho_smooth_100.tif)
#   --sites <list>    Space-separated list of site folder names to process
#                     (default: all site folders found in PROJECT_ROOT)
#   --aoi-name <f>    Filename to look for as site-level AOI (default: AOI.gpkg)
#   --no-delete-tiles Keep intermediate tile folders (deleted by default)
#   --dry-run         Preview without running inference
#   --force           Re-run even if output already exists
#   --skip-errors     Continue to next site if a site fails
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCH_SCRIPT="$SCRIPT_DIR/batch_infer_rgb_temporal.sh"

# ── Defaults ──────────────────────────────────────────────────────────────────
CONFIG="preset_seg_multi_NQOS_selvamask_SAM3_FT_quality_1cm"
TIF_PATTERN="*_rgb_model_ortho_smooth_100.tif"
AOI_FILENAME="AOI.gpkg"
SITE_FILTER=""       # empty = all sites
DRY_RUN=false
FORCE=false
SKIP_ERRORS=false
DELETE_TILES=true    # delete intermediate tiles by default

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()    { echo -e "${RED}[FAIL]${NC}  $*"; }
header()  { echo -e "\n${BOLD}${CYAN}======================================================================${NC}"; \
            echo -e "${BOLD}${CYAN}  $*${NC}"; \
            echo -e "${BOLD}${CYAN}======================================================================${NC}"; }

# ── Argument parsing ──────────────────────────────────────────────────────────
PROJECT_ROOT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c)              CONFIG="$2";          shift 2 ;;
        -p)              TIF_PATTERN="$2";     shift 2 ;;
        --sites)         SITE_FILTER="$2";     shift 2 ;;
        --aoi-name)      AOI_FILENAME="$2";    shift 2 ;;
        --no-delete-tiles) DELETE_TILES=false; shift ;;
        --dry-run)       DRY_RUN=true;         shift ;;
        --force)         FORCE=true;           shift ;;
        --skip-errors)   SKIP_ERRORS=true;     shift ;;
        -h|--help)
            sed -n '3,43p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *)  PROJECT_ROOT="$1"; shift ;;
    esac
done

[ -z "$PROJECT_ROOT" ] && { echo "Usage: bash batch_infer_all_sites.sh [OPTIONS] <PROJECT_ROOT>"; exit 1; }
[ -d "$PROJECT_ROOT" ] || { echo "Project root not found: $PROJECT_ROOT"; exit 1; }
[ -x "$BATCH_SCRIPT" ] || chmod +x "$BATCH_SCRIPT"

# Output subfolder name = config name (makes runs with different configs coexist)
OUTPUT_SUBDIR="CanopyRS/$CONFIG"

# ── Discover site folders ─────────────────────────────────────────────────────
header "CanopyRS All-Sites Batch"
info "Project root : $PROJECT_ROOT"
info "Config       : $CONFIG"
info "TIF pattern  : $TIF_PATTERN"
info "Output subdir: <date>/$OUTPUT_SUBDIR/"
info "AOI file     : $AOI_FILENAME (searched at site level)"
[ "$DELETE_TILES" = true ] && info "Delete tiles : YES (intermediate tile folders removed after inference)"
[ "$DRY_RUN"  = true ]     && warn "DRY-RUN mode — no inference will be run"
[ "$FORCE"    = true ]     && warn "FORCE mode — existing outputs will be overwritten"
echo ""

# Build site list
if [ -n "$SITE_FILTER" ]; then
    # User-supplied list
    read -ra ALL_SITES <<< "$SITE_FILTER"
    SITE_DIRS=()
    for s in "${ALL_SITES[@]}"; do
        d="$PROJECT_ROOT/$s"
        [ -d "$d" ] && SITE_DIRS+=("$d") || warn "Site not found, skipping: $d"
    done
else
    # All subdirs that contain at least one YYYYMMDD subfolder
    mapfile -t SITE_DIRS < <(
        find "$PROJECT_ROOT" -maxdepth 1 -mindepth 1 -type d | sort | while read -r d; do
            if find "$d" -maxdepth 1 -mindepth 1 -type d | grep -qE '/[0-9]{8}$'; then
                echo "$d"
            fi
        done
    )
fi

if [ ${#SITE_DIRS[@]} -eq 0 ]; then
    echo "No site folders with YYYYMMDD date subfolders found in $PROJECT_ROOT"
    exit 1
fi

info "Sites to process (${#SITE_DIRS[@]}):"
for d in "${SITE_DIRS[@]}"; do
    SITE_AOI="$d/$AOI_FILENAME"
    AOI_STATUS="no AOI — full ortho"
    [ -f "$SITE_AOI" ] && AOI_STATUS="AOI: $SITE_AOI"
    echo "    $(basename "$d")  ($AOI_STATUS)"
done
echo ""

# ── Global counters ───────────────────────────────────────────────────────────
TOTAL_OK=0
TOTAL_FAIL=0
MASTER_LOG="/tmp/canopyrs_all_sites_$(date +%Y%m%d_%H%M%S).log"
echo "Master log: $MASTER_LOG"
echo ""

# ── Per-site loop ─────────────────────────────────────────────────────────────
for SITE_DIR in "${SITE_DIRS[@]}"; do
    SITE_NAME=$(basename "$SITE_DIR")
    SITE_AOI="$SITE_DIR/$AOI_FILENAME"

    header "Site: $SITE_NAME"

    # Build batch_infer_rgb_temporal.sh arguments
    BATCH_ARGS=(
        -c "$CONFIG"
        -p "$TIF_PATTERN"
        -o "$OUTPUT_SUBDIR"
        --skip-errors       # let the per-date script handle date-level errors
    )

    [ -f "$SITE_AOI" ]         && BATCH_ARGS+=( --aoi "$SITE_AOI" )
    [ "$DRY_RUN"  = true ]     && BATCH_ARGS+=( --dry-run )
    [ "$FORCE"    = true ]     && BATCH_ARGS+=( --force )
    [ "$DELETE_TILES" = true ] && BATCH_ARGS+=( --delete-tiles )

    if [ -f "$SITE_AOI" ]; then
        info "AOI: $SITE_AOI"
    else
        warn "No $AOI_FILENAME found at site level — running on full orthomosaic"
    fi

    # Run
    if bash "$BATCH_SCRIPT" "${BATCH_ARGS[@]}" "$SITE_DIR" 2>&1 | tee -a "$MASTER_LOG"; then
        success "Site $SITE_NAME completed"
        echo "[SITE:$SITE_NAME] OK" >> "$MASTER_LOG"
        TOTAL_OK=$((TOTAL_OK + 1))
    else
        fail "Site $SITE_NAME had failures (see log)"
        echo "[SITE:$SITE_NAME] FAIL" >> "$MASTER_LOG"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        if [ "$SKIP_ERRORS" = false ]; then
            echo -e "\n${RED}Stopping. Use --skip-errors to continue on site failure.${NC}"
            break
        fi
    fi
    echo ""
done

# ── Final summary ─────────────────────────────────────────────────────────────
header "All-Sites Batch Complete"
echo -e "  ${GREEN}Sites OK     : $TOTAL_OK${NC}"
[ $TOTAL_FAIL -gt 0 ] && echo -e "  ${RED}Sites failed : $TOTAL_FAIL${NC}"
echo ""
echo -e "  Config used  : $CONFIG"
echo -e "  Output subdir: <date>/$OUTPUT_SUBDIR/"
echo -e "  Master log   : $MASTER_LOG"
echo ""

[ $TOTAL_FAIL -gt 0 ] && exit 1 || exit 0
