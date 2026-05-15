"""
tools/temporal_crown_matcher.py — CanopyRS temporal crown matcher.

Aligns tree-crown polygon GeoPackages produced by independent per-date
CanopyRS inference runs, assigns a stable ``crown_id`` to each tree across
the time series, computes temporal confidence metrics, and writes:

  * ``reference_crowns_{SITE}.gpkg``   — one consensus polygon per crown
  * ``{SITE}_{DATE}_matched.gpkg``     — per-date original polygons with
                                         ``crown_id`` joined back

The script is intentionally self-contained: it inlines its own geometry
helpers and imports only standard scientific packages already present in
``canopyrs_env`` (geopandas, pandas, shapely).  No canopyrs pipeline imports.

Usage
-----
::

    python tools/temporal_crown_matcher.py \\
        --site PFY \\
        --dates 20240823 20241015 \\
        -i /path/20240823.gpkg /path/20241015.gpkg \\
        -o /mnt/m/.../Pfynwald/temporal/ \\
        [--match-metric ioa|iou]       # default: ioa
        [--ioa-threshold 0.2]          # IoA threshold (primary metric)
        [--iou-threshold 0.3]          # IoU threshold (if --match-metric iou)
        [--centroid-distance 2.0]      # fallback: max centroid dist (m)
        [--min-dates 2] \\
        [--geometry-method best_score|union|largest]

Output columns — reference_crowns_{SITE}.gpkg
----------------------------------------------
crown_id              str     Site-prefixed stable ID, e.g. ``PFY_000042``
geometry              Polygon Consensus polygon (configurable method)
observation_count     int     Number of distinct dates the crown was detected
date_coverage         str     Comma-separated sorted list of dates observed
temporal_confidence   float   run_agreement x (1 - merge_penalty)  [0, 1]
run_agreement         float   Intersection/Union across all observations
merge_ambiguity       bool    True when >1 polygon from the same date ended
                              up in the same cluster (potential over-merge)
mean_segmenter_score  float   Mean of best available score column (optional)

Output columns added to per-date matched GDFs
----------------------------------------------
crown_id              str     Matched crown_id (None for unmatched)
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union


# =============================================================================
# Minimal geometric helpers (inlined — no dependency on pipeline internals)
# =============================================================================

class _UnionFind:
    """Minimal union-find / disjoint-set with path compression and union-by-rank."""

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path compression
            x = self._parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1


def _iou(geom_a, geom_b) -> float:
    """Return the intersection-over-union of two Shapely geometries."""
    try:
        inter = geom_a.intersection(geom_b).area
        if inter == 0.0:
            return 0.0
        union = geom_a.union(geom_b).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def _ioa(geom_a, geom_b) -> float:
    """Return intersection over area of the *smaller* polygon.

    Unlike IoU, this scores a fully-contained leaf-off crown at 1.0 even when
    the summer crown is much larger — better suited for phenological matching.
    """
    try:
        inter = geom_a.intersection(geom_b).area
        if inter == 0.0:
            return 0.0
        min_area = min(geom_a.area, geom_b.area)
        return inter / min_area if min_area > 0 else 0.0
    except Exception:
        return 0.0


def _run_agreement(geoms: list) -> float:
    """
    Return intersection.area / union.area across all geometries in a cluster.

    This measures how consistently the same spatial extent was detected across
    dates: 1.0 = perfect overlap, 0.0 = no shared intersection.
    """
    if len(geoms) == 1:
        return 1.0
    try:
        intersection = geoms[0]
        for g in geoms[1:]:
            intersection = intersection.intersection(g)
            if intersection.is_empty:
                return 0.0
        union_area = unary_union(geoms).area
        return float(intersection.area / union_area) if union_area > 0 else 0.0
    except Exception:
        return 0.0


def _best_score_col(gdf: gpd.GeoDataFrame) -> Optional[str]:
    """
    Return the most informative score column present in the GDF.

    Priority: aggregator_score > segmenter_score > detector_score.
    Returns None if no score column is found.
    """
    for col in ("aggregator_score", "segmenter_score", "detector_score"):
        if col in gdf.columns and gdf[col].notna().any():
            return col
    return None


# =============================================================================
# Core matching logic
# =============================================================================

def match_crowns(
    gdfs: List[gpd.GeoDataFrame],
    dates: List[str],
    iou_threshold: float = 0.3,
    ioa_threshold: float = 0.2,
    match_metric: str = "ioa",
    centroid_distance_m: float = 2.0,
    max_date_gap: int = 1,
    min_dates: int = 1,
    geometry_method: str = "best_score",
    site: str = "SITE",
) -> Tuple[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame]]:
    """
    Match tree crowns across multiple per-date GeoDataFrames.

    Algorithm
    ---------
    1. Tag each GDF with ``_date``, ``_run_index``, ``_orig_row``.
    2. Concatenate into a flat GDF; reproject to EPSG:3857 if CRS is geographic.
    3. Pass 1 — per date-pair greedy 1-to-1 matching: for each pair of dates
       whose positional gap in the sorted date list is <= ``max_date_gap``,
       build a candidate list of polygon pairs above the overlap threshold,
       sort by score descending, and claim greedily so each polygon is matched
       to at most one partner per date.  Matched pairs are linked via Union-Find.
       Enforcing the gap constraint means a crown that goes undetected for more
       than ``max_date_gap`` dates cannot be silently merged back — a new
       detection at the same location will receive a fresh ``crown_id``.
    4. Pass 2 — centroid fallback: polygons still unmatched (cluster spans only
       one date) are linked to the nearest unmatched polygon from a different
       date within both ``centroid_distance_m`` AND ``max_date_gap``.
    5. Collect connected components (clusters).
    6. For each cluster:
       - Compute consensus geometry (``geometry_method``).
       - Detect merge ambiguity (>1 polygon from the same date → potential
         tree split/merge between flights).
       - Compute ``temporal_confidence = run_agreement × (1 − merge_penalty)``.
       - Assign ``crown_id = {site}_{counter:06d}``.
    7. Write consensus rows into ``reference_gdf`` (filtered by ``min_dates``).
    8. Join crown_id back to original per-date GDFs.

    Parameters
    ----------
    gdfs:
        One GeoDataFrame per date, in the same order as ``dates``.
    dates:
        Date strings for each GDF (e.g. ``["20240823", "20241015"]``).
    iou_threshold:
        Minimum IoU threshold (used when ``match_metric='iou'``). Default 0.3.
    ioa_threshold:
        Minimum IoA threshold (used when ``match_metric='ioa'``). Default 0.2.
    match_metric:
        ``'ioa'`` (default) — intersection over area of the smaller polygon;
        handles leaf-on/off size differences well.
        ``'iou'`` — classic intersection over union.
    centroid_distance_m:
        Maximum centroid-to-centroid distance (metres) for the fallback pass.
        Polygons still unmatched after the primary metric pass are linked if
        centroids are within this distance. Set to 0 to disable. Default 2.0.
    max_date_gap:
        Maximum positional gap (in the sorted date list) between two dates
        that are allowed to be matched.  ``1`` (default) means only consecutive
        dates can be linked — if a crown disappears it cannot reappear and be
        matched to a later date.  ``2`` allows one missed date (e.g. a failed
        flight or low-quality raster).
    min_dates:
        Minimum observation count for a crown to appear in the reference output.
        Default 1 (include all crowns, even singletons).
    geometry_method:
        ``"best_score"`` — polygon with the highest score column value;
        ``"largest"``    — largest polygon across all dates;
        ``"union"``      — unary union of all matching polygons.
    site:
        Short site code prepended to crown_id (e.g. ``"PFY"`` → ``"PFY_000042"``).

    Returns
    -------
    reference_gdf:
        One consensus polygon per crown where observation_count >= min_dates.
    matched_per_date:
        Dict mapping each date string to the original GDF augmented with a
        ``crown_id`` column.  All polygons receive a crown_id regardless of
        whether the crown passes the ``min_dates`` filter.
    """
    if len(gdfs) != len(dates):
        raise ValueError(
            f"Number of GDFs ({len(gdfs)}) must equal number of dates ({len(dates)})"
        )
    if not gdfs:
        raise ValueError("No GDFs provided.")

    # ── 1. Tag each GDF ───────────────────────────────────────────────────────
    tagged_parts: List[gpd.GeoDataFrame] = []
    for run_idx, (gdf, date) in enumerate(zip(gdfs, dates)):
        g = gdf.copy().reset_index(drop=True)
        g["_date"] = date
        g["_run_index"] = run_idx
        g["_orig_row"] = range(len(g))
        tagged_parts.append(g)

    flat = gpd.GeoDataFrame(pd.concat(tagged_parts, ignore_index=True))
    original_crs = gdfs[0].crs

    # ── 2. Reproject to metric CRS ────────────────────────────────────────────
    needs_reproject = (original_crs is not None) and original_crs.is_geographic
    if needs_reproject:
        flat = flat.to_crs(epsg=3857)

    geometries = list(flat.geometry)
    run_indices = flat["_run_index"].tolist()
    n = len(flat)

    # ── 3 & 4. Greedy 1-to-1 matching per date-pair (pass 1) ─────────────────
    # Process each pair of distinct dates independently and do a greedy
    # highest-score-first 1-to-1 assignment.  This prevents the Union-Find
    # chaining problem in dense forests: adjacent trees that share a small edge
    # overlap cannot "steal" a match already claimed by a better-scoring pair.
    from shapely.strtree import STRtree

    uf = _UnionFind(n)

    threshold = ioa_threshold if match_metric == "ioa" else iou_threshold
    metric_fn = _ioa if match_metric == "ioa" else _iou

    # Group flat indices by run (date)
    run_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx in range(n):
        run_to_indices[run_indices[idx]].append(idx)

    unique_runs = sorted(run_to_indices.keys())
    for pos_i, ri in enumerate(unique_runs):
        for pos_j, rj in enumerate(unique_runs):
            if pos_j <= pos_i:
                continue
            if pos_j - pos_i > max_date_gap:
                continue  # enforce temporal gap constraint
            indices_i = run_to_indices[ri]
            indices_j = run_to_indices[rj]
            geoms_j = [geometries[k] for k in indices_j]
            tree_j = STRtree(geoms_j)

            # Collect all candidate pairs above threshold
            pairs: list = []
            for local_i, global_i in enumerate(indices_i):
                geom_i = geometries[global_i]
                for local_j in tree_j.query(geom_i):
                    local_j = int(local_j)
                    global_j = indices_j[local_j]
                    score = metric_fn(geom_i, geometries[global_j])
                    if score >= threshold:
                        pairs.append((score, local_i, local_j, global_i, global_j))

            # Greedy 1-to-1: highest score first; each polygon claimed at most once
            pairs.sort(reverse=True)
            claimed_i: set = set()
            claimed_j: set = set()
            for _score, local_i, local_j, global_i, global_j in pairs:
                if local_i in claimed_i or local_j in claimed_j:
                    continue
                uf.union(global_i, global_j)
                claimed_i.add(local_i)
                claimed_j.add(local_j)

    # ── 4b. Centroid-distance fallback (pass 2) ───────────────────────────────
    # Polygons whose cluster still spans only one date are candidates.
    # Greedy one-to-one nearest-neighbour: each polygon is linked to at most ONE
    # partner from a different date.  Pairs are sorted by distance and claimed
    # greedily so chains cannot form (unlike a free Union-Find pass).
    if centroid_distance_m > 0:
        clusters_p1: Dict[int, list] = defaultdict(list)
        for idx in range(n):
            clusters_p1[uf.find(idx)].append(idx)
        unmatched_idx: List[int] = [
            m
            for members in clusters_p1.values()
            if len({run_indices[m] for m in members}) == 1
            for m in members
        ]
        if unmatched_idx:
            centroids = [geometries[i].centroid for i in unmatched_idx]
            centroid_tree = STRtree(centroids)
            local_to_global = list(unmatched_idx)

            # Collect all valid cross-date pairs within the distance budget
            pairs = []
            for local_i, global_i in enumerate(local_to_global):
                centroid_i = centroids[local_i]
                for local_j in centroid_tree.query(centroid_i.buffer(centroid_distance_m)):
                    local_j = int(local_j)
                    if local_j <= local_i:
                        continue
                    global_j = local_to_global[local_j]
                    if run_indices[global_j] == run_indices[global_i]:
                        continue
                    if abs(run_indices[global_i] - run_indices[global_j]) > max_date_gap:
                        continue  # enforce temporal gap constraint in fallback too
                    dist = centroid_i.distance(centroids[local_j])
                    if dist <= centroid_distance_m:
                        pairs.append((dist, local_i, local_j, global_i, global_j))

            # Greedy assignment: closest pair first; each polygon matched at most once
            pairs.sort()
            claimed: set = set()
            for _dist, local_i, local_j, global_i, global_j in pairs:
                if local_i in claimed or local_j in claimed:
                    continue
                uf.union(global_i, global_j)
                claimed.add(local_i)
                claimed.add(local_j)

    # ── 5. Collect connected components ───────────────────────────────────────
    clusters: dict = defaultdict(list)
    for idx in range(n):
        clusters[uf.find(idx)].append(idx)

    score_col = _best_score_col(flat)

    # ── 6. Build consensus rows ────────────────────────────────────────────────
    reference_rows: list = []
    flat_idx_to_crown: Dict[int, str] = {}
    crown_counter = 0

    for _root, members in clusters.items():
        cluster_df = flat.iloc[members].copy()
        distinct_dates = sorted(cluster_df["_date"].unique().tolist())
        obs_count = len(distinct_dates)

        # Merge ambiguity: >1 polygon from the same date ended up in one cluster
        # (suggests two trees were merged into one in that date's segmentation,
        # or the same tree was detected twice).
        date_counts = cluster_df["_date"].value_counts()
        merge_ambiguity = bool((date_counts > 1).any())
        # Penalty proportional to fraction of "extra" polygons beyond 1 per date
        merge_penalty = (
            float((date_counts - 1).clip(lower=0).sum() / len(members))
            if merge_ambiguity else 0.0
        )

        # Consensus geometry
        cluster_geoms = list(cluster_df.geometry)
        if (
            geometry_method == "best_score"
            and score_col
            and score_col in cluster_df.columns
        ):
            best_loc = cluster_df[score_col].idxmax()
            consensus_geom = cluster_df.loc[best_loc, "geometry"]
        elif geometry_method == "largest":
            consensus_geom = max(cluster_geoms, key=lambda g: g.area)
        else:  # "union"
            consensus_geom = unary_union(cluster_geoms)

        # Temporal confidence: high when polygons agree well AND no merge issues
        agreement = _run_agreement(cluster_geoms)
        temporal_confidence = agreement * (1.0 - merge_penalty)

        # Mean score across all observations
        mean_score: Optional[float] = None
        if score_col and score_col in cluster_df.columns:
            valid_scores = cluster_df[score_col].dropna()
            if not valid_scores.empty:
                mean_score = round(float(valid_scores.mean()), 4)

        crown_id = f"{site}_{crown_counter:06d}"
        crown_counter += 1

        # Every member gets a crown_id so per-date files have full coverage
        for flat_i in members:
            flat_idx_to_crown[flat_i] = crown_id

        # Only include in reference if observation threshold is met
        if obs_count >= min_dates:
            row: dict = {
                "crown_id": crown_id,
                "geometry": consensus_geom,
                "observation_count": obs_count,
                "date_coverage": ",".join(distinct_dates),
                "temporal_confidence": round(temporal_confidence, 4),
                "run_agreement": round(agreement, 4),
                "merge_ambiguity": merge_ambiguity,
            }
            if mean_score is not None:
                row["mean_segmenter_score"] = mean_score
            reference_rows.append(row)

    # ── 7. Build reference GDF ────────────────────────────────────────────────
    if reference_rows:
        ref_gdf = gpd.GeoDataFrame(reference_rows, geometry="geometry")
        if needs_reproject and original_crs is not None:
            ref_gdf = ref_gdf.set_crs(epsg=3857).to_crs(original_crs)
        elif original_crs is not None:
            ref_gdf = ref_gdf.set_crs(original_crs)
    else:
        ref_gdf = gpd.GeoDataFrame(columns=["crown_id", "geometry"])
        if original_crs is not None:
            ref_gdf = ref_gdf.set_crs(original_crs)

    # ── 8. Join crown_id back to per-date original GDFs ──────────────────────
    flat["_crown_id"] = [flat_idx_to_crown.get(i) for i in flat.index]

    matched_per_date: Dict[str, gpd.GeoDataFrame] = {}
    for run_idx, (orig_gdf, date) in enumerate(zip(gdfs, dates)):
        date_rows = flat[flat["_run_index"] == run_idx]
        orig_row_to_crown = dict(zip(date_rows["_orig_row"], date_rows["_crown_id"]))
        result = orig_gdf.copy().reset_index(drop=True)
        result["crown_id"] = [orig_row_to_crown.get(i) for i in range(len(result))]
        matched_per_date[date] = result

    return ref_gdf, matched_per_date


# =============================================================================
# I/O helpers
# =============================================================================

def _save_gdf(gdf: gpd.GeoDataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GPKG")
    print(f"  Saved {len(gdf):,} features -> {path}  [{label}]")


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="temporal_crown_matcher",
        description=(
            "Match CanopyRS crown polygons across multiple overflight dates, "
            "assign stable crown_id values, and write reference + per-date outputs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--discover", metavar="DIR", default=None,
        help=(
            "Auto-discover dates by scanning DIR for YYYYMMDD/ subfolders, "
            "then finding the *_inferfinal.gpkg inside each. "
            "Mutually exclusive with -i / --dates. "
            "E.g. --discover /mnt/m/.../tmp"
        ),
    )
    p.add_argument(
        "-i", "--input", dest="inputs", nargs="+", required=False,
        metavar="GPKG",
        help="One GeoPackage per date, in the same order as --dates. Required unless --discover is used.",
    )
    p.add_argument(
        "--dates", nargs="+", required=False, metavar="YYYYMMDD",
        help=(
            "Date string for each input GeoPackage, same order as -i. "
            "E.g. --dates 20240823 20241015. Required unless --discover is used."
        ),
    )
    p.add_argument(
        "-o", "--output", dest="output_dir", required=True, metavar="DIR",
        help="Output directory. Created if it does not exist.",
    )
    p.add_argument(
        "--site", default="SITE", metavar="CODE",
        help=(
            "Short site code prepended to crown_id "
            "(e.g. PFY -> PFY_000042). Default: SITE"
        ),
    )
    p.add_argument(
        "--match-metric",
        choices=["ioa", "iou"],
        default="ioa",
        help=(
            "Primary overlap metric: "
            "ioa (intersection over area of the smaller polygon, default) or "
            "iou (intersection over union). "
            "IoA handles leaf-on/off size changes better than IoU."
        ),
    )
    p.add_argument(
        "--ioa-threshold", type=float, default=0.2,
        help=(
            "Minimum IoA for two polygons (from different dates) to be "
            "considered the same tree. Used when --match-metric=ioa. Default: 0.2"
        ),
    )
    p.add_argument(
        "--iou-threshold", type=float, default=0.3,
        help=(
            "Minimum IoU for two polygons (from different dates) to be "
            "considered the same tree. Used when --match-metric=iou. Default: 0.3"
        ),
    )
    p.add_argument(
        "--centroid-distance", type=float, default=2.0,
        help=(
            "Maximum centroid-to-centroid distance (metres) for the fallback "
            "matching pass. Polygons still unmatched after the primary metric "
            "pass are linked if centroids are within this distance. "
            "Set to 0 to disable. Default: 2.0"
        ),
    )
    p.add_argument(
        "--max-date-gap", type=int, default=1,
        help=(
            "Maximum positional gap between two dates (in the sorted date list) "
            "that are allowed to be matched. "
            "1 (default) = consecutive dates only: a crown that goes undetected "
            "cannot reappear and be merged with a later detection. "
            "2 = one missed date tolerated (e.g. failed flight). "
            "Set to a large number to match all date pairs (original behaviour)."
        ),
    )
    p.add_argument(
        "--min-dates", type=int, default=1,
        help=(
            "Minimum number of dates a crown must appear in to be included "
            "in the reference_crowns output. Default: 1 (all crowns included)."
        ),
    )
    p.add_argument(
        "--geometry-method",
        choices=["best_score", "union", "largest"],
        default="best_score",
        help=(
            "Consensus polygon selection method: "
            "best_score (polygon with highest segmenter/aggregator score), "
            "union (unary union of all matching polygons), "
            "or largest (largest polygon). Default: best_score"
        ),
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)

    # ── Resolve inputs/dates (manual list or auto-discovery) ────────────────
    if args.discover is not None:
        if args.inputs or args.dates:
            parser.error("--discover is mutually exclusive with -i / --dates")
        discover_root = Path(args.discover)
        if not discover_root.is_dir():
            parser.error(f"--discover: directory not found: {discover_root}")
        import re
        date_dirs = sorted(
            d for d in discover_root.iterdir()
            if d.is_dir() and re.fullmatch(r"[0-9]{8}", d.name)
        )
        if not date_dirs:
            parser.error(f"--discover: no YYYYMMDD subdirs found in {discover_root}")
        dates_found: List[str] = []
        inputs_found: List[Path] = []
        missing: List[str] = []
        for d in date_dirs:
            gpkg_files = sorted(d.glob("*_inferfinal.gpkg"))
            if not gpkg_files:
                print(f"  [discover] {d.name}: no *_inferfinal.gpkg — skipping", file=sys.stderr)
                missing.append(d.name)
                continue
            dates_found.append(d.name)
            inputs_found.append(gpkg_files[0])
            print(f"  [discover] {d.name}: {gpkg_files[0].name}")
        if not inputs_found:
            print("ERROR: --discover found no usable GeoPackages.", file=sys.stderr)
            return 1
        if missing:
            print(f"  [discover] Skipped {len(missing)} date(s) with no output: {', '.join(missing)}", file=sys.stderr)
        inputs = inputs_found
        dates = dates_found
    else:
        if not args.inputs or not args.dates:
            parser.error("-i / --dates are required unless --discover is used")
        inputs = [Path(p) for p in args.inputs]
        dates = args.dates

    if len(inputs) != len(dates):
        parser.error(
            f"Number of inputs ({len(inputs)}) must equal "
            f"number of dates ({len(dates)})"
        )

    # Load GeoPackages
    print(f"\nLoading {len(inputs)} GeoPackage(s)...")
    gdfs: List[gpd.GeoDataFrame] = []
    for path, date in zip(inputs, dates):
        if not path.exists():
            print(f"  ERROR: file not found - {path}", file=sys.stderr)
            return 1
        gdf = gpd.read_file(path)
        print(f"  {date}: {len(gdf):,} crowns  ({path.name})")
        gdfs.append(gdf)

    # Run matching
    active_threshold = args.ioa_threshold if args.match_metric == "ioa" else args.iou_threshold
    print(
        f"\nMatching crowns across {len(dates)} dates "
        f"(metric={args.match_metric} >= {active_threshold}, "
        f"centroid_fallback={args.centroid_distance}m, "
        f"max_date_gap={args.max_date_gap}, "
        f"min_dates={args.min_dates}, geometry={args.geometry_method}) ..."
    )
    ref_gdf, matched_per_date = match_crowns(
        gdfs=gdfs,
        dates=dates,
        iou_threshold=args.iou_threshold,
        ioa_threshold=args.ioa_threshold,
        match_metric=args.match_metric,
        centroid_distance_m=args.centroid_distance,
        max_date_gap=args.max_date_gap,
        min_dates=args.min_dates,
        geometry_method=args.geometry_method,
        site=args.site,
    )

    # Summary
    total = len(ref_gdf)
    if total > 0 and "observation_count" in ref_gdf.columns:
        multi = int((ref_gdf["observation_count"] > 1).sum())
        all_d = int((ref_gdf["observation_count"] == len(dates)).sum())
        amb = int(ref_gdf.get("merge_ambiguity", pd.Series([False] * total)).sum())
        print(f"\nReference crowns     : {total:,} total")
        print(f"  Seen in all {len(dates)} dates : {all_d:,}")
        print(f"  Seen in >1 date    : {multi:,}")
        print(f"  Merge ambiguity    : {amb:,}  (flag for manual review)")
    else:
        print(f"\nReference crowns: {total:,} total")

    # Save
    print("\nWriting outputs...")
    _save_gdf(
        ref_gdf,
        output_dir / f"reference_crowns_{args.site}.gpkg",
        "reference crowns",
    )
    for date, matched_gdf in matched_per_date.items():
        _save_gdf(
            matched_gdf,
            output_dir / f"{args.site}_{date}_matched.gpkg",
            f"matched {date}",
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
