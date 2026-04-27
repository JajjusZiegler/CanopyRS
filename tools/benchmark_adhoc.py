"""
Ad-hoc raster-level benchmark script for CanopyRS.

Runs one or more pipeline presets on an orthomosaic and evaluates each against
a ground-truth GeoPackage.  Results are saved per-pipeline and compared in a
summary table.

─── Single pipeline (reuse existing run) ─────────────────────────────────────
    conda run -n canopyrs_env python tools/benchmark_adhoc.py \
        --preds   '/mnt/m/.../canopyRS_rgb' \
        --truth   '/mnt/m/.../martelloskop_20250807.gpkg' \
        --aoi     '/mnt/m/.../AOI_Marteloskop.gpkg' \
        --output  /tmp/benchmark_marteloskop

─── Run + evaluate multiple presets ──────────────────────────────────────────
    conda run -n canopyrs_env python tools/benchmark_adhoc.py \
        --input   '/mnt/m/.../20250807_Martelloskop_rgb_model_ortho_smooth_100.tif' \
        --truth   '/mnt/m/.../martelloskop_20250807.gpkg' \
        --aoi     '/mnt/m/.../AOI_Marteloskop.gpkg' \
        --output  /tmp/benchmark_marteloskop \
        --presets preset_seg_multi_NQOS_selvamask_SAM3_FT_quality \
                  preset_seg_multi_NQOS_selvamask_SAM3_FT_fast \
                  preset_seg_multi_NQOS_SAM2

─── MS pipeline (needs both RGB + MS ortho) ──────────────────────────────────
    conda run -n canopyrs_env python tools/benchmark_adhoc.py \
        --input   '/mnt/m/.../20250807_Martelloskop_rgb_model_ortho_smooth_100.tif' \
        --ms      '/mnt/m/.../20250807_Martelloskop_multispec_ortho_100cm.tif' \
        --truth   '/mnt/m/.../martelloskop_20250807.gpkg' \
        --aoi     '/mnt/m/.../AOI_Marteloskop.gpkg' \
        --output  /tmp/benchmark_marteloskop \
        --presets preset_seg_multi_NQOS_selvamask_SAM3_FT_quality_ms_ndvi_altumpt

Optional flags:
    --iou-thresholds 0.25 0.5 0.75   (default: 0.25 0.5 0.75)
    --iou-type       segm | bbox     (default: segm)
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

import pandas as pd

# Ensure repo root is on path when run directly
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from canopyrs.engine.benchmark.base.evaluator import CocoEvaluator
from canopyrs.engine.config_parsers import PipelineConfig, InferIOConfig
from canopyrs.engine.config_parsers.base import get_config_path
from canopyrs.engine.pipeline import Pipeline


# ─── Local copy helper ───────────────────────────────────────────────────────

def copy_local(src: Path, local_dir: Path) -> Path:
    """Copy *src* to *local_dir* and return the local path (skips if already there)."""
    local_dir.mkdir(parents=True, exist_ok=True)
    dst = local_dir / src.name
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        print(f"  [copy-local] Already cached: {dst}")
        return dst
    print(f"  [copy-local] Copying {src.name} ({src.stat().st_size / 1e9:.2f} GB) → {dst}")
    shutil.copy2(src, dst)
    print(f"  [copy-local] Done.")
    return dst


# ─── Inference ────────────────────────────────────────────────────────────────

def run_infer(
    input_tif: Path,
    preset: str,
    output_folder: Path,
    ms_tif: Path | None = None,
    aoi_gpkg: Path | None = None,
) -> Path:
    """Run a pipeline preset and return the path to the output predictions .gpkg."""
    print(f"\n  Running inference: {preset}")
    yaml_name = preset if preset.endswith(".yaml") else preset + ".yaml"
    pipeline_config = PipelineConfig.from_yaml(get_config_path(yaml_name))

    config_args = dict(
        input_imagery=str(input_tif),
        tiles_path=None,
        output_folder=str(output_folder),
    )
    if ms_tif:
        config_args["multispectral_imagery"] = str(ms_tif)
    if aoi_gpkg:
        config_args["aoi_config"] = "package"
        config_args["aoi"] = str(aoi_gpkg)

    io_config = InferIOConfig(**config_args)
    pipeline = Pipeline.from_config(io_config, pipeline_config)
    pipeline.run(strict_rgb_validation=(ms_tif is None))

    return _find_preds_gpkg(output_folder)


def _find_preds_gpkg(folder: Path) -> Path:
    """Return the last (deepest-stage) non-AOI .gpkg under folder."""
    gpkg_files = sorted(folder.rglob("*.gpkg"))
    gpkg_files = [f for f in gpkg_files if "aoi" not in f.name]
    if not gpkg_files:
        raise FileNotFoundError(
            f"No prediction .gpkg found under {folder}.\n"
            "Make sure inference completed successfully."
        )
    # Last aggregator output is typically the deepest-numbered subfolder
    return gpkg_files[-1]


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(
    label: str,
    preds_gpkg: Path,
    truth_gpkg: Path,
    aoi_gpkg: Path | None,
    iou_thresholds: list[float],
    iou_type: str,
    output_folder: Path,
) -> dict:
    print(f"\n  Evaluating: {label}")
    print(f"    preds : {preds_gpkg}")
    print(f"    truth : {truth_gpkg}")
    print(f"    aoi   : {aoi_gpkg or 'full extent'}")

    metrics = CocoEvaluator.raster_level_multi_iou_thresholds(
        iou_type=iou_type,
        preds_gpkg_path=str(preds_gpkg),
        truth_gpkg_path=str(truth_gpkg),
        aoi_gpkg_path=str(aoi_gpkg) if aoi_gpkg else None,
        iou_thresholds=iou_thresholds,
    )
    metrics["label"] = label
    metrics["preds_gpkg"] = str(preds_gpkg)

    output_folder.mkdir(parents=True, exist_ok=True)
    with open(output_folder / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, cls=_NumpyEncoder)

    return metrics


# ─── Summary table ────────────────────────────────────────────────────────────

KEY_METRICS = ["precision", "recall", "f1", "num_preds", "num_truths"]

def print_summary(all_results: list[dict]):
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    rows = []
    for r in all_results:
        row = {"pipeline": r["label"]}
        for k in KEY_METRICS:
            if k in r:
                v = r[k]
                row[k] = f"{v:.4f}" if isinstance(v, float) else v
        rows.append(row)
    df = pd.DataFrame(rows).set_index("pipeline")
    print(df.to_string())
    print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ad-hoc multi-pipeline benchmark for CanopyRS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Inputs
    parser.add_argument("--input", help="RGB orthomosaic .tif (required unless --preds is used)")
    parser.add_argument("--ms", default=None, help="Co-registered MS orthomosaic .tif (optional)")
    parser.add_argument(
        "--copy-local", metavar="DIR", default=None,
        help=(
            "Copy --input (and --ms if provided) to this local directory before "
            "running inference. Avoids network-mount read errors on large TIFs. "
            "Example: --copy-local /tmp/canopyrs_inputs"
        ),
    )
    parser.add_argument("--truth", required=True, help="Ground-truth crowns .gpkg")
    parser.add_argument("--aoi", default=None, help="AOI .gpkg (optional but recommended)")
    parser.add_argument("--output", default="/tmp/canopyrs_benchmark", help="Root output folder")

    # Pipeline selection
    parser.add_argument(
        "--presets", nargs="+",
        default=["preset_seg_multi_NQOS_selvamask_SAM3_FT_quality"],
        help="One or more pipeline preset names to run and compare",
    )

    # Skip-infer: point directly at an existing predictions folder or gpkg
    parser.add_argument(
        "--preds", default=None,
        help=(
            "Path to an existing predictions folder (from a previous run) OR "
            "directly to a .gpkg file. Skips inference entirely. "
            "When used with --presets, the label will be taken from the preset name."
        ),
    )

    # Eval options
    parser.add_argument("--iou-thresholds", nargs="+", type=float,
                        default=[0.25, 0.5, 0.75])
    parser.add_argument("--iou-type", default="segm", choices=["segm", "bbox"])
    parser.add_argument(
        "--append", action="store_true",
        help=(
            "Append new results to an existing benchmark_summary.json in --output "
            "instead of overwriting it. Useful to accumulate multiple pipeline runs."
        ),
    )
    args = parser.parse_args()

    truth_gpkg = Path(args.truth)
    aoi_gpkg = Path(args.aoi) if args.aoi else None
    output_root = Path(args.output)

    if not truth_gpkg.exists():
        parser.error(f"Ground truth not found: {truth_gpkg}")
    if aoi_gpkg and not aoi_gpkg.exists():
        parser.error(f"AOI not found: {aoi_gpkg}")

    # Load existing results if appending
    summary_path = output_root / "benchmark_summary.json"
    if args.append and summary_path.exists():
        with open(summary_path) as f:
            all_results = json.load(f)
        print(f"Appending to existing summary with {len(all_results)} result(s): {summary_path}")
    else:
        all_results = []

    # ── Mode A: --preds supplied → skip inference, evaluate directly ──────────
    if args.preds:
        preds_path = Path(args.preds)
        if preds_path.suffix == ".gpkg":
            preds_gpkg = preds_path
        else:
            preds_gpkg = _find_preds_gpkg(preds_path)

        label = args.presets[0] if args.presets else preds_path.name
        result = evaluate(
            label=label,
            preds_gpkg=preds_gpkg,
            truth_gpkg=truth_gpkg,
            aoi_gpkg=aoi_gpkg,
            iou_thresholds=args.iou_thresholds,
            iou_type=args.iou_type,
            output_folder=output_root / label,
        )
        all_results.append(result)

    # ── Mode B: run one or more presets ───────────────────────────────────────
    else:
        if not args.input:
            parser.error("--input is required when --preds is not supplied.")
        input_tif = Path(args.input)
        if not input_tif.exists():
            parser.error(f"Input raster not found: {input_tif}")
        ms_tif = Path(args.ms) if args.ms else None
        if ms_tif and not ms_tif.exists():
            parser.error(f"MS raster not found: {ms_tif}")

        if args.copy_local:
            local_dir = Path(args.copy_local)
            input_tif = copy_local(input_tif, local_dir)
            if ms_tif:
                ms_tif = copy_local(ms_tif, local_dir)

        for preset in args.presets:
            infer_folder = output_root / preset / "infer"
            preds_gpkg = run_infer(
                input_tif=input_tif,
                preset=preset,
                output_folder=infer_folder,
                ms_tif=ms_tif,
                aoi_gpkg=aoi_gpkg,
            )
            result = evaluate(
                label=preset,
                preds_gpkg=preds_gpkg,
                truth_gpkg=truth_gpkg,
                aoi_gpkg=aoi_gpkg,
                iou_thresholds=args.iou_thresholds,
                iou_type=args.iou_type,
                output_folder=output_root / preset,
            )
            all_results.append(result)

    # ── Save combined summary ─────────────────────────────────────────────────
    output_root.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=_NumpyEncoder)

    print_summary(all_results)
    print(f"Full results saved to: {summary_path}")


if __name__ == "__main__":
    main()
