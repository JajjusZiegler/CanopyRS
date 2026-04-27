"""
Visualize CanopyRS benchmark results from benchmark_summary.json.

Usage:
    conda run -n canopyrs_env python tools/visualize_benchmark.py \
        --summary /tmp/benchmark_marteloskop/benchmark_summary.json \
        --output  /tmp/benchmark_marteloskop/benchmark_viz.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


SIZES = ["tiny", "small", "medium", "large", "giant"]
IOUs  = [0.25, 0.5, 0.75]
SIZE_COLORS = {
    "tiny":   "#e8a838",
    "small":  "#4c9be8",
    "medium": "#5cb85c",
    "large":  "#d9534f",
    "giant":  "#9b59b6",
}
IOu_COLORS = {0.25: "#5cb85c", 0.5: "#f0ad4e", 0.75: "#d9534f"}


def short_label(label: str) -> str:
    replacements = {
        "preset_seg_multi_NQOS_selvamask_SAM3_FT_quality": "SAM3 quality",
        "preset_seg_multi_NQOS_selvamask_SAM3_FT_fast":    "SAM3 fast",
        "preset_seg_multi_NQOS_SAM2":                      "SAM2",
        "preset_seg_multi_NQOS_unet_treecrown_quality":    "U-Net quality",
        "preset_seg_standalone_detectree2_rgb":             "Detectree2 RGB",
        "preset_seg_standalone_detectree2_ms":              "Detectree2 MS",
        "preset_seg_multi_NQOS_selvamask_SAM3_FT_quality_ms_ndvi_altumpt": "SAM3+NDVI (Altum-PT)",
        "preset_seg_multi_NQOS_selvamask_SAM3_FT_quality_ms_ndvi_mx_dual": "SAM3+NDVI (MX Dual)",
        "preset_seg_multi_NQOS_selvamask_SAM3_FT_quality_ensemble_altumpt": "SAM3 ensemble (Altum-PT)",
    }
    return replacements.get(label, label.replace("preset_", "").replace("_", " "))


def load(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def make_figure(results: list[dict]) -> plt.Figure:
    n = len(results)
    labels = [short_label(r["label"]) for r in results]
    x = np.arange(n)
    width = 0.25

    fig = plt.figure(figsize=(max(14, n * 3.5), 18))
    fig.suptitle("CanopyRS Benchmark — Marteloskop 20250807", fontsize=15, fontweight="bold", y=0.98)

    # ── 1. P / R / F1 at 3 IoU thresholds ─────────────────────────────────
    ax1 = fig.add_subplot(3, 2, 1)
    for i, iou in enumerate(IOUs):
        tag = "" if iou == 0.5 else f"_{int(iou*100)}" if iou != 0.5 else "_50"
        tag = f"_{int(iou*100)}"
        p  = [r.get(f"precision_{tag.lstrip('_')}", r["precision_per_iou"][i]) for r in results]
        r_ = [r.get(f"recall_{tag.lstrip('_')}",   r["recall_per_iou"][i])    for r in results]
        f1 = [r.get(f"f1_{tag.lstrip('_')}",        r["f1_per_iou"][i])        for r in results]
        offset = (i - 1) * width
        bars = ax1.bar(x + offset, f1, width, label=f"F1 @IoU{iou}", color=IOu_COLORS[iou], alpha=0.85)
        ax1.plot(x + offset, p,  "v", color=IOu_COLORS[iou], ms=6, label=f"P @IoU{iou}" if i == 0 else "")
        ax1.plot(x + offset, r_, "^", color=IOu_COLORS[iou], ms=6, label=f"R @IoU{iou}" if i == 0 else "")

    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylim(0, 1.05); ax1.set_ylabel("Score")
    ax1.set_title("Precision (▼) / Recall (▲) / F1 (bar) by IoU threshold")
    ax1.yaxis.grid(True, alpha=0.4)
    # legend for IoU colours only
    patches = [mpatches.Patch(color=IOu_COLORS[t], label=f"IoU {t}") for t in IOUs]
    ax1.legend(handles=patches, fontsize=8, loc="upper right")

    # ── 2. Detection counts ────────────────────────────────────────────────
    ax2 = fig.add_subplot(3, 2, 2)
    tp = [r["tp"]          for r in results]
    fp = [r["fp"]          for r in results]
    fn = [r["fn"]          for r in results]
    gt = [r["num_truths"]  for r in results]
    ax2.bar(x - width, tp, width, label="TP",       color="#5cb85c")
    ax2.bar(x,         fp, width, label="FP",       color="#d9534f")
    ax2.bar(x + width, fn, width, label="FN (miss)", color="#f0ad4e")
    for xi, g in zip(x, gt):
        ax2.axhline(g, xmin=(xi - width * 1.5) / (n + 1), xmax=(xi + width * 1.5) / (n + 1),
                    color="black", linewidth=1.5, linestyle="--")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("Count"); ax2.set_title("TP / FP / FN (dashed = GT count)")
    ax2.legend(fontsize=8); ax2.yaxis.grid(True, alpha=0.4)

    # ── 3. F1 @ IoU 0.5 per size class ────────────────────────────────────
    ax3 = fig.add_subplot(3, 2, 3)
    bar_width = 0.8 / len(SIZES)
    for si, sz in enumerate(SIZES):
        vals = [r.get(f"f1_{sz}", 0.0) for r in results]
        offset = (si - len(SIZES) / 2 + 0.5) * bar_width
        ax3.bar(x + offset, vals, bar_width,
                label=sz.capitalize(), color=SIZE_COLORS[sz], alpha=0.85)
    ax3.set_xticks(x); ax3.set_xticklabels(labels, rotation=20, ha="right")
    ax3.set_ylim(0, 1.05); ax3.set_ylabel("F1"); ax3.set_title("F1 (mean IoU) per size class")
    ax3.legend(fontsize=8); ax3.yaxis.grid(True, alpha=0.4)

    # ── 4. Precision per size class ────────────────────────────────────────
    ax4 = fig.add_subplot(3, 2, 4)
    for si, sz in enumerate(SIZES):
        vals = [r.get(f"precision_{sz}", 0.0) for r in results]
        offset = (si - len(SIZES) / 2 + 0.5) * bar_width
        ax4.bar(x + offset, vals, bar_width,
                label=sz.capitalize(), color=SIZE_COLORS[sz], alpha=0.85)
    ax4.set_xticks(x); ax4.set_xticklabels(labels, rotation=20, ha="right")
    ax4.set_ylim(0, 1.05); ax4.set_ylabel("Precision"); ax4.set_title("Precision per size class")
    ax4.legend(fontsize=8); ax4.yaxis.grid(True, alpha=0.4)

    # ── 5. Recall per size class ───────────────────────────────────────────
    ax5 = fig.add_subplot(3, 2, 5)
    for si, sz in enumerate(SIZES):
        vals = [r.get(f"recall_{sz}", 0.0) for r in results]
        offset = (si - len(SIZES) / 2 + 0.5) * bar_width
        ax5.bar(x + offset, vals, bar_width,
                label=sz.capitalize(), color=SIZE_COLORS[sz], alpha=0.85)
    ax5.set_xticks(x); ax5.set_xticklabels(labels, rotation=20, ha="right")
    ax5.set_ylim(0, 1.05); ax5.set_ylabel("Recall"); ax5.set_title("Recall per size class")
    ax5.legend(fontsize=8); ax5.yaxis.grid(True, alpha=0.4)

    # ── 6. Summary table ──────────────────────────────────────────────────
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.axis("off")
    col_labels = ["Pipeline", "P@50", "R@50", "F1@50", "P@75", "R@75", "F1@75", "#Pred", "#GT"]
    rows = []
    for r in results:
        rows.append([
            short_label(r["label"]),
            f"{r['precision_50']:.3f}",
            f"{r['recall_50']:.3f}",
            f"{r['f1_50']:.3f}",
            f"{r['precision_75']:.3f}",
            f"{r['recall_75']:.3f}",
            f"{r['f1_75']:.3f}",
            str(r["num_preds"]),
            str(r["num_truths"]),
        ])
    tbl = ax6.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)
    ax6.set_title("Summary table", pad=12)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize CanopyRS benchmark results.")
    parser.add_argument("--summary", default="/tmp/benchmark_marteloskop/benchmark_summary.json",
                        help="Path to benchmark_summary.json")
    parser.add_argument("--output", default=None,
                        help="Output PNG path (default: next to summary file)")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        sys.exit(f"Not found: {summary_path}")

    results = load(summary_path)
    print(f"Loaded {len(results)} pipeline result(s) from {summary_path}")

    fig = make_figure(results)

    out_path = Path(args.output) if args.output else summary_path.parent / "benchmark_viz.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
