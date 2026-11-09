"""Report figures from the CSVs in results/.

  fig_outliers.png : accuracy@3px vs injected outlier ratio (SIFT | DISK panels)
  fig_ksweep.png   : pairwise-Hough sensitivity to neighbourhood size k
  fig_multiobj.png : multi-object benchmark (false hypotheses, object recall)

Run:  .venv/bin/python scripts/make_figures.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"

# palette: reference categorical slots 1-4 (fixed order) + chrome/ink tokens
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100"}
SEQ = ["#9ec5f4", "#3987e5", "#184f95"]          # sequential blue, light -> dark (k = 8, 16, 32)
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

# method -> (colour, marker, label). Colour follows the method, in every panel.
STYLE = {
    "Hough(pairs)": (C["blue"], "o", "Hough, pairwise (ours)"),
    "MAGSAC": (C["orange"], "s", "MAGSAC"),
    "RANSAC": (C["aqua"], "^", "RANSAC"),
    "Hough(single)": (C["yellow"], "D", "Hough, single (Lowe)"),
}


def style_axes(ax, title):
    ax.set_facecolor(SURF)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def read(name):
    p = RES / name
    return list(csv.DictReader(open(p))) if p.exists() else None


def direct_label(ax, xs, ys, text, color):
    ax.annotate(text, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8.5, color=INK2)
    del color  # text wears ink tokens, the marker carries identity


def fig_outliers():
    fracs = [0, 0.5, 0.8, 0.9, 0.95]
    acc = defaultdict(dict)                              # (det, method) -> frac -> acc3
    for f in fracs:
        rows = read(f"hpatches_of{f:g}.csv")
        if rows is None:
            print(f"missing hpatches_of{f:g}.csv - skipping fig_outliers")
            return
        by = defaultdict(list)
        for r in rows:
            by[r["method"]].append(float(r["err"]))
        for m, e in by.items():
            det, meth = m.split("+")
            acc[(det, meth)][f] = 100 * np.mean(np.array(e) < 3)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, facecolor=SURF)
    for ax, det in zip(axes, ["SIFT", "DISK"]):
        style_axes(ax, f"{det} features")
        for meth in ["RANSAC", "MAGSAC", "Hough(single)", "Hough(pairs)"]:
            if (det, meth) not in acc:
                continue
            col, mk, lab = STYLE[meth]
            ys = [acc[(det, meth)][f] for f in fracs]
            xs = list(range(len(fracs)))          # evenly spaced: 90/95 would collide otherwise
            ax.plot(xs, ys, color=col, marker=mk, markersize=7, linewidth=2,
                    markeredgecolor=SURF, markeredgewidth=1.5, label=lab, zorder=3)
            direct_label(ax, xs, ys, lab.split(",")[0] if meth != "Hough(pairs)" else "ours", col)
        ax.set_xticks(range(len(fracs)))
        ax.set_xticklabels([f"{100 * f:g}" for f in fracs])
        ax.set_xlim(-0.2, len(fracs) - 1 + 0.9)
        ax.set_ylim(0, 102)
        ax.set_xlabel("injected outliers (%)", color=INK2, fontsize=9)
    axes[0].set_ylabel("pairs with corner error < 3 px (%)", color=INK2, fontsize=9)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("HPatches homography accuracy under outlier contamination (580 pairs)",
                 color=INK, fontsize=12, y=1.09, x=0.02, ha="left")
    save(fig, "fig_outliers.png")


def fig_ksweep():
    rows = read("sweep_k.csv")
    if rows is None:
        print("missing sweep_k.csv - skipping")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, facecolor=SURF)
    for ax, det in zip(axes, ["SIFT", "DISK"]):
        style_axes(ax, f"{det} + pairwise Hough")
        for col, k in zip(SEQ, [8, 16, 32]):
            rs = sorted([r for r in rows if r["det"] == det and int(r["k"]) == k],
                        key=lambda r: float(r["outlier_frac"]))
            xs = list(range(len(rs)))
            ys = [float(r["acc3"]) for r in rs]
            ticks = [f"{100 * float(r['outlier_frac']):g}" for r in rs]
            ax.plot(xs, ys, color=col, marker="o", markersize=7, linewidth=2,
                    markeredgecolor=SURF, markeredgewidth=1.5, label=f"k = {k}", zorder=3)
            if det == "DISK":                      # SIFT lines nearly coincide; legend suffices
                direct_label(ax, xs, ys, f"k={k}", col)
        ax.set_xticks(xs)
        ax.set_xticklabels(ticks)
        ax.set_xlim(-0.2, len(xs) - 1 + 0.7)
        ax.set_ylim(0, 102)
        ax.set_xlabel("injected outliers (%)", color=INK2, fontsize=9)
    axes[0].set_ylabel("pairs with corner error < 3 px (%)", color=INK2, fontsize=9)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Neighbourhood size k: robustness vs speed (HPatches, 100 pairs)",
                 color=INK, fontsize=12, y=1.09, x=0.02, ha="left")
    save(fig, "fig_ksweep.png")


def fig_multiobj():
    """Multi-object test set: recall and wrong hypotheses vs injected outliers (DISK | SIFT)."""
    fracs = [0, 0.5, 0.8]
    data = {}
    for f in fracs:
        rows = read(f"multiobj_test_of{f:g}.csv")
        if rows is None:
            print(f"missing multiobj_test_of{f:g}.csv - skipping fig_multiobj")
            return
        data[f] = rows
    # colour follows the method; fixed slot order
    meth = [("RANSAC", C["aqua"], "^", "RANSAC (sequential)"),
            ("Hough", C["yellow"], "D", "Hough"),
            ("Sem+RANSAC+M", C["orange"], "s", "Semantic + RANSAC + merge"),
            ("Sem+Hough+M", C["blue"], "o", "Semantic + Hough + merge (ours)")]
    metrics = [("obj_recall", "objects recovered (%)", 100), ("n_wrong", "wrong hypotheses per image", 1)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), facecolor=SURF)
    for i, det in enumerate(["DISK", "SIFT"]):
        for j, (key, ylab, sc) in enumerate(metrics):
            ax = axes[j, i]
            style_axes(ax, f"{det}: {ylab}")
            for mname, col, mk, lab in meth:
                ys = [sc * np.nanmean([float(r[key]) for r in data[f]
                                       if r["det"] == det and r["method"] == mname]) for f in fracs]
                xs = [100 * f for f in fracs]
                ax.plot(xs, ys, color=col, marker=mk, markersize=7, linewidth=2,
                        markeredgecolor=SURF, markeredgewidth=1.5, label=lab, zorder=3)
            ax.set_xticks([100 * f for f in fracs])
            ax.set_xlim(-5, 85)
            ax.set_ylim(0, 100 if sc == 100 else None)
            if j == 1:
                ax.set_xlabel("injected outliers (%)", color=INK2, fontsize=9)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Multi-object scenes from COCO (150 test images, 2-3 independently moving objects)",
                 color=INK, fontsize=12, x=0.02, ha="left", y=1.05)
    fig.tight_layout()
    save(fig, "fig_multiobj.png")


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print(f"wrote {FIG / name}")


if __name__ == "__main__":
    fig_outliers()
    fig_ksweep()
    fig_multiobj()
