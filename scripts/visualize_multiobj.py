"""Qualitative figure: hypotheses found by plain Hough vs semantic Hough on COCO scenes.

Each recovered hypothesis is drawn in its own colour (inlier matches as lines
between the two images). Title shows how many hypotheses were correct / false.

Run:  .venv/bin/python scripts/visualize_multiobj.py --n 6 --det disk
Output: results/figures/qual_<imgid>.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_hpatches import inject_outliers  # noqa: E402
from eval_multiobject import coco_scenes, load_scene_cache, make_scene, run_methods, score  # noqa: E402

HYP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def draw(ax, img1, img2, p1, p2, hyps, title):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas = np.full((max(h1, h2), w1 + w2 + 10, 3), 255, np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1 + 10:] = img2
    ax.imshow(canvas)
    for ci, (H, inl) in enumerate(hyps[:8]):
        col = HYP_COLORS[ci]
        sel = inl if len(inl) <= 60 else np.random.default_rng(0).choice(inl, 60, replace=False)
        for i in sel:
            ax.plot([p1[i, 0], p2[i, 0] + w1 + 10], [p1[i, 1], p2[i, 1]], color=col, lw=0.6, alpha=0.8)
        ax.scatter(p1[inl, 0], p1[inl, 1], s=4, color=col)
        ax.scatter(p2[inl, 0] + w1 + 10, p2[inl, 1], s=4, color=col)
    ax.set_title(title, fontsize=10, loc="left", color="#0b0b0b")
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--det", default="disk")
    ap.add_argument("--outlier-frac", type=float, default=0.8)
    ap.add_argument("--only", type=int, nargs="*", default=None, help="COCO image ids to render")
    args = ap.parse_args()
    out = ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for iid, img, masks, rng in coco_scenes(args.n):
        img2, Hs, label1, _ = make_scene(img, masks, rng)
        if args.only and iid not in args.only:
            continue
        c = load_scene_cache(iid, img, img2, [args.det])
        seg1, seg2 = c["seg"]
        f1, f2, m = c[args.det]
        n_real = len(m)
        f1.shape, f2.shape = img.shape, img2.shape
        f1, f2, m = inject_outliers(f1, f2, m, args.outlier_frac, np.random.default_rng(iid))
        res, p1 = run_methods(args.det, img, f1, f2, m, seg1, seg2)
        p2 = f2.xy[m[:, 1]].astype(float)
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        for ax, name, nice in zip(axes, ["Hough", "Sem+Hough+M"], ["Plain Hough", "Semantic Hough + merge (ours)"]):
            hyps = res[name][0]
            sc = score(hyps, p1, Hs, label1, np.random.default_rng(0), n_real=n_real)
            n_ok = sc["n_hyp"] - sc["n_false"]
            draw(ax, img, img2, p1, p2, hyps,
                 f"{nice}: {n_ok} correct, {sc['n_false']} false hypotheses "
                 f"(purity {100 * sc['purity']:.0f}%), {len(Hs) - 1} moving objects")
        fig.tight_layout()
        fig.savefig(out / f"qual_{iid}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        a = score(res["Hough"][0], p1, Hs, label1, np.random.default_rng(0), n_real=n_real)
        b = score(res["Sem+Hough+M"][0], p1, Hs, label1, np.random.default_rng(0), n_real=n_real)
        print(f"qual_{iid}.png  objs={len(Hs)-1}  plain: recall={a['obj_recall']:.2f} wrong={a['n_wrong']}"
              f"  ours: recall={b['obj_recall']:.2f} wrong={b['n_wrong']}", flush=True)


if __name__ == "__main__":
    main()
