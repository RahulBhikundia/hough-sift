"""Evaluate all (detector x verifier) combos on HPatches.

Run:
  .venv/bin/python scripts/eval_hpatches.py                      # clean matches
  .venv/bin/python scripts/eval_hpatches.py --outlier-frac 0.8   # add fake matches so 80% are outliers
  .venv/bin/python scripts/eval_hpatches.py --limit 20           # quick run on 20 scenes

Matches are cached per detector in data/cache/, so re-runs only redo verification.
Results: results/hpatches_<tag>.csv (one row per pair x method) and printed summary.
"""
import argparse
import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hough_sift import pipeline as P  # noqa: E402
from hough_sift.features import Features  # noqa: E402
from hough_sift.hpatches import corner_error, pairs  # noqa: E402

METHODS = {  # name -> (detector, method, vote_mode)
    "SIFT+RANSAC": ("sift", "ransac", None),
    "SIFT+MAGSAC": ("sift", "magsac", None),
    "SIFT+Hough(single)": ("sift", "hough", "single"),
    "SIFT+Hough(pairs)": ("sift", "hough", "pairs"),
    "DISK+RANSAC": ("disk", "ransac", None),
    "DISK+MAGSAC": ("disk", "magsac", None),
    "DISK+Hough(pairs)": ("disk", "hough", "pairs"),
}


def inject_outliers(f1, f2, m, frac, rng):
    """Append random wrong correspondences so that `frac` of all matches are outliers.
    Fake keypoints get random positions, and (for SIFT) random scale/orientation."""
    if frac <= 0 or len(m) == 0:
        return f1, f2, m
    n_fake = int(round(len(m) * frac / (1 - frac)))

    def extend(f, shape):
        xy = rng.uniform([0, 0], [shape[1], shape[0]], (n_fake, 2)).astype(np.float32)
        sc = None if f.scale is None else np.concatenate([f.scale, rng.choice(f.scale, n_fake)])
        ori = None if f.ori is None else np.concatenate([f.ori, rng.uniform(0, 2 * np.pi, n_fake).astype(np.float32)])
        return Features(np.concatenate([f.xy, xy]), f.desc, sc, ori)

    n1, n2 = len(f1.xy), len(f2.xy)
    f1, f2 = extend(f1, f1.shape), extend(f2, f2.shape)
    fake = np.stack([n1 + np.arange(n_fake), n2 + np.arange(n_fake)], 1)
    return f1, f2, np.concatenate([m, fake])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outlier-frac", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0, help="only first N scenes")
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--thr", type=float, default=3.0)
    args = ap.parse_args()

    cache_dir = ROOT / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    dets = sorted({METHODS[k][0] for k in args.methods})
    caches = {}
    for d in dets:
        p = cache_dir / f"hpatches_{d}.pkl"
        caches[d] = pickle.loads(p.read_bytes()) if p.exists() else {}

    rng = np.random.default_rng(0)
    rows = []
    scenes_seen = set()
    for scene, k, im1, imk, H_gt in tqdm(pairs(), total=580):
        scenes_seen.add(scene)
        if args.limit and len(scenes_seen) > args.limit:
            break
        for d in dets:
            key = (scene, k)
            if key not in caches[d]:
                f1, f2, m = P.extract_and_match(im1, imk, d)
                caches[d][key] = (f1, f2, m)
        for name in args.methods:
            d, method, vm = METHODS[name]
            f1, f2, m = caches[d][(scene, k)]
            f1.shape, f2.shape = im1.shape, imk.shape
            f1, f2, m = inject_outliers(f1, f2, m, args.outlier_frac, np.random.default_rng(hash((scene, k)) % 2**32))
            r = P.estimate(im1, f1, f2, m, method, thr=args.thr, vote_mode=vm)
            rows.append(dict(scene=scene, k=k, type=scene[0], method=name,
                             err=corner_error(r["H"], H_gt, im1.shape),
                             n_matches=len(m), n_inliers=len(r["inliers"]), ms=1000 * r["time"]))
    for d in dets:
        (cache_dir / f"hpatches_{d}.pkl").write_bytes(pickle.dumps(caches[d]))

    tag = f"of{args.outlier_frac:g}" + (f"_lim{args.limit}" if args.limit else "")
    out = ROOT / "results" / f"hpatches_{tag}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarize(rows)
    print(f"\nsaved {out}")


def summarize(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    print(f"\n{'method':20} {'n':>4} | {'all @1/3/5px (%)':^20} | {'illum @3':>8} {'view @3':>8} | {'inl':>5} {'ms':>6}")
    for name, rs in by.items():
        e = np.array([r["err"] for r in rs])
        t = np.array([r["type"] for r in rs])
        acc = lambda x, th: 100 * np.mean(x < th) if len(x) else float("nan")  # noqa: E731
        print(f"{name:20} {len(rs):4d} | {acc(e,1):5.1f} {acc(e,3):6.1f} {acc(e,5):6.1f}   | "
              f"{acc(e[t=='i'],3):8.1f} {acc(e[t=='v'],3):8.1f} | "
              f"{np.mean([r['n_inliers'] for r in rs]):5.0f} {np.median([r['ms'] for r in rs]):6.1f}")


if __name__ == "__main__":
    main()
