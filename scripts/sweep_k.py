"""Sensitivity of pairwise Hough to the neighbourhood size k, across outlier ratios.

Uses the HPatches match caches built by eval_hpatches.py (run that first).
Output: results/sweep_k.csv  (det, k, outlier_frac, acc@1/3/5, median ms)

Run:  .venv/bin/python scripts/sweep_k.py --limit 20
"""
import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_hpatches import inject_outliers  # noqa: E402
from hough_sift import hough as Hg  # noqa: E402
from hough_sift import verify as V  # noqa: E402
from hough_sift.hpatches import corner_error, pairs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20, help="number of scenes (x5 pairs)")
    ap.add_argument("--ks", type=int, nargs="*", default=[8, 16, 32])
    ap.add_argument("--fracs", type=float, nargs="*", default=[0, 0.5, 0.8, 0.9, 0.95])
    args = ap.parse_args()

    # load images only to get shapes + GT; keep just what we need in RAM
    items = []
    for n, (scene, k, im1, imk, H) in enumerate(pairs()):
        if n >= 5 * args.limit:
            break
        items.append((scene, k, im1.shape, imk.shape, H))

    rows = []
    for det in ["sift", "disk"]:
        cache = pickle.loads((ROOT / "data" / "cache" / f"hpatches_{det}.pkl").read_bytes())
        for frac in args.fracs:
            for K in args.ks:
                errs, ts = [], []
                for scene, k, sh1, sh2, H in items:
                    f1, f2, m = cache[(scene, k)]
                    f1.shape, f2.shape = sh1, sh2
                    f1, f2, m = inject_outliers(f1, f2, m, frac,
                                                np.random.default_rng(hash((scene, k)) % 2**32))
                    p1, p2 = f1.xy[m[:, 0]].astype(float), f2.xy[m[:, 1]].astype(float)
                    t = time.perf_counter()
                    cl = Hg.hough_clusters(*Hg.votes_pairs(f1, f2, m, k=K), sh1)
                    Hest, _ = V.verify_clusters(p1, p2, cl)
                    ts.append(time.perf_counter() - t)
                    errs.append(corner_error(Hest, H, sh1))
                e = np.array(errs)
                r = dict(det=det.upper(), k=K, outlier_frac=frac,
                         acc1=100 * np.mean(e < 1), acc3=100 * np.mean(e < 3), acc5=100 * np.mean(e < 5),
                         ms=1000 * np.median(ts))
                rows.append(r)
                print(f"{r['det']} k={K:2d} outliers={frac:4.2f}  @3px={r['acc3']:5.1f}%  {r['ms']:7.1f} ms",
                      flush=True)
        del cache
    out = ROOT / "results" / "sweep_k.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
