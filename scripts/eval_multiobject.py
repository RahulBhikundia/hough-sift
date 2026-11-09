"""Multi-object benchmark built from COCO val2017 (tests the SEMANTIC idea).

For each COCO image with >= 2 suitable objects:
  * background moves by a random camera homography H0
  * 2-3 objects (cut out with COCO ground-truth masks) additionally move by their
    OWN random rotation/scale/translation:  H_i = T_i @ H0
  * holes left behind are inpainted, moved objects are pasted on top
So every pixel of image 1 has a known motion -> we know which matches are correct
and which homography each object should get.

Task: recover ALL motions (multi-model fitting). Each method returns a list of
hypotheses (H, inlier matches). A hypothesis is CORRECT if it agrees with the true
motion of its majority label (mean error < 5 px on that region) and that motion
was not already claimed. Everything else is a FALSE hypothesis - typically a
cross-object cluster mixing matches of different objects.

Semantic priors come from Mask R-CNN run on the images (NOT the COCO GT masks).

Run:  .venv/bin/python scripts/eval_multiobject.py --n 150
"""
import argparse
import csv
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hough_sift import hough as Hg  # noqa: E402
from hough_sift import pipeline as P  # noqa: E402
from hough_sift import semantic as S  # noqa: E402
from hough_sift import verify as V  # noqa: E402

COCO_DIR = ROOT / "data" / "coco"


# ------------------------------------------------------------------ scene synthesis
def warp_mask(mask, H, shape):
    return cv2.warpPerspective(mask.astype(np.float32), H, (shape[1], shape[0])) > 0.5


def make_scene(img, masks, rng):
    h, w = img.shape[:2]
    # camera motion (background)
    A = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-10, 10), rng.uniform(0.9, 1.1))
    A[:, 2] += rng.uniform(-0.05, 0.05, 2) * [w, h]
    H0 = np.vstack([A, [0, 0, 1]])
    img2 = cv2.warpPerspective(img, H0, (w, h))
    hole = np.zeros((h, w), bool)
    for m in masks:
        hole |= warp_mask(m, H0, img.shape)
    hole = cv2.dilate(hole.astype(np.uint8) * 255, np.ones((7, 7), np.uint8), iterations=2)
    img2 = cv2.inpaint(img2, hole, 5, cv2.INPAINT_TELEA)

    Hs, vis2 = [H0], np.zeros((h, w), np.int32)
    vis2[hole > 0] = -1                       # inpainted: no valid background here
    for i, m in enumerate(masks, start=1):
        ys, xs = np.nonzero(m)
        c = H0 @ [xs.mean(), ys.mean(), 1]
        c = c[:2] / c[2]
        for _ in range(20):                   # keep the moved object mostly in frame
            ang = rng.uniform(5, 20) * rng.choice([-1, 1])
            T = np.vstack([cv2.getRotationMatrix2D(tuple(c), ang, rng.uniform(0.85, 1.15)), [0, 0, 1]])
            d = rng.uniform(0.1, 0.25) * max(h, w)
            phi = rng.uniform(0, 2 * np.pi)
            T[:2, 2] += d * np.array([np.cos(phi), np.sin(phi)])
            Hi = T @ H0
            mw = warp_mask(m, Hi, img.shape)
            if mw.sum() > 0.7 * m.sum():
                break
        ow = cv2.warpPerspective(img, Hi, (w, h))
        img2[mw] = ow[mw]
        vis2[mw] = i
        Hs.append(Hi)
    label1 = np.zeros((h, w), np.int32)
    for i, m in enumerate(masks, start=1):
        label1[m] = i
    return img2, Hs, label1, vis2


def coco_scenes(n, seed=0, offset=0):
    """offset: skip the first `offset` eligible images (validation split = offset 150)."""
    from pycocotools.coco import COCO
    coco = COCO(str(COCO_DIR / "annotations" / "instances_val2017.json"))
    rng = np.random.default_rng(seed)
    ids = sorted(coco.getImgIds())
    rng.shuffle(ids)
    out, skipped = 0, 0
    for iid in ids:
        info = coco.loadImgs(iid)[0]
        area = info["height"] * info["width"]
        anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=iid, iscrowd=False))
                if 0.03 * area < a["area"] < 0.25 * area]
        if len(anns) < 2:
            continue
        anns = sorted(anns, key=lambda a: -a["area"])[:3]
        masks = [coco.annToMask(a).astype(bool) for a in anns]
        # drop objects that overlap each other heavily (ambiguous ground truth)
        if any((masks[i] & masks[j]).sum() > 0.1 * min(masks[i].sum(), masks[j].sum())
               for i in range(len(masks)) for j in range(i)):
            continue
        if skipped < offset:
            skipped += 1
            continue
        img = cv2.cvtColor(cv2.imread(str(COCO_DIR / "val2017" / info["file_name"])), cv2.COLOR_BGR2RGB)
        yield iid, img, masks, np.random.default_rng(iid)
        out += 1
        if out >= n:
            return


# ------------------------------------------------------------------ multi-model fitting
def ransac_multi(p1, p2, idx_pool=None, max_models=8, min_inl=8):
    rem = np.arange(len(p1)) if idx_pool is None else np.asarray(idx_pool)
    hyps = []
    while len(rem) >= min_inl and len(hyps) < max_models:
        H, inl = V.ransac(p1[rem], p2[rem])
        if H is None or len(inl) < min_inl:
            break
        hyps.append((H, rem[inl]))
        rem = np.delete(rem, inl)
    return hyps


def hough_multi(p1, p2, clusters, match_group=None, max_models=8, min_inl=8, top=300):
    """Greedy multi-model extraction from Hough clusters (largest first).
    top=300: with semantic gating the 5th dimension splits background 'stuff'
    instances into many large clusters, so small objects can rank far down;
    an earlier top=60 cut-off silently dropped them (diagnosed: 69% of lost objects).
    Clusters with < 5 still-unclaimed matches are skipped cheaply."""
    used = np.zeros(len(p1), bool)
    hyps = []
    for cl in clusters[:top]:
        cand = ~used
        if match_group is not None:          # gated: inliers only from the cluster's instance group
            cand &= match_group == match_group[cl[0]]
        if cand[cl].sum() < 5:
            continue
        H, inl = V.refine_cluster(p1, p2, cl, cand=cand)
        if H is None or len(inl) < min_inl:
            continue
        hyps.append((H, inl))
        used[inl] = True
        if len(hyps) >= max_models:
            break
    return hyps


# ------------------------------------------------------------------ post-processing ("+V")
def merge_duplicates(p1, p2, hyps, thr=5.0):
    """Merge a hypothesis into an earlier one if the earlier homography already
    explains its inliers (median transfer error < thr px). Standard in multi-model
    fitting; removes the duplicates created when gating splits one motion
    (e.g. background) across several instances."""
    kept = []
    for H, inl in hyps:
        for ki, (Hk, inlk) in enumerate(kept):
            if np.median(V.reproj_err(Hk, p1[inl], p2[inl])) < thr:
                u = np.union1d(inlk, inl)
                Hn, _ = cv2.findHomography(p1[u], p2[u], 0)
                kept[ki] = (Hn if Hn is not None else Hk, u)
                break
        else:
            kept.append((H, inl))
    return kept


def mask_verify(hyps, i1, seg1, seg2, tau):
    """Semantic verification of a hypothesis (new): if its inliers sit on object k
    in image 1, warping k's mask with H must land on an image-2 instance of the
    same class with IoU >= tau. Wrong hypotheses throw the mask elsewhere.
    Hypotheses on background / undetected regions cannot be checked and are kept
    (same 'unknown' philosophy as soft gating)."""
    (inst1, cls1), (inst2, cls2) = seg1, seg2
    h2, w2 = inst2.shape
    out = []
    for H, inl in hyps:
        ids = i1[inl]
        k = np.bincount(ids).argmax()
        if k == 0 or np.mean(ids == k) < 0.5:
            out.append((H, inl))
            continue
        warped = cv2.warpPerspective((inst1 == k).astype(np.uint8), H, (w2, h2)) > 0
        cands = [j for j in range(1, len(cls2)) if cls2[j] == cls1[k]]
        if not cands:                        # object not detected in image 2: unknown, keep
            out.append((H, inl))
            continue
        best = 0.0
        for j in cands:
            mj = inst2 == j
            union = (warped | mj).sum()
            best = max(best, (warped & mj).sum() / union if union else 0.0)
        if best >= tau:
            out.append((H, inl))
    return out


# ------------------------------------------------------------------ scoring
def score(hyps, p1, Hs, label1, rng, n_real=None):
    """n_real: matches beyond this index are injected outliers (label = 'outlier')."""
    labels = S.lookup(label1, p1)
    OUT = len(Hs)
    if n_real is not None:
        labels[n_real:] = OUT
    regions = []
    for L in range(len(Hs)):
        ys, xs = np.nonzero(label1 == L)
        sel = rng.choice(len(xs), min(300, len(xs)), replace=False)
        regions.append(np.stack([xs[sel], ys[sel]], 1).astype(float))
    claimed, n_false, n_dup, n_wrong, purities = set(), 0, 0, 0, []
    for H, inl in hyps:
        lab = labels[inl]
        L = np.bincount(lab, minlength=OUT + 1).argmax()
        purities.append(np.mean(lab == L))
        if L == OUT:                         # mostly injected outliers
            n_false += 1
            n_wrong += 1
            continue
        gt = np.hstack([regions[L], np.ones((len(regions[L]), 1))]) @ Hs[L].T
        gt = gt[:, :2] / gt[:, 2:]
        err = V.reproj_err(H, regions[L], gt).mean()
        if err < 5 and L not in claimed:
            claimed.add(L)
        else:
            n_false += 1
            if err < 5:
                n_dup += 1      # correct motion, found again (redundant; mergeable)
            else:
                n_wrong += 1    # matches no true motion: mixed objects / bad fit
    n_obj = len(Hs) - 1
    return dict(obj_recall=len(claimed - {0}) / n_obj, bg_found=float(0 in claimed),
                n_false=n_false, n_dup=n_dup, n_wrong=n_wrong, n_hyp=len(hyps),
                purity=float(np.mean(purities)) if purities else np.nan)


# ------------------------------------------------------------------ methods
def run_methods(det, im1, f1, f2, m, seg1, seg2, tau=None, strict=False):
    """2 x 2 x 2 design: {RANSAC, Hough} x {no gate, semantic gate} x {no post, +M}.
    +M = duplicate merging, applied identically to every method.
    tau: if given, also apply mask verification first. Validation (100 held-out
    scenes) showed it costs 4-5 points of recall even at tau=0.05 for little gain
    (moved objects often get a different class in image 2), so it is OFF by default."""
    p1, p2 = f1.xy[m[:, 0]].astype(float), f2.xy[m[:, 1]].astype(float)
    out = {}
    if len(m) < 8:
        return out, p1
    mode = "strict" if strict else "soft"
    i1, _, ok, mg = S.match_instances(f1, f2, m, seg1, seg2, mode)

    t = time.perf_counter()
    out["RANSAC"] = (ransac_multi(p1, p2), time.perf_counter() - t)

    t = time.perf_counter()
    hy = []
    for g in np.unique(mg[ok]):
        hy += ransac_multi(p1, p2, np.flatnonzero(ok & (mg == g)))
    out["Sem+RANSAC"] = (hy, time.perf_counter() - t)

    votes = Hg.votes_single(f1, f2, m) if det == "sift" else Hg.votes_pairs(f1, f2, m)
    t = time.perf_counter()
    out["Hough"] = (hough_multi(p1, p2, Hg.hough_clusters(*votes, im1.shape)), time.perf_counter() - t)

    t = time.perf_counter()
    s, th, a1, a2, mem = votes
    group, keep = S.make_group_fn(f1, f2, seg1, seg2, mode)(mem, m)
    cl = Hg.hough_clusters(s[keep], th[keep], a1[keep], a2[keep], mem[keep], im1.shape, group=group[keep])
    out["Sem+Hough"] = (hough_multi(p1, p2, cl, match_group=np.where(ok, mg, -1)), time.perf_counter() - t)

    for name in list(out):
        hyps, dt = out[name]
        t = time.perf_counter()
        h = hyps if tau is None else mask_verify(hyps, i1, seg1, seg2, tau)
        out[name + "+M"] = (merge_duplicates(p1, p2, h), dt + time.perf_counter() - t)
    return out, p1


# ------------------------------------------------------------------ main
def load_scene_cache(iid, img, img2, dets):
    """One small pickle per image (keeps RAM low)."""
    p = ROOT / "data" / "cache" / "multiobj" / f"{iid}.pkl"
    p.parent.mkdir(parents=True, exist_ok=True)
    c = pickle.loads(p.read_bytes()) if p.exists() else {}
    dirty = False
    if "seg" not in c:
        c["seg"] = (S.instance_map(img), S.instance_map(img2))
        dirty = True
    for d in dets:
        if d not in c:
            c[d] = P.extract_and_match(img, img2, d)
            dirty = True
    if dirty:
        p.write_bytes(pickle.dumps(c))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--offset", type=int, default=0, help="150 = validation split")
    ap.add_argument("--dets", nargs="*", default=["sift", "disk"])
    ap.add_argument("--outlier-frac", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=None, help="mask-verification IoU threshold (off by default)")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_hpatches import inject_outliers

    rows = []
    for iid, img, masks, rng in tqdm(coco_scenes(args.n, offset=args.offset), total=args.n):
        img2, Hs, label1, vis2 = make_scene(img, masks, rng)
        c = load_scene_cache(iid, img, img2, args.dets)
        seg1, seg2 = c["seg"]
        for det in args.dets:
            f1, f2, m = c[det]
            n_real = len(m)
            f1.shape, f2.shape = img.shape, img2.shape
            f1, f2, m = inject_outliers(f1, f2, m, args.outlier_frac, np.random.default_rng(iid))
            res, p1 = run_methods(det, img, f1, f2, m, seg1, seg2, tau=args.tau, strict=args.strict)
            for name, (hyps, dt) in res.items():
                r = score(hyps, p1, Hs, label1, np.random.default_rng(0), n_real=n_real)
                rows.append(dict(img=iid, det=det.upper(), method=name, n_obj=len(Hs) - 1,
                                 n_matches=len(m), ms=1000 * dt, **r))
    split = "val" if args.offset else "test"
    tau = "" if args.tau is None else f"_tau{args.tau:g}"
    name = f"multiobj_{split}_of{args.outlier_frac:g}{tau}{'_strict' if args.strict else ''}{args.tag}"
    out = ROOT / "results" / f"{name}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarize(rows)
    print(f"\nsaved {out}")


def summarize(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r["det"], r["method"])].append(r)
    print(f"\n{'det':5} {'method':14} {'n':>4} | {'obj recall %':>12} {'bg %':>6} "
          f"{'wrong/img':>9} {'dup/img':>8} {'purity %':>9} | {'ms':>6}")
    for (d, mth), rs in by.items():
        f = lambda k: np.nanmean([r[k] for r in rs])  # noqa: E731
        print(f"{d:5} {mth:14} {len(rs):4d} | {100*f('obj_recall'):12.1f} {100*f('bg_found'):6.1f} "
              f"{f('n_wrong'):9.2f} {f('n_dup'):8.2f} {100*f('purity'):9.1f} | {np.median([r['ms'] for r in rs]):6.1f}")


if __name__ == "__main__":
    main()
