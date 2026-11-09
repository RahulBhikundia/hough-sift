"""Glue: one function that runs any (detector, verifier) combination on an image pair."""
import time

import numpy as np

from . import features as F
from . import hough as Hg
from . import verify as V

RATIO = {"sift": 0.8, "disk": 0.95}   # DISK descriptors are less "peaky"; looser ratio


def extract_and_match(img1, img2, det="sift", n=4000, slim=True):
    f1, f2 = F.EXTRACTORS[det](img1, n), F.EXTRACTORS[det](img2, n)
    m = F.match(f1.desc, f2.desc, ratio=RATIO[det], mutual=True)
    return slim_matches(f1, f2, m) if slim else (f1, f2, m)


def slim_matches(f1, f2, m):
    """Keep only matched keypoints and drop descriptors (verification never needs them).
    Cuts memory from MBs to KBs per pair; afterwards match k is simply (k, k)."""
    def take(f, idx):
        return F.Features(f.xy[idx], np.zeros((len(idx), 0), np.float32),
                          None if f.scale is None else f.scale[idx],
                          None if f.ori is None else f.ori[idx])
    k = np.arange(len(m))
    return take(f1, m[:, 0]), take(f2, m[:, 1]), np.stack([k, k], 1)


def estimate(img1, f1, f2, m, method="hough", thr=3.0, vote_mode=None, group_fn=None):
    """method: 'hough' | 'ransac' | 'magsac'.
    vote_mode: 'single' (needs SIFT scale/ori) or 'pairs'; default = what the detector allows.
    group_fn : optional callable(votes_members, matches) -> per-vote int label (semantic gating),
               or None to disable. May also return a keep-mask over matches as 2nd value.
    Returns dict with H, inlier match indices, timing and #clusters.
    """
    p1, p2 = f1.xy[m[:, 0]].astype(np.float64), f2.xy[m[:, 1]].astype(np.float64)
    t0 = time.perf_counter()
    info = {}
    if method == "ransac":
        H, inl = V.ransac(p1, p2, thr)
    elif method == "magsac":
        import cv2
        H, inl = V.ransac(p1, p2, thr, method=cv2.USAC_MAGSAC)
    else:
        if vote_mode is None:
            vote_mode = "single" if f1.scale is not None else "pairs"
        if vote_mode == "single":
            s, th, a1, a2, mem = Hg.votes_single(f1, f2, m)
        else:
            s, th, a1, a2, mem = Hg.votes_pairs(f1, f2, m)
        group = None
        if group_fn is not None:
            group, keep = group_fn(mem, m)
            s, th, a1, a2, mem, group = s[keep], th[keep], a1[keep], a2[keep], mem[keep], group[keep]
        clusters = Hg.hough_clusters(s, th, a1, a2, mem, img1.shape, group=group)
        info["n_clusters"] = len(clusters)
        info["n_votes"] = len(s)
        H, inl = V.verify_clusters(p1, p2, clusters, thr=thr)
    info.update(H=H, inliers=inl, time=time.perf_counter() - t0, n_matches=len(m))
    return info
