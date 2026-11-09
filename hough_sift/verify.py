"""Step 3: geometric verification - turn a Hough cluster into a homography.

A Hough bin is coarse (30 deg, factor-2 scale), so a cluster still holds a few
wrong matches and the similarity model is only approximate. Following Lowe we
refine each cluster with *least squares + iterative outlier removal*:

  1. fit an AFFINE transform to the cluster (6 DOF, needs >= 3 matches)
  2. re-check ALL matches against it, keep those within a loose threshold
  3. once enough inliers, fit a full HOMOGRAPHY (8 DOF) and tighten the threshold
  4. repeat until the inlier set stops changing

No random sampling anywhere -> the result is deterministic (unlike RANSAC).
The best of the top-K clusters (most final inliers) wins.
"""
import cv2
import numpy as np


def _affine_lstsq(p1, p2):
    A = np.hstack([p1, np.ones((len(p1), 1))])          # (n, 3)
    X, *_ = np.linalg.lstsq(A, p2, rcond=None)           # (3, 2)
    H = np.eye(3)
    H[:2] = X.T
    return H


def reproj_err(H, p1, p2):
    q = np.hstack([p1, np.ones((len(p1), 1))]) @ H.T
    q = q[:, :2] / np.where(np.abs(q[:, 2:]) < 1e-9, 1e-9, q[:, 2:])
    return np.linalg.norm(q - p2, axis=1)


def refine_cluster(p1, p2, idx, thr=3.0, thr_affine=10.0, iters=10, cand=None):
    """cand: optional bool mask over matches - only these may become inliers."""
    if cand is None:
        cand = np.ones(len(p1), bool)
    idx = np.asarray(idx)
    idx = idx[cand[idx]]
    H = None
    for it in range(iters):
        if len(idx) < 3:
            return None, np.zeros(0, int)
        if len(idx) >= 8 and it > 0:
            H, _ = cv2.findHomography(p1[idx], p2[idx], 0)   # 0 = plain least squares
            t = thr
        else:
            H, t = _affine_lstsq(p1[idx], p2[idx]), thr_affine
        if H is None:
            return None, np.zeros(0, int)
        new = np.flatnonzero((reproj_err(H, p1, p2) < t) & cand)
        if np.array_equal(new, idx):
            break
        idx = new
    if H is None or len(idx) < 4:
        return None, np.zeros(0, int)
    # final homography + inlier set at the tight threshold
    H, _ = cv2.findHomography(p1[idx], p2[idx], 0)
    if H is None:
        return None, np.zeros(0, int)
    return H, np.flatnonzero((reproj_err(H, p1, p2) < thr) & cand)


def verify_clusters(p1, p2, clusters, top_k=10, thr=3.0):
    best_H, best_in = None, np.zeros(0, int)
    for cl in clusters[:top_k]:
        H, inl = refine_cluster(p1, p2, cl, thr=thr)
        if H is not None and len(inl) > len(best_in):
            best_H, best_in = H, inl
    return best_H, best_in


def ransac(p1, p2, thr=3.0, method=cv2.RANSAC):
    if len(p1) < 4:
        return None, np.zeros(0, int)
    H, mask = cv2.findHomography(p1, p2, method, thr, maxIters=10000, confidence=0.999)
    if H is None:
        return None, np.zeros(0, int)
    return H, np.flatnonzero(mask.ravel())
