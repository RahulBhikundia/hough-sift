"""HPatches sequences (Balntas et al., CVPR 2017).

116 scenes, each with 6 images. Image 1 is the reference; for images 2..6 the
ground-truth homography H_1_k is given (files H_1_2 ... H_1_6). That gives
116 x 5 = 580 image pairs.
  i_*  (57 scenes): illumination changes, camera fixed
  v_*  (59 scenes): viewpoint changes, lighting fixed

Metric: warp the 4 corners of image 1 with the estimated H and with the true H;
mean corner distance (pixels) = homography error. Report % of pairs with error
below 1 / 3 / 5 px (as in SuperPoint, DISK, LightGlue papers).
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "data" / "hpatches_raw" / "hpatches-sequences-release"


def load_rgb(path, max_dim):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    f = min(1.0, max_dim / max(h, w))
    if f < 1.0:
        img = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
    sx, sy = img.shape[1] / w, img.shape[0] / h
    return img, np.diag([sx, sy, 1.0])          # S maps original -> resized pixels


def pairs(root=ROOT, max_dim=1024):
    """Yield (scene, k, img1, imgk, H_gt) with H_gt adapted to the resized images."""
    for scene in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        im1, S1 = load_rgb(scene / "1.ppm", max_dim)
        for k in range(2, 7):
            imk, Sk = load_rgb(scene / f"{k}.ppm", max_dim)
            H = np.loadtxt(scene / f"H_1_{k}")
            yield scene.name, k, im1, imk, Sk @ H @ np.linalg.inv(S1)


def corner_error(H_est, H_gt, shape):
    if H_est is None:
        return np.inf
    h, w = shape[:2]
    c = np.array([[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1]], float)
    a, b = c @ H_est.T, c @ H_gt.T
    a, b = a[:, :2] / a[:, 2:], b[:, :2] / b[:, 2:]
    return float(np.linalg.norm(a - b, axis=1).mean())
