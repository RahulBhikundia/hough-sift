"""Sanity check: warp a textured image by a KNOWN transform and see if every pipeline recovers it.

Run:  .venv/bin/python scripts/sanity_synthetic.py [optional_image_path]
If no image given, a random texture is generated.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hough_sift import pipeline as P  # noqa: E402
from hough_sift.hpatches import corner_error  # noqa: E402


def random_texture(h=600, w=800, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 255, np.uint8)
    for _ in range(400):
        c = tuple(int(x) for x in rng.integers(0, 255, 3))
        if rng.random() < 0.5:
            cv2.circle(img, tuple(int(v) for v in rng.integers(0, [w, h])), int(rng.integers(5, 40)), c, -1)
        else:
            p = rng.integers(0, [w, h], (2, 2))
            cv2.rectangle(img, tuple(int(v) for v in p[0]), tuple(int(v) for v in p[1]), c, -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def main():
    img1 = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2RGB) if len(sys.argv) > 1 and sys.argv[1] != "-" else random_texture()
    h, w = img1.shape[:2]
    # true transform: rotate 35 deg, scale 0.7, about the centre, plus mild perspective
    ROT = float(sys.argv[2]) if len(sys.argv) > 2 else 35.0
    A = np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), ROT, 0.7), [0, 0, 1]])
    persp = np.array([[1, 0.05, 0], [0.02, 1, 0], [1e-4, 5e-5, 1]])
    H_gt = persp @ A
    img2 = cv2.warpPerspective(img1, H_gt, (w, h))

    print(f"{'detector':8} {'method':7} {'matches':>7} {'inliers':>7} {'err(px)':>8} {'ms':>7}")
    for det in ["sift", "disk"]:
        f1, f2, m = P.extract_and_match(img1, img2, det)
        for method in ["ransac", "hough"]:
            r = P.estimate(img1, f1, f2, m, method)
            print(f"{det:8} {method:7} {len(m):7d} {len(r['inliers']):7d} "
                  f"{corner_error(r['H'], H_gt, img1.shape):8.2f} {1000 * r['time']:7.1f}")
        if det == "sift":   # SIFT can also use pair voting -> isolates the voting scheme
            r = P.estimate(img1, f1, f2, m, "hough", vote_mode="pairs")
            print(f"{det:8} {'hough-p':7} {len(m):7d} {len(r['inliers']):7d} "
                  f"{corner_error(r['H'], H_gt, img1.shape):8.2f} {1000 * r['time']:7.1f}")


if __name__ == "__main__":
    main()
