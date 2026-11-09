"""Step 1: local features and descriptor matching.

Two detectors live here:
  * SIFT  (classical)  -> every keypoint has (x, y, scale, orientation)
  * DISK  (learned)    -> every keypoint has only (x, y). No scale, no orientation.

That difference is the whole reason hough.py has two voting schemes.
"""
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import kornia.feature as KF

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Features:
    xy: np.ndarray            # (N, 2) pixel coordinates, float32
    desc: np.ndarray          # (N, D) descriptors, float32
    scale: np.ndarray | None  # (N,) keypoint size in pixels   (SIFT only)
    ori: np.ndarray | None    # (N,) orientation in radians    (SIFT only)


# ---------------------------------------------------------------- SIFT
def sift(img_rgb: np.ndarray, n: int = 4000) -> Features:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    det = cv2.SIFT_create(nfeatures=n)
    kps, desc = det.detectAndCompute(gray, None)
    if desc is None:
        return Features(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32),
                        np.zeros(0, np.float32), np.zeros(0, np.float32))
    xy = np.array([k.pt for k in kps], np.float32)
    scale = np.array([k.size for k in kps], np.float32)
    # OpenCV's angle is in degrees, measured in image coordinates (y points down).
    ori = np.deg2rad(np.array([k.angle for k in kps], np.float32))
    # RootSIFT (Arandjelovic & Zisserman 2012): L1-normalise then sqrt.
    # Free accuracy gain; makes Euclidean distance behave like Hellinger kernel.
    desc = desc / (desc.sum(1, keepdims=True) + 1e-7)
    desc = np.sqrt(desc).astype(np.float32)
    return Features(xy, desc, scale, ori)


# ---------------------------------------------------------------- DISK
_disk_model = None


def disk(img_rgb: np.ndarray, n: int = 4000) -> Features:
    """DISK (Tyszkiewicz et al., NeurIPS 2020), pretrained weights via kornia."""
    global _disk_model
    if _disk_model is None:
        _disk_model = KF.DISK.from_pretrained("depth").to(DEVICE).eval()
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float()[None] / 255.0
    with torch.inference_mode():
        f = _disk_model(t.to(DEVICE), n, pad_if_not_divisible=True)[0]
    return Features(f.keypoints.cpu().numpy().astype(np.float32),
                    f.descriptors.cpu().numpy().astype(np.float32), None, None)


EXTRACTORS = {"sift": sift, "disk": disk}


# ---------------------------------------------------------------- matching
def match(d1: np.ndarray, d2: np.ndarray, ratio: float = 0.8, mutual: bool = True) -> np.ndarray:
    """Nearest-neighbour matching with Lowe's ratio test (+ optional mutual check).

    Returns (M, 2) int array of index pairs (i into image 1, j into image 2).
    ratio test : keep i only if  dist(best) < ratio * dist(second best)
    mutual     : keep (i, j) only if i is also j's nearest neighbour in image 1
    """
    if len(d1) < 2 or len(d2) < 2:
        return np.zeros((0, 2), int)
    a = torch.from_numpy(d1).to(DEVICE)
    b = torch.from_numpy(d2).to(DEVICE)
    dist = torch.cdist(a, b)                       # (N1, N2)
    two = dist.topk(2, dim=1, largest=False)
    nn12 = two.indices[:, 0]
    keep = two.values[:, 0] < ratio * two.values[:, 1]
    if mutual:
        nn21 = dist.argmin(dim=0)
        keep &= nn21[nn12] == torch.arange(len(a), device=DEVICE)
    i = torch.nonzero(keep).squeeze(1)
    return torch.stack([i, nn12[i]], 1).cpu().numpy()
