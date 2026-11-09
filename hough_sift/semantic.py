"""Step 4 (novelty): semantic-constrained Hough voting.

Plain Hough lets ANY two matches vote together. In a cluttered scene that means
pairs straddling two different objects produce meaningless poses, and two
objects that happen to move similarly can merge into one cluster.

We use an off-the-shelf instance segmenter (Mask R-CNN, COCO-pretrained, no
training) to build a per-pixel instance map for each image, then apply 3 rules:

  R1 match rule : class(p1) must equal class(p2)       (background==background ok)
  R2 pair rule  : all matches in a vote lie on the same instance in image 1
                  AND on the same instance in image 2
  R3 Hough rule : the (instance1, instance2) id is appended as a 5th Hough
                  dimension -> different objects can never share a bin

R1/R2 remove votes, R3 separates the remaining ones. Implemented as a
`group_fn` that pipeline.estimate() / hough_clusters() accept.
"""
import numpy as np
import torch
import torchvision

from .features import DEVICE

_model = None


def instance_map(img_rgb, score_thr=0.5, mask_thr=0.5):
    """Returns inst (H, W) int32 with 0 = background, and cls (n_inst+1,) class per id."""
    global _model
    if _model is None:
        w = torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        _model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=w).to(DEVICE).eval()
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float().to(DEVICE) / 255.0
    with torch.inference_mode():
        out = _model([t])[0]
    keep = out["scores"] > score_thr
    masks = (out["masks"][keep, 0] > mask_thr).cpu().numpy()
    labels = out["labels"][keep].cpu().numpy()
    scores = out["scores"][keep].cpu().numpy()
    inst = np.zeros(img_rgb.shape[:2], np.uint8)   # <= 63 instances -> uint8 saves RAM
    cls = [0]
    # paint low-score first so confident instances win on overlaps
    for n, idx in enumerate(np.argsort(scores)[: 63]):
        inst[masks[idx]] = n + 1
        cls.append(int(labels[idx]))
    return inst, np.array(cls)


def lookup(inst, xy):
    h, w = inst.shape
    x = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
    return inst[y, x].astype(np.int64)    # uint8 storage -> int64 so id arithmetic can't overflow


def match_instances(f1, f2, m, seg1, seg2, mode="soft"):
    """Per-match instance ids (i1, i2), R1 flag, and the group id used by R3.

    strict : background is a class like any other. A match cup<->background is
             rejected; group = (instance1, instance2).
    soft   : background (0) means "segmenter saw nothing here" = UNKNOWN, a
             wildcard. R1 only rejects when BOTH sides are detected objects of
             DIFFERENT classes; group = instance in image 1.
             Rationale: a missed detection should not delete a real object.
    """
    (inst1, cls1), (inst2, cls2) = seg1, seg2
    i1 = lookup(inst1, f1.xy[m[:, 0]])
    i2 = lookup(inst2, f2.xy[m[:, 1]])
    c1, c2 = cls1[i1], cls2[i2]
    if mode == "strict":
        return i1, i2, c1 == c2, i1 * 64 + i2
    ok = (c1 == c2) | (i1 == 0) | (i2 == 0)
    return i1, i2, ok, i1


def make_group_fn(f1, f2, seg1, seg2, mode="soft"):
    def group_fn(members, m):
        i1, i2, ok, g = match_instances(f1, f2, m, seg1, seg2, mode)
        g1, g2 = i1[members], i2[members]                    # (V, k)
        keep = ok[members].all(1)                            # R1
        keep &= (g1 == g1[:, :1]).all(1)                     # R2, image-1 side
        if mode == "strict":
            keep &= (g2 == g2[:, :1]).all(1)                 # R2, image-2 side
        else:                                                # image-2 side: equal or unknown
            keep &= ((g2 == g2[:, :1]) | (g2 == 0) | (g2[:, :1] == 0)).all(1)
        return g[members[:, 0]], keep                        # R3 group per vote
    return group_fn
