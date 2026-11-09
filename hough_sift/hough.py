"""Step 2: Hough voting for a 4-DOF similarity pose (scale, rotation, x, y).

Idea (Lowe, IJCV 2004, Sec. 7.3): a correct match tells us *how the object moved*.
If many matches agree on the same movement, they are probably all correct.
So each match casts a vote in a 4-D "pose space"; correct matches pile up in one
bin, wrong matches scatter randomly. Peaks = candidate objects.

A similarity pose needs 4 numbers: scale s, rotation theta, translation (tx, ty).

  * SIFT single-match voting: ONE match gives all 4 numbers, because each keypoint
    carries its own scale and orientation:
        s = scale2 / scale1,  theta = ori2 - ori1,  translation from positions.

  * Pairwise voting (OUR ADAPTATION for learned features): DISK/SuperPoint keypoints
    have no scale or orientation, so one match only gives translation (2 numbers).
    But TWO matches (i, j) define the segment p1_i -> p1_j in image 1 and
    p2_i -> p2_j in image 2. Comparing the two segments gives
        s = |v2| / |v1|,   theta = angle(v2) - angle(v1)
    i.e. the full similarity. So we vote with pairs of nearby matches instead.

Translation is not voted as raw (tx, ty) - that depends on where the origin is.
Following Lowe, we vote for "where would the centre of image 1 land in image 2",
which is the same for every correct match regardless of its position.
"""
import numpy as np
from scipy.spatial import cKDTree


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ------------------------------------------------------------- vote generation
def votes_single(f1, f2, m):
    """SIFT: one vote per match. Returns pose params + which matches each vote supports."""
    i, j = m[:, 0], m[:, 1]
    s = f2.scale[j] / f1.scale[i]
    # OpenCV measures angles with y pointing down, so the in-image rotation is the
    # negative of the angle difference (verified in tests/test_hough.py).
    theta = wrap_angle(-(f2.ori[j] - f1.ori[i]))
    members = np.arange(len(m))[:, None]           # each vote supports 1 match
    return s, theta, f1.xy[i], f2.xy[j], members


def votes_pairs(f1, f2, m, k=16, min_dist=10.0, mutual=True):
    """Learned features: one vote per PAIR of spatially-neighbouring matches.

    k        : each match is paired with its k nearest matched neighbours in image 1
               (local pairs are likely to lie on the same object).
    mutual   : ALSO require the pair to be within each other's k nearest neighbours
               in image 2. A rigid object keeps neighbours together; a random wrong
               match lands anywhere in image 2, so its pairs are discarded.
               (Same intuition as GMS, Bian et al. CVPR 2017.)
               Without this, wrong matches create pairs that are tiny in image 1 but
               huge in image 2 -> absurd scales -> giant garbage clusters.
    min_dist : pairs closer than this (pixels) give unstable scale/angle -> skipped.
    """
    p1, p2 = f1.xy[m[:, 0]], f2.xy[m[:, 1]]
    M = len(m)
    if M < 2:
        return (np.zeros(0),) * 2 + (np.zeros((0, 2)),) * 2 + (np.zeros((0, 2), int),)

    def knn_pairs(p, kk):
        kk = min(kk + 1, M)
        d, nn = cKDTree(p).query(p, kk)
        a = np.repeat(np.arange(M), kk - 1)
        b = nn[:, 1:].ravel()
        ok = d[:, 1:].ravel() > min_dist
        return np.unique(np.sort(np.stack([a[ok], b[ok]], 1), 1), axis=0)

    pairs = knn_pairs(p1, k)                       # (a,b) and (b,a) merged
    if mutual:
        # a pair survives if it is a k-NN pair in image 2 too (use 2k there: the
        # local neighbourhood is reshuffled a little by perspective/scale change)
        pairs2 = knn_pairs(p2, 2 * k)
        key = lambda P: P[:, 0].astype(np.int64) * M + P[:, 1]  # noqa: E731
        pairs = pairs[np.isin(key(pairs), key(pairs2))]
    a, b = pairs[:, 0], pairs[:, 1]
    v1, v2 = p1[b] - p1[a], p2[b] - p2[a]
    n1, n2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
    s = n2 / np.maximum(n1, 1e-6)
    theta = wrap_angle(np.arctan2(v2[:, 1], v2[:, 0]) - np.arctan2(v1[:, 1], v1[:, 0]))
    good = n2 > 1e-3                                # degenerate: both map to same point
    return s[good], theta[good], p1[a][good], p2[a][good], pairs[good]


# ------------------------------------------------------------- accumulation
def hough_clusters(s, theta, a1, a2, members, img1_shape,
                   ori_bin_deg=30.0, loc_frac=0.25, group=None, min_support=3,
                   scale_range=(1 / 8, 8)):
    """Accumulate votes into 4-D bins and return clusters of match indices.

    Bin sizes are Lowe's: orientation 30 deg, scale factor 2, location 0.25 x the
    (scaled) size of image 1. Each vote goes into the 2 nearest bins in EVERY
    dimension (2^4 = 16 bins) so a pose sitting on a bin boundary is not split.

    group : optional (V,) int label per vote. If given it becomes a 5th Hough
            dimension, so votes with different labels can NEVER share a bin.
            This is the hook the semantic gating (semantic.py) uses.

    Returns list of int arrays (match indices), sorted by number of distinct
    supporting matches, largest first.
    """
    # scale prior: votes implying an absurd zoom are physically implausible.
    # Also essential because location bins grow with scale, so absurd-scale votes
    # would fall into a few gigantic bins and form fake peaks.
    ok = (s >= scale_range[0]) & (s <= scale_range[1])
    s, theta, a1, a2, members = s[ok], theta[ok], a1[ok], a2[ok], members[ok]
    if group is not None:
        group = group[ok]
    V = len(s)
    if V == 0:
        return []
    h1, w1 = img1_shape[:2]
    centre1 = np.array([w1 / 2.0, h1 / 2.0])

    # predicted position of image-1 centre in image 2:  c2 = a2 + s R(theta) (c1 - a1)
    d = centre1 - a1
    c, si = np.cos(theta), np.sin(theta)
    c2 = a2 + s[:, None] * np.stack([c * d[:, 0] - si * d[:, 1],
                                     si * d[:, 0] + c * d[:, 1]], 1)

    # continuous bin coordinates
    b_s = np.log2(np.maximum(s, 1e-6))              # bin width 1 in log2 = factor 2
    n_ori = int(round(360 / ori_bin_deg))
    b_t = (theta + np.pi) / np.deg2rad(ori_bin_deg)

    def two_nearest(x):
        """Lower bin + the neighbouring bin on the side x is closer to."""
        lo = np.floor(x)
        other = np.where(x - lo >= 0.5, lo + 1, lo - 1)
        return np.stack([lo, other], 1).astype(np.int64)   # (V, 2)

    S, T = two_nearest(b_s), two_nearest(b_t) % n_ori
    keys, owners = [], []
    for si_ in range(2):
        sb = S[:, si_]
        # location bin size uses the scale of this scale-bin's centre
        loc = loc_frac * max(h1, w1) * 2.0 ** (sb + 0.5)
        X, Y = two_nearest(c2[:, 0] / loc), two_nearest(c2[:, 1] / loc)
        for ti in range(2):
            for xi in range(2):
                for yi in range(2):
                    # pack the bin index (s, theta, x, y[, group]) into ONE int64
                    # -> np.unique on a 1-D array is ~20x faster than on rows
                    key = (np.clip(sb, -63, 63) + 64) * 16 + T[:, ti]
                    key = key * 4096 + np.clip(X[:, xi], -2047, 2047) + 2048
                    key = key * 4096 + np.clip(Y[:, yi], -2047, 2047) + 2048
                    if group is not None:
                        key = key * 4096 + group
                    keys.append(key)
                    owners.append(np.arange(V))
    keys = np.concatenate(keys)
    owners = np.concatenate(owners)
    _, bin_id = np.unique(keys, return_inverse=True)

    # expand votes -> supported matches, count DISTINCT matches per bin
    k = members.shape[1]
    n_match = int(members.max()) + 1
    bm = np.unique(np.repeat(bin_id, k) * n_match + members[owners].ravel())
    b, mt = bm // n_match, bm % n_match           # already sorted by bin
    splits = np.flatnonzero(np.diff(b)) + 1
    groups = np.split(mt, splits)
    groups = [g for g in groups if len(g) >= min_support]
    groups.sort(key=len, reverse=True)
    return groups
