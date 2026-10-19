"""
Scanpath prediction metrics — ScanMatch, MultiMatch, SED, STDE.

Implementations based on GazeSearch paper's reference code.
All metrics operate on fixation sequences with pixel coordinates (X, Y, T).
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/home/myid/yl80817/GazeSearch/src/common")

import numpy as np
from .multimatch import docomparison
from .scanmatch import ScanMatch
# Paper-standard SED + STDE (GazeSearch reference implementations)
from visual_attention_metrics import (
    string_edit_distance as _paper_sed,
    scaled_time_delay_embedding_similarity as _paper_stde,
)


# ---------------------------------------------------------------------------
# MultiMatch: 5-dim vector (shape, length, direction, position, duration)
# ---------------------------------------------------------------------------

def compute_multimatch(pred_scanpath, gt_scanpath, im_w=224, im_h=224):
    """
    Args:
        pred_scanpath, gt_scanpath: each a dict with 'X', 'Y', 'T' lists (pixel coords)
        im_w, im_h: image width/height
    Returns:
        [mm_shape, mm_length, mm_direction, mm_position, mm_duration] — 5 floats in [0,1]
    """
    def _to_array(s):
        X, Y, T = s['X'], s['Y'], s.get('T', [0.0] * len(s['X']))
        L = len(X)
        if L < 3:
            arr = np.ones((3, 3), dtype=np.float32)
            arr[:L, 0] = X
            arr[:L, 1] = Y
            arr[:L, 2] = T[:L] if len(T) >= L else 0.0
        else:
            arr = np.ones((L, 3), dtype=np.float32)
            arr[:, 0] = X
            arr[:, 1] = Y
            arr[:, 2] = T
        return arr

    try:
        s1 = _to_array(pred_scanpath)
        s2 = _to_array(gt_scanpath)
        mm = docomparison(s1, s2, sz=[im_h, im_w])
        return mm[0]  # [shape, length, direction, position, duration]
    except Exception:
        return [0.0, 0.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# ScanMatch: sequence alignment similarity
# ---------------------------------------------------------------------------

_SCANMATCH_OBJ = None


def _get_scanmatch_obj(im_w=224, im_h=224, Xbin=7, Ybin=7, Threshold=1.5):
    global _SCANMATCH_OBJ
    if _SCANMATCH_OBJ is None:
        _SCANMATCH_OBJ = ScanMatch(
            Xres=im_w, Yres=im_h, Xbin=Xbin, Ybin=Ybin,
            Offset=(0, 0), Threshold=Threshold,
        )
    return _SCANMATCH_OBJ


def compute_scanmatch(pred_scanpath, gt_scanpath, im_w=224, im_h=224, Xbin=7, Ybin=7):
    """
    Args:
        pred_scanpath, gt_scanpath: dicts with 'X', 'Y' (pixel coords)
    Returns:
        ScanMatch score in [0, 1]
    """
    try:
        smObj = _get_scanmatch_obj(im_w, im_h, Xbin, Ybin)
        fix1 = np.column_stack([pred_scanpath['X'], pred_scanpath['Y']])
        fix2 = np.column_stack([gt_scanpath['X'], gt_scanpath['Y']])
        seq1 = smObj.fixationToSequence(fix1).astype(np.int32)
        seq2 = smObj.fixationToSequence(fix2).astype(np.int32)
        score, _, _ = smObj.match(seq1, seq2)
        return float(score)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# SED: String Edit Distance on quantized sequences
# ---------------------------------------------------------------------------

def _quantize_fixations(scanpath, grid_size=7, im_w=224, im_h=224):
    """Convert fixation (x, y) to patch index on grid."""
    X = np.array(scanpath['X']) / im_w
    Y = np.array(scanpath['Y']) / im_h
    col = np.clip((X * grid_size).astype(int), 0, grid_size - 1)
    row = np.clip((Y * grid_size).astype(int), 0, grid_size - 1)
    return row * grid_size + col


def _edit_distance(s1, s2):
    """Levenshtein distance."""
    m, n = len(s1), len(s2)
    if m == 0: return n
    if n == 0: return m
    dp = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(m + 1): dp[i, 0] = i
    for j in range(n + 1): dp[0, j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return int(dp[m, n])


def compute_sed(pred_scanpath, gt_scanpath, grid_size=5, im_w=224, im_h=224):
    """
    Paper-standard SED: raw Levenshtein distance on scanpath → string
    (quantized to grid_size x grid_size cells). Lower is better.
    Uses GazeSearch reference implementation. Paper range: ~4.7-5.1.
    """
    try:
        # Reference impl expects (fixations, stimulus_shape, n=grid_size)
        stimulus = np.zeros((im_h, im_w, 3), dtype=np.float32)
        hs = np.column_stack([gt_scanpath['X'], gt_scanpath['Y']]).astype(np.float64)
        ss = np.column_stack([pred_scanpath['X'], pred_scanpath['Y']]).astype(np.float64)
        return float(_paper_sed(stimulus, hs, ss, n=grid_size))
    except Exception:
        return float('nan')


# ---------------------------------------------------------------------------
# STDE: Spatiotemporal Dynamic Edit distance
# (approximate via weighted Euclidean distance per position)
# ---------------------------------------------------------------------------

def compute_stde(pred_scanpath, gt_scanpath, im_w=224, im_h=224):
    """
    Paper-standard STDE: scaled time-delay embedding similarity from GazeSearch ref code.
    Range [0, 1], HIGHER is better. Paper range: ~0.80-0.86.
    """
    try:
        stimulus = np.zeros((im_h, im_w, 3), dtype=np.float32)
        hs = np.column_stack([gt_scanpath['X'], gt_scanpath['Y']]).astype(np.float64).tolist()
        ss = np.column_stack([pred_scanpath['X'], pred_scanpath['Y']]).astype(np.float64).tolist()
        r = _paper_stde(hs, ss, stimulus)
        return float(r) if r is not None and r is not False else float('nan')
    except Exception:
        return float('nan')


# ---------------------------------------------------------------------------
# Aggregate evaluation over batches
# ---------------------------------------------------------------------------

def evaluate_scanpath_metrics(pred_paths, gt_paths, masks, im_w=224, im_h=224):
    """
    Args:
        pred_paths: (N, T, 2) normalized [0,1] coords
        gt_paths:   (N, T, 2) normalized coords
        masks:      (N, T) valid mask
    Returns:
        dict with keys: scanmatch, mm_shape, mm_length, mm_direction, mm_position, mm_duration, sed, stde
    """
    scanmatches, mms, seds, stdes = [], [], [], []

    for i in range(len(pred_paths)):
        m = masks[i]
        if m.sum() < 3:
            continue
        # Convert normalized → pixel
        px = (pred_paths[i, :m.sum(), 0] * im_w).tolist()
        py = (pred_paths[i, :m.sum(), 1] * im_h).tolist()
        gx = (gt_paths[i, :m.sum(), 0] * im_w).tolist()
        gy = (gt_paths[i, :m.sum(), 1] * im_h).tolist()

        pred = {'X': px, 'Y': py, 'T': [1.0] * len(px)}
        gt = {'X': gx, 'Y': gy, 'T': [1.0] * len(gx)}

        scanmatches.append(compute_scanmatch(pred, gt, im_w, im_h))
        mms.append(compute_multimatch(pred, gt, im_w, im_h))
        seds.append(compute_sed(pred, gt))
        stdes.append(compute_stde(pred, gt, im_w, im_h))

    if not scanmatches:
        return {k: 0.0 for k in ['scanmatch', 'mm_shape', 'mm_length', 'mm_direction',
                                  'mm_position', 'mm_duration', 'sed', 'stde']}

    mms = np.array(mms)  # (N, 5)
    return {
        'scanmatch': float(np.mean(scanmatches)),
        'mm_shape':    float(np.nanmean(mms[:, 0])),
        'mm_length':   float(np.nanmean(mms[:, 1])),
        'mm_direction':float(np.nanmean(mms[:, 2])),
        'mm_position': float(np.nanmean(mms[:, 3])),
        'mm_duration': float(np.nanmean(mms[:, 4])),
        'sed':  float(np.nanmean(seds)),
        'stde': float(np.nanmean(stdes)),
    }
