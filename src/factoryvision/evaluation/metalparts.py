"""Descriptive source-development ranking; bounded-memory native pixel statistics."""

import numpy as np

BINS = 65536


def ranking_metrics(labels, scores):
    """Exact tie-aware binary AUROC and non-interpolated average precision."""
    y, s = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if (
        y.ndim != 1
        or y.shape != s.shape
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(s).all()
    ):
        raise ValueError("Require aligned finite scores and binary labels.")
    if len(y) == 0:
        return _counts_metrics(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(y) - 1]
    tp = np.cumsum(y, dtype=np.int64)[ends]
    fp = ends + 1 - tp
    return _counts_metrics(np.diff(np.r_[0, tp]), np.diff(np.r_[0, fp]))


def _counts_metrics(positive, negative):
    p, n = int(positive.sum()), int(negative.sum())
    result = {"positive": p, "negative": n, "prevalence": p / (p + n) if p + n else None}
    if not p or not n:
        return {
            **result,
            "auroc": None,
            "average_precision": None,
            "reason": "Both classes required for the reported paired ranking metrics.",
        }
    tp, fp = np.cumsum(positive, dtype=np.float64), np.cumsum(negative, dtype=np.float64)
    auc = np.sum(negative * (tp - positive / 2)) / (p * n)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    ap = np.sum(positive * precision) / p
    return {**result, "auroc": float(auc), "average_precision": float(ap), "reason": None}


class PixelRanking:
    """Every native pixel included, fixed 16-bit quantization; no sampled pixels.

    Metrics are exact for floor(score*65535), approximate for original float scores.
    Larger bins rank higher; ties receive half credit in ROC and grouped precision in AP.
    """

    def __init__(self):
        self.positive = np.zeros(BINS, dtype=np.int64)
        self.negative = np.zeros(BINS, dtype=np.int64)

    def add(self, spatial, mask):
        s, m = np.asarray(spatial), np.asarray(mask)
        if (
            s.ndim != 2
            or s.shape != m.shape
            or not np.isin(m, [0, 1]).all()
            or not np.isfinite(s).all()
            or np.any(s < 0)
            or np.any(s > 1)
        ):
            raise ValueError("Require aligned native binary masks and finite MSE maps in [0,1].")
        # Float64 conversion fixes the binning rule independently of input dtype.
        bins = np.floor(s.astype(np.float64) * (BINS - 1)).astype(np.int64)
        selected = m.astype(bool)
        self.positive += np.bincount(bins[selected], minlength=BINS)
        self.negative += np.bincount(bins[~selected], minlength=BINS)

    def result(self):
        return {
            **_counts_metrics(self.positive[::-1], self.negative[::-1]),
            "bins": BINS,
            "score_quantization": "floor(float64(MSE)*65535)",
            "interpretation": (
                "Exact on quantized scores; approximate unquantized native-pixel ranking; "
                "no sampling."
            ),
        }
