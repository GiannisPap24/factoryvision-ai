"""Explicit binary metric semantics using scikit-learn's tested ranking implementations."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def checked_scores(labels, scores):
    y, s = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if (
        y.ndim != 1
        or s.shape != y.shape
        or not y.size
        or y.size > 3_000_000
        or not np.isin(y, (0, 1)).all()
        or not np.isfinite(s).all()
    ):
        raise ValueError("Require aligned nonempty binary labels and finite scores, <=3M values.")
    return y.astype(np.uint8), s


def binary_metrics(labels, scores, threshold=None) -> dict:
    y, s = checked_scores(labels, scores)
    positive = int(y.sum())
    negative = int(y.size - positive)
    result = {
        "samples": y.size,
        "positives": positive,
        "negatives": negative,
        "auroc": float(roc_auc_score(y, s)) if positive and negative else None,
        "average_precision": float(average_precision_score(y, s))
        if positive and negative
        else None,
    }
    if threshold is not None:
        if type(threshold) not in (float, int) or not np.isfinite(threshold):
            raise ValueError("Threshold must be finite.")
        predicted = s > threshold
        tp = int(((y == 1) & predicted).sum())
        fp = int(((y == 0) & predicted).sum())
        fn = positive - tp
        tn = negative - fp
        result.update(
            threshold=threshold,
            comparison="score > threshold",
            confusion_matrix=[[tn, fp], [fn, tp]],
            tn=tn,
            fp=fp,
            fn=fn,
            tp=tp,
            precision=tp / (tp + fp) if tp + fp else None,
            recall=tp / positive if positive else None,
            f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            normal_fpr=fp / negative if negative else None,
        )
    return result


def paired_bootstrap(labels, ae_scores, mean_scores, repeats=1000, seed=42) -> dict:
    y, a = checked_scores(labels, ae_scores)
    _, b = checked_scores(labels, mean_scores)
    if len(np.unique(y)) != 2 or type(repeats) is not int or not 2 <= repeats <= 2000:
        raise ValueError("Paired stratified bootstrap requires both classes and 2..2000 repeats.")
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == c) for c in (0, 1)]
    samples = np.empty((repeats, 2), dtype=np.float64)
    functions = (roc_auc_score, average_precision_score)
    for i in range(repeats):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        samples[i] = [f(y[idx], a[idx]) - f(y[idx], b[idx]) for f in functions]
    result = {
        "resamples": repeats,
        "seed": seed,
        "unit": "paired stratified image",
        "limitation": "Conditional on image independence; not physical/acquisition groups.",
    }
    for j, (name, f) in enumerate(zip(("auroc", "average_precision"), functions, strict=True)):
        low, high = np.quantile(samples[:, j], [0.025, 0.975], method="linear")
        result[name] = {
            "ae_minus_mean": float(f(y, a) - f(y, b)),
            "ci95_percentile": [float(low), float(high)],
        }
    return result
