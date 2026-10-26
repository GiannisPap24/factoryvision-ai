"""Same 3E native-pixel bins; extend integer range for unbounded feature distances."""

import numpy as np

from factoryvision.evaluation.metalparts import _counts_metrics

LABEL = "EXPLORATORY SOURCE-DOMAIN COMPARISON — NOT INDEPENDENT TEST PERFORMANCE"
MAX_BINS = 1048576


class DistancePixelRanking:
    def __init__(self):
        self.positive = np.zeros(65536, dtype=np.int64)
        self.negative = np.zeros(65536, dtype=np.int64)

    def add(self, spatial, mask):
        s, m = np.asarray(spatial), np.asarray(mask)
        if (
            s.ndim != 2
            or s.shape != m.shape
            or s.size == 0
            or not np.isin(m, [0, 1]).all()
            or not np.isfinite(s).all()
            or np.any(s < 0)
        ):
            raise ValueError("Require aligned finite nonnegative maps and binary masks.")
        largest = float(s.max()) * 65535
        if largest >= MAX_BINS:
            raise MemoryError("Fixed pixel histogram budget exceeded; scores were not clipped.")
        bins = np.floor(s.astype(np.float64) * 65535).astype(np.int64)
        required = int(bins.max()) + 1
        if required > len(self.positive):
            self.positive = np.pad(self.positive, (0, required - len(self.positive)))
            self.negative = np.pad(self.negative, (0, required - len(self.negative)))
        selected = m.astype(bool)
        self.positive += np.bincount(bins[selected], minlength=len(self.positive))
        self.negative += np.bincount(bins[~selected], minlength=len(self.negative))

    def result(self):
        return {
            **_counts_metrics(self.positive[::-1], self.negative[::-1]),
            "label": LABEL,
            "bins": len(self.positive),
            "bin_width": 1 / 65535,
            "quantization": "floor(float64(raw_score)*65535); no clipping or normalization",
            "interpretation": "Exact quantized ranking; approximate unquantized ranking.",
        }
