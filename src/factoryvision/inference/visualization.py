"""Display-only spatial reprojection; original scores/maps remain unnormalized."""

import numpy as np
from PIL import Image


def panels(result):
    height, width = result.original.shape[:2]
    scale = min(1, 1024 / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    original = Image.fromarray(result.original).resize(size, Image.Resampling.BILINEAR)
    spatial = np.asarray(Image.fromarray(result.spatial).resize(size, Image.Resampling.BILINEAR))
    ceiling = float(result.spatial.max())
    intensity = np.clip(spatial / ceiling, 0, 1) if ceiling > 0 else np.zeros_like(spatial)
    # Pale yellow -> amber -> orange. Display only; never probability.
    stops = np.array(
        [[255, 247, 188], [254, 196, 79], [217, 95, 14], [153, 52, 4]], dtype=np.float32
    )
    rgb = np.stack(
        [np.interp(intensity, [0, 1 / 3, 2 / 3, 1], stops[:, c]) for c in range(3)], axis=-1
    )
    heatmap = Image.fromarray(np.round(rgb).astype(np.uint8))
    overlay = Image.blend(original, heatmap, 0.45)
    output = {"original": original, "heatmap": heatmap, "overlay": overlay, "ceiling": ceiling}
    for name in ("reconstruction", "reference"):
        array = getattr(result, name)
        output[name] = (
            None
            if array is None
            else Image.fromarray((np.clip(array, 0, 1) * 255).round().astype(np.uint8)).resize(
                size, Image.Resampling.BILINEAR
            )
        )
    return output
