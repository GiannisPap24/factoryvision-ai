"""Pure protocol-V2 transforms of admitted synthetic RGB8 arrays, one condition at a time."""

import io
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from factoryvision.inference.uploads import readonly
from factoryvision.robustness.contracts import (
    Axis,
    Budgets,
    Kind,
    Parent,
    ParentID,
    Perturbation,
    Plan,
    RobustnessInvalid,
    _require,
)

BLOCK_ROWS = 64
GEOMETRIC = (Kind.ROTATION, Kind.TRANSLATION)


def _shape(height, width):
    _require(type(height) is int and type(width) is int)
    _require(1 <= height <= 4096 and 1 <= width <= 4096)
    _require(height * width <= Budgets().max_source_pixels)


def _rgb(array):
    _require(type(array) is np.ndarray and array.dtype == np.uint8 and array.ndim == 3)
    _require(array.shape[2] == 3)
    _shape(*array.shape[:2])


def _spec(spec):
    _require(type(spec) is Perturbation)
    spec.__post_init__()


def _rgb8(values):
    return np.rint(np.clip(values, 0.0, 255.0)).astype(np.uint8)


@dataclass(frozen=True, slots=True, eq=False)
class TransformResult:
    """Detached read-only pixels and conservative mask; no source metadata."""

    pixels: np.ndarray
    valid: np.ndarray

    def __post_init__(self):
        _rgb(self.pixels)
        _require(type(self.valid) is np.ndarray and self.valid.dtype == np.bool_)
        _require(self.valid.shape == self.pixels.shape[:2])
        object.__setattr__(self, "pixels", readonly(self.pixels))
        object.__setattr__(self, "valid", readonly(self.valid))

    @property
    def coverage(self) -> float:
        return float(np.count_nonzero(self.valid) / self.valid.size)


@dataclass(frozen=True, slots=True)
class Geometry:
    """Source pixel centers: x right, y down, positive angles visually counterclockwise."""

    width: int
    height: int
    perturbation: Perturbation

    def __post_init__(self):
        _shape(self.height, self.width)
        _spec(self.perturbation)
        _require(self.perturbation.kind in (*GEOMETRIC, Kind.IDENTITY))

    def points(self, points: np.ndarray, *, inverse: bool = False) -> np.ndarray:
        """Map explicit N-by-2 float points; this is not a general registration API."""
        self.__post_init__()
        _require(type(inverse) is bool)
        _require(type(points) is np.ndarray and points.dtype in (np.float32, np.float64))
        _require(points.ndim == 2 and points.shape[1] == 2)
        _require(1 <= len(points) <= Budgets().max_source_pixels)
        _require(np.isfinite(points).all() and (np.abs(points) <= 8192).all())
        x, y = self._xy(points[:, 0].astype(np.float64), points[:, 1].astype(np.float64), inverse)
        return readonly(np.column_stack((x, y)))

    def _xy(self, x, y, inverse=False):
        spec = self.perturbation
        if spec.kind is Kind.IDENTITY:
            return x, y
        direction = -1 if inverse else 1
        if spec.kind is Kind.TRANSLATION:
            dx = spec.severity * self.width if spec.axis is Axis.X else 0.0
            dy = spec.severity * self.height if spec.axis is Axis.Y else 0.0
            return x + direction * dx, y + direction * dy
        angle = math.radians(direction * spec.severity)
        c, s = math.cos(angle), math.sin(angle)
        cx, cy = (self.width - 1) / 2, (self.height - 1) / 2
        return c * (x - cx) + s * (y - cy) + cx, -s * (x - cx) + c * (y - cy) + cy


def _bilinear(source, x, y, *, mask=None, fill=0.0):
    """Bounded row-block sampling; validity requires all positive-weight contributors."""
    h, w = source.shape[:2]
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - x0, y - y0
    output = np.zeros((*x.shape, *source.shape[2:]), dtype=np.float64)
    valid = np.ones(x.shape, dtype=bool)
    for dx, dy, weight in (
        (0, 0, (1 - fx) * (1 - fy)),
        (1, 0, fx * (1 - fy)),
        (0, 1, (1 - fx) * fy),
        (1, 1, fx * fy),
    ):
        xx, yy = x0 + dx, y0 + dy
        inside = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        xx, yy = np.clip(xx, 0, w - 1), np.clip(yy, 0, h - 1)
        supported = inside if mask is None else inside & mask[yy, xx]
        valid &= (weight == 0) | supported
        if source.ndim == 3:
            weight, inside = weight[..., None], inside[..., None]
        output += np.where(inside, source[yy, xx], fill) * weight
    return output, valid


def _geometry(rgb, spec):
    h, w = rgb.shape[:2]
    geometry = Geometry(w, h, spec)
    result = np.empty_like(rgb)
    valid = np.empty((h, w), dtype=bool)
    for start in range(0, h, BLOCK_ROWS):
        stop = min(h, start + BLOCK_ROWS)
        x, y = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(start, stop, dtype=np.float64))
        sx, sy = geometry._xy(x, y, inverse=True)
        pixels, mask = _bilinear(rgb, sx, sy, fill=128.0)
        result[start:stop], valid[start:stop] = _rgb8(pixels), mask
    return TransformResult(result, valid)


def _blur(rgb, sigma):
    h, w = rgb.shape[:2]
    radius = math.ceil(3 * sigma)
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    # One full float64 intermediate; bounded row blocks for each convolution pass.
    horizontal = np.empty(rgb.shape, dtype=np.float64)
    padded = np.pad(rgb, ((0, 0), (radius, radius), (0, 0)), mode="edge")
    for start in range(0, h, BLOCK_ROWS):
        stop = min(h, start + BLOCK_ROWS)
        block = np.zeros((stop - start, w, 3), dtype=np.float64)
        for index, weight in enumerate(kernel):
            block += weight * padded[start:stop, index : index + w]
        horizontal[start:stop] = block
    del padded
    result = np.empty_like(rgb)
    for start in range(0, h, BLOCK_ROWS):
        stop = min(h, start + BLOCK_ROWS)
        block = np.zeros((stop - start, w, 3), dtype=np.float64)
        for offset, weight in zip(offsets, kernel, strict=True):
            rows = np.clip(np.arange(start, stop) + offset, 0, h - 1)
            block += weight * horizontal[rows]
        result[start:stop] = _rgb8(block)
    return result


def _jpeg(rgb, quality):
    with io.BytesIO() as buffer:
        with Image.frombytes("RGB", (rgb.shape[1], rgb.shape[0]), rgb.tobytes()) as image:
            image.save(
                buffer,
                format="JPEG",
                quality=quality,
                subsampling=0,
                progressive=False,
                optimize=False,
                exif=b"",
                icc_profile=None,
            )
        buffer.seek(0)
        with Image.open(buffer, formats=("JPEG",)) as decoded:
            return np.array(decoded.convert("RGB"), dtype=np.uint8, copy=True)


def perturb(
    rgb: np.ndarray, specification: Perturbation, *, plan: Plan, parent: Parent
) -> TransformResult:
    """Transform the original admitted parent. Provenance must be established by the caller.

    No runner, automatic expansion, image decoding or inference. Never pass a previous result
    as a parent: arrays alone cannot prove synthetic origin or detect arbitrary chaining.
    """
    _require(type(plan) is Plan and type(parent) is Parent)
    plan.__post_init__()
    parent.__post_init__()
    _spec(specification)
    _require(parent in plan.parents and specification in plan.perturbations)
    _rgb(rgb)
    _require(rgb.shape[:2] == (parent.height, parent.width))
    spec = specification
    if spec.kind in GEOMETRIC:
        return _geometry(rgb, spec)
    if spec.kind is Kind.IDENTITY:
        result = rgb
    elif spec.kind is Kind.BLUR:
        result = _blur(rgb, spec.severity)
    elif spec.kind is Kind.JPEG:
        try:
            result = _jpeg(rgb, spec.severity)
        except (OSError, ValueError, SyntaxError):
            raise RobustnessInvalid() from None
    else:
        result = np.empty_like(rgb)
        if spec.kind is Kind.NOISE:
            # Stable public case ordinal and approved-seed replicate ordinal, no private IDs.
            case = tuple(ParentID).index(parent.parent_id) + 1
            replicate = (42, 1729, 2026).index(spec.seed)
            generator = np.random.Generator(
                np.random.PCG64(np.random.SeedSequence([spec.seed, case, replicate]))
            )
        for start in range(0, rgb.shape[0], BLOCK_ROWS):
            stop = min(rgb.shape[0], start + BLOCK_ROWS)
            block = rgb[start:stop].astype(np.float64)
            if spec.kind is Kind.BRIGHTNESS:
                values = block * spec.severity
            elif spec.kind is Kind.CONTRAST:
                values = ((block / 255.0 - 0.5) * spec.severity + 0.5) * 255.0
            else:
                values = (
                    block / 255.0 + generator.standard_normal(block.shape) * spec.severity
                ) * 255.0
            result[start:stop] = _rgb8(values)
    return TransformResult(result, np.ones(rgb.shape[:2], dtype=bool))
