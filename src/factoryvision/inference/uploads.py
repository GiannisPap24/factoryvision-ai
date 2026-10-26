"""Bounded, memory-only image decoding. No filenames, paths, persistence or network."""

import hashlib
import io
import json
import warnings
from importlib.resources import files

import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_BYTES = 8 * 1024**2
MAX_SIDE = 4096
MAX_PIXELS = 16_000_000


class InputRejected(ValueError):
    """Safe user-facing input error; never includes metadata or a user filename."""


def protected_hashes():
    data = json.loads(
        files("factoryvision.inference").joinpath("protected_inputs.json").read_text()
    )
    metal = json.loads(
        files("factoryvision.inference").joinpath("metalparts_protected_inputs.json").read_text()
    )
    return (
        frozenset(data["byte_sha256"]) | frozenset(metal["byte_sha256"]),
        frozenset(data["pixel_sha256"]) | frozenset(metal["pixel_sha256"]),
    )


def decode_upload(payload: bytes) -> Image.Image:
    if not isinstance(payload, bytes) or not 0 < len(payload) <= MAX_BYTES:
        raise InputRejected("Choose a nonempty JPEG or PNG file no larger than 8 MiB.")
    byte_blocks, pixel_blocks = protected_hashes()
    if hashlib.sha256(payload).hexdigest() in byte_blocks:
        raise InputRejected("This known official test image is excluded from the local demo.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload), formats=("JPEG", "PNG")) as source:
                if source.format not in {"JPEG", "PNG"}:
                    raise InputRejected("Only decoded JPEG and PNG formats are supported.")
                if (
                    max(source.size) > MAX_SIDE
                    or source.width * source.height > MAX_PIXELS
                    or getattr(source, "n_frames", 1) != 1
                ):
                    raise InputRejected(
                        "Use a single image, at most 4096 pixels per side and 16M pixels."
                    )
                source.verify()
            with Image.open(io.BytesIO(payload), formats=("JPEG", "PNG")) as source:
                if source.getexif().get(274, 1) != 1:
                    raise InputRejected("Export the image upright without an EXIF rotation tag.")
                source.load()
                if "A" in source.getbands() or "transparency" in source.info:
                    alpha = source.convert("RGBA").getchannel("A")
                    if alpha.getextrema() != (255, 255):
                        raise InputRejected(
                            "Export transparent images onto an opaque RGB background."
                        )
                rgb = source.convert("RGB")
                # A fresh image carries pixels only, without EXIF, ICC, comments or text chunks.
                image = Image.frombytes("RGB", rgb.size, rgb.tobytes())
    except InputRejected:
        raise
    except (
        OSError,
        ValueError,
        SyntaxError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise InputRejected(
            "The image is corrupt, truncated, unsafe to decode or unsupported."
        ) from exc
    digest = hashlib.sha256(
        f"RGB:{image.width}:{image.height}:".encode() + image.tobytes()
    ).hexdigest()
    if digest in pixel_blocks:
        raise InputRejected("These decoded pixels match an excluded official test image.")
    return image


def readonly(array):
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result
