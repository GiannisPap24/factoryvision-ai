"""Offline, packaged typography. No user input, font service or static file server."""

from base64 import b64encode
from functools import lru_cache
from importlib.resources import files

FONT_FACES = (
    ("Regular", 400, "normal"),
    ("Medium", 500, "normal"),
    ("SemiBold", 600, "normal"),
    ("Bold", 700, "normal"),
    ("Italic", 400, "italic"),
    ("SemiBoldItalic", 600, "italic"),
)


@lru_cache(maxsize=1)
def font_styles() -> str:
    """Embed reviewed original font bytes; cache only constant public assets."""
    assets = files("factoryvision.app").joinpath("fonts")
    rules = []
    for name, weight, style in FONT_FACES:
        encoded = b64encode(assets.joinpath(f"IBMPlexSans-{name}.ttf").read_bytes()).decode("ascii")
        rules.append(
            '@font-face{font-family:"IBM Plex Sans";'
            f'src:url("data:font/ttf;base64,{encoded}") format("truetype");'
            f"font-weight:{weight};font-style:{style};font-display:swap;}}"
        )
    return "<style>" + "".join(rules) + "</style>"
