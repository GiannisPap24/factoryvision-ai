"""Licensed offline font bytes and native theme contracts, without a font service."""

import base64
import hashlib
import json
import re
import tomllib
from importlib.resources import files
from pathlib import Path

from factoryvision.app.typography import FONT_FACES, font_styles


def test_embedded_fonts_are_exact_reviewed_resources_and_have_no_remote_source():
    assets = files("factoryvision.app").joinpath("fonts")
    manifest = json.loads(assets.joinpath("manifest.json").read_text(encoding="utf-8"))
    css = font_styles()
    payloads = re.findall(r"data:font/ttf;base64,([A-Za-z0-9+/=]+)", css)
    assert len(payloads) == len(FONT_FACES) == 6
    for (name, weight, style), payload in zip(FONT_FACES, payloads, strict=True):
        filename = f"IBMPlexSans-{name}.ttf"
        raw = assets.joinpath(filename).read_bytes()
        assert base64.b64decode(payload, validate=True) == raw
        assert hashlib.sha256(raw).hexdigest() == manifest["files"][filename]
        assert f"font-weight:{weight};font-style:{style};font-display:swap" in css
    assert (
        hashlib.sha256(assets.joinpath("LICENSE.txt").read_bytes()).hexdigest()
        == (manifest["licenseSha256"])
    )
    assert not any(value in css for value in ("http:", "https:", "local(", "<script"))
    assert font_styles() is css


def test_native_typography_preserves_security_and_legible_scale():
    config = tomllib.loads(Path(".streamlit/config.toml").read_text(encoding="utf-8"))
    theme = config["theme"]
    assert theme["font"] == theme["headingFont"] == "IBM Plex Sans"
    assert theme["baseFontSize"] == 16
    assert set(theme["headingFontWeights"]) <= {400, 500, 600, 700}
    assert config["server"]["enableStaticServing"] is False
    assert config["server"]["enableCORS"] is True
    assert config["server"]["enableXsrfProtection"] is True
