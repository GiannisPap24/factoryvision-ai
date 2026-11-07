"""Display invariants: numerical maps remain untouched and UI colors exclude purple."""

import colorsys
import re
import tomllib
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

import numpy as np
from test_inference import encoded, engine  # noqa: F401

from factoryvision.inference.visualization import panels


def test_heatmap_color_and_numeric_invariants(engine):  # noqa: F811 - imported pytest fixture
    result = engine.inspect(encoded(), "autoencoder")
    ramp = np.linspace(0, 1, 128 * 128, dtype=np.float32).reshape(128, 128)
    ramp.setflags(write=False)
    result = replace(result, spatial=ramp)
    before = ramp.copy()
    view = panels(result)
    assert np.array_equal(before, result.spatial) and not result.spatial.flags.writeable
    assert view["ceiling"] == 1 and view["original"].size == view["heatmap"].size
    for rgb in set(view["heatmap"].get_flattened_data()):
        hue, saturation, _ = colorsys.rgb_to_hsv(*(v / 255 for v in rgb))
        assert saturation < 0.15 or not 245 <= hue * 360 <= 345


def test_packaged_theme_no_purple_and_source_theme_agree():
    theme_text = files("factoryvision.app").joinpath("theme.toml").read_text(encoding="utf8")
    theme = tomllib.loads(theme_text)["theme"]
    source = tomllib.loads(Path(".streamlit/config.toml").read_text(encoding="utf8"))["theme"]
    assert theme == source
    for value in re.findall(r"#[0-9A-Fa-f]{6}", theme_text):
        rgb = tuple(int(value[i : i + 2], 16) / 255 for i in (1, 3, 5))
        hue, saturation, _ = colorsys.rgb_to_hsv(*rgb)
        assert saturation < 0.15 or not 245 <= hue * 360 <= 345, value
    assert len(theme["chartSequentialColors"]) == len(theme["chartDivergingColors"]) == 10
    assert theme["base"] == "light"


def _contrast(first, second):
    def luminance(color):
        values = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in values]
        return sum(v * factor for v, factor in zip(linear, (0.2126, 0.7152, 0.0722), strict=True))

    high, low = sorted((luminance(first), luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_surface_palette_preserves_text_and_interaction_contrast():
    """Contrast regression guard; actual compositing and focus need browser checks."""
    css = files("factoryvision.app").joinpath("workbench.css").read_text(encoding="utf8")
    colors = dict(re.findall(r"--palette-(\w+):(#[0-9A-Fa-f]{6});", css))
    for text in ("text", "muted"):
        for background in ("canvas", "workspace", "surface", "surfaceSecondary", "diagnostic"):
            assert _contrast(colors[text], colors[background]) >= 4.5, (text, background)
    for text, background in (
        ("navText", "nav"),
        ("navText", "navRaised"),
        ("navText", "image"),
        ("navMuted", "image"),
        ("surface", "primary"),
        ("surface", "hover"),
        ("hover", "hoverBg"),
        ("hover", "selected"),
        ("info", "infoBg"),
        ("warning", "warningBg"),
        ("error", "errorBg"),
        ("success", "successBg"),
        ("disabled", "disabledBg"),
    ):
        assert _contrast(colors[text], colors[background]) >= 4.5, (text, background)
    for indicator, background in (
        ("control", "surface"),
        ("control", "workspace"),
        ("focusLight", "surface"),
        ("focusLight", "selected"),
        ("focusDark", "image"),
        ("focus", "nav"),
        ("primary", "selected"),
    ):
        assert _contrast(colors[indicator], colors[background]) >= 3, (indicator, background)


def test_native_theme_and_owned_surface_palette_agree():
    css = files("factoryvision.app").joinpath("workbench.css").read_text(encoding="utf8")
    colors = dict(re.findall(r"--palette-(\w+):(#[0-9A-Fa-f]{6});", css))
    theme = tomllib.loads(
        files("factoryvision.app").joinpath("theme.toml").read_text(encoding="utf8")
    )["theme"]
    for key, token in (
        ("backgroundColor", "canvas"),
        ("secondaryBackgroundColor", "surface"),
        ("textColor", "text"),
        ("borderColor", "control"),
        ("primaryColor", "primary"),
        ("dataframeHeaderBackgroundColor", "diagnostic"),
        ("blueBackgroundColor", "infoBg"),
        ("yellowBackgroundColor", "warningBg"),
        ("redBackgroundColor", "errorBg"),
        ("greenBackgroundColor", "successBg"),
    ):
        assert theme[key] == colors[token], key
    assert theme["sidebar"]["backgroundColor"] == colors["nav"]
    assert theme["sidebar"]["secondaryBackgroundColor"] == colors["navRaised"]
