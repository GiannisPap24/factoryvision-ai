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
