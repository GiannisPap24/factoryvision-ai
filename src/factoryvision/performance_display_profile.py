"""Fixed diagnostic line attribution of the existing display functions, not a renderer."""

import ast
import inspect
import sys
import time

from factoryvision.inference.batch import png
from factoryvision.inference.visualization import panels

LABELS = (
    "geometry",
    "original_rgb_resize",
    "spatial_resize",
    "normalization",
    "palette",
    "color_interpolation",
    "rgb_conversion",
    "overlay_blend",
    "output_auxiliary",
    "preview_copy",
    "preview_resize",
    "png_encoding",
    "buffer_output",
)
REPETITIONS = 10


def prepare():
    """Resolve two fixed installed source functions before the worker read guard."""
    groups = {}
    names = {
        "original": "original_rgb_resize",
        "spatial": "spatial_resize",
        "ceiling": "normalization",
        "intensity": "normalization",
        "stops": "palette",
        "rgb": "color_interpolation",
        "heatmap": "rgb_conversion",
        "overlay": "overlay_blend",
        "output": "output_auxiliary",
    }
    for function in (panels, png):
        source, first = inspect.getsourcelines(function)
        mapping = {}
        for node in ast.parse("".join(source)).body[0].body:
            label = "geometry" if function is panels else "buffer_output"
            if function is panels:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    label = names.get(node.targets[0].id, label)
                elif isinstance(node, (ast.For, ast.Return)):
                    label = "output_auxiliary"
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "copy":
                    label = "preview_copy"
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                label = {"thumbnail": "preview_resize", "save": "png_encoding"}.get(
                    getattr(node.value.func, "attr", ""), label
                )
            for line in range(node.lineno, node.end_lineno + 1):
                mapping[first + line - 1] = label
        groups[function.__code__] = mapping
    return groups


def measure(result, groups):
    """Ten fixed instrumented display calls; no pixel data or source metadata returned."""
    totals = dict.fromkeys(LABELS, 0)
    active = {}
    previous_trace = sys.gettrace()

    def trace(frame, event, arg):
        mapping = groups.get(frame.f_code)
        if mapping is None:
            return None
        now = time.perf_counter_ns()
        if frame in active:
            label, started = active.pop(frame)
            totals[label] += now - started
        if event == "line":
            active[frame] = (mapping[frame.f_lineno], time.perf_counter_ns())
        return trace

    try:
        sys.settrace(trace)
        for _ in range(REPETITIONS):
            view = panels(result)
            for name in ("original", "heatmap", "overlay"):
                png(view[name], 384)
    finally:
        sys.settrace(previous_trace)
    return {key: value / REPETITIONS / 1e6 for key, value in totals.items()}
