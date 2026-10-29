# Interface and visual interpretation

The application has eight workspaces: Overview, Inspect, Compare, Batch Inspection,
Reports, Model Catalog, Research and Demo. Use Overview to choose a category and
understand model availability. Inspect runs one model; Compare aligns outputs from
multiple compatible methods. Batch Inspection uses one model sequentially on up to
20 images. Reports retains that completed batch until it is cleared or the user
leaves the Batch Inspection / Reports pair. Catalog cards and the synthetic Demo
work without model files. Research presents recorded results with their limitations.

Navigation uses navy, controls blue and teal, and semantic feedback uses blue,
green, amber or red. Fonts and theme assets are local. Purple-family colors are
excluded from application-controlled styling. Narrow screens collapse navigation
and stack columns; tables retain horizontal scrolling where needed.

## Scientific maps

Warm yellow–orange maps are display renderings of original anomaly arrays.
The legend identifies a relative display scale; it is not a probability or a
shared calibrated scale across models or images. Display normalization never
changes raw arrays, scores, frozen thresholds or exported research values.
See [inference contracts](inference.md) for preprocessing and numerical behavior.

## Published screenshots and review evidence

The README includes five reviewed screenshots: source-only Overview,
Model Catalog, generated Demo, synthetic Human Review and synthetic exports.
They show the interface without weights and
original synthetic geometry, with no real predictions or third-party dataset
imagery. Their [capture provenance](https://github.com/GiannisPap24/factoryvision-ai/blob/main/docs/images/README.md)
is recorded alongside the repository images. These presentation assets are
excluded from wheel and sdist packages.

The larger engineering review set and model-enabled captures remain private.
Future public captures should retain the same clear synthetic labels and undergo
review of visible filenames, browser chrome, paths and image metadata.
