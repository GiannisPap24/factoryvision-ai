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

## Inspection Workbench V2 — first implementation slice

Overview, the shared shell and single-image inspection follow the reviewed local
UX-1R direction. The native sidebar remains the single eight-workspace navigation
control and the authority for category selection. A navy workspace header and
packaged `app/workbench.css` provide the shared visual foundation; no remote font,
JavaScript component or additional frontend is used. The header stays inside the
native content region, rather than replacing Streamlit browser/sidebar controls.

Overview offers Synthetic Demo first when no methods are ready. Its original
geometry thumbnails are illustrative only: they do not generate a Demo session,
review records or exports. Local readiness details and Model Catalog remain
available. Real inference actions appear only when verified methods are available.

Single-image inspection places the validated RGB preview on the left and setup
on the right, stacking preview first on narrow screens. Category stays in the
sidebar; its current value is repeated in setup. Preview validation does not load
models or run inference. No weights means no anomaly output. Both categories keep
the original limits, consent and explicit execution gates; MetalParts also keeps
its minimum image size and exploratory-only restrictions. Clear releases upload
widget state, results, review state and consent through the existing cleanup helper.
Empty and invalid files show validation errors and cannot retain a previous preview.

Native sidebar collapse/expand, keyboard operation and contextual loss notices are
retained. Closing the sidebar does not clear work. Workspace/category changes
retain their original cleanup rules, including the completed Batch/Reports exception.
The other six workspaces keep their existing content, exports and serializers.
The design prototype banner is not part of the application. Native widget spacing,
sidebar geometry and controls intentionally differ from the static design reference.

The bounded density polish uses Streamlit 1.66 theme padding in both the source
and packaged themes. Setup uses compact native container gaps and a wider desktop
column; Run spans that column. Detailed input limits remain in native disclosures,
with upload validation, consent and unavailable-model gates unchanged. Clear keeps
its existing callback and precedes Run. Mobile Overview actions stack vertically;
only the empty preview is shortened, not uploaded image geometry.

The native sidebar supports its own click/Enter controls. Escape does not reliably
close it or restore focus as in the design prototype; no DOM script overrides that
behavior. Keyboard focus styling and navigation cleanup remain native.
