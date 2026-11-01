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
excluded from application-controlled styling. Narrow screens wrap the visible
navigation above the workspace and stack columns; tables retain horizontal scrolling where needed.

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
UX-1R direction. One always-rendered native content rail holds the eight-workspace
radio and Category selector. It wraps above the workspace on narrow screens;
there are no actionable controls in a collapsed sidebar. A navy workspace header and
packaged `app/workbench.css` provide the shared visual foundation; no remote font,
JavaScript component or additional frontend is used. The header stays inside the
native content region, rather than replacing Streamlit browser/sidebar controls.

Overview offers Synthetic Demo first when no methods are ready. Its original
geometry thumbnails are illustrative only: they do not generate a Demo session,
review records or exports. Local readiness details and Model Catalog remain
available. Real inference actions appear only when verified methods are available.

Single-image inspection places the validated RGB preview on the left and setup
on the right, stacking preview first on narrow screens. Category stays in the
navigation rail; its current value is repeated in setup. Preview validation does not load
models or run inference. No weights means no anomaly output. Both categories keep
the original limits, consent and explicit execution gates; MetalParts also keeps
its minimum image size and exploratory-only restrictions. Clear releases upload
widget state, results, review state and consent through the existing cleanup helper.
Empty and invalid files show validation errors and cannot retain a previous preview.

Native keyboard operation and contextual loss notices are retained. Resizing or
focusing navigation does not clear work. Workspace/category changes
retain their original cleanup rules, including the completed Batch/Reports exception.
Batch Inspection and Reports use the UX-4 presentation below; Research retains
its existing content. All export schemas and serializers are unchanged. Catalog and Demo use the compact UX-2B presentation
described in [Model Catalog and synthetic demo](model-catalog-demo.md), with
the same scientific and session contracts.
The design prototype banner is not part of the application. Native widget spacing,
navigation geometry and controls intentionally differ from the static design reference.

The bounded density polish uses Streamlit 1.66 theme padding in both the source
and packaged themes. Setup uses compact native container gaps and a wider desktop
column; Run spans that column. Detailed input limits remain in native disclosures,
with upload validation, consent and unavailable-model gates unchanged. Clear keeps
its existing callback and precedes Run. Mobile Overview actions stack vertically;
only the empty preview is shortened. Uploaded image geometry is preserved inside
the viewer using contain sizing.

The content navigation replaces the prototype's collapsible menu to avoid hidden
keyboard-actionable sidebar controls. Category and workspace selection remain
native widgets, with a single authority and no injected script or custom router.

## Image Viewer and Compare (UX-3)

Inspect and Compare share a large input preview beside compact setup controls.
Preview decoding remains separate from inference, including when all models are
unavailable. A completed single inspection replaces the preview with the result
viewer. Original, Heatmap, Overlay and Reconstruction / reference are distinct
native tabs; images precede model diagnostics. Unsupported reconstructions have
an explicit unavailable explanation.

Compare uses one view selector for all selected models, in selection order, with
at most two result cards per row. A link beneath Run jumps to the results.
Mobile stacks the cards and wraps the selector. Unsupported reconstructions use
a labelled empty panel to retain alignment with other methods.
Changing the view only changes presentation; it does not rerun inference or clear
results. Each model retains its own map legend, score units, frozen threshold or
unavailable decision, measured timing and artifact identity. Raw scores cannot
rank different methods; relative map colors are not a shared calibrated scale.

Viewers contain the complete image without cropping or stretching its geometry.
Input previews are reduced to at most 1024 pixels per side; result display pixels,
per-map maximum normalization and the 45% overlay remain unchanged. Desktop image
surfaces are 400 pixels high (440 for input), adapting to 280 pixels on mobile.
These display bounds do not alter numerical arrays, inference inputs or exports.
No zoom/pan interaction or new inference capability is introduced.

## Batch Inspection and Reports (UX-4)

Batch setup and input status sit in adjacent native panels on desktop and stack
on mobile. Setup keeps one compatible model, the existing image limit, upload,
consent, clear and explicit Run controls. Validation counts and ordered thumbnails
are input status, not inspection progress. Progress uses actual completed-row
callbacks from the existing sequential batch service.

A completed snapshot exposes three native tabs: **Summary & exports**, **Image
evidence** and **Human review**. Summary distinguishes inspected, rejected, failed
and skipped rows and keeps validated research CSV/JSON downloads near the status.
Image evidence uses the accepted four-view viewer with compact retained previews;
only successful predictions have evidence or a review form. Human Review exports
remain separate and include pending records before any review is saved. Save is
explicit; unsaved edits are not exported. No score, schema or history changes.

A visible session notice explains that Batch Inspection and Reports retain the
snapshot and saved reviews. Leaving that pair, changing category or clearing
removes them; refresh, expiry or restart may also discard them. Export results and
saved review history before leaving. Returning from Reports shows a read-only
snapshot until Clear batch; it never resumes inference. Stopped batches retain
successful earlier rows without describing the whole batch as successful.

## Research and scientific evidence (UX-5)

Research leads with evidence status and study membership for the selected category.
Cashew separates historical image/pixel rankings from outcomes at frozen thresholds,
retaining six-decimal published metrics and exact thresholds. Its observed official
test, missed anomalies, false positives and independence limitations remain explicit.
Feature-memory and student–teacher diagnostics are development-only evidence;
PCB1 remains BLOCKED and PCB2 INCONCLUSIVE.

MetalParts presents the observed source-domain comparison with full recorded
precision, methodology, uncertainty and provenance. Independent target evaluation
is NOT CLEARED; there is no binary threshold. Separate-run timings are contextual
diagnostics, not a controlled speed comparison. Earlier normal reconstruction
losses and historical examples remain expandable; absent or unverified images
have an unavailable message. No research output is generated by this workspace.

Native evidence tables and expanders retain keyboard access, with contained
horizontal scrolling for long numeric values on mobile. Model Catalog continues
to provide the frozen artifact/split/protocol identities and model-card downloads.
The existing navigation, session lifecycle and all eight workspaces are unchanged.
