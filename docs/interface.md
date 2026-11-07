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

The shared surface hierarchy separates the cool outer canvas and workspace from
white task cards, quieter supporting panels and pale diagnostic groups. Header,
navigation and image workspaces have distinct dark surfaces. Blue identifies
primary actions and the Overview starting card; teal remains a structural accent,
not evidence of successful inference or scientific clearance. Native informational,
warning, error and success notices retain their text and semantic roles.
Only outer task groups receive subtle elevation; document sections and nested
result cards do not stack shadows. Decorative panel dividers are separate from
the stronger borders needed for interactive controls. Light and dark surfaces
have separate focus colors; selected tabs retain their native indicator and labels.
Disabled buttons remain disabled and have readable muted text. These style rules
change no image pixels, scalar precision, workflow order or mobile control sizes.

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

## Inspection Workbench V2 — Industrial Precision foundation

The shared shell and single-image inspection develop the recovered UX-1R direction
into the Industrial Precision foundation; no exact match to an unavailable original
mockup is claimed. One full-width navy header identifies the application and joins
the navy navigation rail inside a common bounded frame. One
always-rendered native content rail holds one eight-workspace selectbox and one
Category selector. Below 1100px these native controls sit side by side; no duplicate
controls or collapsed sidebar are used. The workspace key and route values retain
their existing session authority. Page headers identify the active category and
route purpose; Catalog's global overview explicitly covers both categories.
Session-loss warnings occupy a
full-width slot before navigation, including the render that creates new work.
Research/session boundaries remain accessible below each workspace.

The synchronized source/packaged themes define native controls; packaged
`app/workbench.css` supplies application-scoped tokens and layout. No remote font,
JavaScript component, additional frontend or custom navigation router is used.
IBM Plex Sans provides one shared family for headings, branding, body and controls.
Original, licensed local font bytes are embedded through the existing trusted
style path; static file serving remains disabled. Regular, Medium, SemiBold and
Bold weights plus Regular/SemiBold italics are packaged. Tables and diagnostic
values use tabular lining numerals without changing scalar strings or precision.
Body text remains 16px, supporting/table text at least 14px on mobile, and primary
targets retain their existing sizes. Modern Greek and English are covered; some
polytonic Greek characters use the native fallback stack. See
[font provenance and licensing](../src/factoryvision/app/fonts/README.md).
The application header remains within Streamlit's content region, above both the
navigation rail and workspace; it does not replace browser chrome.

Overview offers Synthetic Demo as the primary first-use action and Model Catalog
as the secondary action. Verified local methods additionally expose Inspect and
Compare entry points. Three workflow entries also allow source-only Inspect,
Compare and Batch preview. A local-readiness panel distinguishes verified artifacts
from registration; a first-session disclosure retains the guided workflow.
Its original geometry thumbnails are illustrative only: they do not generate a Demo session,
review records or exports. Local readiness details and Model Catalog remain
available. Real inference actions appear only when verified methods are available.

Single-image inspection places the validated RGB preview on the left and a bounded
20–23rem setup panel on the right, stacking evidence before setup below 1050px.
The image keeps its aspect ratio inside the canvas. Native role tabs, captions,
full-precision disclosure and grouped diagnostics retain their original values.
Category stays in the navigation rail; its current value is repeated in setup. Preview validation does not load
models or run inference. No weights means no anomaly output. Both categories keep
the original limits, consent and explicit execution gates. Model selection,
image/permission and Run/review have distinct section labels. Expected missing
artifacts use informational guidance in Inspect, with verification details and
setup help available; actual input and execution failures remain errors. Compare
uses the same setup and informational missing-artifact presentation. MetalParts also keeps
its minimum image size and exploratory-only restrictions. Clear releases upload
widget state, results, review state and consent through the existing cleanup helper.
Invalid files show validation errors and cannot retain a previous preview.
With no file selected, the empty viewer explains preview versus inference.
An early native link goes to the existing Inspection setup heading in both Inspect
and Compare, for both categories. Keyboard Tab continues into its controls. The
narrow empty state is compact; loaded images retain their dominant, stable keyed
preview and aspect ratio. Reading order is unchanged and there is only one uploader.

Native keyboard operation and contextual loss notices are retained. Resizing or
focusing navigation does not clear work. Workspace/category changes
retain their original cleanup rules, including the completed Batch/Reports exception.
Batch Inspection and Reports use the UX-4 presentation below; Research retains
its existing content. All export schemas and serializers are unchanged. Catalog and Demo use the compact UX-2B presentation
described in [Model Catalog and synthetic demo](model-catalog-demo.md), with
the same scientific and session contracts.
Batch discloses selected-model availability before upload; preview and validation
remain usable when inference is unavailable. Empty Reports explains the verified
model prerequisite and links to Catalog without promising reports from Demo.
Demo keeps its synthetic warning, shows replacement/loss warnings whenever content
exists, and disables Clear only when empty. Generate and Clear retain their review
reset semantics through native callbacks.
The design prototype banner is not part of the application. Native widget spacing,
navigation geometry and controls intentionally differ from the static design reference.

The bounded density polish uses Streamlit 1.66 theme padding in both the source
and packaged themes. Setup uses native container gaps;
Run spans its column. Detailed input limits remain in native disclosures,
with upload validation, consent and unavailable-model gates unchanged. Clear keeps
its existing callback and precedes Run. Mobile Overview actions stack vertically;
only the empty preview is shortened. Uploaded image geometry is preserved inside
the viewer using contain sizing.

These are implementation rules, not a claim of complete accessibility acceptance.
Prior spoken observations apply to their recorded UI state and do not certify a
redesigned state. Screen-reader behavior requires separate verification from
responsive screenshots and AppTest checks.

Image expansion uses native in-page disclosures. The complete fitted image is
primary; optional magnified detail includes a whole-image context graphic.
The preview is replaced visually by the complete view while expanded, avoiding
two consecutive copies of the same primary graphic. Enter or Space on the same
heading opens or collapses it; Escape is not a close action.
Tab enters the detail scroll area, arrow keys pan, and Shift+Tab returns to its
heading. Secondary graphics have distinct role names and retain the original
caption. Closed content is excluded by the native disclosure semantics.
Display scaling adds no scientific resolution. Original image objects and the
existing Streamlit encoding settings are preserved, including native thumbnail
resizing and AUTO encoding; this is not a claim of lossless transport.
The stock fullscreen toolbar is hidden within the shared image wrapper. No
injected focus script, model execution, session-state image copy or export
mutation is used. Demo image cards stack below 800px; compact secondary
disclosures and a narrow
padding budget preserve useful detail width. Collapsed containers still render:
the four image representations add encoding/DOM work even when hidden.
Preview visibility and expanded card layout follow only the outer disclosure's
own open state. Closing it restores the preview even when the inner detail
remains logically open; reopening retains that native inner state. Scoped
sizing/toolbar hooks require review after upgrades.

Compare presents complete method panels in selection order, at most two per row,
stacking below 800px. Diagnostics adapt to their available panel width and retain
complete numbers; the precision disclosure remains authoritative for raw values.
Batch groups setup, validation and explicit execution, with a responsive thumbnail
grid. Its interrupted-run and no-replay rules are unchanged. Reports place model
identity, outcome counts and provenance before exports. Empty Reports explains
its completed-batch requirement and separate synthetic exports.

Catalog uses balanced category groups and distinguishes registration, verified
local artifacts and research status. Research uses document-style sections with
unchanged historical values, precision and limitations. Demo retains its explicit
generation, synthetic labels, reset warnings, validation and separate exports.
Tables retain local horizontal scrolling and at least 14px mobile type. Scoped
Streamlit caption and metric hooks prevent faded caveats and concealed numbers;
these framework-specific hooks must be rechecked on a Streamlit upgrade.

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
per-map maximum normalization and the 45% overlay remain unchanged. Desktop result
surfaces are 400 pixels high; Inspect and Compare inputs use a 360–520px
responsive height. All adapt to 280 pixels on mobile.
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
