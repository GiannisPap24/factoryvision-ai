# Model Catalog and synthetic demonstration

Run the existing app from the repository root with
`.venv-31410/Scripts/python.exe -m factoryvision.app`. It binds `127.0.0.1:8501`.
No research data or weights are needed for these two pages.

## Model Catalog

Choose **Model Catalog** in the sidebar. Compact Cashew and MetalParts sections
group four and two methods, with learning method, scientific status and a local Markdown
model-card download. These presentation copies preserve accepted scientific text;
relative links become explicit source-distribution paths so standalone downloads
do not contain broken local hyperlinks. The overview lists all six registered
methods; the detail selector follows the selected category. It does not change
the model selections used by image or batch inspection. Category remains the shared
sidebar control; **Catalog method** selects details within that category. A shared
unavailable notice appears only when every method in a group is unavailable;
mixed availability shows each method's status. Details retain family, outputs,
runtime and expandable artifact identity, provenance and scientific limitations.

The typed projection in `inference/model_catalog.py` reads compatible IDs from
`batch.models`, identity/threshold metadata from `reporting.model_info` and its
existing packaged manifests, and availability from the original service verifiers.
Only descriptive method annotations are added. There is no second registration,
checkpoint search path, threshold source, or model-loading implementation.

Details distinguish trained autoencoders/student, fitted statistical reference,
and frozen-feature memory banks. They show preprocessing, outputs, pretrained
provenance, frozen threshold availability, research status and model-card paths.
Expected artifact SHA-256 and locally verified SHA-256 are separate fields.
Verified means file and metadata checks passed; runtime tensors are checked when
explicit inference loads the model. Catalog viewing never deserializes weights.

Missing, changed or incomplete artifacts are unavailable, with setup guidance;
the expected identity is never presented as proof of local availability. There
are no downloads on navigation. The upstream provenance link opens only when
clicked and is not an instruction to acquire or redistribute weights.

## Synthetic demo

Choose **Demo**, then **Generate synthetic demo**. The page prominently
states: **Synthetic workflow demonstration — not real model inference or defect detection.**

Two original geometric images are generated in memory (384 × 320 RGB rectangle
and ellipse), together with deliberately invalid bytes. The real bounded image
decoder validates them: two accepted, one rejected. No model is loaded or called.
No training, anomaly scoring, probabilities, thresholding or defect judgments occur.
Category selection does not make these examples category-matched research data.

The two images remain paired on desktop and mobile. **Input validation** separates
valid and rejected inputs; **Not computed** is a display label, while exported
scores, thresholds and research flags remain null. **Review & exports** jumps to
the demo downloads and expandable **Human Review & review exports** section.
Review exports remain available before saving: they contain two pending records
with empty notes, null review timestamps and no revision history. Saving remains
explicit; regeneration starts a fresh review of the new demo object.

Explicit CSV/JSON buttons export the separate `synthetic-workflow-v1` schema.
Every export retains its warning, `inference_executed=false`, null model/artifact
identity and null scores, thresholds and research flags. The deterministic ID
is a digest of the validated demo document. CSV uses the existing spreadsheet
cell guard. The real research report builder rejects this unrelated demo schema.

Nothing is saved automatically. **Clear synthetic demo**, page changes and category
changes invalidate the demo state/downloads. Demo state never populates image or
batch results. Browser downloads are explicit files under the user's control.
For actual inference, independently acquire the permitted prerequisites described
in [reproduction](reproduction.md); demo records cannot replace that evidence.

## Verification

`tests/test_catalog_demo.py` tests identities, unavailable/verified separation,
determinism, invalid exports and no model calls. `tests/test_app.py` exercises
navigation, warnings, explicit generation, state resets and unavailable batches
with Streamlit AppTest. `scripts/smoke_installed.py` repeats asset-free workflows
from an installed wheel in an empty directory with network connections denied.
The synthetic debug configuration uses `scripts/debug_catalog_demo.py`.

## Registry and historical evidence (5B)

Model Catalog now includes **Version and recorded provenance** for the selected
model, a metadata-only JSON download, and **Historical experiment evidence**
with category/status filters. Six registrations and their original availability
checks are preserved. Missing checkpoints disable real inference but do not hide
accepted historical evidence. Frozen pretrained extractors are components of
feature-memory and student methods, not newly trained models.

Cashew image/pixel AUROC and AP retain the model card's printed precision.
MetalParts values retain the packaged exploratory diagnostics and quantization
scope. PCB1/PCB2 show integrity status and missing metrics, never zero metrics.
Other endpoints and detailed training settings remain in downloadable model cards.
See [contracts and verification](milestone-5b-model-registry.md).
