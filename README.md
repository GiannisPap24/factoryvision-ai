# FactoryVision AI

**Runtime:** Windows AMD64, Python 3.14.10, uv 0.13.5.
Owned source is MIT licensed. This source-only candidate contains no datasets or weights.

A local computer vision research application for visual anomaly inspection.
Choose Cashew or experimental MetalParts, select frozen models, upload a permitted
image and explicitly run CPU inference. Compare aligned anomaly maps, overlays,
reconstructions where supported, model-specific scores and measured latency.

**Batch inspection:** select one compatible model, upload up to 20 images (32 MiB
total), review validation and explicitly run sequential CPU inspection. Results
stay in the local session. Completed batches provide a validated Quality Control
Research Report with explicit, deterministic CSV/JSON downloads. See the
[reporting contract](docs/reporting.md),
[batch workflow](docs/local-demo.md) and the
[authoritative development roadmap](docs/roadmap.md).

**Research demonstration. Not validated for industrial acceptance, food safety or
other consequential decisions.** A high anomaly score can reflect lighting,
geometry or blur; a low score does not establish defect absence.

## Industrial Precision interface

Eight workspaces separate Overview, Inspect, Compare, Batch Inspection, Reports,
Model Catalog, Research and Demo. The light canvas, navy navigation and blue/teal
controls use a packaged local theme. Heatmaps use a yellow–orange display scale;
raw anomaly maps and frozen scores are unchanged. Model cards can be downloaded
without research artifacts. Completed reports remain available when moving between
Batch Inspection and Reports; leaving that pair clears the batch.
See the [interface guide](docs/interface.md).

## Run locally

Verified: Windows AMD64, Python **3.14.10**, uv **0.13.5**.
Complete the [project-local setup](docs/local-demo.md) once, then run:

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path (Get-Location) '.venv-31410'
.venv-31410/Scripts/python.exe -m factoryvision.app
```

Open **http://127.0.0.1:8501**. If uv or Python is missing, follow the
[project-local setup guide](docs/local-demo.md); it does not require global
security changes. Opening the workspace does not install, download or train.

**A fresh checkout starts without model files.** It displays recorded research
and actionable model-unavailable states. Private datasets, checkpoints and
pretrained weights are deliberately excluded. Inference requires the original
verified artifacts under their approved relative paths. There are no placeholder
predictions or automatic dataset downloads. See [research reproduction](docs/reproduction.md)
for prerequisites and the distinction between a new experiment and registered demo weights.

## Model Catalog and dataset-free demo

**Model Catalog** lists all six existing methods with expected and verified local
artifact identities, preprocessing, outputs, frozen thresholds, pretrained
provenance, model-card references and scientific limitations. It distinguishes
trained neural models, fitted statistics and frozen-feature memory. A missing
checkpoint stays explicitly unavailable; navigation never downloads weights.

**Demo → Generate synthetic demo** creates two original geometric images
and one invalid input, exercises image validation, and offers deterministic CSV/JSON
examples. **Synthetic workflow demonstration — not real model inference or defect
detection.** Every score, threshold and research flag is null; no model is called.
Demo exports use a separate schema and cannot become research reports.
[Catalog and demo guide](docs/model-catalog-demo.md).

## Batch inspection and reporting

Real single-image, comparison and batch inference require compatible, verified
local artifacts. Batch results retain rejected/error rows and distinguish completed
inspection from validation. Explicit downloads use the validated research schema,
artifact identity and original frozen thresholds; scores are not probabilities.
CSV cells are guarded against spreadsheet formula execution. Changing relevant
inputs invalidates stale results and downloads. [Report contract](docs/reporting.md).

## Models and what was learned

| Category | Method | Learning procedure | Output |
|---|---|---|---|
| Cashew | Convolutional autoencoder | Trained neural reconstruction model | RGB reconstruction, MSE map, frozen normal-calibration threshold |
| Cashew | Spatial statistical baseline | Fitted mean normal image; no gradient training | Reference image, squared-error map, frozen threshold |
| Cashew | ResNet18 feature memory | Frozen pretrained backbone; fitted normal descriptor bank | Nearest-neighbor feature distances, frozen threshold |
| Cashew | Student-teacher | Compact student trained against frozen ResNet18 teacher | Feature mismatch map, frozen threshold |
| MetalParts | Experimental autoencoder | Four epochs, 864 updates on 144 normal source images | Native patch reconstruction and MSE; **no validated threshold** |
| MetalParts | Experimental feature memory | Frozen ResNet18; 6,912 descriptors from the same 144 normals | Native patch feature distances; **no validated threshold** |

Raw scores have different units across methods and categories. Heatmaps use
separate relative scales, not calibrated defect probabilities. Category selection
is manual; the application does not reliably detect wrong-category or shifted inputs.

## Recorded research

Cashew uses VisA: 315 normal fitting, 67 validation and 68 calibration images.
The historical held-out evaluation contained 50 normal and 100 anomalous images.
Only the autoencoder and spatial baseline have the recorded official defect
metrics below. The official set has since been observed.

| Historical Cashew metric | Autoencoder | Spatial baseline |
|---|---:|---:|
| Image AUROC / AP | 0.93760 / 0.94322 | 0.73680 / 0.84122 |
| Pixel AUROC / AP | 0.81812 / 0.08094 | 0.89530 / 0.26730 |
| Recall at frozen threshold | 0.52 | 0.15 |
| False positives / 50 normals | 3 | 0 |

The autoencoder missed 48 of 100 anomalies at its unchanged threshold.
Feature-memory and student-teacher development diagnostics do not establish
comparable independent defect accuracy. [Cashew model card](docs/model-card.md).

MetalParts uses the pinned AmazonScience RobustAD source domain: 144 fitting
normals, 36 reused validation normals and 51 previously observed source anomalies.

**EXPLORATORY SOURCE-DOMAIN COMPARISON — NOT INDEPENDENT TEST PERFORMANCE**

| MetalParts metric | Autoencoder | Feature memory |
|---|---:|---:|
| Image AUROC / AP | 0.59695 / 0.71427 | 0.80174 / 0.88751 |
| Pixel AUROC / AP | 0.86717 / 0.02154 | 0.96350 / 0.30512 |

Feature memory has lower pixel AP on 10 of 51 anomalies; both methods place their
maximum outside the defect mask on 22. Aggregate improvement does not establish
an independent winner. [Comparison evidence](docs/milestone-3f-metalparts-feature-memory.md).

**PCB1: BLOCKED** by confirmed cross-partition source overlap.
**PCB2: INCONCLUSIVE**, with 56 similarity pairs unresolved without independent
human adjudication. **RobustAD target-domain evaluation: NOT CLEARED**, because
training/test independence remains unresolved. No new official evaluation is
performed by this application. [Research chronology](docs/reproduction.md).

## Engineering and privacy

One Streamlit frontend delegates to category-specific inference services. Immutable
registries verify artifact size, SHA-256 and model/config provenance before use.
Checkpoints load on CPU with `weights_only=True`; approved models are cached,
uploaded images are not cached as resources or saved to disk by the app.

Uploads: JPEG/PNG, at most 8 MiB, 4096 pixels per side and 16 million decoded pixels.
MetalParts requires at least 256 pixels per side. Corrupt, animated, transparent,
non-upright and recognized protected test inputs are rejected. Transformed test
copies cannot all be detected; only upload permitted images. Changing category
clears uploads and results. Clear the image when finished; process memory is not
securely erased. The launcher binds loopback, disables usage telemetry and static
serving, and retains CORS/XSRF protections. Do not expose it publicly.

[Architecture](docs/architecture.md) | [Inference contracts](docs/inference.md) |
[Development and offline checks](docs/development.md) |
[Local setup](docs/local-demo.md)

## Source layout

```text
src/factoryvision/    app, inference, data contracts, models, training, evaluation
scripts/             explicit research and synthetic debugging entry points
tests/               offline tests with fabricated images
integration_tests/   opt-in real-data checks; read scope before running
docs/                methods, model cards, protocols, milestone evidence
.vscode/             manual run/debug configurations
```

`data/`, `runs/`, `reports/`, `.venv/`, `.tools/` and `dist/` remain local and ignored.
This release candidate is independently assembled from reviewed source files.
Private experiment outputs and engineering audit records are excluded.
See the [publication checklist](docs/publication.md).

## Tests and CI status

For the complete development environment, omit `--no-dev` from the sync command.
Run `.venv-31410/Scripts/python.exe scripts/ci_check.py --uv uv` after installation.
The shared command checks offline fabricated-data tests and branch coverage, Ruff,
lock consistency, dependency compatibility, wheel/sdist contents and a fresh
installed-package Streamlit smoke without datasets or weights.

GitHub Actions workflow files target Windows 2025 and Python 3.14.10 with pinned
actions and uv, read-only permissions and no shared cache or artifact uploads.
**Hosted CI: NOT RUN — REQUIRES REMOTE REPOSITORY.**
[CI setup and security](docs/ci.md) | [Verification procedure](docs/ci.md).

## Methodology and roadmap

Normal-only fitting, validation and frozen normal-calibration thresholds are
documented separately from historical defect evaluation. Follow the
[reproduction guide](docs/reproduction.md) and model cards before any eligible
experiment. A newly trained model is a new experiment, not the registered frozen
checkpoint; never replace artifacts or tune thresholds silently. Independent
target-domain reliability remains **NOT ESTABLISHED**.

The [roadmap](docs/roadmap.md) records completed research and engineering work,
the current 4E local release candidate, and a separately gated 4F community launch. Later 5A–5F work covers
human review, registry, robustness, performance, camera and pilot discovery.
Those milestones require separate authorization. See [contributing preparation](CONTRIBUTING.md)
and [proposed repository presentation](docs/github-preparation.md).

## Licensing and attribution

VisA (Zou, Jeong, Pemula, Zhang and Dabeer, 2022) and AmazonScience RobustAD are
third-party datasets with separately documented CC BY 4.0 terms. Preserve their
attribution and pinned provenance; do not infer ownership of their imagery.
ResNet18 weights originate from torchvision's ImageNet-pretrained release and
have separate provenance and usage considerations from torchvision source code.
No trained weights or third-party images are distributed here. FactoryVision-owned
source is licensed under [MIT](LICENSE), Copyright (c) 2026 GiannisPap24.
Third-party datasets, images, weights and dependencies retain their own terms;
MIT does not authorize their redistribution. The [licensing inventory](docs/licensing.md)
records dependency notices and research-asset boundaries. Public publication remains
unauthorized.
