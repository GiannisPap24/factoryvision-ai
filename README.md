# FactoryVision AI

[![Source verification](https://github.com/GiannisPap24/factoryvision-ai/actions/workflows/verify.yml/badge.svg?branch=main)](https://github.com/GiannisPap24/factoryvision-ai/actions/workflows/verify.yml)
[![Python 3.14.10](https://img.shields.io/badge/Python-3.14.10-blue)](docs/local-demo.md)
[![MIT licence](https://img.shields.io/badge/License-MIT-teal)](LICENSE)

**A local visual anomaly inspection research app built with PyTorch and Streamlit.**
Explore six frozen methods, compare spatial evidence and export research results.
Try the interface and synthetic workflow after installation—no dataset or model weights needed.

[Quick start](#quick-start-clone-to-first-demo) · [Available workflows](#what-you-can-run) · [Research results](#research-results-and-limits) · [Documentation](docs/README.md)

![FactoryVision Overview with six registered methods and explicit research status](https://raw.githubusercontent.com/GiannisPap24/factoryvision-ai/main/docs/images/overview.jpg)
*Overview from the public app without weights. Zero models are ready for real inference.*

**Research demonstration. Not validated for industrial acceptance, food safety or automated product-quality decisions.**

## What it does

- [Synthetic robustness diagnostics](docs/protocols/robustness-source-only.md): deterministic perturbations, scalar stability reports and a bounded Windows analytical worker. No actual model execution or validated OOD detector is provided by this workflow.
- Model Registry V1: artifact versions, provenance and eight historical evidence summaries, available without weights. [Scope and verification](docs/milestone-5b-model-registry.md).
- Eight workspaces: Overview, Inspect, Compare, Batch Inspection, Reports, Model Catalog, Research and Demo.
- Six registered methods: four Cashew methods (autoencoder, spatial baseline, ResNet18 feature memory and student–teacher) and two experimental MetalParts methods.
- With verified local artifacts: explicit CPU inference, aligned heatmaps, overlays, reconstructions where supported, model-specific scores and latency.
- Batch inspection of up to 20 images, with validation, retained error rows and deterministic CSV/JSON research reports.
- [Human Review V1](docs/milestone-5a-human-review.md): session-local observations on individual results, revision history and explicit JSON/CSV exports. Predictions remain unchanged; feedback is not verified ground truth.

## What you can run

| Available after a fresh installation | Requires verified local research artifacts |
|---|---|
| Overview and historical Research summaries | Real single-image inspection |
| Model Catalog and downloadable model cards | Multi-model comparison and anomaly maps |
| Synthetic Demo and its CSV/JSON examples | Batch inference and completed research reports |
| Synthetic human-review practice, with null prediction fields | Human review of actual single-image and batch results |

The separate [robustness Python examples](docs/protocols/robustness-source-only.md)
also run without weights. They generate their own synthetic pixels; real-development
admission and model execution remain disabled.

The public repository contains **no datasets, checkpoints or pretrained weights**. Unavailable models remain disabled; there are no fabricated predictions or automatic weight downloads. See [artifact prerequisites](docs/reproduction.md).

## Quick start: clone to first demo

Verified platform: **Windows AMD64 · Python 3.14.10 · uv 0.13.5**. Git and an internet connection are needed for the initial checkout and dependencies.

1. In PowerShell, choose a working folder and clone the public repository:

   ```powershell
   git clone https://github.com/GiannisPap24/factoryvision-ai.git
   if ($LASTEXITCODE -ne 0) { throw 'Clone failed' }
   Set-Location factoryvision-ai
   ```

2. Follow [download and verify local tools](docs/local-demo.md#download-and-verify-local-tools) once to prepare project-local Python and uv. Start at that section if you have already cloned; do not repeat the clone. The guide includes exact download URLs, SHA-256 checks and the Python signature check. Then install from the repository root:

   ```powershell
   $env:UV_PROJECT_ENVIRONMENT = Join-Path (Get-Location) '.venv-31410'
   $env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
   $env:UV_PYTHON_DOWNLOADS = 'never'
   .tools/uv-0.13.5/uv.exe sync --locked --extra app --python .tools/python-3.14.10/python.exe
   if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
   ```

3. Start the app:

   ```powershell
   .venv-31410/Scripts/python.exe -m factoryvision.app
   ```

   Open **http://127.0.0.1:8501**, choose **Demo → Generate synthetic demo**, then download the synthetic CSV or JSON. Stop the server with **Ctrl+C**. For later sessions, repeat only this launch command from the repository root.

![Synthetic demo with original geometry and empty prediction fields](https://raw.githubusercontent.com/GiannisPap24/factoryvision-ai/main/docs/images/demo.jpg)
*Original geometry tests validation and exports only. No inference occurs; scores, thresholds and research flags are null. Demo exports cannot become research reports.*

<details>
<summary>Model Catalog preview</summary>

![Model Catalog showing method provenance and unavailable local artifacts](https://raw.githubusercontent.com/GiannisPap24/factoryvision-ai/main/docs/images/catalog.jpg)

Method descriptions and model cards remain accessible without weights.

</details>

If a model is unavailable, that is expected in a fresh checkout. There is currently no public weight bundle. See [missing-model guidance](docs/local-demo.md#missing-or-damaged-model-files) and [troubleshooting](docs/local-demo.md#troubleshooting) before changing local settings.

## Architecture and local privacy

One Streamlit frontend calls reusable inference services. Immutable registries verify artifact size, SHA-256 and provenance before CPU loading with `weights_only=True`. The launcher binds to localhost, disables usage telemetry and retains CORS/XSRF protection. Use one trusted local session; do not expose it publicly.

Uploads stay in process memory and are not saved by the app. Clear them when finished; secure memory erasure is not claimed. Only upload permitted images. Input limits, session lifecycle and the limits of protected-test detection are documented in [architecture](docs/architecture.md), [inference](docs/inference.md) and [reporting](docs/reporting.md).

Human reviews and notes also stay in the current session. Save a decision explicitly, then download its JSON/CSV if needed. Exports include all saved revisions and notes, so enter no private information. Changing inputs, models or categories clears associated reviews; no database, feedback upload or automatic training is involved. [Review lifecycle and privacy limits](docs/milestone-5a-human-review.md).

## Tests and CI

The [verified hosted run for 8e8dade](https://github.com/GiannisPap24/factoryvision-ai/actions/runs/40380407763) passed **711 tests with 98.09% combined statement and branch coverage**, plus lint, lock/dependency checks, wheel/sdist inspection and fresh-install smoke checks. The badge above links to the current branch status. The percentage combines executed statements and branch outcomes; it is not the branch-only percentage. See [coverage interpretation](docs/ci.md#interpreting-coverage).

After the installation above (including development dependencies), run:

```powershell
.venv-31410/Scripts/python.exe scripts/ci_check.py --uv .tools/uv-0.13.5/uv.exe
```

Default tests use fabricated data and require no private assets. The local Human Review V1 candidate passes **797 tests (one Windows symlink-privilege skip)** with **98.18% combined coverage**; this is separate from the published baseline CI above. See the [5A verification record](docs/milestone-5a-human-review.md) and [CI details](docs/ci.md).

## Research results and limits

| Scope | Evidence and limitation |
|---|---|
| Cashew | Historical autoencoder image AUROC **0.93760**, recall **0.52** at the frozen threshold; **48/100 anomalies missed**. The official test set has since been observed. |
| Feature memory / student–teacher | Development diagnostics do not establish comparable independent defect accuracy. |
| MetalParts | **Exploratory source-domain comparison**, not independent test performance; no validated binary threshold. |
| PCB1 | **BLOCKED** by cross-partition source overlap. |
| PCB2 | **INCONCLUSIVE**; 56 similarity pairs await independent human adjudication. |
| RobustAD target domain | Evaluation **NOT CLEARED**; training/test independence remains unresolved. |

Scores have different units across methods and are not defect probabilities. Heatmaps use separate relative scales. A low score does not establish defect absence; category selection is manual and wrong-category inputs are not reliably detected. Read the [Cashew model card](docs/model-card.md), [MetalParts comparison](docs/milestone-3f-metalparts-feature-memory.md) and [reproduction guide](docs/reproduction.md) before interpreting results.

## Documentation and licence

Start with the [documentation index](docs/README.md) for user guides, engineering references and historical research records.

[Setup and workflows](docs/local-demo.md) · [Architecture](docs/architecture.md) · [Development](docs/development.md) · [Roadmap](docs/roadmap.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

FactoryVision-owned source and original synthetic content: [MIT](LICENSE), Copyright (c) 2026 GiannisPap24. VisA (Zou et al., 2022), AmazonScience RobustAD, pretrained weights and dependencies retain separate terms and attribution. No third-party dataset images or weights are redistributed. See the [licensing inventory](docs/licensing.md). Screenshots show only the public interface and original geometry; these repository presentation assets are excluded from wheel and sdist packages.
