# FactoryVision AI

[![Source verification](https://github.com/GiannisPap24/factoryvision-ai/actions/workflows/verify.yml/badge.svg?branch=main)](https://github.com/GiannisPap24/factoryvision-ai/actions/workflows/verify.yml)
[![Python 3.14.10](https://img.shields.io/badge/Python-3.14.10-blue)](docs/local-demo.md)
[![MIT licence](https://img.shields.io/badge/License-MIT-teal)](LICENSE)

**A local visual anomaly inspection research app built with PyTorch and Streamlit.**
Explore six frozen methods, compare spatial evidence and export research results.
Try the interface and synthetic workflow after installation—no dataset or model weights needed.

![FactoryVision Overview with six registered methods and explicit research status](https://raw.githubusercontent.com/GiannisPap24/factoryvision-ai/main/docs/images/overview.jpg)
*Overview from the public app without weights. Zero models are ready for real inference.*

**Research demonstration. Not validated for industrial acceptance, food safety or automated product-quality decisions.**

## What it does

- Eight workspaces: Overview, Inspect, Compare, Batch Inspection, Reports, Model Catalog, Research and Demo.
- Six registered methods: four Cashew methods (autoencoder, spatial baseline, ResNet18 feature memory and student–teacher) and two experimental MetalParts methods.
- With verified local artifacts: explicit CPU inference, aligned heatmaps, overlays, reconstructions where supported, model-specific scores and latency.
- Batch inspection of up to 20 images, with validation, retained error rows and deterministic CSV/JSON research reports.

## What you can run

| Available after a fresh installation | Requires verified local research artifacts |
|---|---|
| Overview and historical Research summaries | Real single-image inspection |
| Model Catalog and downloadable model cards | Multi-model comparison and anomaly maps |
| Synthetic Demo and its CSV/JSON examples | Batch inference and completed research reports |

The public repository contains **no datasets, checkpoints or pretrained weights**. Unavailable models remain disabled; there are no fabricated predictions or automatic weight downloads. See [artifact prerequisites](docs/reproduction.md).

![Model Catalog showing method provenance and unavailable local artifacts](https://raw.githubusercontent.com/GiannisPap24/factoryvision-ai/main/docs/images/catalog.jpg)
*Catalog: method descriptions and model cards remain accessible without weights.*

## Quick start: clone to first demo

Verified platform: **Windows AMD64 · Python 3.14.10 · uv 0.13.5**. Git and an internet connection are needed for the initial checkout and dependencies.

1. In PowerShell, choose a working folder and clone the public repository:

   ```powershell
   git clone https://github.com/GiannisPap24/factoryvision-ai.git
   if ($LASTEXITCODE -ne 0) { throw 'Clone failed' }
   Set-Location factoryvision-ai
   ```

2. Follow [first installation](docs/local-demo.md#first-installation) to download and verify project-local Python and uv. It includes exact download URLs, SHA-256 checks and the Python signature check; no activation or security-policy change is needed. Then, from the repository root:

   ```powershell
   $env:UV_PROJECT_ENVIRONMENT = Join-Path (Get-Location) '.venv-31410'
   $env:UV_PYTHON_DOWNLOADS = 'never'
   .tools/uv-0.13.5/uv.exe sync --locked --extra app --python .tools/python-3.14.10/python.exe
   if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
   .venv-31410/Scripts/python.exe -m factoryvision.app
   ```

3. Open **http://127.0.0.1:8501**, choose **Demo → Generate synthetic demo**, then download the synthetic CSV or JSON. Stop the server with **Ctrl+C**.

![Synthetic demo with original geometry and empty prediction fields](https://raw.githubusercontent.com/GiannisPap24/factoryvision-ai/main/docs/images/demo.jpg)
*Original geometry tests validation and exports only. No inference occurs; scores, thresholds and research flags are null. Demo exports cannot become research reports.*

## Architecture and local privacy

One Streamlit frontend calls reusable inference services. Immutable registries verify artifact size, SHA-256 and provenance before CPU loading with `weights_only=True`. The launcher binds to localhost, disables usage telemetry and retains CORS/XSRF protection. Use one trusted local session; do not expose it publicly.

Uploads stay in process memory and are not saved by the app. Clear them when finished; secure memory erasure is not claimed. Only upload permitted images. Input limits, session lifecycle and the limits of protected-test detection are documented in [architecture](docs/architecture.md), [inference](docs/inference.md) and [reporting](docs/reporting.md).

## Tests and CI

Published on **2026-10-26**. The [verified hosted run for f461b4f](https://github.com/GiannisPap24/factoryvision-ai/actions/runs/40377255928) passed **711 tests with 98.09% branch coverage**, plus lint, lock/dependency checks, wheel/sdist inspection and fresh-install smoke checks. The badge above links to the current branch status; historical results do not assert that later runs passed.

After the installation above (including development dependencies), run:

```powershell
.venv-31410/Scripts/python.exe scripts/ci_check.py --uv .tools/uv-0.13.5/uv.exe
```

Default tests use fabricated data and require no private assets. See [CI details](docs/ci.md).

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

[Setup and workflows](docs/local-demo.md) · [Interface](docs/interface.md) · [Catalog and Demo](docs/model-catalog-demo.md) · [Development](docs/development.md) · [Roadmap](docs/roadmap.md) · [Publication record](docs/publication.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

FactoryVision-owned source and original synthetic content: [MIT](LICENSE), Copyright (c) 2026 GiannisPap24. VisA (Zou et al., 2022), AmazonScience RobustAD, pretrained weights and dependencies retain separate terms and attribution. No third-party dataset images or weights are redistributed. See the [licensing inventory](docs/licensing.md). Screenshots show only the public interface and original geometry; these repository presentation assets are excluded from wheel and sdist packages.
