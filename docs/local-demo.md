# Local demonstration setup

**Runtime:** Windows AMD64, Python 3.14.10, uv 0.13.5.
Owned source is MIT licensed. This source-only release contains no datasets or weights.

## First installation

Verified: Windows AMD64, Python 3.14.10, uv 0.13.5. Other platforms and
Python versions are not verified. You need Git and internet access for the first
installation. Open PowerShell in a folder where you want the checkout:

```powershell
git clone https://github.com/GiannisPap24/factoryvision-ai.git
if ($LASTEXITCODE -ne 0) { throw 'Clone failed' }
Set-Location factoryvision-ai
```

If already cloned, open PowerShell at that repository root instead.

### Download and verify local tools

Run these bootstrap blocks once for a new checkout. Existing verified tools can
be reused; the guards deliberately refuse to overwrite their directories. The following
blocks use project-local tools, without global installation or activation. Obtain
and verify the official standalone
[uv 0.13.5](https://github.com/astral-sh/uv/releases/tag/0.13.5):

```powershell
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force '.tools' | Out-Null
$uvArchive = Join-Path (Get-Location) '.tools/uv-0.13.5-windows.zip'
if (Test-Path '.tools/uv-0.13.5') { throw 'uv directory already exists; verify and reuse it explicitly' }
Invoke-WebRequest 'https://github.com/astral-sh/uv/releases/download/0.13.5/uv-x86_64-pc-windows-msvc.zip' -OutFile $uvArchive
if ((Get-FileHash -LiteralPath $uvArchive -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'a38b8d1902b7cfbc76b423ba1bb456f623f42064cdc72628b6075b89afd1019a') { throw 'uv archive digest mismatch' }
Expand-Archive -LiteralPath $uvArchive -DestinationPath '.tools/uv-0.13.5'
$uvVersion = & .tools/uv-0.13.5/uv.exe --version
if ($LASTEXITCODE -ne 0 -or $uvVersion -notmatch '^uv 0\.13\.5(?: |$)') { throw 'uv version verification failed' }
```

Next obtain the official full Python Windows runtime (not the embeddable ZIP).
This verifies the pinned digest and PSF signature before execution. Do not overwrite
an existing runtime directory.

```powershell
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force '.tools' | Out-Null
$archive = Join-Path (Get-Location) '.tools/python-3.14.10-amd64.zip'
Invoke-WebRequest 'https://www.python.org/ftp/python/3.14.10/python-3.14.10-amd64.zip' -OutFile $archive
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'cc28026a650970cc1570bb127312f1f9634eb67abfe157ae6ca1acbbc627d091') { throw 'Python archive digest mismatch' }
if (Test-Path '.tools/python-3.14.10') { throw 'Runtime directory already exists; verify and reuse it explicitly' }
Expand-Archive -LiteralPath $archive -DestinationPath '.tools/python-3.14.10'
$signature = Get-AuthenticodeSignature '.tools/python-3.14.10/python.exe'
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'O=Python Software Foundation') { throw 'Python signer verification failed' }
& .tools/python-3.14.10/python.exe -c "import sys; assert sys.version_info[:3] == (3, 14, 10)"
if ($LASTEXITCODE -ne 0) { throw 'Python version verification failed' }
```

### Install locked dependencies

Run from the repository root after verifying the tools. The default sync includes
development dependencies so the same environment supports the demo and CI checks.
Initial installation needs internet access; `UV_OFFLINE` must not be set to `1`
unless all required dependencies are already cached.

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path (Get-Location) '.venv-31410'
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
$env:UV_PYTHON_DOWNLOADS = 'never'
.tools/uv-0.13.5/uv.exe sync --locked --extra app --python .tools/python-3.14.10/python.exe
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
```

Set `UV_PROJECT_ENVIRONMENT` in
external terminals before every project uv operation; VS Code supplies it to new
integrated terminals. Workspace automatic activation is disabled; the configured
PATH selects the interpreter without running an activation script. In an external terminal, verify with
`.venv-31410/Scripts/python.exe -c "import sys; print(sys.executable, sys.version)"`
and `.tools/uv-0.13.5/uv.exe run --locked --offline python -c "import sys; print(sys.executable)"`.

Initial dependency installation uses upstream package sources. Cached checks can
use `--offline`. The explicit PyTorch CPU index is preserved. No dataset or model
weights are downloaded. No activation, global PATH edit or security-policy change
is required. The app listens only on `127.0.0.1:8501`; stop it with Ctrl+C.
Run from the research root only when using approved existing artifacts.

## First synthetic demo

Start the app from the repository root:

```powershell
.venv-31410/Scripts/python.exe -m factoryvision.app
```

Open **http://127.0.0.1:8501**. Choose **Demo** in the sidebar, then **Generate
synthetic demo**. Two original geometric images pass validation and one deliberately
invalid input is rejected. Choose **Download synthetic demo CSV** or **Download
synthetic demo JSON** to save the examples explicitly. No file is written automatically.

This is **not real inference or defect detection**. Scores, thresholds and research
flags are null; the demo schema cannot be used as a Quality Control Research Report.
No datasets, weights or private files are needed. Model Catalog and its model-card
downloads are also available. Inspect, Compare and batch research reports require
the verified artifacts described below.

For later launches, run `.venv-31410/Scripts/python.exe -m factoryvision.app` from
the repository root. You do not need to repeat downloads or installation.

## Troubleshooting

| Symptom | What to do |
|---|---|
| Models show **Unavailable** and inspection is disabled | Expected in the public checkout. Use Catalog and Demo; real inference requires the exact verified artifacts below. |
| `uv` or `python` is not recognized | Use the explicit `.tools/uv-0.13.5/uv.exe` and `.venv-31410/Scripts/python.exe` paths from the repository root. Activation and global PATH changes are unnecessary. |
| The Python environment or Streamlit is missing | Complete **Install locked dependencies** with `--extra app`; launch with that environment's interpreter. |
| An uncached dependency cannot be found in offline mode | For an online installation, remove the terminal-only override with `Remove-Item Env:UV_OFFLINE -ErrorAction SilentlyContinue`, then repeat the locked sync. Do not change the lockfile to resolve a cache miss. |
| Port 8501 is already in use | Stop the previous FactoryVision session with Ctrl+C in its terminal. If another application owns the port, resolve that conflict before launching; do not terminate unrelated processes. |
| An archive digest, Python signer or version check fails | Stop setup. Check the official source and pinned values; do not bypass verification or disable Windows protections. |

Linux, macOS, GPU execution and other Python versions are not verified release
targets. See [development checks](development.md) for the supported local gate.

## Missing or damaged model files

Overview lists availability. Inspection stays disabled when a selected model is
unavailable. Research summaries remain readable without assets; local example
images show an unavailable message. No predictions are fabricated.

Restore the original approved files at the relative paths recorded in
`src/factoryvision/inference/*artifacts.json`, including their dependent metadata
and pretrained backbone. Every model must pass hash and semantic checks.
Do not edit hashes to force an arbitrary checkpoint to load. A missing download
link is intentional: no model redistribution package has been licensed or published.
[Reproduction prerequisites](reproduction.md) describe creating new research runs.
Fresh source installation is supported; one-command reproduction of sealed private
experiments from source alone is not currently provided.

## Inspecting an image

Select Category, open Inspect, select one model, upload JPEG/PNG, confirm
permitted use and press Run inspection. Use Compare to select multiple compatible
models and view their separately scaled maps side by side. Uploads remain in process memory; results update only
on explicit execution. Changing image, models or consent invalidates results.
Changing category clears both categories' result and upload references. A browser
reload starts a new Streamlit session; disconnected session retention follows
Streamlit's session timeout. Clear explicitly when done; no secure erasure is claimed.
The app has no login and is intended for one trusted local owner.

Use self-created, non-sensitive images for demonstrations. Synthetic inputs test
software behavior only and say nothing about real defect performance. Scores are
method-specific; read the heatmap ceiling and interpretation beside each result.
Large MetalParts images require many native tiles and more time and RAM. If CPU
inference fails, clear and retry with one model or a smaller supported input.
Do not resize an official test image to bypass its protection.

## Batch inspection

Open **Batch Inspection** in the sidebar, choose the category and one compatible
model, and use **Choose batch images** to select multiple PNG/JPEG files. Reduce
the image limit if desired (1–20, default/hard maximum 20). Review accepted and
rejected previews, confirm permitted category-matched inputs, then press **Run
batch inspection**. Processing is sequential on CPU. Completion lists successful,
rejected, failed and skipped rows; choose an individual result to see its aligned
original, heatmap, overlay and reconstruction/reference where supported.

Changing category, model, uploads or limit invalidates results. Details and other
reruns do not repeat inference. Use **Clear batch** to release the selection and
results, or to explicitly retry an interrupted/completed batch. Navigation away
clears batch state except between Batch Inspection and Reports, which retain the
completed snapshot. Returning to Batch Inspection shows a retained report until
Clear batch starts a new selection. Reports offers deterministic CSV/JSON exports.
There is no cancellation, resume or background processing. Cashew flags use frozen research thresholds only;
MetalParts shows continuous scores without a binary decision.

Uploads are capped at 8 MiB each and the accepted selection at 32 MiB total.
The aggregate/count checks occur after Streamlit receives uploads: they are not
network transport or operating-system memory quotas. Use one trusted local
session; many simultaneous uploads/sessions can still exhaust memory. Display
previews have a maximum side of 384 pixels; full-resolution maps are discarded
after each image. Original validation caps remain 4096 per side and 16M pixels;
MetalParts requires at least 256 per side. See the
[batch and report contract](reporting.md).

## Licensing and release

- [VisA official registry](https://registry.opendata.aws/visa/): CC BY 4.0,
  Zou et al. (2022), *SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection
  and Segmentation*. Attribute the source and indicate modifications.
- [AmazonScience RobustAD](https://huggingface.co/datasets/AmazonScience/RobustAD):
  dataset card declares CC BY 4.0. The preserved MetalParts revision is
  `82c642b2d18be60f2dab104f8dd4dea1f4ccfacb`. Review the full card and included notices
  before any redistribution; historical local acquisition is not a publication licence review.
- [Torchvision pretrained-model notice](https://github.com/pytorch/vision#pre-trained-models):
  weights can carry terms derived from their training datasets, separate from source-code licensing.
  ResNet18 `IMAGENET1K_V1` originates at
  `https://download.pytorch.org/models/resnet18-f37072fd.pth`, SHA-256
  `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
  Commercial redistribution rights have not been cleared by this milestone.
- FactoryVision-owned source and synthetic content use MIT, Copyright (c) 2026
  GiannisPap24. Third-party dependencies, datasets and weights keep separate terms.

The public README screenshots show the source-only interface and original synthetic
geometry. Private research previews, audit reports and workstation details are
excluded. See the [publication record](publication.md) and [successful hosted CI](ci.md).

## Batch research reports

After explicit batch execution, choose Open quality report or Reports. Review
the Quality Control Research Report and
choose Download research CSV or Download research JSON. Exports use the completed
snapshot and never rerun inference. Invalid results disable downloads; partial
failures remain visible with null predictions. Switching inputs/model/category or
clearing invalidates exports. Save only when needed and handle downloaded results
as local research records. See [schema, privacy and workflow](reporting.md).
