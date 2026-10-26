# Milestone 3A — Local visual inspection application

25 October 2026. Scope: local Cashew inference only. This milestone adds an interface
and verification evidence, not new model accuracy. PCB1 remains **BLOCKED**; PCB2
remains **INCONCLUSIVE — 56 unresolved pairs**, with no independent human adjudication.

## A. Recovery checkpoint

`bb820b874b98a014ba654b12c817c89e56773c39` —
`docs: preserve PCB2 independent adjudication status`.
Exactly the nine pending 2F0-A files were committed after baseline checks, using the
approved repository-local noreply identity. Parent:
`b708c6dee86e44bb50e87ad902e6ebe91cb05d55`. No remote push occurred.
Baseline: 448 tests passed, one existing symlink skip; Ruff and lock/dependency checks passed.

## B. Exact created and modified files

Modified (8): `.vscode/launch.json`, `.vscode/tasks.json`, `README.md`,
`docs/architecture.md`, `docs/development.md`, `docs/model-card.md`,
`pyproject.toml`, `uv.lock`.

Created (16):

- `.streamlit/config.toml`
- `docs/inference.md`
- `docs/milestone-3a-completion.md`
- `integration_tests/test_inference_real.py`
- `scripts/check_application_quality.py`
- `scripts/debug_inference.py`
- `src/factoryvision/app/__init__.py`
- `src/factoryvision/app/__main__.py`
- `src/factoryvision/app/ui.py`
- `src/factoryvision/inference/cashew_artifacts.json`
- `src/factoryvision/inference/protected_inputs.json`
- `src/factoryvision/inference/service.py`
- `src/factoryvision/inference/uploads.py`
- `src/factoryvision/inference/visualization.py`
- `tests/test_app.py`
- `tests/test_inference.py`

Generated evidence is ignored under `reports/application-3a-preflight-v1/` and
`reports/application-3a-verification-v1/`; distributions and the isolated environment
are ignored under `.tmp/dist-3a-final/` and `.tmp/wheel-env-3a/`. No images, model weights,
reviewer ZIPs or generated outputs are added to Git.

## C. Application architecture

The optional Streamlit interface delegates byte validation, pinned artifact loading,
CPU inference and display reprojection to separate inference modules. Original
model, training, data and evaluation sources remain authoritative and unchanged.
Only model objects use the process cache; uploads/results stay in session memory.
An explicit button performs inference in a lock and `torch.inference_mode()`.
See [inference design](inference.md) for contracts and the complete threat boundary.

## D. Available models and artifact identities

All four existing Cashew artifacts are available and verified. Full checkpoint,
teacher, recipe and dataset identities and thresholds are recorded in
[inference design](inference.md#architecture-and-frozen-identities) and the packaged
`cashew_artifacts.json`. Safe original loaders use CPU `weights_only=True`; no
pretrained weights are downloaded implicitly. Availability and every explicit run
check file length, SHA-256, dataset membership and frozen calibration identity.

Of 2,459 baseline files, 2,451 remain byte-identical. The only eight changed baseline
files are the authorized app docs, dependency metadata and VS Code entries above.
No missing or unexpected changed files were found. Historical full-source seals
include earlier versions of these eight files: their original bytes remain at the
recovery checkpoint, and the entire old seal is not claimed to match the evolved
workspace. All data, models, protocols, prior experimental sources and historical
reports remain unchanged. Evidence: `preservation.json` in the verification directory.

## E. Actual inference capabilities

Each method returns a model-specific score, strict frozen-threshold comparison,
128×128 raw spatial map, original RGB, timing and checkpoint identity. AE additionally
returns reconstruction; spatial mean returns its frozen reference. Feature methods
do not claim a reconstruction. Selection order is retained without raw-score ranking.
No training, fitting, recalibration, masks or official evaluation occurs.

The explicit real test read only `visa-cashew-normal-000` and `-001`, both normal
validation members, twice per model. Repeated scores/maps were identical and model
parameter hashes stayed unchanged. An audit hook allows only those image paths and
rejects all other dataset images/masks; network connections are blocked. This is a
functionality check, not an accuracy or throughput study. Evidence: `real-inference.json`.

## F. Screens and visualizations

Overview, Image inspection and Research results provide model availability, bounded
upload/preview, explicit single/multiple-model selection, scores, threshold wording,
latency, original/heatmap pairs, optional overlay, AE reconstruction or mean reference,
and model identities. Aspect ratio is restored for display with bilinear interpolation.
The navy–teal–yellow legend is normalized per image with its raw maximum stated;
colors are not defect probabilities. Exact numerical captions remain readable when
metric tiles truncate in a narrow window. Input/selection changes clear stale results.

## G. Local startup and application verification

`.venv/Scripts/python.exe -m factoryvision.app` started the Uvicorn-backed Streamlit
server on `127.0.0.1:8501`; the listening address was checked. The launcher disables
telemetry, static serving and developer toolbar actions, while preserving CORS/XSRF.

The callable computer/browser tool and local browser skill were present. The `iab`
backend explicitly returned unavailable and the browser connection inventory was
empty; Chrome connection was not established. This does not establish an uninstalled
browser plugin. A successful native VS Code integrated-browser fallback rendered
the app. Its file chooser uploaded permitted normal validation image `000.JPG`;
preview appeared and Run remained disabled until acknowledgement. Clicking Run
produced the actual AE result: score 0.00231656, threshold 0.00315313, forward/score
6.4 ms, total service 43.7 ms. The original and anomaly heatmap were inspected.
This was a real browser interaction, separate from automated AppTest coverage.

## H. Input safety and privacy

Synthetic tests exercise bytes/type/size limits, decoded format, dimension/pixel
limits, transparency, orientation, animation, truncation, corruption, bomb warnings
and errors, RGB conversion and metadata stripping. Metadata-derived exact byte/pixel
hashes reject known official test images without opening those files during this
milestone. Transformed images can evade that exact-match guard; no OOD claim is made.

Inference accepts no arbitrary user paths/model files. Tests cover containment,
linked/missing/modified assets, invalid outputs, strict threshold semantics and no
filesystem writes. Default tests prohibit network connections. UI AppTest permits
only Windows asyncio's private loopback socket pair; application connections remain
blocked. No upload bytes, filename or EXIF are logged/persisted by application code.
Framework/browser transient memory remains; clearing is not secure memory erasure.

## I. Actual CPU latency observations

Four Torch threads, batch one, two normal validation images; repeated cached-model
forward-plus-score observations (not a performance benchmark):

| Model | Repeated inference range | Repeated total service range |
|---|---:|---:|
| Autoencoder | 1.45–1.76 ms | 34.82–35.13 ms |
| Spatial mean | 0.61–0.61 ms | 33.58–34.76 ms |
| Feature memory | 9.55–11.13 ms | 70.47–89.95 ms |
| Student–teacher | 5.43–5.68 ms | 62.91–63.25 ms |

Total includes decode/preprocess and rechecking artifact hashes. Browser/upload/render
and debugger overhead are excluded. Cold starts, operating-system cache and contention
affect these timings. Historical benchmark numbers are not replaced.

## J. Offline tests and coverage

Complete default suite: **496 passed, one existing Windows symlink privilege skip**.
The new focused suites add 48 passing synthetic tests. The explicit real development
test passes separately. Statement and branch coverage retain the existing 95% combined
gate; no source exclusions or weakened assertions were introduced. Final percentages
are **98.94% statement, 97.63% branch, 98.58% combined**, recorded in `coverage.json`
and `quality-console.txt`.

Decoder and visualization coverage is 100%. Tests exercise actual Streamlit navigation,
all four synthetic adapter outputs, explicit run controls, errors, absent capabilities,
stale-state clearing and launch flags. Initial validation found and fixed PNG EXIF/
verification ordering and bounded Torch test threads to resolve Windows threaded
AppTest instability. Those failed attempts are not counted as passing checks.

## K. Ruff, lockfile and packaging

Ruff lint/format, offline uv lock consistency and all 61 installed dependency checks
pass. Streamlit 1.66.1 is an optional app dependency; all 33 original dependency
versions are unchanged. Wheel and source distribution build offline. Archive member
inspection checks packaged catalogs and excludes real data, weights, environments,
reports and caches; the generic uv cache-location warning was checked against actual
archive members. A separate runtime/app environment installs the wheel without the
editable project; `python -I` imports site-packages and verifies all four artifact sets.

The new quality runner preserves historical reports and targets maintained source
directories, avoiding frozen generated snapshots. Every command exit code is stored
in `quality.json`; builds/install evidence is in the verification directory.

## L. VS Code debugging verification

The explicit startup task, focused inference tests, existing pytest integration and
Ruff remain available. The new debug configuration runs a synthetic image through
the actual pinned AE model. No task starts on workspace open. Native VS Code debugging
stopped at `inference/service.py:236` with reason **breakpoint**, in `inspect`, called
from `scripts/debug_inference.py:14`. Locals showed `model_id='autoencoder'` and
`flag=True` for the fabricated RGB probe. Step Over advanced through the inference
context exit; Continue completed the run. This is actual debugpy execution, not a
simulated breakpoint or a training run. The probe score was 0.00552295 with a
128×128 map. The normal-image browser check and synthetic debugging are separate.

## M. Scientific limitations displayed

The interface states Cashew-only support, no reliable OOD detector, research status,
no industrial validation, scores/colors not probabilities, and possible defects below
threshold. Historical Cashew AE/mean image and pixel metrics are labelled as recorded
evidence. Feature/student official defect performance remains unestablished. PCB1 is
BLOCKED; PCB2 is INCONCLUSIVE with 56 unresolved pairs. No new PCB metrics, human
reviews, official test predictions or independence guarantees are invented.

## N. Final Git status

HEAD remains the recovery checkpoint. Eight tracked files are modified and sixteen
new source/config/docs/test files are untracked; none are staged. Milestone 3A is not
committed or pushed. Exact final status and preservation comparison are retained in
the ignored verification directory. No dataset download, driver change, training,
official test reevaluation or cloud deployment occurred.

## O. Recommended Milestone 3B improvements

Separately authorize bounded local result export, broader accessibility/interaction
checks and camera-interface design using provenance-controlled demonstration data.
Consider model-load progress and resource profiling before any multiuser interface.
Resolve physical-group/acquisition independence before new accuracy claims. PCB2
requires independent human adjudication or acquisition evidence, not another model
run. No 3B implementation, training, evaluation or publication has begun.
