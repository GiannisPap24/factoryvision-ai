# Milestone 3E — MetalParts development diagnostics and experimental app

25 October 2026. **COMPLETE — METALPARTS DEVELOPMENT DIAGNOSTICS & EXPERIMENTAL APP VERIFIED.**
Local source-development work only.

**EXPLORATORY SOURCE-DOMAIN DIAGNOSTICS — NOT INDEPENDENT TEST PERFORMANCE**

## A. Reviewed Milestone 3D checkpoint

Local commit `2282bfb811c38ff26440b0cae56b8bdc053f9661`:
`feat: train experimental MetalParts autoencoder`, parent
`5452bb46df1c2a0e363f9730e4250f41fbda69da`. Approved repository-local GitHub
noreply identity was used. All eleven expected 3D files were reviewed; added
content passed credential/private-path screening and staged whitespace checks.
Before committing, 551 tests passed with one existing skip, Ruff lint/format,
lock and dependency checks passed, all 56 sealed execution sources matched,
and the pilot/source evidence and four Cashew artifact checks passed. The tree
was clean immediately after the commit. No push or publication occurred.

## B. Exact source-domain inventory

Pinned AmazonScience/RobustAD revision
`82c642b2d18be60f2dab104f8dd4dea1f4ccfacb`; source metadata SHA-256
`7d587bc34e31e29d2fae95f7486d024d5e20b8477ab24048a33f2e5d387e65f1`.
The source audit checked all 231 train images and 51 masks against the saved 3C
file inventory, opening 283 source files including metadata. Every image is
2681 × 1500 RGB. No missing/inconsistent source record or mask was found.

| Source records | Count | 3E use |
|---|---:|---|
| 3D normal fitting parents | 144 | Membership/content audit only |
| 3D normal validation parents | 36 | Frozen full-image comparison |
| Source-training anomalies | 51 | Frozen full-image diagnostics |
| Official test0–test6 images/masks | 0 opened | Excluded |

The complete stable IDs, full source-relative image/mask paths, dimensions,
byte/pixel hashes, positive-mask counts and original 3D assignments are in ignored
`reports/metalparts-3e-v1/diagnostic-manifest.json`. The 180 normal assignments
exactly match the 3D split; all 51 anomalies match its excluded IDs. No anomalous
record entered fitting or checkpoint selection. No sample was relabelled or
removed to improve a metric. Scoring opened 138 source files: 87 images and 51 masks.

## C. Frozen checkpoint and protocol

Unchanged checkpoint `runs/metalparts-ae-pilot-v1/best.pt`, SHA-256
`c23543a8d22a305193e95a36985331c57b86ecb8b71e86c58b29a82fb1ab8651`.
The separate [3E protocol](metalparts-development-diagnostic-protocol.md), source
inventory, selected IDs, preview IDs, execution-source hashes and artifact catalog
were sealed before any real-anomaly forward pass. Manifest SHA-256:
`9811dc9222b6241fb284bbfe27b8286905d26b398e7c38c37e2dfc603dcc48d4`.
The seal command refuses an existing manifest; evaluation refuses code/input drift
and any existing output directory. All sealed files still matched after execution.

Unchanged conv-ae-v1, 37,763 parameters, CPU float32, deterministic algorithms,
four threads, eval/inference mode. Batch one; native 256-pixel crops, stride 192,
row-major full grid with exact trailing-edge coverage (112 tiles per source parent).
RGB uint8 becomes float32 [0,1]. Uniformly average overlapping reconstructed RGB,
then compute channel-mean squared residual; image score is the native map mean.
No whole-image resize, padding, smoothing or fitted normalization. Synthetic and
actual first-validation-normal results match the accepted 3D native diagnostic
exactly, including map, reconstruction and score. No model, threshold or inference
setting was tuned after viewing anomaly results.

## D. Real source-anomaly evidence

All 51 source anomaly masks were hash-verified, single-channel, native-aligned,
nonempty and limited to 0/255. Binarization uses >0. Normals use derived zero masks.
No mask resizing enters metrics. Official target masks were not opened. Source
anomalies contain 84,903 positive pixels in total; they are real publisher
annotations, unlike the fabricated default-test and browser-debug probes.

## E. Exploratory image-level results

**EXPLORATORY SOURCE-DOMAIN DIAGNOSTICS — NOT INDEPENDENT TEST PERFORMANCE**

87 images: 51 anomalies, 36 validation normals; anomaly prevalence **58.6207%**.
Exact tie-aware AUROC **0.5969498910675382**; non-interpolated average precision
**0.7142665461395848**. Cached scores were independently checked with scikit-learn.
No binary threshold, classification accuracy, threshold recall or false-alarm rate
was calculated. Image AP depends on this comparison's class prevalence.

| Continuous native MSE | Minimum | Q25 | Median | Q75 | Maximum |
|---|---:|---:|---:|---:|---:|
| Validation normal | 0.00179812 | 0.00251831 | 0.00299550 | 0.00355605 | 0.00645152 |
| Source anomaly | 0.00187949 | 0.00258710 | 0.00335824 | 0.00457516 | 0.00756749 |

The distributions overlap substantially. These observations do not justify a
reliable image-level acceptance/rejection rule.

## F. Exploratory pixel-level results

**EXPLORATORY SOURCE-DOMAIN DIAGNOSTICS — NOT INDEPENDENT TEST PERFORMANCE**

All **349,870,500 native pixels** included: 84,903 positive and 349,785,597 negative;
positive prevalence **0.0002426697878** (0.024267%). Fixed 65,536-bin AUROC
**0.8671747847310998**; AP **0.02154141500770638**.

These are exact ranking metrics for `floor(float64(MSE)*65535)`, with bin width
1/65535, and explicitly approximate unquantized-score metrics. Fixed bins were
declared before scoring; no pixel sampling or data-selected edges. Ties receive
half credit for AUROC and grouped precision for AP. Synthetic tests match
scikit-learn on the same quantized values. The positive and negative histograms
are saved. Metrics return null/reason for absent classes. Low pixel AP despite
higher AUROC illustrates the importance of overwhelming background prevalence.

All 87 saved float32 maps retain 1500 × 2681 geometry. Their hashes and exact mean
scores were verified after the run without re-running real-anomaly inference.

## G. Validation, dependence and uncertainty limits

The 36 comparison normals already influenced checkpoint selection. Source
anomalies may depict related physical objects; no publisher capture/physical-item
IDs establish independence. Prior findings remain intact: 145 source/test0
basename correspondences, 626 heuristic train/test similarity candidates, zero
exact-byte or native-pixel image duplicates in the old audit. Only two
same-basename pairs also met its similarity rule. These are distinct evidence
types; none was dismissed or promoted to an independent grouping claim.
Existing relationship metadata is preserved in the manifest without opening targets.

Pixels and images are dependent. No independent confidence intervals, shifted
target-domain performance or production claims are justified. Official MetalParts
independent evaluation remains **NOT CLEARED**. Original 3C **BLOCKED**, PCB1
**BLOCKED**, PCB2 **INCONCLUSIVE** are unchanged. No new acquisition or training.

## H. Experimental Streamlit integration

A separate category selector defaults to Cashew. MetalParts uses its own pinned
artifact catalog and CPU service, sharing the existing result and visualization
contracts. Explicit upload, acknowledgement and Run action are required; selecting
a category, previewing an image or viewing research does not run a model.
Category/input changes clear stale results. Original, reconstruction, native error
heatmap, overlay, continuous MSE, measured timings and checkpoint/dataset provenance
are shown. Missing/incompatible artifacts produce a safe availability message.

The interface displays: **Experimental MetalParts model — trained on normal
source-domain images. Independent defect detection performance has not been
established.** Threshold and binary decision remain null; raw scores are not
compared with Cashew. All four Cashew methods and their controls remain available.

## I. Heatmaps and local visualizations

Numerical maps are native, unnormalized reconstruction errors. Display thumbnails
use at most 1024 pixels per side, bilinear image/map reduction and per-image
navy/teal/yellow scaling; overlays blend 45% heatmap. Display masks use nearest
neighbor only. The displayed ceiling is shown; brightness is not comparable across
images or a probability of defect. Substantial blur, object edges and patch seams
can dominate residuals; small defects can be diluted by global mean scoring.

Before inference, one normal validation ID and the first three anomaly IDs in
lexical order were fixed as examples. Research shows their verified local original,
reconstruction, heatmap, overlay and mask thumbnails. Selection was not based on
favorable performance. Third-party image bytes and weights are excluded from Git
and distributions. Preview paths/hashes are packaged; missing local files are
handled gracefully. No images were published.

## J. Actual timing and memory

Per-image native forward/stitch/map time: minimum **255.42 ms**, median
**358.63 ms**, p95 **363.30 ms**, maximum **371.35 ms**. Complete diagnostics
(source decode, maps, histograms, previews and normal replay) took **39.93556 s**,
with aggregate process CPU **155.34375 s** and peak working set **802,607,104 bytes**
(765.43 MiB). Preparation and initial model loading precede that timed section.
Four CPU threads and streaming histograms were used. Host-specific observations,
not a controlled performance benchmark. The browser synthetic 384 × 320 probe
reported 23.3 ms inference and 35.1 ms total on its first run.

## K. Cashew and historical preservation

The four Cashew implementations, pinned artifacts, thresholds, original metrics,
historical model cards and research evidence remain unchanged. The isolated installed package reproduced all four prior synthetic Cashew
scores, thresholds and artifact hashes exactly. Preservation compared 770 baseline
files: 761 unchanged, nine intended maintained-file edits, zero missing or
unexpected changes. Another 168 historical reports match the older preservation
inventory. The acquisition/source evidence and all historical run files are intact.
The original 3D protocol/checkpoint and both smoke/pilot runs are immutable.
The broad 3D execution seal included the UI; its original UI bytes are recoverable
at the 3D Git checkpoint. Current app changes belong to 3E and do not revise that
historical seal. No raw target payload was read for preservation checks.

## L. Offline quality and packaging

Final full suite: **586 passed, one existing skip**, 51.64 seconds. Combined
statement/branch coverage **98.530365%** (statements 98.877400%, branches 97.584541%).
The existing 95% gate remains intact. Skip: host lacks Windows symlink privilege;
junction tests pass without security changes. Ruff lint and formatting pass across
127 Python files. Offline lock consistency and all 61 development packages pass
compatibility checks. An earlier formatting-only check failure was repaired; the
complete subsequent quality run passed. No existing test was weakened.

Tests cover native overlap/edges/alignment, accepted-3D agreement, finite scores,
source-only guards and membership drift, invalid masks, metric ties/undefined
classes/quantization, pinned model/category compatibility, changed/missing
artifacts, safe loading, cached re-verification, upload limits, explicit app actions,
category switching, absent models and unchanged Cashew behavior. Default fixtures
are synthetic; real source checks use the explicitly invoked diagnostic runner.

Offline wheel and source-distribution builds passed. Archive inspection found
67 wheel members and 184 sdist members, with no images, weights, raw data, reports,
environments or caches. The generic build cache warning did not correspond to
cache content in either archive. A fresh `.tmp/wheel-env-3e` installed runtime/app
dependencies and the wheel offline, without the editable project or development
group; all 49 installed packages passed compatibility checks. `python -I` imported
site-packages and verified all four frozen Cashew regression outputs exactly.
The installed MetalParts service verified the checkpoint, returned a finite native
320 × 384 map on fabricated RGB and replayed it bit-for-bit from cache, with null
threshold/decision. Packaged research metadata and all 1,721 protected-image hashes
were present. Evidence is in `reports/metalparts-3e-v1/isolated-probe.json` and logs.

## M. VS Code and actual browser verification

Existing workspace, project Python **3.12.15**, local uv **0.13.4**, official
`openai.chatgpt@26.1021.22027`, Python/debugpy and Ruff extensions verified.
VS Code Refresh Tests completed, and Test Explorer displayed the new
`test_metalparts_inference.py` alongside the existing tests. The full suite was
run through the project quality command. Pytest configuration points to `tests`;
Ruff uses the local executable; the
explicit Streamlit task remains localhost-only with no folder-open launch.

The new **FactoryVision: debug frozen MetalParts inference** launch loaded the
actual frozen checkpoint using fabricated 384 × 320 RGB. VS Code stopped at
`scripts/debug_metalparts_inference.py:19` after inference. Locals showed native
shape (320,384), score 0.0902247279882431, null threshold/decision and the correct
checkpoint hash. F10 advanced to line 20; Continue completed with the Debug
Console confirmation `score=0.09022473; no dataset opened`. No training ran.

The in-app browser successfully loaded the running app, verified four Cashew
artifacts, selected MetalParts, uploaded fabricated pixels, acknowledged eligibility
and clicked Run. It displayed actual reconstruction/map/overlay, score, timings and
verified provenance. Research tables and a preselected local anomaly preview were
opened. Clearing removed the upload and results and disabled Run. A browser
WebSocket warning occurred during the intentional local server restart; the
reconnected final flow completed successfully. Browser controls are available and exercised in this session; the local
compatibility skill exists. No connected Chrome backend was listed; Chrome
installation was not assessed. This is not a Browser-plugin absence claim.

## N. Security and privacy

Existing 8 MiB encoded limit, 4096-pixel side limit, 16M-pixel limit, JPEG/PNG
format checks, single-frame/opaque/upright requirements, corruption/truncation and
decompression-bomb protections remain active. RGB normalization strips metadata;
backend inference takes bytes, never user paths or filenames. MetalParts additionally
requires 256 pixels per side, preventing hidden upscaling. The user's uploader
widget may show its selected filename locally; no image/filename is persisted by
the backend. Explicit clearing is not a secure memory-erasure guarantee.

The unchanged 550-entry original official-test hash registry is supplemented by
1,171 MetalParts target image byte/pixel hashes derived solely from prior 3C
metadata. Re-encoded exact pixels are blocked in either category. Crops and other
transformations cannot all be recognized, so prohibited test inputs still require
explicit user acknowledgement. No target image or mask was read to build this list.
The original source-only 3D guard was not weakened; 3E has a separate allowlist.

Only package-pinned, project-local, link/size/hash-checked CPU state_dict artifacts
load through weights_only and strict finite tensor validation. Uploaded checkpoints
and arbitrary file paths are unsupported. No network call, cloud inference,
telemetry, new dependency, public deployment, driver change or Windows security
change was introduced. The app remains bound to 127.0.0.1:8501.

## O. Exact changes and final Git status

Modified existing files (nine):

- `.vscode/launch.json`
- `README.md`
- `docs/architecture.md`
- `docs/development.md`
- `docs/inference.md`
- `pyproject.toml` (source-distribution inclusion only)
- `src/factoryvision/app/ui.py` (additive category routing; line endings normalized to LF)
- `src/factoryvision/inference/uploads.py` (additional protected hashes)
- `tests/test_app.py` (additional tests; existing tests preserved)

Added maintained files (thirteen):

- `docs/metalparts-development-diagnostic-protocol.md`
- `docs/metalparts-model-card.md`
- `docs/milestone-3e-metalparts-development-evaluation.md`
- `scripts/debug_metalparts_inference.py`
- `scripts/metalparts_diagnostic.py`
- `src/factoryvision/app/metalparts_ui.py`
- `src/factoryvision/data/metalparts_diagnostic.py`
- `src/factoryvision/evaluation/metalparts.py`
- `src/factoryvision/inference/metalparts.py`
- `src/factoryvision/inference/metalparts_artifacts.json`
- `src/factoryvision/inference/metalparts_protected_inputs.json`
- `src/factoryvision/inference/metalparts_research.json`
- `tests/test_metalparts_inference.py`

Final intended Git state: nine modified and thirteen untracked maintained text
files, none staged. HEAD remains the reviewed 3D commit. New 3E work is uncommitted;
no push or publication. Full inventories, maps, previews, logs, builds and isolated
environments remain ignored. Credential/private-path and whitespace checks passed.
No third-party pixels or trained weights were added to Git or either distribution.

## P. Recommended next milestone

Request a separately scoped development review of the frozen results and known
blur/seam/small-defect failure modes. Any changed inference rule or new training
must be a new protocol and explicitly authorized exploratory experiment, with
these observed source anomalies acknowledged as development evidence. Seek
publisher physical-item/capture provenance before considering independent
target-domain evaluation. Do not optimize thresholds or open test masks to improve
the current results. No later milestone was begun.

**COMPLETE — METALPARTS DEVELOPMENT DIAGNOSTICS & EXPERIMENTAL APP VERIFIED.**
