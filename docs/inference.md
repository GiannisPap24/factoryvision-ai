# Local Cashew inference — Milestone 3A

## Milestone 4A application behavior

Category precedes navigation; model selection precedes upload. Explicit runs produce at most two result columns per row with aligned RGB/maps, overlay, labelled per-image ceiling, model-specific score and CPU timing. MetalParts has no threshold. Missing-model help links the setup and reproduction guides; fresh installations can read research without assets. Runtime failures clear attempted results. Category changes discard uploads and consent as well as results; uploads are not persisted by the app.

FactoryVision AI is a CPU research demonstration, not an industrial acceptance system.
Only Cashew images under comparable imaging conditions are supported. Valid JPEG/PNG
syntax does not establish category membership or distribution compatibility; there is
no reliable OOD detector. PCB1 remains **BLOCKED** due to source overlap. PCB2 remains
**INCONCLUSIVE**, with all 56 flagged pairs awaiting independent human adjudication.

## Launch and use

From the authoritative repository root, on Windows AMD64 / Python 3.12.15:

```powershell
.tools/uv-0.13.4/uv.exe sync --locked --extra app
.venv/Scripts/python.exe -m factoryvision.app
```

The optional app extra pins Streamlit 1.66.1. Existing ML dependency versions remain
unchanged. After initial dependency installation, inference requires no internet.
Visit `http://127.0.0.1:8501`. Stop the server with Ctrl+C. The VS Code task
**FactoryVision: local inspection app** runs the same entrypoint, explicitly only.

Overview shows artifact availability. In Image inspection, upload a permitted Cashew
image, choose one or more available methods, acknowledge the category/test-data
restriction, and press **Run inspection**. Neither upload nor model selection runs
inference. Changing image, selection or acknowledgement clears stale results.
Clear image and results removes current application references and resets the uploader.
Research results displays historical metrics, never recomputes evaluation.

## Architecture and frozen identities

`inference/service.py` accepts bytes and fixed model IDs, independent of Streamlit.
`uploads.py` validates/decodes in memory. `visualization.py` creates display panels.
`app/ui.py` handles presentation and session state. `app/__main__.py` enforces a
loopback-only launch, disabled telemetry and static serving, and retained CORS/XSRF
protection. Existing model/training/evaluation implementations are unchanged.

The package catalog fixes dataset, fitting IDs, preprocessing resolution, model
artifacts, metadata, recipes, calibration records and pretrained backbone hashes.
Files are checked for containment, links, length and SHA-256 before deserialization;
the original loaders use CPU `weights_only=True`. This is local hash pinning, not a
publisher signature or protection against a malicious owner who changes the code.
Concurrent local modification between hash checking and loading is outside this
single-user prototype's threat boundary. No arbitrary path/model upload is accepted.
Missing or incompatible files disable the method with an explanation; nothing is
downloaded, manufactured, fitted or calibrated as a fallback.

| Method | Weight SHA-256 | Frozen threshold |
|---|---|---:|
| Autoencoder | `da7e0caed8e365a08a3d301ec7e88536bc5df8e3f8b59f27635853e15ab4cdbc` | 0.0031531278509646654 |
| Spatial mean | `04dc72dd50c54214d7bf874f650da58bf53eb80c803d550f8a75c7368bfd4c63` | 0.017185416119173163 |
| Feature memory | `4acd638fbef5c167c5121a12adf6dd796773cf0205420f6d317ace64fee1b1b0` | 0.08742312490940093 |
| Student–teacher | `d5b087c3c1a1a1a3937cd2d4b275cf9c605b774e760e7200f2a7262fa0f95ed9` | 0.9195413202047348 |

Teacher SHA-256: `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
Canonical Cashew manifest SHA-256:
`80bac86ae2fc1b0ad7e0e046597fb4a3a1e1f4f92ce556d5dddbfdd37d51f84c`.
The complete file hashes live in `inference/cashew_artifacts.json`. Local artifacts
remain in their existing ignored `runs/`, `data/manifests/` and `data/pretrained/`
locations. Launch from the repository root; a wheel does not contain these assets.

All models use the original 128×128 PIL RGB bilinear antialiased resize and float32
division by 255. The feature extractor retains its own ImageNet normalization.
The AE and mean score is the spatial mean of channel-mean squared residuals.
Feature memory uses the original maximum raw feature-grid nearest-neighbor RMS score;
student–teacher uses the maximum raw normalized feature discrepancy. Their existing
bilinear 128×128 maps are preserved. A threshold flag uses strict `score > threshold`.
Equality is below-or-at. Scores and colors are **not probabilities**, and raw scores
across methods are not comparable. The UI does not rank or select winning models.

Training provenance: all methods fit 315 normal development images; validation has
67 and calibration 68. The 37,763-parameter AE completed 12 real CPU epochs; the
112,128-parameter student completed 480 updates over 12 epochs with frozen teacher.
Feature memory stores 5,040 selected normal patches; spatial mean is a fitted RGB
reference with no optimizer. See the existing Model Cards for accepted provenance.

## Image safety, privacy and scope

Only decoded single-frame JPEG/PNG files up to 8 MiB, 4096 pixels per side and 16M
pixels are accepted. Headers, full decoding and PNG integrity are checked; corrupt,
truncated, decompression-bomb, non-upright EXIF and nonopaque inputs are rejected.
Grayscale/palette/opaque RGBA become RGB. A new pixel-only image drops EXIF, ICC,
comments and text. No metadata is executed and filenames are not passed to inference.

The packaged exclusion list contains metadata-derived hashes for 150 Cashew, 200 PCB1
and 200 PCB2 official test images. Exact bytes are rejected before decoding; exact
dimension-qualified decoded RGB matches are also rejected. Generating the list read
existing inventories only. Reencoded exact pixels are covered, but transformed images
are not reliably detected. This is not a category/OOD detector or a public upload firewall.

Uploads and results live in memory; only models use the process cache. The app does
not write uploads, persist scores, or log filenames/EXIF/payloads. Streamlit and the
browser hold transient upload/media buffers, and the browser may display its chosen
filename. Clearing references/disconnecting is not secure memory erasure; browser/OS
memory and crash behavior remain outside the application's guarantee. No user data
is sent to an external inference service. The server is unauthenticated and intended
for one trusted local workstation, not unrestricted public exposure. No deployment,
database, remote API, camera access or automated acceptance action is included.

## Visualization and resources

Preprocessing uses the historical square geometry. For display, maps/reconstruction
are bilinearly reprojected onto the original aspect ratio, preview max edge 1024.
The original appears beside the heatmap; the overlay uses 45% heatmap opacity.
Navy→teal→yellow maps zero through each image's maximum raw map value. A zero map
stays navy, and a constant positive map is uniform. The legend gives the raw maximum.
Per-image brightness cannot compare severity between images or models. These are
coarse visual cues, not calibrated defect masks; small defects and object boundaries
can be misleading. The AE shows reconstruction; the mean shows a **reference**;
feature methods do not claim an RGB reconstruction.

Four CPU threads, batch size one, local model cache and serialized per-service
inference are used. A modern CPU and several GB of free RAM are appropriate; no GPU
is required. The UI separately reports preprocessing, artifact verification/loading,
forward-plus-score and total service time. Browser/upload/render time is not included.
Cold loading and repeated hashing add overhead; observations are not a throughput SLA.

## Scientific limits and next work

Historical 2B Cashew image AUROC/AP: AE 0.937600/0.943219, mean 0.736800/0.841216.
Frozen-threshold recall: 0.52/0.15, false positives 3/50 and 0/50. Pixel AUROC/AP:
AE 0.818119/0.080936, mean 0.895301/0.267302. These are the existing single observed
test run, with independence and distribution limitations described in the
[Model Card](model-card.md). Feature/student official Cashew defect metrics are not
established; artificial development probes cannot substitute for real defects.

A separately authorized 3B could add bounded result export, camera-interface design,
better accessibility/interaction checks and an independently collected demonstration
set with acquisition provenance. Any new category, training, threshold tuning or
accuracy benchmark requires separate authorization and a valid integrity protocol.
Unresolved PCB2 adjudication cannot be replaced by application demonstrations.

## Experimental MetalParts (Milestone 3E)

Choose Category → MetalParts (experimental), then Image inspection. Upload an
explicitly permitted MetalParts JPEG/PNG, confirm it is not an official test image,
and press Run MetalParts inspection. Category recognition is not automatic.
The result shows original, reconstruction, native residual heatmap, overlay,
continuous MSE, measured CPU time and verified provenance. There is no binary
classification or calibrated threshold. Cashew remains a separate default option
with its original methods, metrics and thresholds.

MetalParts uses the frozen 3D checkpoint, native 256-pixel patches, stride 192,
batch one and full edge coverage. Average overlapping reconstructed RGB before
squared channel residuals; average the native map for the image score. Inputs must
be 256–4096 pixels per side, at most 16M decoded pixels and 8 MiB encoded. No
whole-image resize or padding is used for scoring. Presentation reduces images to
at most 1024 pixels and scales color per image; intensity is not defect probability.

Model bytes, metadata and source/split/category identity are pinned separately from
Cashew. No user model path is accepted. Existing safe decoding controls and CPU
weights_only loading apply. Known MetalParts test hashes supplement the unchanged
Cashew/PCB registry, including native-pixel matches after re-encoding. Transformed
copies cannot all be identified; official test0–test6 inputs remain prohibited.
Uploads and results stay in session memory; category changes clear results, and
Clear removes the current upload widget/results. Widget filenames may be visible
in the user's browser; the backend does not retain filenames or metadata.

Research results shows the original 3D losses, explicitly labelled descriptive
source metrics and four preselected local-only examples. Missing model artifacts
or preview assets produce an availability message. All official target-domain
performance remains NOT CLEARED. See the [3E report](milestone-3e-metalparts-development-evaluation.md).

## MetalParts feature-memory comparison (3F)

Choose MetalParts (experimental), Image inspection, then select Convolutional
Autoencoder, Feature-Memory Detector or both. Upload a permitted image and press
Run MetalParts inspection. Selection changes clear stale results. The two methods
show separately labelled heatmaps, continuous scores, latency and provenance;
AE reconstruction remains available. There is no binary decision or probability.
AE mean RGB MSE and maximum feature RMS distance have different units, and heatmap
colors scale per method/image. Do not rank methods by raw scores or color brightness.

Feature inference requires the pinned local `runs/metalparts-feature-v1/memory.pt`,
its metadata and existing pretrained ResNet18 file. Assets are not bundled and
are never downloaded implicitly. Missing or modified files disable that method.
Research results preserves historical AE evidence and adds the labelled exploratory
comparison with shared failures and dependence limits. Independent MetalParts
evaluation remains NOT CLEARED. See [feature model card](metalparts-feature-model-card.md).

## Bounded batch inputs (4B1)

The Batch inspection page uses the same byte-only decoder and artifact-verified
services as single-image inspection. It accepts no user filesystem paths. Count
(1–20 configurable), 32 MiB aggregate bytes and existing per-image checks apply
before preview/inference; corrupt or prohibited inputs become rejected rows.
Per-image failures continue; artifact/CPU failures stop further inference and
mark remaining accepted rows skipped. Error text from unexpected exceptions is
not exposed. Safe positional IDs replace filenames in results and previews.
No images are written by this workflow, logged or sent to external AI services.
Clear/navigation/category changes drop references; secure memory erasure is not
claimed. Streamlit's upload receipt and browser cache are separate from the
application's aggregate acceptance budget. No multi-user deployment is claimed.
See [local usage](local-demo.md) and [4B1 evidence](local-demo.md).

## Research report exports (4B2)

Completed batches now expose a Quality Control Research Report and explicit CSV/JSON
downloads. Inference and frozen artifacts are unchanged. Failed/rejected/skipped
images retain null predictions; MetalParts retains no threshold. Validation occurs
before bytes are exposed, and arbitrary errors, paths, filenames, EXIF and image
bytes are excluded. Spreadsheet cells receive formula/control-character protection
in addition to CSV quoting. See [report schema and security](reporting.md).
