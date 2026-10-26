# FactoryVision AI architecture

## Milestone 4C catalog and demonstration

The single Streamlit frontend adds Model Catalog and Synthetic demo pages.
`inference/model_catalog.py` projects existing compatible IDs, reporting identity
metadata and original service availability verification into frozen typed entries.
Descriptive method annotations do not become a second artifact registry. Catalog
inspection never deserializes checkpoints or executes inference.

`demo.py` generates original geometric inputs and validates a separate immutable
demo document with no scores or model identity. `app/demo_ui.py` stores its prepared
CSV/JSON in a dedicated session key; page/category changes clear it. Research
results and schema-v1 reports never accept this schema. Existing six inference
services, batch coordinator and report contracts are unchanged.

`scripts/ci_check.py` is shared by local source-only verification and the proposed
Windows workflow. It installs the built wheel into an isolated runtime and checks
the app from an empty directory. See [catalog/demo](model-catalog-demo.md) and
[CI](ci.md); hosted Actions has not run.

## Milestone 4A presentation and session lifecycle

The single Streamlit frontend shares `app/presentation.py` for local CSS, availability/help, workflow cards and category switching. Category changes clear both result sets, upload widget references and consent. Model/image/consent changes invalidate stale results. The UI catches CPU runtime/memory failures without exposing private exception text. No new service, model or dataset was introduced. Pillow was updated from 12.3.0 to 13.0.0 for documented decoder advisories; upload parsing now restricts formats before decoding. The application still performs no downloads or network inference. Model resources remain cached separately from per-session images; all existing identity checks still execute. See [local demo](local-demo.md).

FactoryVision AI is a local research package for future industrial visual anomaly detection and localization. Milestone 2A adds a compact autoencoder, real CPU training, reconstruction scores and provisional normal calibration. Milestone 2B adds a frozen held-out evaluator and one normal-only spatial mean baseline. Further models or deployment require separate authorization.

## Implemented package

~~~text
src/factoryvision/
  config/__init__.py      immutable validation settings
  domain/
    contracts.py         source, dataset, sample, split provenance and manifest
    errors.py            stable Issue and ContractError
    paths.py             pure cross-platform lexical path checks
    validation.py        pure manifest consistency/leakage rules
  data/
    archive.py           bounded tar inspection and selective Cashew extraction
    images.py            bounded RGB/mask decode, Dataset and CPU DataLoader
    visa.py              official CSV ingestion and exact-content integrity audit
    manifest.py          strict bounded JSON I/O and canonical fingerprints
    filesystem.py        opt-in file existence and containment checks
    splits.py            deterministic normal-only group assignment
  models/autoencoder.py compact convolutional RGB autoencoder
  training/config.py    bounded immutable experiment configuration
  training/engine.py    normal-only optimizer steps, validation and early stopping
  training/checkpoint.py atomic state_dict artifacts and compatibility checks
  evaluation/reconstruction.py image/spatial errors and normal quantile calibration
  evaluation/metrics.py standard ranking, threshold counts and paired bootstrap
  evaluation/pipeline.py bounded ordered inference and report aggregation
  evaluation/protocol.py immutable protocol and model artifact verification
  models/spatial_mean.py training-only spatial reference
  inference/             frozen six-model services, uploads, catalog, batches and reports
  app/                   optional local Streamlit interface
  demo.py                explicit synthetic workflow without model inference
  cli.py                 offline validate-manifest command
  __main__.py            python -m factoryvision entry point
tests/
  fixtures/              12 fabricated metadata records; no dataset images
scripts/
  check_quality.py       common terminal/VS Code checks
  debug_manifest.py      small debugger target
~~~

The domain depends only on the standard library. Data adapters depend on domain contracts; CLI orchestrates settings, loading and optional filesystem validation. No domain operation downloads data or reads image contents.

Constructors check individual types/values. Manifest construction checks shape and immutability; call validate_manifest to obtain sorted relational issues, or require_valid_manifest to raise ContractError. The JSON loader and split function always enforce relational validation. Keeping this distinction explicit allows tests and callers to inspect contradictory candidate manifests without mutation.

All containers inside accepted contracts are immutable tuples or frozen dataclasses. JSON decoding rejects missing and unknown fields. Source records distinguish genuine VisA provenance from fabricated fixtures. Canonical serialization sorts samples by stable ID before hashing; returned dictionaries are detached copies.

Filesystem resolution is optional and separate from pure validation. It rejects links/junctions escaping the root and requires existing files. It does not decode images, verify supplied hashes or provide an atomic secure-open operation against concurrent filesystem changes.

## Technology decisions

Python 3.12.15, local uv 0.13.4, pytest, pytest-cov and Ruff. Hatchling builds wheels and source distributions. Runtime dependencies are torch 2.13.0+cpu, torchvision 0.28.0+cpu, NumPy, Pillow and scikit-learn 1.9.1. The explicit CPU package index and Windows AMD64 lock prevent accidental GPU selection. The optional Streamlit frontend calls reusable in-process inference services. There is no database, remote inference service or experiment tracker. The sections below retain the sequence of research boundaries; the application and reporting sections describe the current implementation.

The domain remains standard-library only. Acquisition is an explicit script, separate from package imports and default tests. Archive preflight examines all members before writing only Cashew and necessary metadata into a new directory. VisA ingestion reconciles the one-class CSV with category annotations, hashes bytes/decoded RGB pixels and enriches the immutable manifest before splitting. Image loading decodes one sample on demand; zero-worker DataLoaders keep ordering and memory predictable. See dataset-preparation.md for resource limits, interfaces and caveats.

The implemented 37,763-parameter autoencoder is trained with Adam and MSE using 315 approved normals. A separate 67-normal validation partition selects the checkpoint, then 68 normals establish the predeclared provisional threshold. The script never constructs a test-model loader. A later PatchCore comparison needs separate pretrained-weight provenance. GPU use requires a new AMD compatibility checkpoint; no GPU backend is configured here.

Use simple explicit functions and contracts until real experiments establish additional requirements. Keep experiment orchestration in scripts/train_cashew.py and reusable mechanisms in the package. The explicit screening script and data/leakage.py compute heuristic cross-partition candidates without changing membership. See training.md for architecture, run evidence and remaining independence limitations.

## Frozen evaluation boundary

The preparation script fits only the mean reference and calibrates on normal calibration records, then seals reference/threshold, source and artifact hashes before test prediction. The one-shot evaluator verifies that seal, uses the same test loader for both models, saves raw maps/per-image results, and computes bounded pooled metrics. Metric replay and synthetic debugging are separate commands. Neither operation trains the AE or changes thresholds. See [evaluation-protocol.md](evaluation-protocol.md).

## Feature-memory development boundary

Milestone 2C adds models/feature_memory.py for frozen ResNet18 layer2/layer3 extraction and bounded 1-NN patch scoring; data/development.py restricts image access to normal development partitions and supplies a Python file-open allowlist plus deterministic checkerboards. evaluation/feature_development.py scores only validation/calibration normals and enforces calibration identity. training/feature_artifacts.py binds a weights-only tensor bank to recipe, pretrained-weight, manifest and fitting-ID provenance. scripts/feature_experiment.py orchestrates explicit one-shot fit/development evaluation. Existing 2A/2B source and artifacts remain unchanged; no new dependency, service or inference API was added. See [feature-memory.md](feature-memory.md) for exact shapes and deviations from published PatchCore.

## Student-teacher development boundary

Milestone 2D adds models/student_teacher.py (frozen raw ResNet18 layer2/layer3 teacher, compact trainable student and unit-feature discrepancy), training/student_engine.py (exact normal membership, finite gradients, bounded configuration and ordered validation/calibration scoring), and training/student_artifacts.py (student state_dict with teacher/dataset/recipe provenance). scripts/student_experiment.py performs explicit smoke, bounded training and development evaluation in new one-shot runs. The accepted feature-memory/AE implementations and artifacts remain unchanged. No service or new dependency is introduced. See [student-training.md](student-training.md).

## Milestone 3A application boundary

The optional Streamlit application sits above a reusable CPU inference service.
`inference/uploads.py` validates memory-only bytes; `inference/service.py` verifies
pinned assets and delegates to the original model implementations;
`inference/visualization.py` reprojects display maps without changing scores.
`app/ui.py` owns session presentation and explicit run controls; `app/__main__.py`
enforces loopback settings. No training or evaluation entrypoint is called.
See [inference design](inference.md) for identities, score semantics and privacy.
Model instances are process-cached, guarded by a lock and inference mode. Uploads
and results are session-local; changing inputs invalidates results.

## Milestone 3E experimental MetalParts path

The existing Cashew InspectionService and four adapters remain intact. A separate
MetalPartsService uses package-pinned metadata and checkpoint hashes, the original
safe CPU loader and complete native reconstruction. It returns the existing
InspectionResult contract with null threshold/decision. The app category selector
routes to separate presentation code and clears stale results when categories
change; model caches contain weights only. No upload is saved to disk.

Source-development diagnostics use a separate source-training allowlist, preserving
the stricter 3D normal-only guard. An audit freezes the exact source inventory,
3D split, code, protocol and checkpoint before one inference pass. Native maps are
streamed to ignored artifacts; fixed pixel histograms bound metric memory. Research
summary JSON contains descriptive numbers and verified local preview references,
not third-party image bytes. Existing official-test upload hashes are supplemented
with MetalParts hashes from prior metadata, without reading target payloads.

See the [3E protocol](metalparts-development-diagnostic-protocol.md) and
[MetalParts model card](metalparts-model-card.md).

## Milestone 3F addition

`models/metalparts_memory.py` adapts the frozen feature extractor to native tiles
and traces each sampled descriptor to its permitted normal fitting parent.
`training/metalparts_memory.py` validates the separate category-bound bank identity
and weights-only artifact. `evaluation/metalparts_comparison.py` preserves the
existing native pixel ranking semantics with a bounded extensible distance histogram.
`inference/metalparts_feature.py` verifies package-pinned assets before each cached
use. The app selects either MetalParts method or both; its view never fits a bank.
Existing AE, Cashew models, upload guards and old result registries are preserved.
See [frozen protocol](metalparts-feature-memory-protocol.md) and
[3F report](milestone-3f-metalparts-feature-memory.md).

## Milestone 4B1 batch orchestration

`app/batch_ui.py` owns one session-local selection and immutable completed report.
`inference/batch.py` validates byte payloads, category/model compatibility, ordered
content identity and budgets. `BatchService` calls the original `inspect` methods
one image at a time. No preprocessing, loader, scoring or threshold is duplicated.
Only compact PNG display previews and scalar results survive each iteration;
decoded images/maps are released. Progress callbacks report finished rows, while
the UI publishes results only after a complete return. A started marker prevents
silent replay after success or interruption. No inference occurs during preview,
widget reruns or startup. See [4B1](local-demo.md) for schema,
error boundaries, measured resources and limitations.

## Scalar reporting boundary (4B2)

`inference/reporting.py` projects the authoritative frozen BatchReport into validated
immutable scalar contracts; it has no inference, Streamlit or output-file dependency.
Approved installed metadata supplies model identity and frozen thresholds.
`app/report_ui.py` stores the report and CSV/JSON bytes beside their exact source
snapshot in session state. Existing batch/category lifecycle reset paths remove
this state. Downloads use prepared bytes without rerunning the script. No global
report cache, service, database or secondary inference path is introduced.
See [reporting and security contract](reporting.md).

## Industrial Precision presentation (4D-UX)

One Streamlit app retains the existing inference boundary. `app/presentation.py`
centralizes headings, overview, native tables, legends and session transitions.
`app/theme.toml` is packaged and explicitly selected by `python -m factoryvision.app`;
the source `.streamlit/config.toml` carries the same theme for direct development.
Static CSS only targets documented keyed-container classes and semantic elements.
It supplies card surfaces, focus/selection cues, wrapped labels and contained table
scrolling; no untrusted HTML, external fonts or UI dependencies are introduced.

Inspect permits one compatible method; Compare permits several. Both use the
original adapters. Page/category changes reset uploads, results and consent. The
Batch Inspection/Reports pair retains a completed immutable snapshot; returning
from Reports displays an archive view until Clear batch. This explicitly handles
Streamlit removing uploader widget state when the widget leaves the page.

Result tabs share aligned originals, warm heatmaps, overlays and supported
reconstructions. Only RGB display stops changed: numerical anomaly maps, per-image
maximum normalization, bilinear display interpolation and 45% overlay alpha are
unchanged. Model-card copies and the theme are owned text resources in the wheel.
