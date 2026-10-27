# Milestone 5B — Model Registry & Experiment Tracking V1

**COMPLETE — MODEL REGISTRY V1 VERIFIED**

Local acceptance: 2026-10-26 to 2026-10-27 (Europe/Athens).
The feature is not pushed, merged or published by this milestone.

## Baseline and scope

Human Review V1 was merged through [PR #1](https://github.com/GiannisPap24/factoryvision-ai/pull/1).
The accepted baseline is `648091dbe2301ee2429b0e4c19576f64479efc88`, with successful
[hosted merge CI](https://github.com/GiannisPap24/factoryvision-ai/actions/runs/40404981472).
The clean public-source checkout uses local branch `feat/model-registry-v1`, with
no upstream configured. The private development repository is outside the scope.
Python 3.14.10, uv 0.13.5, dependencies and the 95% coverage gate are unchanged.
The completed baseline CI was reused rather than repeated before implementation.

This is a read-only extension of the Model Catalog, not a second registration
database. No model training, evaluation, weight download or private-directory
discovery occurs. The original eight workspaces and six registrations remain.

## Authorities and smallest extension

| Concept | Existing authority | 5B projection |
|---|---|---|
| Model/category identity | `batch.models`, `reporting.model_info` | Existing CatalogEntry inside RegistryRecord |
| Family, method, preprocessing, outputs, limitations | `model_catalog.definitions` | Preserved Definition |
| Expected weights, frozen threshold, catalog identity | Packaged artifact manifests and ModelInfo | Artifact version `sha256:<expected digest>` |
| Local availability | Original inference service status/checkpoint verification | Verified file digest or null; no deserialization |
| Pretrained provenance | Catalog ResNet18 identity and pinned original manifests | Frozen extractor component, separate from student or memory fitting |
| Run, split, recipe/protocol identity | Existing packaged artifact manifests | Portable recorded IDs/digests; no path dereference |
| Cashew historical metrics | Packaged `model_cards/model-card.md` table | Image/pixel AUROC and AP at published precision |
| MetalParts diagnostics | Packaged `metalparts_comparison.json` | Original values, class counts and quantization scope |
| PCB1/PCB2 integrity status | Packaged Cashew card's application addendum | BLOCKED/INCONCLUSIVE summaries, no model registration or metrics |

The existing training/checkpoint contracts remain authoritative for training and
loading. The metadata view does not load checkpoints to recover missing fields.
`model_registry.py` adds immutable EvidenceReference, Metric, ExperimentEvidence
and RegistryRecord dataclasses. Schema-v1 evidence must equal its reviewed source
projection, including exact Python types. Invalid model/version relationships,
dataset substitutions, fabricated timestamps or changed metrics are rejected.

Each source reference contains the package resource and its SHA-256. The fixed
allowlist binds the adapters to reviewed source bytes. Source changes require an
explicit adapter/pin review; pins are consistency checks, not signatures against
an attacker who can replace the application code itself. JSON is deterministic,
scalar metadata only, and generated only for explicit download.

## Historical discovery and scientific meaning

Six evidence IDs use `history-<existing model ID>`. Two additional IDs,
`integrity-pcb1` and `integrity-pcb2`, describe integrity findings. These are index
identities, not invented historical execution/run identifiers. Recorded run IDs
are taken from manifest references. Missing timestamps, canonical configuration
fingerprints and unsupported provenance remain null. Recipe file SHA-256 and
protocol file SHA-256 are labelled separately from a canonical config hash.

- Cashew AE and spatial mean: the one historical official evaluation, now
  observed; 150 images, 100 anomalous and 50 normal. Pixel metrics cover
  2,457,600 resized pixels with 27,672 positives. No rerun or recalibration.
- Cashew feature memory and student: development-only evidence. Real-defect
  accuracy remains unavailable; artificial-probe and calibration diagnostics
  stay in the associated model cards and are not treated as real-defect metrics.
- MetalParts: exploratory source-domain diagnostics on 51 anomalies and 36
  previously used validation normals. Native pixel ranking uses 65,536-bin
  quantization; exact quantized and approximate unquantized interpretation is
  retained. There is no binary threshold. Independent target evaluation remains
  **NOT CLEARED**.
- PCB1 remains **BLOCKED** by overlap; PCB2 remains **INCONCLUSIVE**, with 56
  unresolved pairs and no independent human adjudication. Missing metrics are
  absent, not zero.

Historical Cashew execution provenance retains Python 3.12.15 from its model
cards. Current application Python 3.14.10 is not substituted for that record.
Other historical endpoints and detailed hyperparameters remain accessible in
the unchanged model cards; this index does not claim exhaustive run telemetry.
Human Review records remain observations, not verified ground-truth labels.

## Interface and portability

The existing **Model Catalog** contains all six cards and its original category
model selector. **Version and recorded provenance** exposes model/artifact and
dataset identity, missing fields and a JSON download. **Historical experiment
evidence** provides category and status filters, scoped metrics, missing-evidence
reasons and source fingerprints. Empty filter results are explicit. No global
metric ranking or production-readiness score is created.

Checkpoint availability and research evidence availability are independent:
source-only installations can inspect all eight summaries, while real inference
remains disabled without verified research artifacts. The navy/blue/teal design,
inspection, comparison, batch, reports, synthetic demo and Human Review flows
are preserved.

## Security and privacy review

The registry has no metadata import, external URL fetch, serialization execution,
checkpoint deserialization, local registration write or automatic upload. Reads
are limited to exact packaged resources and at most 256 KiB per resource. The
allowlist rejects traversal, absolute paths and URLs before file I/O. Manifest
asset paths are never followed by the evidence index. Package digest mismatch
and missing resources fail closed with a fixed public error message.

New UI data uses text, code, JSON and escaped table rendering, without dynamic
HTML. Export filenames use existing fixed model IDs. No notes, images, private
paths or secrets are collected by the registry. Source distributions exclude
datasets, model weights, local reports, temporary environments and screenshots.

No append-only runtime store was added: it would introduce competing state
without a V1 requirement. Future registration, import, persistence and cloud
tracking remain unimplemented. Local verification artifacts stay Git-ignored.

## Verification results

The complete local GitHub Actions equivalent finished with exit code **0**:

```powershell
.venv-31410/Scripts/python.exe scripts/ci_check.py --uv .tools/uv-0.13.5/uv.exe
```

| Check | Actual result |
|---|---|
| Full offline suite | 828 passed, 1 skipped; 12,597 existing warnings |
| Combined statement/branch coverage | 98.24%; required 95% unchanged |
| Statement / branch coverage | 98.61% / 97.12% |
| New registry contract and UI coverage | 100% statements and branches for both modules |
| Ruff lint and format | Passed; 171 Python files formatted |
| uv lock and dependency compatibility | Offline lock verification passed; development and isolated app dependencies compatible |
| Wheel / source distribution | Built offline; 93 / 263 members; no privacy, secret, forbidden-artifact or broken-document-link findings |
| Fresh isolated wheel | Installed without project source on the import path; catalog, registry, PCB1 status, demo, review, exports and resets passed |
| Source-only behavior | Verified from an empty directory with network connections denied; all six artifacts unavailable, all eight evidence summaries inspectable |
| Browser | Actual in-app browser; desktop 1280×720 and narrow 390×844, zero page-width overflow at 390 |
| VS Code | Real debugpy stop at `scripts/debug_registry.py:17`, inspected live values, continued to PASS and normal termination |
| Diff/security/privacy | Reviewed all 17 changed files; no unresolved security findings; scientific manifests/cards and inference/review code unchanged |

The original baseline passed 797 tests plus one skip at 98.18% coverage. 5B adds
31 tests. Targeted validation passed 63 tests; the final full run completed in
30.19 seconds. The one existing skip is the Windows symlink test because the host
does not grant symlink-creation privileges. A first malformed-package test needed a short pytest parameter ID
because Windows limits environment-variable length; no application defect was
involved. Browser inspection found automatic metric-table rounding; the new
table now renders exact numeric strings, with a regression assertion. The full
offline gate was rerun after that code correction and passed.

Browser verification exercised all six catalog cards, historical/blocked/
inconclusive/exploratory evidence, empty filter results, metadata JSON download,
missing-checkpoint handling and the source-only synthetic demo. The installed
app first ran without configuration or artifacts; visual verification then used
only the public `.streamlit/config.toml` copied into its isolated working folder
to preserve the existing navy/blue/teal theme. No dataset or checkpoint was added.
The browser control tool is available and was verified in this session. The
local Browser compatibility skill is installed. Chrome was not present in the
connected browser inventory; no claim of a Chrome extension connection is made.

The existing conspicuously labelled, ignored QA harness supplied synthetic test
doubles and original geometry for single-image inspection, four-method Cashew
comparison, mixed batch (two valid images and one rejected input), human review,
CSV/JSON downloads and review persistence between Reports and Batch Inspection.
Downloaded JSON retained the original score and separate human decision. These
checks demonstrate workflow preservation, not real-checkpoint accuracy. The
actual synthetic demo retained null scores/thresholds and no model identity.
Both inspected browser tabs reported zero error-level console entries.

At the actual debugger breakpoint, the live expression
`(len(records), len(evidence), tuple(e.status for e in evidence),
all(not r.catalog.available for r in records))` returned
`(6, 8, ('HISTORICAL', 'HISTORICAL', 'DEVELOPMENT ONLY', 'DEVELOPMENT ONLY',
'EXPLORATORY', 'EXPLORATORY', 'BLOCKED', 'INCONCLUSIVE'), True)`.
Continuing printed `PASS: six registry records and eight evidence summaries;
no model loaded.` Existing VS Code windows and unrelated sessions were preserved.

Local logs, coverage, screenshots, downloaded synthetic verification payloads
and the security/privacy audit are under Git-ignored
`reports/model-registry-5b/evidence/`. They are excluded from both distributions.
The final source distribution was rebuilt after completing this report.

## Change inventory

Created:

- `src/factoryvision/inference/model_registry.py`
- `src/factoryvision/app/registry_ui.py`
- `tests/test_model_registry.py`
- `tests/test_registry_app.py`
- `scripts/debug_registry.py`
- `docs/milestone-5b-model-registry.md`

Modified:

- `src/factoryvision/app/catalog_ui.py`
- `scripts/smoke_installed.py`
- `.vscode/launch.json`
- `pyproject.toml`
- `README.md`
- `docs/README.md`
- `docs/roadmap.md`
- `docs/architecture.md`
- `docs/model-catalog-demo.md`
- `docs/reproduction.md`
- `docs/milestone-5a-human-review.md`

All accepted artifact manifests, packaged model cards, scores, thresholds,
training/checkpoint code and Human Review contracts remain unchanged.

## Handoff and next scope

Proposed commit: `feat: add source-only model registry and historical evidence index`.
After owner review, stage only the listed source/documentation files, commit on
the existing feature branch, then request separate publication approval before
pushing or creating a remote PR. Proposed PR title: **Model Registry and
Experiment Tracking V1**. Include the acceptance results and scientific limits;
leave public main unchanged until the normal review/CI/merge procedure completes.

Recommended 5C: a separately approved robustness assessment plan with permitted
development inputs, explicit perturbation scope, immutable provenance and
predeclared acceptance criteria. No retraining, target-test opening or production
claims are implied. Milestone 5C has not started.
