# Milestone 2C completion — 24 October 2026

Completed: a frozen, CPU feature-memory prototype, one normal-development experiment, artificial localization probes, reproducibility checks and future experiment designs. Real industrial anomaly accuracy remains unestablished. No new official Cashew test evaluation occurred. Machine-readable evidence is in milestone-2c-verification.json (local evidence, not distributed: `docs/evidence/milestone-2c-verification.json`).

## A. Git checkpoint and final status

The accepted 2B inventory and evidence were verified before implementation, including 26 file records, preserved artifacts and frozen execution sources. Baseline quality passed: 300 offline tests, one skip, 98.70484% combined coverage, Ruff, lock verification and dependency compatibility. The authorized local commit is **a0e89c2f8e86444fd3f498de3c0949fa5e832e73**, message **feat: evaluate initial visual anomaly detection models**, using the approved repository-local noreply identity. The tree was clean immediately after that checkpoint.

Final branch is main, with four local commits, no remotes and nothing staged. The 25 Milestone 2C files below remain uncommitted. There was no push or publication. Historical autoencoder weights, preprocessing, calibration, metrics, baseline, Model Card, frozen protocol and run artifacts are preserved byte-for-byte; hashes were checked without replaying official-test metrics.

## B. Exact created/modified files

Seven modified files:

- `.vscode/launch.json` — explicit fabricated-input feature debugging.
- `.vscode/tasks.json` — feature tests, fit, development evaluation, permitted-data checks and Ruff.
- `AGENTS.md` — current scope, preservation and next-milestone boundaries.
- `README.md` — current development outcome, historical distinction and links.
- `docs/architecture.md` — feature modules and data isolation.
- `docs/development.md` — commands, packaging and VS Code guidance.
- `pyproject.toml` — include new methods/model/planning documents in source distributions; no dependency change.

Eighteen created files:

- `docs/evidence/milestone-2c-verification.json`
- `docs/evidence/resnet18-provenance.json`
- `docs/feature-memory.md`
- `docs/feature-model-card.md`
- `docs/milestone-2c-completion.md`
- `docs/protocols/feature-memory-development-v1.json`
- `docs/protocols/feature-memory-development-v1.json.sha256`
- `docs/protocols/pcb1-confirmatory-v1.md`
- `docs/protocols/pcb1-confirmatory-v1.sha256`
- `docs/student-teacher-proposal.md`
- `integration_tests/test_feature_development_real.py`
- `scripts/debug_features.py`
- `scripts/feature_experiment.py`
- `src/factoryvision/data/development.py`
- `src/factoryvision/evaluation/feature_development.py`
- `src/factoryvision/models/feature_memory.py`
- `src/factoryvision/training/feature_artifacts.py`
- `tests/test_feature_memory.py`

Ignored local outputs comprise the pretrained weight/provenance files under `data/pretrained`, 20 files under `runs/cashew-feature-v1`, quality/coverage and wheel-check reports, two distribution archives, and temporary verification scripts/environments/caches. The evidence JSON enumerates every feature-run file with its size and hash. Originals, weights, maps and generated figures remain outside Git. No earlier package source, old regression test, lockfile or historical experiment document was rewritten.

## C. Implemented feature-memory architecture

The [methods](feature-memory.md) and immutable [recipe](protocols/feature-memory-development-v1.json) specify full-frame RGB128, bilinear antialiased resize, division by 255 and ImageNet channel normalization. Frozen ResNet18 through layer3 supplies layer2 `[128,16,16]` and layer3 `[256,8,8]`. Each receives 3x3 local average pooling; layer3 is bilinearly aligned to 16x16 and concatenated into 384-dimensional descriptors. The retained extractor has 2,782,784 frozen parameters; zero were optimized.

Sample-ID order and per-image SHA-256-derived Torch seeds select 16 of 256 patches from each training image: 5,040 patches from 80,640 candidates. The hard cap is 8,192. Exact chunked RMS Euclidean nearest-neighbor distances use default query/bank blocks 256/512, maximum 512 per side. Maximum patch distance is the image score; bilinear interpolation reconstructs the continuous 128x128 map. There is no pixel threshold.

This is PatchCore-inspired, with random deterministic sampling rather than greedy coreset selection, a smaller backbone, unweighted maximum scoring and no Gaussian smoothing. It is not an exact published PatchCore reproduction. Recipe SHA-256 is `62ebced7c38f87a73a31099895de9120dc2320a0906bf3ed7263523f13b656fa`; it was frozen before fitting and score inspection.

## D. Pretrained weight provenance

Installed torchvision 0.28.0 identifies official `ResNet18_Weights.IMAGENET1K_V1`. The provenance record (local evidence, not distributed: `docs/evidence/resnet18-provenance.json`) records the official HTTPS URL, default certificate validation, refused redirects, 46,830,571 bytes and SHA-256 **f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec**. The upstream filename's published eight-character hash prefix was verified; the full hash is locally recorded, not independently signed. The download was below the authorized 100 MB bound and required no credentials.

The loader performs no implicit download and validates size, hash, keys, tensor shapes/types/finiteness before strict state loading with weights_only. Official legacy V1 omits 20 BatchNorm counters; the narrowly defined compatibility path restores those counters while requiring all learned weights and running statistics.

Torchvision's BSD code license does not establish unrestricted weight rights. Official documentation warns of dataset-derived terms; ImageNet's separate access-agreement endpoint could not be retrieved. Local research use and no redistribution are recorded; no blanket commercial clearance is claimed. Source links and review details are in the methods document.

## E. Fitting data and leakage protections

Only the existing **315 normal training images** fit the bank. The **67 normal validation** and **68 normal calibration** images have disjoint IDs and never enter fitting. Normal calibration alone sets the threshold; validation remains diagnostic. Manifest fingerprint stays `80bac86ae2fc1b0ad7e0e046597fb4a3a1e1f4f92ce556d5dddbfdd37d51f84c`.

DevelopmentDataset rejects the test partition before decoding. Batch checks enforce expected partition, label and ID order. A Python file-open audit hook allowlists exactly the 450 normal development paths, with source-image hash checks. Both fit and development logs record 450 unique permitted files and **zero official-test image/mask files**. This includes preflight hashing; it does not mean all 450 images were used to fit. Metadata contains historical membership information but did not guide model selection. The guard protects this Python pipeline, not arbitrary native access at OS level.

Source snapshots, manifest/recipe/weight identities and artifact hashes guard loading and evaluation. Existing run directories and evaluation-start markers prevent overwriting results. Historical exact-duplicate checks found none; prior near-duplicate review remains limited and physical/acquisition independence is unverified. No new claim of independence follows from disjoint IDs.

## F. Actual development-stage results

Provisional image threshold: **0.08742312490940093**, the float64 linear 95th percentile of 68 calibration scores, strict greater-than.

| Normal partition | n | Minimum | Median | q95 | Maximum | Above threshold |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 67 | 0.06195803 | 0.07229325 | 0.08581289 | 0.10171185 | 2/67 (2.985%) |
| Calibration | 68 | 0.06304479 | 0.07153935 | 0.08742312 | 0.10850630 | 4/68 (5.882%) |

Calibration exceedances describe the same data used to set the threshold; they are not an independent false-positive estimate. Neither subset estimates real anomaly recall.

Sixteen deterministic 16x16 magenta/green checkerboards, with 4x4 cells and ID-derived locations, were inserted into the first 16 normal validation images. Their masks cover exactly the overwritten pixels. All 16 increased the score, exceeded the frozen image threshold and placed a map maximum within the artificial mask. Median paired score increase was **0.15725876**. Synthetic-only pooled pixel AUROC was **0.99798822**, AP **0.88751292**, over 262,144 pixels with 4,096 artificial positives. Four predetermined example panels were visually inspected; responses are spatially broad.

These are mechanics/geometry checks on conspicuous artificial defects. They neither predict real industrial performance nor establish superiority over the autoencoder. No new official Cashew AUROC, AP or pixel metric was computed.

## G. Memory and CPU measurements

Measured on Ryzen 7 9700X, Windows AMD64, Python 3.12.15, four Torch CPU threads, batch eight for fitting/scoring, zero workers. Successful fit took **3.1513 s**; development evaluation/reporting took **3.1321 s**, excluding imports/preflight. Peak working sets were **478,789,632 bytes (456.61 MiB)** and **468,221,952 bytes (446.53 MiB)** respectively, within the 2 GiB bound. The 600 s fit and 300 s evaluation budgets were not approached.

The bank tensor is **7,741,440 bytes**; serialized artifact is **7,743,017 bytes**, SHA-256 `4acd638fbef5c167c5121a12adf6dd796773cf0205420f6d317ace64fee1b1b0`. Default pairwise distance matrices occupy 0.5 MiB, hard maximum 1 MiB each; this is not a total-RAM bound.

| Batch-one operation | Median ms | p95 ms |
|---|---:|---:|
| Feature extraction | 3.40775 | 4.930205 |
| Nearest-neighbor search | 4.71600 | 5.163330 |
| Full preloaded feature/search/map | 9.44965 | 11.029735 |
| Warm-cache decode/resize | 7.59400 | 8.073910 |

Inference components used five warmups and 30 serial repeats on the first normal validation image; decode used 30 already-warm repeats. Component medians are not additive. Startup, cold disk, camera I/O and debugger time are excluded. No production throughput guarantee is implied.

## H. Offline tests and coverage

The complete offline suite passed **338 tests, one skipped**. All 300 historical tests remain; 38 new tests cover shapes, frozen parameters/buffers, partition/ID isolation, deterministic sampling, numerical distances, map/image scores, chunk equivalence and bounds, invalid/missing/corrupt weights, artifact identity and synthetic masks. Default tests use fabricated images/weights and no network or VisA files. The existing Windows symlink privilege skip remains; relevant junction containment checks pass.

Combined statement/branch coverage is **98.92229%**, above the unchanged 95% gate: **1,264/1,275 statements** and **480/488 branches**. Each of the four new package modules has 100% statement and branch coverage. The five explicitly invoked real-development checks passed: historical byte preservation, frozen sources/artifacts/read logs, all 450 permitted normals' hashes/geometry, real bank reload and normal calibration, and saved synthetic map replay. The unrestricted historical integration suite was deliberately not run because it decodes official-test images.

## I. Ruff, lockfile and packaging

Ruff lint and formatting passed for all 85 Python files. Offline uv lock verification and the 33-package development compatibility check passed. No dependency pin or lockfile changed; lock SHA-256 remains `d3b2b5efb9996aa50ef74e2019a64ca917a6a4a59d6090153e13ce0592b89168`.

Offline source/wheel builds passed. Archive inspection checked exact source bytes and excluded datasets, images, weights, outputs, environments and caches. uv warned that its in-source cache might be included; the actual archives exclude it. A fresh runtime-only `.tmp/wheel-env-2c` installed the built wheel and passed compatibility for 20 packages. Isolated Python imports resolve to installed site-packages; dev test packages are absent. Module CLI help and the real frozen-backbone/bank probe succeeded on fabricated RGB, with score 0.24296748638153076 and map `[1,128,128]`. That check denied all real-image reads and matched all installed source bytes. The historically blocked Windows console launcher was not treated as verified; Python module/scripts remain the supported local invocation.

## J. VS Code verification

The existing workspace exposes feature-model unit tests, feature-memory fit, development-only evaluation, permitted-data verification and Ruff tasks. Fit/evaluation entries match the completed one-shot CLI commands and were not rerun to duplicate the experiment. No task runs when the folder opens; no global settings changed.

Using the Computer Use skill, the real feature-scoring breakpoint at `models/feature_memory.py:140` was reached with the pretrained backbone and fitted bank, but fabricated RGB. Stepping computed the image score; continuing printed the finite score/map/bank shapes and ended normally. The new breakpoint was disabled afterwards. The focused VS Code task passed **38 tests in 2.67 s**. The selected interpreter was the existing project `.venv`. Browser access was not required or claimed.

## K. Limitations and scientific validity

Mechanics, isolation and small CPU resource use are demonstrated. Real-defect detection and localization are not. Limitations include one category/recipe, deliberately obvious synthetic squares, coarse 16x16 feature geometry, distortion from square resizing, ImageNet domain shift, deterministic random patch subsampling, small calibration/validation populations, and unknown physical/acquisition independence. The historical observed Cashew test motivated this work and cannot become an unbiased selection benchmark again.

Failures and fixes are retained in the account: initial lint/format findings were corrected before successful fitting; the first fit attempt stopped before run creation because the custom loader rejected legacy BatchNorm counters, then a narrow compatibility fix and regression test resolved it. An initial real-data test transposed expected height/width; its assertion was corrected to `[1176,1274]`, with preprocessing unchanged. There was one successful fit and one development evaluation, without score-driven retuning. A UI state query used an incorrect argument shape during verification; rereading the documented API recovered it. The ImageNet terms endpoint limitation and existing Windows restrictions remain disclosed. No security policy was changed.

## L. Proposed student-teacher experiment

The [proposal](student-teacher-proposal.md) specifies planning only: frozen ResNet18 teacher, separately randomly initialized matching student, layer2/layer3 normalized feature matching, training normals only, Adam 0.001, batch eight, seed42, four threads, maximum 20 epochs/600 s. Select the strict best normal-validation loss; calibrate only afterwards on the separate normal calibration subset. Record teacher immutability, student updates, reload equivalence, resource use and artificial diagnostics. Do not tune against official Cashew results or synthetic AP. No student network was trained or implemented in 2C.

## M. Pre-registered untouched-category plan

The local [pcb1 v1 design](protocols/pcb1-confirmatory-v1.md) is sealed by SHA-256 **5e2a40897a880fcb2741d09ed8cc701d187d9001d15a0d3ecb130151fcdb0c1a**. It enrolls feature memory and spatial mean, requiring fresh pcb1 normal fitting and calibration. It defines deterministic development splits, capacity rules, artifact/threshold freezing before test access, exact/near-duplicate gates, pooled pixel AP as primary endpoint, comparator AP difference, secondary image/pixel metrics, fixed mask geometry and conditional uncertainty. It is a local dated design, not a publicly registered trial.

Student-teacher enrollment requires a pre-test v2 amendment. Tested resource/cap implementations are required before executing the plan; the old evaluator cannot silently discard pixels to fit its smaller ceiling. Confirmed leakage stops the run before metrics; unknown independence remains explicit. This would evaluate category-specific fitting, not zero-shot Cashew-to-pcb1 transfer.

No pcb1 data was downloaded, extracted, decoded or evaluated during 2C. The archive acquired in 1B already contains all categories; that historical presence is not new access or evidence that pcb1 has been evaluated.

## N. Recommended Milestone 2D scope

Subject to a new instruction, implement and test the single bounded student-teacher development recipe, using only permitted Cashew normals and artificial diagnostic probes. Preserve all historical/2C artifacts, compare CPU resources and mechanics, and amend the future confirmatory design before any pcb1 test access if a third model is to be enrolled. Keep actual pcb1 acquisition/extraction/evaluation as a separately gated later experiment. No API, frontend, GPU installation or deployment belongs in that development scope. Work stops after Milestone 2C.
