# Milestone 2E2 — first opening stopped at the integrity gate

**BLOCKED — OFFICIAL EVALUATION NOT COMPLETED**

The official PCB1 set was **partially unblinded**: all 200 test images and 100 masks were extracted and decoded for integrity. The registered similarity review identified a visually confirmed near-duplicate/repeated-acquisition pair across fitting and test partitions. Evaluation stopped before the `evaluate` command, so **zero official model predictions and zero performance metrics** were computed. No sample was removed, no model changed, and no retry occurred.

## A. Milestone 2E1 checkpoint

Local commit: `306ee79d8712272622d7d49384a326d831c24dc3` on `main`, message `feat: prepare frozen PCB1 confirmatory evaluation`. Parent: `960b76d57c834c0b8f73d19538c5a7a013c92f26`. The exact reviewed 27-file inventory was committed using the approved repository-local identity. The tree was clean immediately afterward. No data, generated outputs or weights were staged; no push or publication occurred.

## B. Final pre-opening audit

PASS, explicitly conditional on the subsequent integrity gate. Recorded at `2026-10-24T20:24:53.557194+00:00` in `reports/pcb1-2e2-preopening.json`, SHA-256 `0142376a5125bdf69f689c94d17de96acf734d4cd9b9c6a7f10f49c8fd9d9cc9`.

All four safely loaded candidates matched their sealed identities, calibration records and normal-only development provenance. Repeated permitted-normal inference preserved model states and matched numerically. Synthetic checks verified mask alignment, ordering, positive-score orientation and strict `score > threshold`: equality is negative. A known four-image example gave AUROC 0.875 and TN/FP/FN/TP 2/0/1/1. The 8,000,000-pixel synthetic metric-cap check passed with peak process memory 892,669,952 bytes, below the frozen 4 GiB limit. Default fixtures and five normal-only checks passed before opening.

The new normal-only audit helper was checked and hashed before opening. It did not alter the registered protocol, models, thresholds or existing runner, and refuses to execute after an opening exists. The owner's explicit authorization was recorded against the seal in `reports/pcb1-owner-approval.json`.

## C. Frozen protocol and artifact identities

All values below are SHA-256. The original 1,032-file seal passed again after the stop.

| Identity | Hash |
|---|---|
| Evaluation seal | `8b8782d73761695ce8d75f612b59cabfec3fc25c717dd4467ac69bada50f2b30` |
| v3 protocol | `c7b59ae7d88e1ec32009067cae7f018e7ad2f5106f2737d6985563fe81ef38bc` |
| Derived manifest | `82a6d379a3cbb3604756c01c0d7898f6be8d2524c92659da00a53eef02dda34c` |
| Original VisA archive | `2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362` |
| Official split CSV | `a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995` |
| Autoencoder best.pt | `6509df8499791fc4a02dd4baf91da691e59d975d279e7ad107e13fcff5cdbe19` |
| Spatial mean best.pt | `224cec850e91dae14562704e5bec2bd7821099c1776d0f340a9fc6d9c66c1394` |
| Feature memory best.pt | `ecb7bdf84f68c24a949cc6083a009c031f0d810494942b2b099e3d71398e6cc0` |
| Student best.pt | `ad568c4ededd8779259fecf8b53efa04427ab5830c501508f9f0d548e76a1cf5` |
| Pretrained ResNet18 | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |
| Frozen opening/evaluation runner | `2ee9ae6b27d1cf966b2061a441d1155dbd793c1a00fb4d95993b0cd5f1f25343` |
| New pre-opening audit helper | `0dc8804ffa79fd9d76a2e5295b7631e4e6a145271a0dc3fe387e7db328cabea7` |

Dataset snapshot: `VisA_20220922.tar`, upstream split revision `2a692ab575001cbde74d402d897a7286086c6199`. Full configurations, calibration hashes, preprocessing, software identities and model states are retained in the original audit and seal. Protocol v3 remains unchanged.

## D. Actual held-out inventory

| Item | Count |
|---|---:|
| Normal test images | 100 |
| Anomalous test images | 100 |
| Associated anomaly masks | 100 |
| Extracted files | 300 |
| Extracted bytes | 53,364,546 |
| Masks disappearing at nearest-neighbor 128×128 resize | 0 |
| Missing/extra files or excluded samples | 0 |

Unique IDs and their deterministic order match the frozen manifest. The quarantined file set exactly equals its declared official test image/mask paths; all extracted hashes were rechecked. Opening validated readability, geometry, labels/mask association and safe archive paths. Development remains 632 fitting, 135 validation and 137 calibration normals. No official membership was changed.

## E. Integrity and leakage finding

No exact file-byte or decoded-RGB duplicates were found across all 1,104 images. The predefined cross-partition screen examined 371,199 pairs, retained 59,150 broad candidates, and found **one strong pair**. All 21 required visual pairs were inspected across three contact sheets. The strong pair was also inspected in both full-resolution originals:

| Finding | Value |
|---|---|
| Held-out sample | `visa-pcb1-normal-0187` |
| Fitting sample | `visa-pcb1-normal-0188` |
| dHash / pHash distance | 4 / 2 |
| Full-frame 128 RGB RMS | 0.02206023782491684 |
| Central-crop 64 RGB RMS | 0.020066535100340843 |
| Crop correlation | 0.9983470480597558 |
| Strong rule triggered | correlation ≥ 0.995 |

Visual evidence shows matching board pose, transducer mesh orientations/reflections, solder and oscillator markings, and background fibers/texture, with small exposure/appearance differences. This supports the repeated-acquisition/near-duplicate finding; it does not provide externally verified capture metadata. Different exact hashes do not clear this pair. No repeated acquisition was confirmed among the other 20 selected pairs, which is not exhaustive clearance of every broad candidate or proof of object/batch independence.

The frozen rule is: **“confirmed repeated acquisition across partitions or incomplete required review blocks fitting/test metrics; no silent removal/reassignment.”** Accordingly, `review.json` records `review_complete: true`, `confirmed_leakage: true`, all required pairs and the finding. The runner rejects this failed review. Its SHA-256 is `299a0fd6b64b5670acf2ff1845c1ff155efc92564711f236b55d50b51a560d6f`. Original `integrity.json` remains unchanged with its historical pending-review status; the separate review and blocked-status record give the final disposition.

## F. Autoencoder results

Official image AUROC/AP, precision, recall, F1, confusion matrix, false-positive rate and pixel AUROC/AP: **NOT COMPUTED — integrity gate blocked evaluation**. Frozen threshold: 0.012788945995271203. Normal-only preparation does not establish defect detection performance.

## G. Spatial baseline results

All official image and pixel metrics: **NOT COMPUTED — integrity gate blocked evaluation**. Frozen threshold: 0.019585608690977096.

## H. Feature-memory results

All official image and pixel metrics: **NOT COMPUTED — integrity gate blocked evaluation**. Frozen threshold: 0.07782351225614548.

## I. Student–teacher results

All official image and pixel metrics, including the primary pooled pixel AP: **NOT COMPUTED — integrity gate blocked evaluation**. Frozen threshold: 0.8243726611137387.

## J. Four-model comparison

No winner can be identified for image discrimination, frozen-threshold detection or localization. Those quantities are unavailable, not zero and not undefined due to a metric denominator. Only pre-opening engineering measurements can be compared.

| Family | 2E1 preparation | Parameters / fitted state | Official metrics |
|---|---|---|---|
| Autoencoder | 12 epochs, 480 updates | 37,763 trainable | Not computed |
| Spatial mean | Mean of 632 fitting normals | 49,152 RGB reference values; no trainable parameters | Not computed |
| Feature memory | 12 patches per fitting image | 2,782,784 frozen extractor parameters; 7,584×384 bank | Not computed |
| Student–teacher | 12 epochs, 948 updates | 112,128 student + 2,782,784 frozen teacher | Not computed |

No training took place in 2E2. Full-frame RGB is resized to 128×128 with bilinear antialiasing; feature/student inputs additionally use frozen ImageNet normalization. Reconstruction families use mean residual image scores. Feature/student image scores use native-map maxima, with bilinear interpolation for 128×128 localization. Binary target masks use nearest-neighbor resizing; normal masks are zero. See the unchanged methods for exact definitions.

## K. Actual pre-opening CPU measurements

AMD Ryzen 7 9700X (8 physical / 16 logical cores), four Torch threads, CPU only; Python 3.12.15, Torch 2.13.0+cpu, torchvision 0.28.0+cpu, NumPy 2.5.4. The frozen benchmark used the first normal validation image `normal-0006`, preloaded 128×128 batch size 1, five warmups and 30 repetitions. These were completed before any test payload access.

| Family | Median ms | p95 ms | Cumulative process peak bytes | Candidate artifact bytes |
|---|---:|---:|---:|---:|
| Autoencoder | 1.01325 | 1.26412 | 363,302,912 | 156,083 |
| Spatial mean | 0.47960 | 0.502205 | 363,302,912 | 198,185 |
| Feature memory | 12.34425 | 16.85863 | 500,801,536 | 11,650,601 |
| Student–teacher | 5.21485 | 6.363265 | 513,118,208 | 453,113 |

Spatial mean was fastest in this normal-image benchmark. Process peaks are cumulative within one process, **not independent per-model memory allocations**. Feature bank storage is 11,649,024 bytes; feature/student methods additionally depend on the shared 46,830,571-byte pretrained weight file. The student artifact alone is not its complete deployment footprint. Warm image decode was measured separately: median 7.16345 ms, p95 7.63034 ms. Inference timing includes model-specific tensor preprocessing but excludes file decoding, loading, plotting and test metrics. No official-test timing or visualization timing exists; no production-line throughput is inferred.

## L. Statistical uncertainty

Not computed because the integrity gate failed. The unchanged registration specifies 1,000 paired, class-stratified whole-image bootstrap replicates, seed 42, percentile 95% intervals for student-minus-comparator image AUROC/AP. Conditional physical-image independence was already a limitation. Pixel AP differences and descriptive ranks were registered, but pixel confidence intervals, pixel thresholds and AUPRO were not. No post hoc statistic was added.

## M. Preserved evidence and visualizations

First opening began at `2026-10-24T20:25:26.517021+00:00` and exited successfully at `20:26:13.257534+00:00`, producing integrity evidence only. The complete command/stdout/exit log plus the later stop record is `reports/pcb1-2e2-execution.log`. No `reports/pcb1-opening/evaluation` directory exists.

Local evidence is preserved under ignored paths:

- `reports/pcb1-2e2-preopening.json` and `reports/pcb1-owner-approval.json`.
- `reports/pcb1-opening/started.json`, `integrity.json`, `screen.json`, `review.json`, `blocked-status.json`.
- `reports/pcb1-opening/review-1.png`, `review-2.png`, `review-3.png`: required integrity contact sheets, not model-localization examples.
- `data/pcb1-test-quarantine`: original extracted payloads, unchanged.

Portable audit and evidence indexes are in `docs/evidence/pcb1-2e2-preopening.json` and `docs/evidence/milestone-2e2-verification.json`. All stored hashes refer to actual local files. No model heatmaps or representative performance figures were generated, because prediction was prohibited. Third-party images remain local and ignored.

## N. Quality and packaging status

Before opening: complete offline suite **396 passed, 1 skipped**, combined branch/statement coverage **98.8051%** (1,741/1,757 statements; 657/670 branches). The skip is the existing Windows symlink privilege restriction. Ruff lint passed; formatting check reported 114 files formatted. Lock consistency and all 33 installed-package compatibility checks passed. Five explicitly invoked PCB1 normal-only integration checks passed in 20.74 seconds. The new audit helper separately passed Ruff after pre-opening formatting corrections; it is outside the package coverage denominator.

While opening ran, source distribution and wheel were built successfully into new `dist/2e2`; the wheel was installed offline into fresh `.tmp/wheel-env-2e2` with 20 compatible packages. Final isolated import/synthetic smoke, archive-member inspection and the later end-of-milestone quality rerun were **not completed before the mandatory integrity stop**. No claim of full Phase 14 completion is made. Model/artifact seal verification passed again during evidence finalization; application code and tests remain unchanged.

## O. VS Code verification

No fresh 2E2 debugger run was completed before the mandatory stop. The accepted 2E1 report documents the official Codex extension and actual normal-only PCB1 breakpoint/step/console verification. That historical check is not presented as a new 2E2 result. Existing launch/task configurations remain sealed and unchanged; default test discovery remains synthetic and no startup/install task runs the held-out evaluation.

## P. Model Card limitations

The new [PCB1 Model Card](pcb1-model-card.md) records the failed integrity gate and absence of performance evidence. Physical-object/production-batch independence remains unverified. Small defects may be lost at 128×128 and coarse feature resolution; normal-calibration thresholds do not guarantee recall or false-positive rates. Scores have model-specific units. No claim of industrial readiness, factory generalization, production reliability or zero-shot transfer is supported.

## Q. Final Git and publication state

HEAD remains the local 2E1 commit above. No 2E2 files are staged or committed. Expected local changes are README.md plus new `scripts/pcb1_preopening.py`, this report, `docs/pcb1-model-card.md`, `docs/evidence/pcb1-2e2-preopening.json` and `docs/evidence/milestone-2e2-verification.json`. Data, audit logs, contact sheets, checkpoints, package outputs and temporary environments remain ignored. No remote, push, public repository or publication was created. The evidence index records final tracked/untracked status and immutable-artifact verification.

## R. Scientific interpretation and next decision

The mandatory integrity gate worked: exact hashes alone would have missed this near-duplicate. No defensible confirmatory performance comparison can be reported under the frozen protocol after this finding. The official PCB1 data is now observed for integrity and must not be described as untouched in future work.

Historical Cashew reconstruction results, especially weak localization, motivated feature-memory and student–teacher exploration. Those results remain a separate, exploratory research history and were not reevaluated. All four PCB1 candidates were fitted specifically on PCB1 normals, so even a successful test would have assessed category-specific detection, not zero-shot transfer.

The recommended next decision is an explicitly authorized data-integrity review and prospective plan for independent confirmation with defensible acquisition grouping. Do not silently delete the pair, revise thresholds, retrain or relabel a rerun as the original blind experiment. No remedy, new experiment, deployment, publication or later milestone was started. Work ends with this blocked report.
