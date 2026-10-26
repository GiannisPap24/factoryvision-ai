# Milestone 2B completion: frozen Cashew evaluation

Completed 24 October 2026. Scope ended at 2B. The autoencoder improves image ranking over the spatial mean baseline but is worse at pixel localization. At its unchanged threshold it misses 48 of 100 anomalies. Neither model is ready for industrial deployment. Full-precision aggregate evidence is in milestone-2b-verification.json (local evidence, not distributed: `docs/evidence/milestone-2b-verification.json`).

## A. Milestone 2A checkpoint

Reviewed the actual model, split, scoring/calibration, all 25 entries of the accepted 2A file inventory and the accepted training artifacts. Verified hashes, ignore rules, local identity and baseline quality (258 passed, one existing skip; 98.54% combined coverage). Created exactly one authorized local commit on `main`:

`e493c943f45f6d6053b43ef4fadeb872f2ea3f18` — `feat: train first CPU convolutional autoencoder`.

The repository-local approved noreply identity was used. The tree was clean immediately after that checkpoint. No push, remote or further commit was created.

## B. Exact 2B change inventory

Modified existing files (11):

- `.vscode/launch.json`
- `.vscode/tasks.json`
- `AGENTS.md`
- `README.md`
- `docs/architecture.md`
- `docs/dataset-preparation.md`
- `docs/development.md`
- `docs/evaluation-protocol.md`
- `docs/training.md`
- `pyproject.toml`
- `uv.lock`

Created trackable files (15):

- `docs/evidence/milestone-2b-verification.json`
- `docs/milestone-2b-completion.md`
- `docs/model-card.md`
- `docs/protocols/cashew-evaluation-v1.json`
- `docs/protocols/cashew-evaluation-v1.sha256`
- `integration_tests/test_evaluation_real.py`
- `scripts/debug_evaluation.py`
- `scripts/evaluate_cashew.py`
- `scripts/prepare_evaluation.py`
- `scripts/review_leakage_2b.py`
- `src/factoryvision/evaluation/metrics.py`
- `src/factoryvision/evaluation/pipeline.py`
- `src/factoryvision/evaluation/protocol.py`
- `src/factoryvision/models/spatial_mean.py`
- `tests/test_evaluation.py`

The evidence JSON records each reviewable file's SHA-256 except its own self-reference. Ignored outputs comprise `runs/cashew-eval-v1/` (reference/calibration, seals, source snapshot, per-image JSON, spatial NPZ, metrics, latency, six figures, artifact hashes), `reports/leakage-2b-screen.json`, `reports/leakage-2b-review.json`, three leakage review JPEGs, packaging/install/precheck reports, coverage output, rebuilt distributions and temporary isolated-install/check utilities. No original images or weights entered the change inventory. Existing training source, architecture and checkpoint artifacts remain unchanged.

## C. Frozen identities and chronology

| Identity | SHA-256 |
|---|---|
| Immutable protocol v1 | `3b89b8447f00e478ab5d7793a50ca4cb92f3d0804bd6bee9fada564618adaa81` |
| AE epoch-12 checkpoint | `da7e0caed8e365a08a3d301ec7e88536bc5df8e3f8b59f27635853e15ab4cdbc` |
| Derived manifest | `80bac86ae2fc1b0ad7e0e046597fb4a3a1e1f4f92ce556d5dddbfdd37d51f84c` |
| Parent manifest | `9d8e56d672589ef1234a9382cc61a52efd29804364b1b265d0cfeb478ce56372` |
| Split assignments | `6ddb3acf4172bad9c4bdd667606f0fcd12941f570884eccb5ee298d2f8470aa0` |
| Spatial mean artifact | `04dc72dd50c54214d7bf874f650da58bf53eb80c803d550f8a75c7368bfd4c63` |
| Frozen artifact seal | `5045dc887b62f202e963ab63405675b9a9686bb4ed640de0d7e9a3c64c034087` |

UTC chronology on 2026-10-24: protocol **15:59:18.699113**, reference/calibration/source seal **16:08:16.373678**, test-start marker **16:08:28.637283**. Hash and chronology assertions passed. No prediction preceded the protocol/threshold freeze. The exclusive marker prevents another official run; it was not removed.

Fixed input is RGB float32 /255, bilinear antialiased square resize to 128x128. Mask classes become binary nonzero and resize nearest-neighbor. Both scores use raw channel-mean squared RGB residuals, then spatial mean for image scores; higher is anomalous. Stable sample-ID order, CPU batch 16, four threads, zero workers. No smoothing, adaptive heatmap normalization or test-driven recipe change.

AE threshold **0.0031531278509646654** was inherited unchanged from 2A. Spatial mean threshold **0.017185416119173163** was calibrated separately. Both use 68 normal calibration scores, linear 95th-percentile interpolation, strict `score > threshold`; equality is normal. Full methods and undefined-metric behavior are in [evaluation-protocol.md](evaluation-protocol.md).

## D. Leakage and duplicate-screening findings

Re-audited all 600 images and 100 masks, including byte/decoded-pixel hashes, original geometry, mask identities, split fingerprints and unchanged official membership. No corrupt/missing files or exact byte/pixel duplicates were found. Original mask identities remain unchanged. No sample was removed, reassigned or relabeled.

The prior 2A screen compared **114,581 cross-partition pairs** and flagged **7,946** with dHash64 distance <=4 OR pHash63 distance <=4 OR RGB32 RMS <=0.02. Its 24 earlier thumbnail reviews were retained as historical evidence, not treated as clearance of every candidate.

Before test prediction, the immutable 2B procedure refined all 7,946 candidates with aligned full RGB128 RMS, center-quarter RGB64 RMS and center grayscale Pearson correlation. A strong candidate required `(full RMS <=0.015 AND crop RMS <=0.025) OR correlation >=0.995`. **Zero** met that criterion. The visual set was the union of the first eight full-RMS ranks, first eight crop-RMS ranks, first eight deterministic SHA-256 pair ranks, and all strong candidates: **20 unique pairs**, all inspected on three review sheets. Selection used neither model scores nor labels.

Visible shape, surface or pose differences supported no confirmed duplicated acquisition among those 20. The closest full-RMS pair (normal-477/normal-478) had full RMS 0.054490, crop RMS 0.091103 and correlation 0.965349. This does not prove distinct physical objects. Alignment sensitivity, unreviewed pairs and missing acquisition/physical-item IDs remain explicit nonblocking limitations. No confirmed leakage triggered the stop rule. Full candidate identities remain local under ignored reports/.

## E. Official held-out membership

**150 images: 100 anomalous, 50 normal**, identical for both models. Development remained 315 training normals, 67 validation normals and 68 calibration normals. No test sample entered fitting, checkpoint selection or calibration. Test prevalence is 66.7% anomaly and is not representative of an ordinary production defect rate.

## F. Autoencoder image metrics

Image AUROC **0.937600**; Average Precision **0.9432187602**. AP is scikit-learn's non-interpolated AP, not trapezoidal PR area. Positive class is anomaly (1). At the frozen threshold:

| | Predicted normal | Predicted anomaly |
|---|---:|---:|
| Actual normal | TN 47 | FP 3 |
| Actual anomaly | FN 48 | TP 52 |

Precision **0.9454545455**, recall **0.52**, F1 **0.6709677419**, normal FPR **0.06** (60 false positives per 1,000 normals by point estimate). The small 50-normal test does not precisely establish an operational false-alarm rate. All 150 scores and decisions are saved in `runs/cashew-eval-v1/per-image.json`.

## G. Autoencoder pixel metrics

Pooled pixel AUROC **0.8181194144**, AP **0.0809360592**. Computed exactly over **2,457,600 pixels**, including normal images: **27,672 positive**, **2,429,928 negative**. Binary masks remain aligned at 128x128; zero anomaly masks became empty. Small boundaries may still be lost or distorted by resizing. Pixel dependence prevents treating millions of pixels as independent evidence.

The metric implementation caps arrays at three million entries. No subsampling was used. AUPRO was deferred before evaluation; no unvalidated substitute was reported. No Dice/IoU or thresholded pixel metric was calculated because no independent pixel threshold exists.

## H. Statistical baseline

The spatial mean fits exclusively on the same 315 training normals in sorted-ID order with sequential float64 accumulation and a final float32 reference. Calibration used only the 68 calibration normals, with four scores above its frozen threshold. It has no optimizer-trained parameters, but **49,152 fitted reference values**; it is not a data-free baseline.

Image AUROC **0.736800**, AP **0.8412160965**; confusion `[[50,0],[85,15]]`; precision **1.0**, recall **0.15**, F1 **0.2608695652**, normal FPR **0.0**. Pixel AUROC **0.8953007580**, AP **0.2673019809** over identical masks/pixels. Zero observed false positives is not proof of zero future false alarms.

## I. Comparison and CPU evidence

| Measure | Autoencoder | Spatial mean |
|---|---:|---:|
| Image AUROC | 0.9376 | 0.7368 |
| Image AP | 0.943219 | 0.841216 |
| Recall / F1 | 0.52 / 0.670968 | 0.15 / 0.260870 |
| False positives / missed anomalies | 3 / 48 | 0 / 85 |
| Pixel AUROC | 0.818119 | 0.895301 |
| Pixel AP | 0.080936 | 0.267302 |
| Forward+score median / p95, ms | 0.8091 / 1.09755 | 0.3818 / 0.499685 |
| Trainable parameters | 37,763 | 0 |
| Fitted reference values | Not applicable | 49,152 |
| Serialized model bytes | 156,083 | 198,185 |

The AE improves image ranking, recall and F1; the baseline has better pixel ranking, fewer observed false alarms and lower inference latency. The measurements support no blanket claim of neural-model superiority.

AE-minus-mean AUROC difference **0.2008**, paired-bootstrap 95% interval **[0.110195, 0.291805]**. AP difference **0.102002664**, interval **[0.032783130, 0.167381233]**. These 1,000 paired stratified image resamples (seed 42) condition on unverified image independence and are not physical-group intervals.

Hardware: AMD Ryzen 7 9700X (8 cores/16 logical processors), Windows AMD64. Python 3.12.15, torch 2.13.0+cpu, torchvision 0.28.0+cpu, NumPy 2.5.4, Pillow 12.3.0, scikit-learn 1.9.1. Four Torch threads; no GPU. Latency used the first normal validation image, preloaded batch one, five warmups and 100 serial repetitions per model, forward plus score including validation overhead, excluding image decoding. Separate same-file warm-cache decode/resize with normal-mask derivation: **7.37465 ms median / 7.97943 ms p95**, 30 repeats. These are microbenchmarks, not camera throughput or end-to-end deployment latency.

The evaluation/reporting block took **6.9630 s**, excluding initial imports/preflight. Process peak working set was **631,361,536 bytes (602.11 MiB)**, below the declared 2 GiB observed ceiling. The mean tensor occupies 196,608 bytes. Debugging time is excluded from latency evidence.

## J. Produced visualizations

Six fixed 1024x294 PNG examples in ignored `runs/cashew-eval-v1/`, selected before predictions and all visually inspected:

- `example-visa-cashew-anomaly-000.png`
- `example-visa-cashew-anomaly-001.png`
- `example-visa-cashew-anomaly-002.png`
- `example-visa-cashew-anomaly-003.png`
- `example-visa-cashew-normal-006.png`
- `example-visa-cashew-normal-012.png`

Each contains resized original RGB, AE reconstruction, raw error heatmap displayed on the fixed [0,0.05] blue-red scale, and binary ground truth. Normal ground truth is zero. Panels are enlarged with nearest-neighbor for display only. They show blurred reconstructions and substantial boundary/background residuals, including on normal images. They are examples, not independent evidence or selected best cases. Three separate JPEG contact sheets document the 20 leakage-review pairs. No figure or original dataset image is committed or packaged.

## K. Tests

Default offline suite: **300 passed, one skipped** (301 collected). Existing tests preserved; **42 new synthetic evaluation tests** cover score direction, exact rankings/counts, absent classes, zero denominators, strict equality boundaries, non-finite/invalid data, binary mask geometry, memory bounds, ordering/identity rejection, training-only baseline fit, independent normal calibration, bootstrap determinism, protocol/checkpoint hash rejection and report correctness.

Explicit real-data/experiment suite: **14 passed** (six dataset, four accepted training checks, four new frozen-evaluation replay checks). The replay verifies chronology/artifact/source hashes, isolated fit/calibration identities, exact membership/labels/example IDs, saved pooled image/pixel metrics and an independent pairwise image-AUROC calculation. It does not retrain or run another official test inference.

## L. Quality, coverage, packaging and failures

Ruff lint and format checks passed (70 Python files). Offline `uv lock --check` passed; all prior lock pins remained unchanged. Development compatibility passed for 33 packages. Six additions: scikit-learn 1.9.1, scipy 1.18.1, joblib 1.6.0, threadpoolctl 3.7.0, narwhals 2.27.1 and cloudpickle 3.1.2. Native imports succeeded.

Combined statement/branch coverage **98.70484%**, unchanged 95% gate: **1,044/1,055 statements**, **404/412 branches**. New metrics, pipeline, protocol and spatial-mean modules each have 100% statement/branch coverage. Remaining misses are existing archive defensive paths and the module entry point; scripts are outside this package coverage denominator.

Offline wheel and sdist builds passed. Explicit member inspection checked 34 wheel entries and 79 sdist entries; every Python wheel source matches the workspace, and original data, weights, generated figures, caches and environments are absent. Model Card and frozen protocol are included in the sdist. A fresh runtime-only `.tmp/wheel-env-2b` installed the actual wheel offline with 20 packages; pytest is absent, compatibility passed, native imports and synthetic fit/evaluation passed, and both saved image metric records replayed exactly through the installed package. No second source checkout was made.

Observed issues, retained rather than hidden:

- The first focused pytest attempt stalled without output; a second quality invocation was stopped to avoid simultaneous use of the shared pytest temporary directory. Only owned processes were stopped. Root cause was not established. Native imports, a diagnostic focused run and later clean sequential standard runs passed. No Windows security/DLL failure was established.
- Initial Ruff E501/E731/format findings were corrected before execution-source freeze; the later replay-test E731 finding was corrected in the new test only. No metric formula or frozen implementation was changed after seeing test results.
- One Windows symlink test remains skipped due to WinError 1314; junction checks pass. Historical Application Control restriction on the generated console launcher remains documented; module/script commands were used. Security settings were unchanged.
- One VS Code accessibility action lacked geometry; fresh screenshot-backed observation recovered it.
- A documentation replacement patch rejected duplicate target operations; no change occurred. A subsequent local text-edit helper hit a Windows decoding error; explicit UTF-8 recovered it. Git whitespace checking exposed CRLF introduced by that helper in documentation/editor JSON; these nonfrozen files were normalized to LF. Final diff checking passes.
- uv warns that its project-local cache is inside the build tree. Explicit distribution-member inspection confirmed that it was excluded.

## M. VS Code

The existing workspace and project interpreter were retained. Added focused evaluation task, explicit one-shot official-evaluation task and cached replay task. The task picker showed both real tasks. The focused UI task passed **42 tests in 1.88 s**, and pytest discovery showed `test_evaluation.py`. Official scoring was executed once from the command line with the task's identical script arguments; the completed one-shot task was not launched again just to encounter its guard. No task runs on folder open.

The synthetic evaluation debug launch stopped at `evaluation/pipeline.py:40`, with two images, labels [0,1], `n=2`, 32 mask pixels and SpatialMean. Stepping computed scores [0,1]; continuing printed perfect synthetic image confusion and pixel AUROC/AP. The verified breakpoint was left disabled. No new training/debug smoke on real data occurred.

Ruff 2026.90.0, Python 2026.10.0 and debugpy 2026.6.0 integrations remain installed/configured. Official `openai.chatgpt` 26.1021.22027 was verified via extension inventory; its existing panel displayed current responses. No separate authentication/billable probe or interaction with Codex UI controls was performed. No global settings or repository location changed.

## N. Model Card and limitations

[model-card.md](model-card.md) documents intended research use, provenance, scope, preprocessing, both frozen thresholds, comparison, resource measurements and restrictions. Reconstruction MSE may emphasize background/pose and dilute small defects. One category/seed, small calibration/test-normal samples, test prevalence, resizing, pixel correlation and unverified physical acquisition groups limit generalization. No operating-cost criterion or independent production data was available. No industrial readiness claim is made.

## O. Git state

`main` remains at the 2A commit above with three local commits and no remotes. **11 modified files and 15 untracked reviewable files** comprise 2B; none are staged or committed. Ignore rules keep original data, weights, reports, figures, caches and environments out of Git. Accepted 2A artifacts, immutable protocol, frozen evaluation sources and result hashes were reverified. No push/publication occurred. Historical local-machine paths remain in prior evidence; new aggregate evidence uses project-relative paths.

## P. Recommended next milestone

Recommend **Milestone 2C: evaluation governance, physical-group provenance and structured error analysis**, separately authorized. Establish a new independent confirmation strategy before choosing any next model from these observed results. Define acceptable miss/false-alarm costs and localize failure types without changing this frozen benchmark. A subsequent predeclared CPU feature-based benchmark may be justified, but the present evidence does not justify deployment or GPU work. No new model, PatchCore, dataset category, training run, threshold tuning, API, frontend, cloud resource or publication was started.
