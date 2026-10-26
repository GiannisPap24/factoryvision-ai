# Milestone 2D completion — 24 October 2026

Completed genuine CPU training of one compact student–teacher architecture, followed by normal-only calibration and explicitly artificial diagnostics. Student learning, frozen-teacher integrity, checkpoint reload and bounded resource use are demonstrated. **Real-defect accuracy is not established.** No official Cashew test re-evaluation or pcb1 access occurred. See machine-readable evidence (local evidence, not distributed: `docs/evidence/milestone-2d-verification.json`), [Model Card](student-model-card.md) and [training methods](student-training.md).

## A. Milestone 2C checkpoint

Verified the exact accepted 25-file inventory, file hashes, historical/feature artifacts, Git ignores, sensitive-content signatures, branch/HEAD and empty staging area before implementation. Baseline quality passed: 338 offline tests, one skip, 98.92229% combined coverage, Ruff, offline lock and 33-package compatibility checks. Created the authorized local commit **151ba0ffee8cb03b132eec18607b3add891b82b8**, **feat: add feature-memory anomaly localization prototype**, with approved repository-local GitHub noreply identity. The tree was clean immediately afterwards. No push or remote was created.

## B. Exact files created and modified

Seven modified files:

- `.vscode/launch.json` — explicit student training smoke debugger.
- `.vscode/tasks.json` — student tests, CPU smoke, bounded training, development evaluation and permitted-data checks.
- `AGENTS.md` — checkpoint and completed 2D preservation/scope.
- `README.md` — current outcome, historical distinction and links.
- `docs/architecture.md` — student modules and boundaries.
- `docs/development.md` — explicit workflow and verification guidance.
- `pyproject.toml` — include student methods/card in source distributions; no dependency changes.

Fifteen created files:

- `docs/evidence/milestone-2d-verification.json`
- `docs/milestone-2d-completion.md`
- `docs/protocols/pcb1-confirmatory-v2.md`
- `docs/protocols/pcb1-confirmatory-v2.sha256`
- `docs/protocols/student-development-v1.json`
- `docs/protocols/student-development-v1.sha256`
- `docs/student-model-card.md`
- `docs/student-training.md`
- `integration_tests/test_student_real.py`
- `scripts/render_student_examples.py`
- `scripts/student_experiment.py`
- `src/factoryvision/models/student_teacher.py`
- `src/factoryvision/training/student_artifacts.py`
- `src/factoryvision/training/student_engine.py`
- `tests/test_student_teacher.py`

Ignored outputs include three new run directories (`cashew-student-smoke-v1`, `cashew-student-v1`, `cashew-student-debug-v1`), corrected illustrations, quality/coverage and installed-wheel reports, distributions, and temporary verification scripts/environment/caches. The evidence JSON enumerates new run artifacts with byte sizes and hashes. No earlier package source, regression test, accepted experiment artifact, historical Model Card or preregistration was modified.

## C. Teacher and student architectures

Teacher: existing verified ImageNet V1 ResNet18 through layer3, **2,782,784 frozen parameters**. Raw layer2 `[B,128,16,16]` and layer3 `[B,256,8,8]` are exposed without the feature-memory pooling. All teacher parameters and BatchNorm statistics remain frozen; parent training mode cannot enable teacher training.

Student: **112,128 trainable parameters**, independently initialized with seed 42. Four biased 3x3 stride-two padding-one convolutions with ReLU use channels 3→16→32→64→96. Separate linear 1x1 heads map 64→128 at 16x16 and 96→256 at 8x8. No student BatchNorm, dropout, pretrained initialization or skip path.

Both use full-frame RGB128, bilinear antialiased resize, /255 and ImageNet channel normalization. Each spatial feature vector is channel-L2-normalized with epsilon 1e-6, without fitted dataset statistics. Loss is channel-summed squared teacher/student unit-feature discrepancy, equally averaged over the two aligned levels and then over batch/spatial positions. Image score is maximum raw 16x16 discrepancy; maps resize bilinearly to 128x128, align_corners=False. No smoothing or pixel threshold.

This is EfficientAD-inspired, not exact reproduction: the [published paper](https://arxiv.org/abs/2303.14535) motivates efficient normal-feature student prediction; its complete teacher design, training objective, auxiliary autoencoder and fusion are not implemented. No published performance claim is inherited. The earlier 2C planning-only matching-ResNet18/20-epoch proposal remains unchanged; the owner's new compact-student/12-epoch instruction superseded it.

## D. Dataset and teacher provenance

Manifest fingerprint remains **80bac86ae2fc1b0ad7e0e046597fb4a3a1e1f4f92ce556d5dddbfdd37d51f84c**. Original parent/assignment fingerprints and image hashes remain unchanged. Exactly 315 training normals fit the student; 67 normal validation images select checkpoints; 68 disjoint normal calibration images set the threshold. Preflight checks all 450 permitted image hashes. Read logs contain exactly these 450 files and zero official-test image/mask reads. This count includes hashing, not optimizer membership.

The metadata manifest retains historical membership, but test labels/scores were not consulted for selection or fitting. No official-test loader was constructed. Exact per-epoch normal IDs, source hashes, recipe, versions and state fingerprints remain in ignored run records. The Python open allowlist and batch membership checks defend this pipeline; they are not an OS sandbox or proof of physical-item independence.

Teacher SHA-256: **f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec**. The verified official local 46,830,571-byte file was reused, with no download. Its provenance and rights caveat (local evidence, not distributed: `docs/evidence/resnet18-provenance.json`) remain: torchvision's code license does not establish blanket commercial weight redistribution rights. No new dependency or GPU component was installed.

## E. Training configuration and optimizer updates

Immutable recipe SHA-256: **fb5fd19e04aaa737de16b0483ca3f218af50aaf47707549dbc7c0345a42b7779**, sealed before the smoke/full experiments and development diagnostics. CPU, four Torch threads, batch 8, zero workers; Python/NumPy/Torch seed 42, deterministic Torch algorithms and dedicated seed-42 training shuffle generator. Validation/calibration use stable ID order.

Adam learning rate 0.001, betas (0.9,0.999), epsilon 1e-8, zero weight decay. Maximum 12 epochs; patience 3, min_delta 0, strict minimum normal validation loss, ties retain earlier checkpoint. Calibration and synthetic probes never select checkpoints. Every epoch processed all 315 unique training IDs and 67 validation IDs, with 40 updates per epoch and a final three-image training batch. Loss averages are sample-weighted. Total: **480 genuine optimizer updates**, all 12 epochs completed, best epoch **12**. No additional architecture, seed or hyperparameter run was trained.

## F. Actual per-epoch losses

Initial untrained normal validation loss: **1.9757610524**.

| Epoch | Training loss | Normal validation loss | Updates |
|---|---:|---:|---:|
| 1 | 0.8297790391 | 0.5643904423 | 40 |
| 2 | 0.5125199504 | 0.4730765913 | 40 |
| 3 | 0.4484045490 | 0.4315676440 | 40 |
| 4 | 0.4096076635 | 0.3964153817 | 40 |
| 5 | 0.3854505385 | 0.3739253340 | 40 |
| 6 | 0.3671516668 | 0.3606921555 | 40 |
| 7 | 0.3523415748 | 0.3523859964 | 40 |
| 8 | 0.3418569315 | 0.3365652174 | 40 |
| 9 | 0.3326347346 | 0.3316549877 | 40 |
| 10 | 0.3293390118 | 0.3255410212 | 40 |
| 11 | 0.3185125084 | 0.3169389803 | 40 |
| 12 | 0.3111041363 | 0.3089492370 | 40 |

All losses were finite. These losses measure normal feature matching, not real anomaly performance.

## G. Duration and resource measurements

Host: Ryzen 7 9700X, Windows AMD64, Python 3.12.15, torch 2.13.0+cpu, torchvision 0.28.0+cpu, NumPy 2.5.4. The preliminary two-update real smoke took **0.4480 s**, peak **496,168,960 bytes (473.18 MiB)**, with changed student, unchanged teacher and exact reload. This measured check supported the fixed batch/configuration; no fabricated training-time estimate was used.

Full training plus initial/epoch validation and checkpoint writes took **59.1796 s**; process CPU time **234.9375 s**, approximately **3.97 logical cores** average (not system-wide CPU utilization). Peak working set **495,120,384 bytes (472.18 MiB)**. Import/preflight and subsequent reload/calibration are excluded from that training timer. Development scoring/reporting took **3.4137 s**, peak **494,264,320 bytes (471.37 MiB)**. Both remained within observed 2 GiB limits; training remained below its 600-second budget. Checks occur at batch boundaries, not through an OS memory/time sandbox.

Debugger smoke elapsed **98.0835 s** includes deliberate pauses and is excluded from benchmarks; it performed only two updates of the same architecture and produced the same smoke checkpoint hash.

## H. Checkpoint and reload

The best student state_dict is **453,113 bytes**, SHA-256 **d5b087c3c1a1a1a3937cd2d4b275cf9c605b774e760e7200f2a7262fa0f95ed9**. It contains no teacher weights, optimizer or executable model object. Separate metadata validates architecture, teacher, dataset, recipe, fitting IDs, selected epoch/loss and file hash. Loading uses weights_only=True, strict keys/shapes/dtypes/finiteness, CPU mapping and bounded artifact sizes. Existing run reuse, invalid names and linked artifact paths are rejected.

Student state changed from **4bd92497bacf86a097757eb0600ec81f6c11a053b304de676a019428581c48a8** to **ca887f47cd298b3bb8d6d2fd6694cab728a80cd9c05a53e005a9121f9a790696**. Teacher state, including buffers, remained **745bc3471c07baab7673fdf201c1a5a48490dedef169782a2246b516009fb7db** before/after. Teacher gradients stayed absent; all student parameters received finite gradients and changed through Adam updates.

The selected eight-image normal-validation probe reloaded with **zero maximum absolute score and map error**, within declared atol 1e-6/rtol 1e-5. A separate real check replayed all 67 normal validation losses, matching the selected checkpoint within 1e-6. This combination supports genuine learning and artifact integrity; loss decrease alone would not suffice. No exact mid-run resume is claimed because optimizer/RNG state is not saved.

## I. Normal calibration and development diagnostics

Provisional threshold **0.9195413202047348** is the float64 linear 95th percentile of the 68 normal calibration scores; strict greater-than.

| Normal partition | n | Minimum | Median | q95 | Maximum | Above threshold |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 67 | 0.63809842 | 0.77558231 | 0.93568237 | 1.15971637 | 4/67 (5.970%) |
| Calibration | 68 | 0.62321150 | 0.76857474 | 0.91954132 | 1.00798321 | 4/68 (5.882%) |

Calibration exceedances are in-sample. Validation also selected the checkpoint, so its exceedances are not an independent unseen-data FPR estimate. Neither provides real anomaly recall or an industrial operating guarantee.

Sixteen fixed 16x16 magenta/green checkerboards on permitted validation normals, with deterministic ID-based locations and exact overwritten masks, were evaluated after selection/calibration. **15/16 scores increased**, **16/16 exceeded the image threshold**, **13/16 map maxima fell inside the artificial mask**. Median paired score increase **0.29038537**. Synthetic-only pixel AUROC **0.99615918**, AP **0.77351594**, over 262,144 pixels including 4,096 artificial positives. These conspicuous artificial probes do not estimate real industrial accuracy or establish superiority over either historical model. No tuning followed these diagnostics.

## J. CPU latency and anomaly-map behavior

| Operation | Median ms | p95 ms |
|---|---:|---:|
| Preloaded teacher+student+score/map, batch 1 | 6.59785 | 7.205395 |
| Preloaded teacher+student+score/map, batch 8 | 25.57985 | 26.404680 |
| Warm-cache decode/resize, one image | 8.27065 | 8.733120 |

Five warmups/30 serial repetitions on the first normal validation image or first eight images; decode used 30 already-warm repeats. No camera, cold-disk, startup or debugger overhead is included. Batch inference is supported within the fixed 1–16 input contract; training used measured batch 8. No production throughput or cross-machine speed claim follows.

Continuous 128x128 maps come from coarse 16x16 discrepancies and show broad responses, including ordinary edges/background; three synthetic peaks missed the injected mask. Four fixed corrected panels were visually inspected. The original helper inherited an incorrect “Feature RMS map” caption. Original outputs/hashes are preserved; the separate rendering script regenerated correctly labelled illustrations from saved maps and permitted normals without inference or metric changes. Fixed display clipping at the image-calibration threshold is display-only, not a pixel decision threshold.

## K. Tests, coverage, Ruff and packaging

**375 offline tests passed, one skipped**, including all 338 earlier tests and **37 new student tests**. Default tests fabricate data and random teacher weights, with no network or real VisA requirement. Coverage: **99.069995% combined**, **1,470/1,481 statements** and **554/562 branches**. All three new package modules have **100% statement/branch coverage**; the 95% gate is unchanged. Tests cover immutable teacher, student gradients/updates, shape/math contracts, finite/invalid inputs, exact membership/order, calibration boundaries, corrupt/incompatible checkpoints, path safety and synthetic maps. The existing Windows symlink privilege skip remains; security settings were not changed.

**Five explicit permitted-development checks passed**: prior artifact preservation/frozen sources; exact epoch membership, update counts and checkpoint selection; read logs/calibration isolation; actual teacher/student reload plus all normal validation loss; and saved synthetic-map/state replay. The unrestricted historical integration suite was not invoked because it decodes official test assets.

Ruff lint/format passed (**95 Python files**); offline lock verification and 33-package development compatibility passed. No dependency pin or lock change. Offline wheel/sdist builds and archive inspection passed; the warning about possible in-source uv cache inclusion was checked against actual archives and the cache was absent. Data, weights, images, outputs, environments and caches are excluded from distributions.

A fresh runtime-only `.tmp/wheel-env-2d` installed the wheel and passed 20-package compatibility. Isolated imports resolved to installed site-packages and every package source matched. The real saved student/teacher loaded; a fabricated-input optimizer step changed only the student, and reloading reproduced scores/maps exactly. No real image reads were allowed; dev test dependencies were absent. Module CLI help passed. The historically blocked generated Windows console launcher was not misrepresented as verified; Python module/scripts remain the documented invocation.

## L. VS Code verification

Existing authoritative workspace and `.venv/Scripts/python.exe` interpreter verified. Explicit unit-test, smoke, full bounded training, development evaluation, permitted-data verification and existing Ruff/pytest tasks are available. The full/evaluation tasks match the already executed commands; they were not rerun. No task runs on folder open and no global settings changed.

Using the Computer Use skill, `training/student_engine.py:51` paused after backward with finite loss **1.9736**, the real Adam optimizer and permitted training IDs. Stepping executed optimizer.step; the breakpoint was disabled and the smoke continued to successful checkpoint reload. The VS Code student test task passed **37 tests in 2.27 s**. Installed extensions include Python, debugpy, Ruff and official **openai.chatgpt 26.1021.22027**; installation was checked with VS Code's CLI. No Codex UI automation, authentication flow or paid service probe was performed. Browser access was not required or claimed.

## M. Scientific limitations and unresolved risks

The student learned normal features under explicit isolation and resource limits. It may still reproduce anomalous features too well, focus on background/edges, or fail on subtle/structural defects. One category, seed and fixed recipe cannot establish generalization. Artificial checkerboards are unrepresentative; coarse geometry, square resizing, ImageNet domain shift and small normal tails remain limitations. Disjoint IDs and heuristic duplicate screening do not establish physical-object/acquisition independence. Observed historical Cashew results cannot become an unbiased model-selection benchmark again.

No training/test-driven model adjustment or second full fit occurred. Initial lint/import/format findings were corrected; a late-bound-loop warning in immediate checkpoint serialization was explicitly bound after the smoke and before full training, without changing its numerical behavior. The final quality run passed. The inherited figure caption was corrected through separate rendering, retaining all raw evidence. No numerical training failure occurred. Existing Windows restrictions and pretrained-rights uncertainty remain. Source/metadata hashing is local integrity evidence, not signing or a hostile-filesystem security guarantee.

## N. Pre-registered pcb1 protocol

The [v2 local preregistration](protocols/pcb1-confirmatory-v2.md), SHA-256 **034ef00ae2a9ff2a80a59ae2bd4488f491cb71b7341627840eaf0db35d936a30**, preserves v1 and enrolls exactly compact student–teacher, feature memory and spatial mean. It defines fresh category-specific normal fits, deterministic development partitions, normal-only student checkpoint selection, separate q95 thresholds, fixed orientation/geometry/metrics and conditional paired uncertainty. Student pooled pixel AP is the predeclared primary endpoint; both comparator contrasts and every family's outcome must be reported without test-driven selection.

Test opening requires separate authorization, normal-only integrity/resource checks, a tested final protocol/evaluator, all three successful category-specific fits and calibration, sealed model/threshold/source/lock identities, and a persisted permission-to-open record. Only then may the predeclared exact/near-duplicate integrity audit read test assets; confirmed leakage or incomplete review stops before predictions/metrics. Tested expansion beyond the historical evaluator's pixel cap is required before access. No silent sample/pixel discard is allowed.

This is fresh **category-specific pcb1 fitting**, not zero-shot Cashew transfer or cross-factory confirmation. No pcb1 data was downloaded, extracted, decoded or evaluated during 2D. The archive retained since 1B already contains category bytes; this is disclosed rather than called a new download or new inspection. The local design is not a publicly time-stamped registry entry.

## O. Final Git status

Branch main, HEAD **151ba0ffee8cb03b132eec18607b3add891b82b8**, five local commits, no remotes. **22 Milestone 2D files: seven modified, fifteen created; all unstaged and uncommitted.** Accepted historical source and artifacts retain their hashes. No weights/images were staged, no additional commit was made, and nothing was pushed or published.

## P. Recommended next narrow milestone

After a new owner instruction, implement and validate the v2 confirmation machinery: generic category-specific adapters, training-normal-only extraction/quarantine, resource-bounded exact metrics, leakage gates and a verifiable pre-test opening record. Use synthetic fixtures and, only if specifically authorized, pcb1 normal development assets. Keep official pcb1 test opening/evaluation behind its separate authorization and sealed gates. Do not add architectures, repeat Cashew tuning, install a GPU backend or build an API/dashboard. **Stop after Milestone 2D.**
