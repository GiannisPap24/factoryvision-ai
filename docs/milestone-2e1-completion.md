# FactoryVision AI — Milestone 2E1 pre-test audit

**Verdict: READY FOR BLINDED PCB1 TEST.** This means the preparation gates passed; it does not authorize test opening or establish real-anomaly accuracy. Work stops at 2E1. Official PCB1 test payloads were not extracted, previewed, decoded or scored. No 2E2 approval file or opening marker was created.

## A. Milestone 2D checkpoint

The accepted 22-file checkpoint was reviewed, its artifact hashes verified, and the baseline passed 375 tests with one platform skip, 99.07% combined coverage, Ruff, offline lock and dependency checks. Local commit `960b76d57c834c0b8f73d19538c5a7a013c92f26` has the exact message `feat: train student-teacher anomaly detection model`, using the approved repository-local noreply identity. The tree was verified clean immediately afterward. No remote/push/publication exists.

Git's initial whitespace preflight flagged CRLF in two accepted checksum companions. Their bytes were preserved and checked with a one-command `cr-at-eol` whitespace setting before committing; persistent configuration was not changed.

## B. Exact created and modified project files

Modified 4 files:

```text
.vscode/launch.json
.vscode/tasks.json
README.md
pyproject.toml
```

Created 23 files:

```text
docs/evidence/milestone-2e1-verification.json
docs/evidence/pcb1-evaluation-seal.json
docs/evidence/pcb1-evaluation-seal.sha256
docs/milestone-2e1-completion.md
docs/pcb1-confirmatory.md
docs/protocols/pcb1-confirmatory-v3.json
docs/protocols/pcb1-confirmatory-v3.sha256
integration_tests/test_pcb1_development_real.py
scripts/debug_pcb1.py
scripts/pcb1_blinded.py
scripts/pcb1_experiment.py
scripts/prepare_pcb1.py
scripts/review_pcb1_normals.py
scripts/seal_pcb1.py
scripts/verify_pcb1_seal.py
src/factoryvision/data/category.py
src/factoryvision/data/normal_review.py
src/factoryvision/evaluation/confirmatory.py
src/factoryvision/models/pcb1.py
tests/test_category.py
tests/test_confirmatory.py
tests/test_pcb1_models.py
tests/test_pcb1_runner.py
```

Ignored generated assets are inventoried separately in the machine-readable evidence/seal: 904 normal JPEGs; official/derived manifests and preparation JSON; similarity screen/review JSON and three normal contact sheets; two smoke runs; four full model run directories; normal-only verification/quality reports; build artifacts under `dist/2e1`; and an isolated package environment plus verification helpers under `.tmp`. No third-party image or weight is added to Git.

## C. Acquisition and verified inventory

Reused the existing official VisA 20220922 archive, 1,929,840,640 bytes, SHA-256 `2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362`. Additional download: **0 bytes**. All archive headers passed the existing path/type/link/size/structure checks before allowlisted extraction.

Official split CSV SHA-256 `a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995`, upstream revision `2a692ab575001cbde74d402d897a7286086c6199`. Actual permitted inventory: 904 training normals, 243,575,802 bytes, all 1404×1070 RGB JPEGs. Internal partitions: 632 fitting, 135 validation, 137 calibration. The 200 official test records remain unchanged metadata; no label breakdown was used for selection. [Methods](pcb1-confirmatory.md) record source URLs, attribution, CC BY 4.0 obligations and original/archive/canonical fingerprints.

## D. Integrity and leakage screening

No missing/corrupt permitted image, exact-byte duplicate group or decoded-RGB duplicate group. Exact-image groups remain within one partition by construction. 190,399 cross-partition development comparisons produced 41,137 broad heuristic candidates, zero strong matches, and 20 unique required visual pairs. All 20 were reviewed without confirming repeated capture; no sample was removed or reassigned. Pose, lighting and sensor-detail differences are observable; physical-object and acquisition-batch independence remain unverified. This is a limited integrity gate, not proof that every heuristic candidate is an independent item. Train/test integrity review remains deferred to the separately authorized opening.

## E. Preregistered protocol

[V3 JSON](protocols/pcb1-confirmatory-v3.json), SHA-256 `c7b59ae7d88e1ec32009067cae7f018e7ad2f5106f2737d6985563fe81ef38bc`, was frozen before normal extraction/fitting. It explicitly enrolls all four owner-requested families and preserves earlier versions. It fixes preprocessing, seeds, budgets, validation selection, patch sampling, calibration, endpoints, ranking, null semantics, bootstrap, sample ordering, failure/exclusion rules and resource limits.

Primary endpoint: student pooled pixel AP. All four methods and student-minus-comparator differences will be reported; image AUROC/AP, pixel AUROC and frozen-threshold confusion statistics are secondary. Image bootstrap intervals are conditional on unverified independence. Pixel intervals/AUPRO/pixel thresholds are deferred. Cashew results influenced the research direction; no new Cashew evaluation occurred.

## F. Four configurations

RGB128, CPU, 4 threads, workers 0, seed 42 throughout. AE: accepted 37,763 parameters, fresh Adam 0.001/batch 16,12 epochs max / patience 3/min-delta1e-5. Spatial mean: all 632 fit images, sequential float64 then float32. Feature memory: accepted frozen ResNet18 layer2/3, 12 seeded patches per fit image, 7,584×384 bank, exact chunked RMS nearest neighbor. Student: fresh 112,128 parameters, accepted frozen 2,782,784-parameter teacher, Adam 0.001/batch 8,12 epochs max / patience 3. Full configuration and score/map semantics are in [methods](pcb1-confirmatory.md) and the immutable JSON. No Cashew-trained candidate was reused.

## G. Actual category-specific fitting

| Model | Fit wall seconds | CPU seconds | Peak bytes during fit | Updates | Best epoch / validation loss |
|---|---:|---:|---:|---:|---|
| Autoencoder |75.7109303|301.65625|435,372,032|480|12 /0.009314460217676781|
| Spatial mean |4.7504418|18.90625|377,741,312|Not optimized|All632 normals|
| Feature memory |6.2621801|24.609375|479,535,104|Not optimized|7,584 patches from632 normals|
| Student–teacher |98.2377799|390.625|497,528,832|948|12 /0.23677749975963874|

AE validation loss began at 0.02478036233396442; student began at 2.0608781584986935. Both completed 12 epochs; all finite checks passed. Teacher state hash stayed `745bc3471c07baab7673fdf201c1a5a48490dedef169782a2246b516009fb7db`. Two-update smoke runs preceded fresh full training. Every epoch's loss, updates, identities and elapsed time is preserved in each run's `history.json` and the portable verification evidence. No architecture/threshold was selected on test predictions.

## H. Normal-only validation/calibration

| Model | Frozen threshold | Validation flags /135 | Calibration flags /137 |
|---|---:|---:|---:|
| Autoencoder |0.012788945995271203|8|7|
| Spatial mean |0.019585608690977096|8|7|
| Feature memory |0.07782351225614548|9|7|
| Student–teacher |0.8243726611137387|2|7|

All thresholds are independently calculated using only 137 calibration normals, float64 linear q95 and strict greater-than. Full score vectors, distributions, counts and normal-derived display scales are retained. No PCB1 synthetic anomaly probes or official-test diagnostics were used. These false alarms cannot establish sensitivity, localization quality or model superiority.

## I. Frozen artifacts and SHA-256

Run directories follow `runs/pcb1-{autoencoder,spatial-mean,feature-memory,student-teacher}-v1/`.

| Model | `best.pt` SHA-256 | Bytes |
|---|---|---:|
| Autoencoder |`6509df8499791fc4a02dd4baf91da691e59d975d279e7ad107e13fcff5cdbe19`|156,083|
| Spatial mean |`224cec850e91dae14562704e5bec2bd7821099c1776d0f340a9fc6d9c66c1394`|198,185|
| Feature memory |`ecb7bdf84f68c24a949cc6083a009c031f0d810494942b2b099e3d71398e6cc0`|11,650,601|
| Student–teacher |`ad568c4ededd8779259fecf8b53efa04427ab5830c501508f9f0d548e76a1cf5`|453,113|

Calibration JSON hashes, respectively: `47142bdc363a24a131b5dcd91e94209884b6785b6ad201e7d0fc96503245f863`; `19decae81e93247915e5fd17e3b00895b13caf8f038fba40b4addf8c48ddc24c`; `09a331b66592d10012ea3098bee1f6680dac4fdb1c69ef494b9ee6e9de53d4be`; `f692e870bebe1228d65f7737693b7351e4950b964ae476a65365688e222bbb92`.

The evaluation seal (local evidence, not distributed: `docs/evidence/pcb1-evaluation-seal.json`) binds 1,032 files and has SHA-256 **`8b8782d73761695ce8d75f612b59cabfec3fc25c717dd4467ac69bada50f2b30`**. Its paths are portable; it contains no local absolute paths or personal information. Read-only verification passed. No model/protocol/threshold changes may occur under this confirmatory seal.

## J. Deferred evaluation runner and safeguards

`scripts/pcb1_blinded.py` has separate read-only `verify`, approval-gated one-time `open`, and review-gated one-time `evaluate` modes. It verifies hashes, compatibility and provenance; refuses overwrite; never invokes optimization/calibration; preserves stable raw results, metrics, latency, CPU/memory, examples and evidence hashes. Opening alone makes zero predictions. Exact duplicates or incomplete visual review block evaluation. The entire deferred workflow was exercised only with fabricated archive members; model and metric functions were also tested offline. Real normal reloads were permitted. The official PCB1 opening/evaluation modes were not executed.

## K. Offline quality and packaging

**396 passed, 1 skipped**; combined coverage **98.805109%**, statements 1741/1757 (99.089357%), branches 657/670 (98.059701%). New confirmatory/normal-review modules reached 100%; all pre-existing tests are retained. The skip is the existing Windows symlink privilege limitation; junction containment checks pass. Ruff lint/format passed 112 files; offline uv lock and all 33 development-environment dependency checks passed. Lock SHA-256 remains `d3b2b5efb9996aa50ef74e2019a64ca917a6a4a59d6090153e13ce0592b89168`.

Five explicit real normal-only checks passed, including complete validation/calibration score reloads for all four methods. Wheel and sdist builds passed offline. A fresh isolated environment installed the wheel with 20 compatible runtime packages, matched all 41 installed Python source files to the workspace, passed the module CLI with a synthetic manifest, and loaded/scored all four PCB1 candidates on one permitted normal validation image. No editable checkout or user site supplied the imported package.

Package contents were inspected: no images, weights, run artifacts, raw datasets, caches or tools. uv's generic cache warning was checked against actual archives; the cache was excluded. The packaging helper initially selected uv's `.gitignore` as an archive and then treated the legitimate `factoryvision/data` Python package as a dataset directory. Both verifier conditions were corrected; final inspection passed without changing package source or model evidence. Final distribution sizes/hashes are recorded in verification evidence (local evidence, not distributed: `docs/evidence/milestone-2e1-verification.json`).

## L. VS Code and debugger

The existing FactoryVision workspace uses project-local Python 3.12.15 and uv 0.13.4. Installed official Codex extension: `openai.chatgpt@26.1021.22027`; Python/debugpy/Ruff extensions remain present. Explicit family fit/calibration, normal-only probe, offline contracts and frozen-manifest validation tasks were added. No automatic held-out task or global settings change was introduced.

Actual debugger verification: `scripts/debug_pcb1.py:16` paused on normal validation `visa-pcb1-normal-0006`; stepping reached line 18 with score 0.008895463310182095 and map shape [1,128,128]. The console completed with `test_images_opened: 0`. The breakpoint was disabled afterward. The final read-only seal-validation task result is recorded in machine-readable evidence.

## M. Compatibility and determinism

All four first-normal reload probes had exactly 0 score/map maximum absolute difference. Complete saved normal validation/calibration vectors also reproduced within frozen atol 1e-6/rtol 1e-5 without changing thresholds. Checkpoints, manifests, fit/calibration IDs, teacher identity and reference-bank membership are validated; offline tests reject cross-family/category mismatches, changed checkpoint bytes, changed threshold/provenance, tampered seals and invalid metrics. Fresh-wheel inference independently reproduced the four normal scores.

Original 2A/2B/2C/2D artifact inventories (12/18/20/44 entries) all retain their accepted hashes. Training-time source snapshots remain in each run. Two non-fitting helpers were completed/refactored while fitting ran: `data/normal_review.py` and `evaluation/confirmatory.py`. Neither is invoked by model fitting/calibration; executed architecture/preprocessing/scoring/training sources did not change. Final execution sources were sealed before any held-out opening. This tooling history is explicitly recorded, not retroactively rewritten in training provenance.

## N. Limitations and risks

No real PCB1 anomaly metrics exist. Independence of physical items/batches is not proven; this is one category and128px preprocessing can erase small defects. Calibration is finite-sample/descriptive; validation also selects trainable checkpoints. Warm CPU benchmarks are workstation-specific and may include concurrent verification activity. Hashes/exclusive markers/Python audit hooks provide reproducibility and accidental-access safeguards, not cryptographic owner authentication or OS isolation. Keep actual owner authorization with any future execution record. Cashew-informed research direction, multiplicity and deferred pixel uncertainty are disclosed in the protocol/methods.

## O. Git status

HEAD remains `960b76d57c834c0b8f73d19538c5a7a013c92f26` on main. New 2E1 work is **uncommitted and unstaged**: 4 modified plus 23 created project files. No data/model/weight files are tracked, no remote configured, no push or publication. Historical checkpoint bytes remain preserved.

## P. Final verdict

**READY FOR BLINDED PCB1 TEST**, subject to separately granted 2E2 owner authorization and the deferred opening integrity gate. This is readiness to perform the frozen experiment, not readiness for deployment. Stop after 2E1.

## Q. Exact separate-approval procedure

The owner must explicitly authorize 2E2 against seal SHA `8b8782d73761695ce8d75f612b59cabfec3fc25c717dd4467ac69bada50f2b30`. The [methods/opening procedure](pcb1-confirmatory.md#seal-operation-and-deferred-opening) specifies the approval record, exact `open` command, mandatory pre-prediction review and separate `evaluate` command. Neither command was run on official PCB1 test payloads. No request to approve2E2 is implied by this completed 2E1 report.
