# Milestone 3D — exploratory MetalParts training

25 October 2026. **COMPLETE — EXPLORATORY METALPARTS TRAINING VERIFIED.**

This is a development-training outcome. The model is an **EXPERIMENTAL MODEL
TRAINED ON VERIFIED NORMAL SOURCE DATA**. MetalParts official independent
evaluation remains **NOT CLEARED**; the original 3C **BLOCKED** verdict is
unchanged. No official test image inference, test-mask access, real-defect metrics
or threshold calibration occurred in 3D.

## A. Reviewed Milestone 3C checkpoint

The nine expected 3C text files were reviewed against the change inventory.
Baseline verification passed 528 tests with one existing Windows symlink skip,
Ruff lint/formatting, offline lock consistency and dependency compatibility.
Staged content contained no images, masks, weights, generated evidence or
credentials. The only newly added absolute-path string was the intentionally
fabricated `C:/outside.png` rejection test, not a private machine path.

The authorized local commit is
`5452bb46df1c2a0e363f9730e4250f41fbda69da`, with message
`feat: add RobustAD MetalParts data adapter and integrity checks`.
Its parent is `2396ab8648788812fd1448d376d03d65c0a0f72f`. The approved project-local
GitHub noreply identity was used. The tree was clean immediately afterwards.
No push, remote addition or publication occurred. The reviewed diff and baseline
checks are retained under ignored `reports/metalparts-3d-v1/`.

## B. Official training metadata and labels

Source: [AmazonScience/RobustAD, pinned MetalParts revision](https://huggingface.co/datasets/AmazonScience/RobustAD/tree/82c642b2d18be60f2dab104f8dd4dea1f4ccfacb/MetalParts).
Revision: `82c642b2d18be60f2dab104f8dd4dea1f4ccfacb`. The previously acquired
official subset was reused locally; no dataset or dependency was downloaded.
Original source/provenance and CC BY 4.0 attribution remain in the 3C report.

The actual `metal_parts_data_dir_train/metadata.jsonl` was read and checked
against SHA-256
`7d587bc34e31e29d2fae95f7486d024d5e20b8477ab24048a33f2e5d387e65f1`.
Its rows, not directory names alone, determine eligibility.

| Source-training population | Count | 3D disposition |
|---|---:|---|
| Integer label 0, null mask | 180 | Eligible normal development parents |
| Integer label 1, anomaly mask reference | 51 | Excluded from fitting, validation and diagnostics |
| Metadata inconsistencies | 0 | None found |
| Official test parents used | 0 | Prohibited |
| Anomaly images or masks opened | 0 | Prohibited |

Each eligible image matched its existing verified source SHA-256 and decoded as
native 2681 × 1500 RGB. Excluded anomaly references were validated structurally,
but their images and masks were not opened. The all-domain 3C adapter/audit was
not called: it would decode official test images and masks.

## C. Repeated filenames and sample identity

Sample identity is preserved from 3C: dataset identity, pinned revision and the
normalized full source-relative path are hashed together. The path includes the
official source partition. Internal fitting/validation membership does not change
that ID. Windows-case-insensitive duplicate full paths are rejected.

A basename repeated in separate directories is a different reference, not proof
of duplicate pixels or leakage. The existing 145 train/test0 basename
correspondences comprise 104 normal and 41 anomalous pairs. They remain unresolved
under their original IDs; no sample was renamed, removed or moved between official
domains to erase a concern.

## D. Confirmed and unconfirmed integrity evidence

Existing 3C evidence was reviewed as metadata only: zero exact-byte image
duplicates, zero native-pixel image duplicates, 626 cross-train/test heuristic
similarity candidates and 145 repeated basenames. Only two same-basename pairs
also met the fixed similarity rule. These categories are not interchangeable.
No new unrestricted all-pairs investigation or visual adjudication was performed.

For the permitted 180 training normals, byte and native RGB pixel hashes were
rechecked. There were 180 distinct exact-content groups: no exact duplicate group
crosses the new fitting/validation split. There are no publisher physical-item or
capture-session identifiers and no newly confirmed repeated-capture relationships.
Heuristic candidates were not promoted to confirmed duplicates or dismissed as
independent objects. Physical relationships remain unverified.

Three conclusions therefore differ: verified label-0 source images are suitable
for this authorized exploratory fit; internal normal validation is **potentially
dependent**; independent official-test generalization is **not established**.
The original 3C protocol and blocked outcome were preserved byte for byte.

## E. New exploratory protocol and seal

The separate [exploratory protocol](metalparts-exploratory-training-protocol.md)
was written and sealed before the first real optimizer update. Its SHA-256 is
`a0cc2bf67fe34e2078c8fae5b2cc0a68d4a3387bddf87aad2013f769cde18162`.
The development inventory/split hash is
`bb87d41b2a8296d91f9f67496128e7f7280c2a7ccb688e78f48c175874874c41`.
The verified source-file evidence hash is
`84f2716eaf3afe766d34e569d4fb917abcd383201ffc5a8dc5f5ca5908b672fb`.

The seal binds the protocol, full development inventory file, split and 56
execution-source files. Both run configurations retain that seal. Verification
after training confirmed all sealed bytes unchanged. The preparer, smoke and
pilot refuse existing output/run locations; no completed run was restarted.

## F. Development and patch inventory

Exact-content groups are ordered using SHA-256 of seed 42 and representative
sample ID. The first ceiling 20% form validation. Explicitly confirmed relations
can also be unioned by the implementation; none were available for this dataset.
Every selected tile stays with its parent.

| Partition | Parent images | Selected patches/parent | Patches |
|---|---:|---:|---:|
| Fitting | 144 | 12 | 1,728 |
| Validation | 36 | 12 | 432 |
| Calibration | 0 | 0 | 0 |

Native 256 × 256 crops use stride 192 and exact trailing-edge coordinates. The
complete grid is 112 tiles/parent; twelve deterministic, spatially spread grid
indices are selected before training, without image-content or mask selection.
Selected coordinates are fixed across epochs. RGB becomes float32 [0,1] without
resizing or aspect-ratio distortion. A two-parent decode cache bounds memory.
Training shuffles parents and their patches reproducibly; validation is ordered
by parent ID then y/x. Patch IDs encode parent and native coordinates.

The bounded training subset does not cover every source pixel. The one native
diagnostic uses all 112 tiles with uniform overlap averaging and checks that no
pixel is uncovered. Complete parent/hash assignments and patch IDs are in
`reports/metalparts-3d-v1/development.json` and each run's `patches.json`.

## G. Architecture, initialization and optimization

The existing `conv-ae-v1` PyTorch autoencoder was reused at its supported
256-pixel resolution, with **37,763 trainable parameters**. No architecture change
or additional model family was introduced. MetalParts used fresh seed-42 random
initialization; no Cashew-trained weight was loaded. The smoke and pilot record
the same fresh-initialization hash, confirming the pilot did not inherit the
smoke's trained state.

Optimization: Adam 0.001, batch eight, four CPU threads, zero loader workers,
deterministic algorithms and four fixed full-pilot epochs. Shared existing
`train_step`, `train_epoch` and normal scoring functions perform actual
forward/backward passes, finite-gradient checks and optimizer updates.
No fitted normalization, augmentation or hyperparameter search was used.

Before the full pilot, a real single-batch probe changed parameters with finite
gradients. A separate one-epoch smoke then completed six updates on 48 patches
from four fitting parents, with 24 validation patches from two parents. Its
checkpoint reloaded exactly and its peak working set was 706,416,640 bytes.
The smoke performed seven updates in total including the preliminary probe.

## H. Actual per-epoch development losses

Untrained full-pilot validation MSE: **0.07434441422433506**.
Every epoch processed all 1,728 fitting patches and 432 validation patches.
Each parent contributes equally because all have twelve selected patches.

| Epoch | Fitting MSE | Validation MSE | Optimizer updates | Epoch wall seconds |
|---|---:|---:|---:|---:|
| 1 | 0.0281510112453597 | 0.015160586958464355 | 216 | 8.805804 |
| 2 | 0.01284813006296409 | 0.009171449737617944 | 216 | 8.695988 |
| 3 | 0.008132544923918667 | 0.00588374514349939 | 216 | 8.748198 |
| 4 | 0.00545794806870011 | 0.004014183262201927 | 216 | 8.831908 |

The pilot completed **864 actual optimizer updates** and 6,912 fitting-patch
presentations. The earliest strict minimum validation loss selected **epoch 4**.
These are normal-data reconstruction losses, not anomaly accuracy measurements.
The final model parameter hash differs from initialization.

## I. CPU and memory measurements

The pilot's timed training/validation/reload/native-diagnostic section took
**38.252989 seconds**. Total script wall time, including preparation and patch
inventory work, was **44.207780 seconds**; process CPU time was **166.03125
seconds**, aggregated over CPU threads. The four epoch intervals total about
35.08 seconds. Final peak working set was **713,900,032 bytes (680.83 MiB)**;
the largest epoch-reported peak was 497,594,368 bytes.

The smoke's corresponding timed section was 0.834699 seconds and total script
wall time 4.624244 seconds. Both stayed below their sealed 900/3,600-second
training/validation budgets and 8 GiB memory limit. Memory/deadline checks run
before each training and validation batch. Measurements are observed on this
host, not a controlled throughput benchmark: the offline quality run overlapped
part of the pilot. No GPU package, driver, global interpreter or security setting
was changed; dependency pins remain unchanged.

## J. Checkpoint integrity and reload

Selected artifact: `runs/metalparts-ae-pilot-v1/best.pt`.
SHA-256:
`c23543a8d22a305193e95a36985331c57b86ecb8b71e86c58b29a82fb1ab8651`.
The separate smoke checkpoint hash is
`a882d4796660f727a374cc1b4925a398219998a655480cd6a9e7f61a9d9ece68`.

Artifacts bind dataset `robustad`, category `metalparts`, pinned revision,
development split, protocol, native-patch recipe and purpose. CPU state_dict
loading uses `weights_only=True`, bounded/link-checked files, checksum and exact
identity matching, and finite tensor key/shape/dtype validation. Cashew identities
and non-null thresholds are rejected by the new artifact contract.

Selected validation scores replayed **exactly**, including order, against the
scores saved with the winning checkpoint. Two independent safe reloads produced
bit-equal probe outputs. The actual installed wheel also loaded this checkpoint
and produced finite synthetic outputs of shape 1 × 3 × 256 × 256.

## K. Development-only diagnostics

One normal validation parent, ID
`metalparts-009c0c2b5fc5b390c74f8bb7008f426da430d916e390affc2ab75df7dc933ee9`,
was reconstructed using complete native tiling. The output and continuous error
map are aligned at 2681 × 1500; the native pixel-mean MSE score is
**0.006451522931456566**. It differs from selected-patch validation MSE because
the diagnostic covers the complete image.

Local artifacts are `normal-reconstruction.png` and `normal-error-map.npy` in the
pilot run. Visual inspection of the generated reconstruction showed preserved
coarse shape but substantial blur and visible tile-grid boundaries. Correct
coordinates do not imply adequate reconstruction fidelity or defect sensitivity.
Synthetic identity-model tests verify exact native reconstruction through
overlapping edges. No pseudo-defect result was used as a real-defect metric.

The logged source access set is exactly **181 files**: training metadata and all
180 normal source images (preparation verifies every eligible normal even for
the smaller smoke). No official test image, source anomaly image or mask was
opened. No independent normal calibration subset was justified, so the model
returns continuous scores with **no validated binary threshold**.

## L. Tests, coverage and packaging

The complete offline suite passed **551 tests, one existing skip**, in 63.63
seconds. It includes 23 new synthetic tests. Combined statement/branch coverage
was **98.667935%** (99.018325% statements, 97.735192% branches). New development
data code reached 99.543379% and new training/artifact code 97.191011% combined
coverage; the existing 95% gate was preserved.

Coverage includes real gradient updates on fabricated RGB, actual metadata
semantics, normal selection, duplicate/group splits, test/calibration exclusion,
read guards, patch coordinates and IDs, source drift, CPU shapes, time/memory
stops, safe tensor loading, incompatible identities, null thresholds and native
stitch alignment. Existing tests were not weakened. The one existing Windows
symlink privilege skip remains; junction checks passed without security changes.

Ruff lint and formatting passed across 120 maintained Python files. Offline
lock verification and compatibility of all 61 development packages passed.
Wheel and sdist built successfully; archive inspection excludes real data,
weights, environments, reports and cache files. The generic build cache warning
did not correspond to cache content in either distribution. A fresh ignored
`.tmp/wheel-env-3d` installed the wheel offline without the editable project or
development group; all 49 installed runtime/app packages passed compatibility.
The isolated interpreter imported site-packages, exercised synthetic source
loading, loaded the new MetalParts checkpoint and ran all four frozen Cashew
models with exact agreement to their previous synthetic regression observations.

Evidence is under `reports/metalparts-3d-v1/final-quality/`, `isolated-probe.json`,
build/install logs, the sealed development evidence and the two ignored run
directories. A formatting issue was fixed before sealing execution code. No
post-training execution-code change or repeat pilot was needed.

## M. VS Code and Streamlit preservation

The existing workspace and official Codex extension remained in use. A separate
**FactoryVision: debug MetalParts synthetic update** launch executes fabricated
256-pixel RGB only. The real debugger stopped at `training/engine.py:51` after
backward with finite loss/gradients, stepped over `optimizer.step()` to line 52,
then continued through the parameter-change assertion. The Debug Console confirmed `loss=0.07936698; real update verified; no dataset opened`, and debugging stopped normally.

The in-app browser successfully loaded the running Streamlit Overview in this
session and showed all four Cashew methods available with verified frozen
artifacts. PCB1 BLOCKED and PCB2 INCONCLUSIVE remained visible. Browser tooling
is available and verified; no Chrome connection was listed, and Chrome
installation was not assessed. No MetalParts application option or UI redesign
was added. Cashew models were not retrained or recalibrated, and historical
metrics were not recomputed.

## N. Remaining benchmark limitations

PCB1: **BLOCKED**. PCB2: **INCONCLUSIVE**. RobustAD MetalParts official independent
evaluation: **NOT CLEARED**. Original Milestone 3C: **BLOCKED — DATASET INTEGRITY
OR ACQUISITION ISSUE**, unchanged. The new completion verdict concerns execution
of the owner-authorized exploratory training only.

Physical-item/capture independence remains unknown. Internal validation may be
dependent; selected patches omit much of each image; uniform tile stitching
leaves visible boundaries; four training epochs do not establish an optimal
model; normal-only reconstruction improvement does not establish anomaly ranking,
localization accuracy, recall, false-alarm rates or production reliability.
No official image AUROC, anomaly recall, pixel AP or other test performance
metric is available from this milestone.

## O. Changed files, preservation and Git

Modified: `.vscode/launch.json`, `README.md`, `docs/development.md`, and
`pyproject.toml` (source-distribution inclusion of the new protocol/report only).

Added:

- `src/factoryvision/data/metalparts_development.py`
- `src/factoryvision/training/metalparts.py`
- `scripts/metalparts_experiment.py`
- `scripts/debug_metalparts.py`
- `tests/test_metalparts_training.py`
- `docs/metalparts-exploratory-training-protocol.md`
- `docs/milestone-3d-metalparts-training.md`

The original 3C report/protocol, existing model implementations, frozen Cashew
artifacts, old run evidence and dependency lock are unchanged. The preservation
inventory covers 2,731 pre-3D files; only the four intentional maintained-file
edits above differ, with zero missing or unexpected changes. The new protocol,
development inventory and all 56 sealed execution files still match their seal.
New datasets were not acquired; all weights, reconstructions, maps, inventories,
logs and distributions remain ignored. Final Git status has four modified and
seven new maintained files, none staged. HEAD remains the reviewed 3C checkpoint;
3D was not committed, pushed or published.

## P. Recommended next milestone

Propose a separately authorized, development-only AE review of native patch
coverage, reconstruction fidelity and seam handling, with decisions sealed
before any additional training. Preserve this pilot as the baseline. Seek
publisher capture/physical-item provenance in parallel if independent benchmark
claims remain the eventual objective. Do not use official test masks or scores
to fix the observed normal-image artifacts. An anomaly sensitivity study,
another training run, threshold calibration or app exposure needs its own
explicit scope and evidence requirements. No later milestone was started.

**COMPLETE — EXPLORATORY METALPARTS TRAINING VERIFIED.**
