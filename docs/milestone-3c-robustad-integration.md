# Milestone 3C — RobustAD MetalParts integration and integrity gate

Date: 25 October 2026. Scope: local preservation, bounded acquisition, source
adapter and model-free audit. **BLOCKED — DATASET INTEGRITY OR ACQUISITION ISSUE.**
Acquisition and structural validation succeeded. Unresolved training/test
similarity and filename correspondence prevent clearance for a confirmatory
experiment. No MetalParts training, model inference or performance evaluation
occurred. Milestone 3D was not started.

## A. Reviewed Milestone 3B checkpoint

The only outstanding 3B file was `docs/milestone-3b-dataset-strategy.md` (580 lines).
It was reviewed in full, including staged content, whitespace, local paths,
personal information, credentials and generated-artifact exclusions. Its SHA-256
was `bf35a99a4ab1a1fd4b7ae0f92640cb9dc69b7d868c76f0e77d5744c483aef94c`.
Baseline checks passed: 496 tests, one existing Windows symlink skip, 98.577929%
combined statement/branch coverage, Ruff, offline lock consistency, compatibility
of 61 installed packages, wheel and source distribution builds.

The authorized local commit is
`2396ab8648788812fd1448d376d03d65c0a0f72f`, with message
`docs: establish alternative industrial dataset strategy`, using the approved
repository-local GitHub noreply identity. Parent:
`bdaa52908869fb2300daced8d6aee11a38d06af3`. The tree was clean immediately after
that one-file commit. No push or publication occurred.

Local evidence: `reports/robustad-3c-v1/checkpoint.json`, `baseline-*`, and
`preservation-before.json`. Generated evidence remains ignored.

## B. Existing Cashew application

All four frozen methods passed actual CPU inference on a fabricated 192 × 128 RGB
image. Each returned a finite score and a 128 × 128 map. Existing synthetic UI,
image-validation and geometry tests verify validation errors and aligned heatmaps
and overlays. No historical Cashew test metrics were replayed or recalculated.

| Frozen method | Unchanged threshold | Checkpoint SHA-256 |
|---|---:|---|
| Autoencoder | 0.0031531278509646654 | `da7e0caed8e365a08a3d301ec7e88536bc5df8e3f8b59f27635853e15ab4cdbc` |
| Spatial mean | 0.017185416119173163 | `04dc72dd50c54214d7bf874f650da58bf53eb80c803d550f8a75c7368bfd4c63` |
| Feature memory | 0.08742312490940093 | `4acd638fbef5c167c5121a12adf6dd796773cf0205420f6d317ace64fee1b1b0` |
| Student–teacher | 0.9195413202047348 | `d5b087c3c1a1a1a3937cd2d4b275cf9c605b774e760e7200f2a7262fa0f95ed9` |

The current-session in-app browser successfully loaded the local Streamlit
Overview and showed all four models available with verified artifacts. PCB1
remains **BLOCKED** and PCB2 **INCONCLUSIVE**. The app was not redesigned and has
no MetalParts inference option. The first navigation preceded server startup and
failed; navigation after startup succeeded. Browser tooling was callable and
verified in this session; Chrome was not connected in the inventory. Chrome
installation was not assessed.

The four full artifact hashes, scores, thresholds and shapes are retained in
`reports/robustad-3c-v1/cashew-regression.json`. Final preservation checks compare
the original evidence inventory and tracked files against their pre-3C hashes;
the intentional documentation and quality-runner edits are separated from frozen
data, models, protocols and historical reports.

## C. Official source and licence

The source is the public, ungated
[AmazonScience/RobustAD repository](https://huggingface.co/datasets/AmazonScience/RobustAD),
owned by the AmazonScience Hugging Face organization. The pinned revision is
`82c642b2d18be60f2dab104f8dd4dea1f4ccfacb` and the selected category is
[MetalParts at that revision](https://huggingface.co/datasets/AmazonScience/RobustAD/tree/82c642b2d18be60f2dab104f8dd4dea1f4ccfacb/MetalParts).
Public repository metadata reports `private: false`, `gated: false` and
`license: cc-by-4.0`. The pinned card declares CC BY 4.0; no additional restrictive
dataset terms or separate root licence file were found in the inspected tree.

Preserve attribution to Amazon Science and Latha Pemula, Dongqing Zhang and Onkar
Dabeer, *Robust AD: A Real World Benchmark Dataset For Robustness in Industrial
Anomaly Detection*, CVPR Workshops 2025. Retain the source/revision, licence link,
notices and an indication of changes for derived material. CC BY 4.0 permits
sharing and adaptation, including commercial use, subject to its conditions;
this is not a grant of unrelated rights or a decision about publishing project
code or pretrained weights. See the [licence deed](https://creativecommons.org/licenses/by/4.0/)
and [official paper](https://openaccess.thecvf.com/content/CVPR2025W/VAND/html/Pemula_Robust_AD_A_Real_World_Benchmark_Dataset_For_Robustness_in_CVPRW_2025_paper.html).

The paper supports both supervised and unsupervised settings. Anomalous source
training rows are therefore preserved, not silently relabeled or treated as a
corrupt one-class split. Physical-item independence is a separate question.

## D. Bounded acquisition and provenance

Only the pinned `MetalParts/` tree was acquired: **1,912 files,
684,375,437 bytes** (684.38 decimal MB; 652.67 MiB). Files are individually hosted;
there was no archive extraction. Exact paths, sizes and official Git/LFS hashes
were enumerated before acquisition through the official recursive repository API.
Free space before acquisition was 519,449,989,120 bytes.

The conservative cumulative download reservation was **705,509,937 bytes**,
below the authorized 1,073,741,824-byte limit. This includes a 20 MiB allowance
for public metadata/source research and reserves the remaining response size
before every attempted request, including retries. Total pinned source-directory
storage, including saved provenance files, is **685,899,888 bytes**, well below
3 GiB. The complete RobustAD repository was not downloaded.

Files reside under ignored
`data/raw/robustad/82c642b2d18be60f2dab104f8dd4dea1f4ccfacb/`.
Each LFS object matched its published SHA-256; non-LFS JSONL files matched the
published Git blob SHA-1. Local SHA-256 values were recorded for all files and
verified again after acquisition. Original files were not transformed.

The downloader supported partial-range resume, rejected unsafe paths and links,
and used revision-pinned official resolve URLs. It used no authentication or
stored tokens, added no dependency and executed no downloaded code. A Windows
ledger-replacement permission error and a transient path-containment assertion
interrupted two attempts. All paths were rechecked within the intended root;
retries retained the containment, budget and checksum checks. Acquisition then
completed successfully. These events remain in `download-ledger.json`.

Saved provenance includes official repository metadata, complete tree pages,
source card, per-file inventory, download ledger and the official paper. Local
evidence is under `reports/robustad-3c-v1/source/` and the ignored revision folder.

## E. Actual inventory and official membership

Every source domain has `metadata.jsonl`, `normal/`, `anomaly/` and `masks/`.
The metadata has exactly `file_name`, integer `label`, and nullable `mask`.
Image filenames are domain-relative; mask references are repository-relative.

| Domain | Source condition | Normal | Anomaly | Images | Masks |
|---|---|---:|---:|---:|---:|
| train | Source acquisition | 180 | 51 | 231 | 51 |
| test0 | Source condition | 120 | 51 | 171 | 51 |
| test1 | Lighting | 120 | 80 | 200 | 80 |
| test2 | Position | 120 | 80 | 200 | 80 |
| test3 | Rotation | 120 | 80 | 200 | 80 |
| test4 | Scale | 120 | 80 | 200 | 80 |
| test5 | Background 1 | 60 | 40 | 100 | 40 |
| test6 | Background 2 | 60 | 40 | 100 | 40 |
| Total | Eight source domains | 900 | 502 | 1,402 | 502 |

Together with eight JSONL files this gives exactly 1,912 files. All images have
native dimensions **2681 × 1500**. There are seven official test domains; none
was assigned to internal fitting, validation or calibration. All original
source memberships remain intact.

Domain names and filenames agree with the paper's broad shift categories.
Position/scale filenames use `+5%_to_-5%`, while the paper's generic ranges differ.
Those exact perturbation magnitudes were not independently established. The
adapter preserves the filenames instead of rewriting them to match a narrative.

## F. Labels, masks and decoding

All 1,402 source images decoded successfully with the existing bounded decoder.
Label `0` maps to the canonical `Label.NORMAL` and requires null mask; label `1`
maps to `Label.ANOMALY` and requires a corresponding PNG. All 502 masks decoded in
single-channel `L` mode, contained nonempty foreground, used only values 0 and
255, and matched native image dimensions. The matching mask must be in the same
source domain and have the same stem as its image.

No missing, unexpected, malformed, undecodable or misaligned files were found.
Source training contains **51 anomalies**; only its **180 normals** are eligible
for a later unsupervised development split. This milestone assigned no internal
split. Test masks were opened only to verify format, alignment, nonemptiness and
class values. No defect-size distribution, fitted preprocessing, threshold,
model prediction or performance metric was derived from them.

## G. Minimal provider adapter

`factoryvision.data.metalparts` adds frozen typed source, sample and manifest
objects with a narrow `factoryvision.robustad-metalparts.source-v1` envelope.
It reuses existing `Label`, `OfficialSplit`, contract diagnostics, portable path
checks, containment resolution and bounded image/mask decoding. The existing
VisA schema, split logic and training pipeline are unchanged: their category
allowlist and normal-only training assumptions cannot truthfully represent this
source's anomalous training rows and domain identities.

The envelope records pinned dataset identity, category, source domain, original
filename/metadata reference, label, image/mask paths, dimensions, licence and
attribution. Stable IDs hash dataset identity, revision and full source path.
The manifest hash binds the canonical exported records. Publisher physical-item
IDs are absent, so `group_id` is explicitly null. A separate synthetic provenance
tuple prevents fabricated fixtures from masquerading as official observations.

The adapter rejects unknown labels, booleans as integer labels, duplicate JSON
keys/paths, missing fields, unsafe paths, source/domain mismatches, bad masks and
unreferenced files. `training_normals` is an explicit eligibility filter, not an
implicit split or authorization to fit. It is not an operational model adapter
and cannot be handed to the existing VisA-only training entry point as though
compatibility had been established.

`factoryvision.data.metalparts_audit` reuses the existing fixed fingerprints and
cross-partition screen. `scripts/audit_metalparts.py` is an explicit local-only
entry point that refuses an existing output directory and source-directory
output. The full source manifest SHA-256 is
`de3f0854b6c1d3c9448329ef7f5d126e37d631a9371277f777076b7f395f7e17`.

## H. Integrity findings and unresolved dependence

The protocol was sealed before image decoding:
`docs/protocols/robustad-metalparts-integrity-v1.json`, SHA-256
`ab1833755a0c6f76317c299addd539c11a67cf3cc7142a3065de749af0dcd18c`.
It distinguishes byte identity, native RGB pixel identity, heuristic similarity
and filename correspondence, and stops for unresolved high-risk cross-train/test
concerns. No threshold was adjusted after looking at the results.

| Check | Finding |
|---|---|
| Published inventory/content hashes | All 1,912 files verified |
| Exact-byte duplicate groups, all source files | 0 |
| Exact-byte duplicate groups, images | 0 |
| Native decoded RGB pixel-identical image groups | 0 |
| Cross-domain image comparisons | 851,501 |
| Fixed-rule similarity candidates | 980 |
| Candidates involving source train and a test domain | 626 |
| Candidates involving eligible source-train normals | 475 |
| Repeated cross-train/test basenames | 145: 104 normal, 41 anomaly |
| Confirmed related physical captures | Not established |

Native pixel hashes include RGB mode, width, height and pixel bytes without
resizing. Similarity uses the existing rule: dHash64 distance ≤4 **or** pHash63
distance ≤4 **or** 32 × 32 RGB RMS ≤0.02. These low-resolution fingerprints are
an integrity heuristic; their thumbnails are not proposed model preprocessing.

The 626 source-train/test candidate pairs comprise 389 normal/normal, 86
normal/anomaly, 88 anomaly/anomaly and 63 anomaly/normal pairs, with training label
listed first. Domain-pair counts are 190 with test0, 346 with test1, 4 with test2,
19 with test3, 19 with test4, 48 with test5 and zero with test6. Zero candidates
for a domain does not prove independence. The remaining 354 candidates link test
domains and matter for any pooled uncertainty estimate.

All 145 basename correspondences link train and test0. Only two of those same-name
pairs also meet the fingerprint rule. Filename reuse therefore must not be
equated with identical content, and these counts must not be summed as though
they were disjoint evidence of confirmed leakage. Repeated manufactured shapes
can trigger similarity without sharing physical identity.

Nevertheless, official metadata supplies no physical-item, capture-session or
acquisition-group key that resolves these relationships. Hashing alone cannot
establish independence. No visual adjudication was used to label similar parts
as confirmed duplicates. The gate stays blocked under the predeclared rule;
this is **unresolved independence**, not a finding of confirmed dataset fraud or
confirmed leakage. No image was removed or reassigned to manufacture clearance.

Full per-image hashes, candidate pairs, repeated basenames, source verification
and summary counts are in `reports/robustad-3c-v1/integrity/`. The audit's nonzero
status is the intended blocked outcome, not a decode failure. Resolution needs
publisher acquisition/group evidence or a separately authorized, explicitly
non-confirmatory research objective. Repeating similarity thresholds or moving
official samples between splits is not a resolution.

## I. CPU, storage and resolution

The existing verified CPU environment remains Windows AMD64/Python 3.12.15,
PyTorch 2.13.0+cpu and torchvision 0.28.0+cpu on the Ryzen 7 9700X host. No GPU
driver, ROCm package or new dependency was installed. The machine has about
31.61 GiB OS-visible RAM from the prior hardware inspection; current free RAM
is variable. This audit establishes decode feasibility, not training throughput.

Each native RGB image contains 4,021,500 pixels: approximately 11.51 MiB as RGB
bytes or 46.02 MiB as float32 RGB, before activations and masks. Streaming is
required; do not predecode all images. Whole-image 128 × 128 resizing reduces
width by about 20.95×, height by 11.72× and distorts aspect ratio, which is
unsuitable as an unexamined tiny-defect preservation choice.

The pre-audit proposal was native 512 × 512 tiles, 128-pixel overlap, full border
coverage, no image rescaling and native-coordinate reconstruction. At these
dimensions, deterministic trailing-edge coverage gives 7 columns × 4 rows =
28 tiles/image. The last stride is shorter where necessary. Uniform averaging
of overlapping maps and a full-coverage assertion would avoid gaps and preserve
coordinates; masks stay native, with nearest-neighbor only if a label transform
is explicitly needed. Non-overlapping patches are cheaper but increase seam
risk. Aspect-ratio-preserving whole-image letterboxing retains global context
but still reduces small defects if heavily downscaled.

Compatibility is **not yet implemented**: `ConvAutoencoder` currently accepts
32–256 pixels and the frozen feature recipe requires exactly 128. A future
versioned 512-pixel variant must extend and test those shape/resource contracts
without changing existing Cashew recipes. Native 256 tiles with 64 overlap are
a technically compatible AE alternative (112 tiles/image), but choosing that
fallback requires a new sealed plan before fitting, not a test-informed switch.

## J. Offline and packaging verification

The final full offline run passed **528 tests, one existing skip**, in 51.71
seconds. Combined statement/branch coverage was **98.686630%** (99.024214%
statements, 97.788462% branches); both new package modules reached 100% combined
coverage. The existing 95% gate is unchanged. The skipped Windows symbolic-link
test still requires a privilege unavailable on this host; security settings were
not changed, and existing junction-containment tests passed.

Ruff lint passed and all 115 maintained Python files passed formatting. Offline
`uv lock --check` resolved the unchanged 61-package lock, and development
`uv pip check` confirmed all 61 installed packages compatible. Both wheel and
source distribution builds passed, including building the wheel from the sdist.
The build emitted a generic warning about the project-local cache; member
inspection confirmed it was absent, along with datasets, runs, weights,
environments and generated reports. Both new provider modules are packaged.

A new ignored `.tmp/wheel-env-3c` was synchronized offline with runtime/app
dependencies and no editable project or development group. The actual wheel was
installed with `--no-index --no-deps`; 49 installed packages passed compatibility
checks. `python -I` imported from that environment's site-packages, validated eight
fabricated source-domain images through the installed adapter and ran all four
real frozen Cashew models on fabricated RGB. Scores, thresholds and artifact
hashes matched the development-environment check exactly. The archive-inspection
helper initially tried to open uv's small `.gitignore` as a tar archive; limiting
inspection to the two named distribution files resolved the helper error. The
distributions themselves were valid.

Evidence: `reports/robustad-3c-v1/final-quality-3/`, `final-quality-3.log`,
`build.log`, `isolated-*.log`, `isolated-probe.json` and the explicit
`integrity/source-verification-final.json`. Model-free integrity completed with
the intended blocked verdict described above; it did not produce accuracy metrics.

The new 32 tests fabricate their own images and masks. They cover preservation of
anomalous train rows, source/domain identity, stable IDs, immutable objects,
malformed/duplicate JSON, path and label rules, missing/corrupt images and masks,
mask alignment, unexpected files, published hash checks, byte-vs-pixel identity,
filename-only uncertainty and refusal to overwrite evidence. Existing tests were
not weakened. Real source verification was explicitly invoked, never added to
default test discovery.

An initial quality pass found only formatting in the edited quality runner; a
later new assertion incorrectly expected fabricated identical masks to have no
byte-duplicate group. The runner was formatted and the test now explicitly
checks that all eight identical fixture masks are detected, plus a separate
two-image group after deliberate duplication. Failed logs are retained alongside
the final passing run. Neither issue changed the actual dataset findings.

## K. VS Code and debugger

The existing canonical VS Code workspace and official Codex extension were used.
The existing **FactoryVision: debug local inference** configuration reached an
actual breakpoint at `inference/service.py:236` with `model_id=autoencoder`.
Stepping reached the context-manager exit at line 210; continuing printed
`Synthetic debug probe: score=0.00552295; map=(128, 128)` and completed. This used
the frozen AE and fabricated RGB, not a MetalParts image. The primary interpreter
and existing launch/task configurations remain unchanged. No task runs on open.
UI and debugger observations are in `application-ui-debug.json`.

## L. Exact maintained files changed in 3C

Modified:

- `README.md`: current blocked status and report links.
- `docs/development.md`: explicit audit and fresh-evidence verification commands.
- `scripts/check_quality.py`: optional report destination and maintained Ruff targets.

Added:

- `src/factoryvision/data/metalparts.py`: immutable source adapter.
- `src/factoryvision/data/metalparts_audit.py`: model-free integrity functions.
- `scripts/audit_metalparts.py`: explicit guarded local audit entry point.
- `tests/test_metalparts.py`: 32 offline fabricated-data tests.
- `docs/protocols/robustad-metalparts-integrity-v1.json`: pre-decode integrity rules.
- `docs/milestone-3c-robustad-integration.md`: this report and conditional plan.

Downloaded data, acquisition scripts/logs, per-image evidence, local paper copy,
quality reports, distributions and temporary installed environments are ignored,
not proposed Git content. The 3B report belongs to the checkpoint, not the 3C diff.

## M. Git status and preservation

Milestone 3C changes remain **uncommitted** and unstaged. HEAD stays at the 3B
checkpoint above. No remote was added and nothing was pushed or published.
Frozen models, thresholds, VisA data, PCB evidence, previous source snapshots and
historical reports remain preserved. Only the three intentional maintained-file
edits listed above should differ among the pre-3C inventory; new files are listed
separately. The final comparison checked **2,493 files**: 2,490 unchanged,
the three intentional changes, zero missing files and zero unexpected changes.
The new integrity protocol still matches its pre-decode seal. Full comparison
results are in `reports/robustad-3c-v1/preservation-after.json`. Git reports three
modified and six new maintained files, with no staged changes; generated data
and evidence remain ignored. Whitespace review covers the final diff.

## N. Conditional first MetalParts experiment

**Recommendation: resolve the integrity gate first; then develop a separate
MetalParts native-tile autoencoder, followed by a frozen feature-memory baseline.**
This is planning only and does not authorize execution. Public test images/masks
have been decoded for this integrity audit; a later experiment must disclose that
exposure and must not claim an unopened blind test.

1. **Eligibility and split:** retain all official domains. Fit, validate and
   calibrate only source-train normals. Exclude its 51 anomalies from all three
   unsupervised subsets. Obtain physical/capture groups before assigning splits;
   hash seed 42 at the group level, retain every tile from an image in its parent
   group, and seal parent/assignment hashes. A 70/15/15 target would give
   126/27/27 images only if independence evidence permits singleton groups;
   actual counts must follow verified groups. Stop if groups cannot support
   meaningful disjoint validation and calibration. Do not silently invent IDs
   from filename suffixes or borrow test normals for calibration.
2. **Compatibility before fitting:** add separately versioned native-512 tile,
   geometry/stitching and checkpoint contracts using synthetic geometry tests.
   Keep the existing AE family and frozen ResNet18 layer2/3 feature family;
   require CPU float32, deterministic seeds 42 for Python/NumPy/Torch, four Torch
   threads, zero loader workers, fixed stable non-training order and no implicit
   weight download. Each artifact must bind `robustad`, `metalparts`, revision,
   source-manifest hash, group split, tile geometry and model recipe.
3. **Predeclared AE candidate:** same compact convolutional family with a new
   512-resolution recipe, RGB [0,1], native 512 tiles/128 overlap, no augmentation,
   Adam learning rate 0.001, batch two and maximum 12 epochs. Select the earliest
   checkpoint achieving the lowest mean parent-image normal validation MSE,
   aggregating tile losses within each image before averaging images. Do not
   choose epochs using test anomalies. Use channel-mean squared reconstruction
   error for tile maps, uniform overlap averaging and the maximum native map
   value for image score. Freeze aggregation before fitting.
4. **Feature-memory candidate after AE feasibility:** freeze the existing
   verified ImageNet ResNet18 weights and their normalization, use layer2/3
   features with the existing averaging/concatenation concept and exact RMS
   nearest-neighbor scoring. A 512 tile yields a 64 × 64 query grid; process one
   tile at a time with query chunks 256 and bank chunks 512. Proposed bank cap:
   4,096 descriptors, deterministically sampled with seed 42 from fitting images
   only, balanced across parent images before tile/position sampling. Freeze the
   selection algorithm and memory cap before fitting; retain selected indices.
   Interpolate tile maps to tile coordinates, average overlaps, then take the
   maximum native map score. Do not treat the old 128-only recipe as compatible.
5. **Normal-only calibration:** reserve calibration images until model selection
   is complete. For each frozen family use its calibration parent-image scores,
   quantile 0.95, linear interpolation and strict `score > threshold`, recorded
   exactly with the comparison rule. Twenty-seven calibration images would give
   a coarse estimate, not a guaranteed 5% production false-alarm rate. Report
   uncertainty and avoid operational claims. Pixel ranking metrics need no
   mask-selected cutoff; any pixel operating threshold requires a separate
   predeclared normal-only rule, never optimization against official test masks.
6. **CPU feasibility gate:** on fitting normals only, time decode, tile generation,
   one AE smoke epoch and feature extraction/query separately after authorization.
   Record cold/warm latency, peak resident memory and estimated full-run cost.
   Abort at 8 GiB process RAM or under 8 GiB available system RAM; smoke cap
   15 minutes, full fit cap two hours per family, and projected single evaluation
   cap two hours per family. These are proposed stop budgets, not demonstrated
   performance. A failed budget returns to planning; it does not justify hidden
   downsampling, test subsampling or an unsealed recipe switch.
7. **Held-out domain evaluation only after gates:** freeze source/config/weights,
   membership and thresholds before any model scoring of test0–test6. Use a
   one-shot guarded runner and preserve per-image predictions. Report per-domain
   image AUROC/AP, false-positive rate, recall and confusion at the frozen
   threshold, plus pixel AUROC/AP and a precisely specified AUPRO integration
   range. Include per-domain counts, worst-domain results and domain-balanced
   macro summaries; do not rely only on pooled metrics. Bootstrap verified
   acquisition groups, not tiles or pixels. If no group identity is available,
   disclose that image bootstrap can understate uncertainty. With correlated
   source/test captures, results can only be descriptive under a separately
   authorized reframing, not evidence of independent generalization.
8. **Limits and stop conditions:** preserve the unresolved 3C findings; stop for
   source/hash/geometry drift, overlap evidence, invalid masks, source-revision
   mismatch, incompatible checkpoint identities, nonfinite scores or resource
   overruns. No official test data may select preprocessing, bank, architecture,
   thresholds or stopping epochs. The paper's robustness summaries depend on a
   trustworthy source-condition reference; test0's unresolved provenance prevents
   a confident confirmatory robustness claim. One category, public labels,
   tile seams, limited global context, small calibration size and manufacturing
   dependence remain limitations. Consider student–teacher only after the two
   simpler families pass compatibility and resource checks. Keep Cashew models
   available throughout; do not expose MetalParts in the app until an appropriate
   verified model and clearly stated limitations exist.

## O. Final verdict

**BLOCKED — DATASET INTEGRITY OR ACQUISITION ISSUE.**

The official acquisition, adapter and structural audit succeeded. The blocker is
unresolved cross-train/test dependence: 626 similarity candidates (475 involving
eligible training normals) and 145 reused basenames, with no publisher group
identity to adjudicate them. No confirmed exact image duplicates were found and
similarity alone was not promoted to confirmed duplication. Existing Cashew work
is preserved. No training, MetalParts test model inference, later milestone,
3C commit or publication was performed.
