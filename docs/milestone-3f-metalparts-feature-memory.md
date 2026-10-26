# Milestone 3F — MetalParts feature-memory comparator

**EXPLORATORY SOURCE-DOMAIN COMPARISON — NOT INDEPENDENT TEST PERFORMANCE**

Verdict: **COMPLETE — METALPARTS FEATURE-MEMORY COMPARATOR VERIFIED**.

## A. Reviewed 3E checkpoint

Local commit `45fcaa36c5ccb72eb4dfa76e7377b9c863d6ad0d`,
`feat: integrate experimental MetalParts inspection`; parent
`2282bfb811c38ff26440b0cae56b8bdc053f9661`. Exact intended 22-file 3E inventory
reviewed; 586 baseline tests passed, one existing Windows symlink skip; Ruff,
formatting, lock and dependency compatibility passed. No assets/secrets/private
paths were staged. Approved repository-local noreply identity was used. Tree was
clean after checkpoint. No push. Review: `reports/metalparts-3f-v1/3e-checkpoint-review.json`.

## B. Exact maintained file inventory

Modified: `.vscode/launch.json`, `README.md`, `docs/architecture.md`,
`docs/development.md`, `docs/inference.md`, `pyproject.toml`,
`src/factoryvision/app/metalparts_ui.py`, `src/factoryvision/app/ui.py`,
`tests/test_app.py`.

Added: `docs/metalparts-feature-memory-protocol.md`,
`docs/metalparts-feature-model-card.md`, this report;
`scripts/debug_metalparts_feature.py`, `scripts/metalparts_feature_experiment.py`,
`scripts/review_metalparts_ae.py`;
`src/factoryvision/evaluation/metalparts_comparison.py`,
`src/factoryvision/inference/metalparts_comparison.json`,
`src/factoryvision/inference/metalparts_feature.py`,
`src/factoryvision/inference/metalparts_feature_artifacts.json`,
`src/factoryvision/models/metalparts_memory.py`,
`src/factoryvision/training/metalparts_memory.py`,
`tests/test_metalparts_feature_service.py`, `tests/test_metalparts_memory.py`.

Ignored local evidence: `reports/metalparts-3f-v1/`,
`runs/metalparts-feature-v1/`, `runs/metalparts-feature-normal-v1/`,
`runs/metalparts-feature-comparison-v1/`, `dist/3f-verification/`,
`.tmp/wheel-env-3f/`. Model/data/report binaries are not Git changes.

## C. Original AE evidence review

No scoring/alignment discrepancy was found. Original checkpoint, maps, metrics,
model card and report remain unchanged. No AE forward pass was needed for the
comparison. Native 256-pixel tiles, stride 192 and appended trailing edges cover
every pixel. Reconstructed RGB is uniformly averaged over overlaps **before**
channel-mean squared RGB error is computed. Image score is the native mean of that
error map. Masks are verified at native geometry and binarized from 0/255; normals
use zero masks. No metric mask resizing. Pixel quantization is fixed floor(MSE×65535).

AE normal scores (min/Q1/median/Q3/max):
0.00179812 / 0.00251831 / 0.00299550 / 0.00355605 / 0.00645152.
Anomalies: 0.00187949 / 0.00258710 / 0.00335824 / 0.00457516 / 0.00756749.
Their overlap limits image ranking. Reconstruction can model defects, while edges,
lighting, blur and seams can produce irrelevant error.

Post-hoc 16-pixel outer-frame review: 11,550,816 of 349,870,500 pixels (3.30145%)
contain 4.70523% of total reconstruction error. Boundary concentration is modest;
it alone does not explain weak localization. This descriptive band did not change
either pipeline. Exact saved-map means equal original image scores.

Post-hoc illustrations, selected by AE per-image pixel AP, not representative samples:

| Source ID suffix | Observation | AE pixel AP | Feature pixel AP | Max inside mask |
|---|---|---:|---:|---|
| bd307ef4 | Tiny edge defect; strong unrelated edges/hole responses | 0.00020195 | 0.00074703 | Neither |
| e3f34f | Small defect near lower center localized | 0.32147963 | 0.63152092 | Both |

Full IDs, overlays/masks and cached-map evidence are in
`reports/metalparts-3f-v1/ae-review/`; cross-method findings in
`failure-comparison.json`. AE maxima fall inside masks in 16/51 anomalies versus
28/51 feature maps; both outside on 22. This peak diagnostic is not a detection
threshold, segmentation recall or substitute for ranking metrics.

## D. Fixed architecture

[Protocol](metalparts-feature-memory-protocol.md) was written before fitting;
SHA-256 `b9cc954774583f1e720cfcf782648161847a0b0dcbdc55c561f44ff6af6f9041`.
Seal SHA-256 `42818f5818245b98741def5deb32f8916f1102a1ea81ac635307c712a01768ec`.
Existing trusted ResNet18/ImageNet preprocessing, frozen layer2/3, 3×3 local average
pooling, bilinear feature alignment, 384 dimensions, 16×16 descriptors. Native
256/stride192 tiles resize to 128 using existing bilinear antialias preprocessing.
No learned normalization, projection, smoothing, coreset optimization or reweighting.
This is PatchCore-inspired, not exact PatchCore.

Bank budget 8,192 descriptors; four per selected tile gives 6,912. Seed is the first
eight SHA-256 bytes of `42|tile-ID`; first four randperm(256) indices, sorted.
Native tile/parent/descriptor indices are recorded. Batch one, four CPU threads,
deterministic algorithms. Chunked RMS 1-NN uses 256 query ×512 bank entries, at most
0.5 MiB per distance block. Image maximum precedes interpolation; native map averages
overlapping expanded distance maps. No threshold. Full source image has 112 tiles.

## E. Weights and dataset provenance

[AmazonScience/RobustAD](https://huggingface.co/datasets/AmazonScience/RobustAD),
CC BY 4.0; Pemula, Zhang, Dabeer, CVPR Workshops 2025. Revision
`82c642b2d18be60f2dab104f8dd4dea1f4ccfacb`. Existing source-verification SHA-256
`84f2716eaf3afe766d34e569d4fb917abcd383201ffc5a8dc5f5ca5908b672fb`.
Pretrained `resnet18-f37072fd.pth`, 46,830,571 bytes, SHA-256
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
No new data/weight/dependency downloads. No Cashew example was reused in MetalParts memory.

## F. Actual fitting inventory

Exactly 144 accepted normal fitting parent IDs, 1,728 selected native tiles;
four descriptors per tile. Fit read allowlist contains exactly those 144 source
images and no masks. Exact identities, source hashes and partitions are sealed in
`reports/metalparts-3f-v1/seal.json`; `runs/metalparts-feature-v1/selection.json`
maps every tile to parent, coordinates and descriptor indices. Split SHA-256
`bb87d41b2a8296d91f9f67496128e7f7280c2a7ccb688e78f48c175874874c41`.
The 36 validation normals, 51 source anomalies and all target payloads are excluded.
No neural-network gradient training took place.

## G. Artifacts and reproducibility

`runs/metalparts-feature-v1/memory.pt`: 6,912×384 float32, 10,618,409 file bytes,
10,616,832 raw tensor bytes, SHA-256
`38d89de7500b8c9c9ae1bb5e2bbdff7a400318f072fbcfba9441aa7e8a713b10`.
Metadata pins category, revision, pretrained hash, recipe, split/protocol and exact
tile IDs. Bounded weights-only CPU loading rejects mismatch, invalid dimensions,
nonfinite data and linked/oversized assets. Extraction replay and serialization
reload matched bit-for-bit. Backbone state before/after remained
`661ce6fdbd2db950486550b98dfccc25e09770a1d742d827cac4b631c0e8c33d`.
First normal inference also matched exactly after independently loading the bank/model.

Recorded one-shot commands: `scripts/metalparts_feature_experiment.py seal`, `fit`,
`normal`, `compare` through `.venv/Scripts/python.exe`. All already completed;
do not rerun or remove run guards. Source/code hashes are checked before each phase.
Default tests use synthetic data. Preserve local assets to replay evidence.

## H. CPU and memory measurements

| Phase | Wall time | Peak process working set |
|---|---:|---:|
| Initial memory fit | 10.118 s | 475,435,008 bytes including replay |
| Fit plus exact extraction replay/reload | 20.008 s | 475,435,008 bytes |
| Normal verification | 36 images, median about 1.55 s/image | 537,673,728 bytes |
| Comparison, cached normals plus 51 new anomalies | 115.824 s | 556,294,144 bytes |

Fit/replay CPU time 79.047 s; comparison CPU time 434.234 s. Feature inference across
87 records: min 1.49146, median 1.64264, p95 2.10200, max 2.19762 seconds. Historical
AE: min 0.25542, median 0.35863, p95 0.36330, max 0.37135; peak 802,607,104 bytes.
Feature timings include normal-pass cached timing records and anomaly-pass measurements;
quality checks overlapped part of the comparison. These separate runs do not isolate
speed or memory differences. Native inference timing excludes image decode, model
load, plotting and metric calculation. Small app uploads are faster.

Feature deployment assets include the 10.6 MB bank plus the existing 46.8 MB full
pretrained file and metadata; AE checkpoint is 156,595 bytes. Four CPU threads,
batch one, bounded bank and tiles are practical for deliberate local use, not a
verified real-time production rate. All predeclared 8 GiB/time budgets passed.

## I. Image ranking

Same 87 stable IDs/order as 3E: 51 anomalies, 36 validation normals.

| Method | AUROC | AP |
|---|---:|---:|
| Preserved AE | 0.5969498911 | 0.7142665461 |
| Feature memory | 0.8017429194 | 0.8875053445 |

Image positive prevalence/constant-score AP reference is 0.5862068966. Feature normal
scores min/Q1/median/Q3/max: 0.14381585 / 0.15444570 / 0.15778381 / 0.16566831 /
0.17539382. Anomalies: 0.14298236 / 0.16176481 / 0.19467521 / 0.21515156 / 0.24655156.
Ranking improved on this observed set; distributions still overlap. Raw feature
distance and AE mean MSE are not common units. No threshold was chosen.

## J. Native pixel ranking

349,870,500 pixels: 84,903 positive, 349,785,597 negative. Prevalence and constant-score
AP reference: 0.0002426698 (0.02426698%).

| Method | AUROC | AP |
|---|---:|---:|
| Preserved AE | 0.8671747847 | 0.0215414150 |
| Feature memory | 0.9635008760 | 0.3051214110 |

Same masks, native resolution, class definitions, tie handling and fixed bin width
1/65535. Both used 65,536 bins in this run. Feature implementation permits values
above one by extending bins without clipping; synthetic tests cover that behavior.
Every native pixel is counted, no sampling. AE cached histograms replay exactly.
Pixel AUROC can look high with many easy negatives; AP better exposes the severe
imbalance but depends on prevalence. Pixel AP improved in aggregate and on 41/51
anomaly images, declined on 10/51. Both methods retain failures. Maps are coarse
feature distances interpolated to native resolution, not native-resolution features.

## K. Interpretation and unresolved dependence

This is one exploratory candidate with no post-result settings changes. The 36
normals influenced earlier AE selection; anomalies and their prior metrics were
already observed. Physical-instance grouping is unavailable. Correlated images
and pixels invalidate simplistic independent confidence intervals. Descriptive
AP/peak comparisons do not establish operational recall/false-alarm performance.

Original 3C evidence includes 145 train/test0 filename correspondences and 626
heuristic train/test similarity candidates, without exact image-byte/pixel duplicates.
Candidates remain unresolved. No target images or masks were opened here to
re-adjudicate or evaluate them. Independent target evaluation remains NOT CLEARED.
No aggregate-metric winner, production readiness or target-generalization claim.

## L. Application and browser verification

Preserved Streamlit app, localhost binding, shared upload protections and four
Cashew methods. MetalParts permits AE, Feature-Memory Detector or both. Explicit
upload, selection and run; native original/heatmap/overlay, separate scales/units,
continuous score, latency and pinned provenance. Reconstruction remains available
for AE. Selection/category changes clear old results; missing artifacts fail safely.
Research view retains the old results and adds the explicitly exploratory comparison.

Current-session in-app browser load and actual two-model inference on fabricated
RGB passed. Native synthetic scores: AE 0.02174602; feature 0.16959599, no threshold.
These are execution probes, not dataset metrics. Screenshot:
`reports/metalparts-3f-v1/browser-comparison.png`. Browser callable and working;
no Chrome connection appeared in inventory. Separate extension/plugin installation
state was not assessed. No claim of a connected Chrome browser is made.

## M. Quality and packaging

Full offline suite: 600 passed, one pre-existing Windows symlink privilege skip.
Branch coverage is 98.29%, exceeding the unchanged 95% gate; measurement is recorded in
`reports/metalparts-3f-v1/final-quality/coverage.json`. Ruff lint/format, offline lock
check and installed dependency compatibility pass. No dependency versions changed.
Wheel/sdist and fresh isolated runtime installation are checked under
`dist/3f-verification/` and `.tmp/wheel-env-3f/`; exact inventories/hashes and
installed-package model replay are in `reports/metalparts-3f-v1/isolated-probe.json`.
No dataset, masks, weights, checkpoints, generated reports or environments enter
distributions. Offline tests exercise isolation, determinism, map direction/alignment,
edge coverage, artifact rejection, category checks, quantization and app selection;
existing NN chunk bounds and target guards remain intact.

## N. Actual VS Code debugger

New **FactoryVision: debug MetalParts feature memory** launch uses the project
interpreter and fabricated pixels only. Real breakpoint reached at
`scripts/debug_metalparts_feature.py:19`, after inference; stepped over the native
320×384 map assertion to line 20, continued successfully. Debug console reported
feature score 0.18294276 and no dataset opened. Screenshot `debugger.png` in the
3F report directory. Existing launch entries and official Codex extension preserved.
No startup installation/training tasks or editor-wide settings were introduced.

## O. Historical preservation

Before-snapshot covers 1,460 files; 1,451 are byte-identical and only the nine
listed maintained files changed. All 154 sealed experiment inputs/code files match;
231 permitted source images and their masks also rehash exactly. Final comparison allows only listed
maintained application/docs/test/launch/packaging changes. Historical datasets,
pretrained weights, runs, saved maps, 3E model card/report/protocol and research
registries remain byte-identical. New bank does not overwrite Cashew memory.
Isolated installed-package fabricated-input regression reproduces all four Cashew
scores, hashes and thresholds, plus original MetalParts AE score/map exactly.
Statuses remain: Cashew previously evaluated research; PCB1 BLOCKED; PCB2
INCONCLUSIVE; MetalParts experimental source research; independent target NOT CLEARED.
Final hash evidence: `reports/metalparts-3f-v1/preservation-after.json`.

## P. Git state

3E is committed locally at 45fcaa36c5ccb72eb4dfa76e7377b9c863d6ad0d.
The 23 maintained 3F files listed above remain uncommitted; no staged assets, no
3F commit, no push/publication. Exact final status is stored in
`reports/metalparts-3f-v1/final-git-status.txt`. No unrelated project edits.

## Q. Recommended next milestone

A separately authorized local release-candidate hardening milestone: reproducible
offline asset setup and integrity diagnostics, application usability/accessibility,
startup/error recovery and documented hardware latency budgets. Retain research
labels and continuous scores. Any future scientific generalization study requires
independent duplicate adjudication or newly collected physically grouped evaluation
data and a fresh prespecified protocol; neither is started here. Stop after 3F.
