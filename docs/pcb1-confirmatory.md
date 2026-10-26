# PCB1 confirmatory preparation — Milestone 2E1

PCB1 is an independent category confirmation of a research direction informed by observed Cashew results. It is not a blinded comparison of architectures chosen without prior dataset experience. The official PCB1 test payloads remain unopened in 2E1. Normal validation false alarms, training losses and timing do not measure anomaly detection accuracy.

## Frozen methodology and data

[Protocol v3](protocols/pcb1-confirmatory-v3.json) supersedes v2's three-family enrollment because the owner explicitly requested all four accepted methods. Earlier protocol files remain unchanged. V3 was hashed before extraction or fitting: `c7b59ae7d88e1ec32009067cae7f018e7ad2f5106f2737d6985563fe81ef38bc`.

The official [VisA registry](https://registry.opendata.aws/visa/) and [Amazon Science repository](https://github.com/amazon-science/spot-diff) identify the source distribution. The retained `VisA_20220922.tar` is 1,929,840,640 bytes, SHA-256 `2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362`. It was originally acquired from the registry's official S3 endpoint; the local digest is a reproducibility identity, not an independently signed upstream checksum. No additional download was needed.

The official one-class CSV at repository revision `2a692ab575001cbde74d402d897a7286086c6199` has SHA-256 `a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995`. It declares 904 permitted PCB1 training normals and 200 held-out records. The held-out records were parsed only to preserve membership; their label distribution was not used for any decision. Whole-archive hashing and header security inspection do not decode or analyze image payloads.

Only 904 training JPEGs were extracted, totaling 243,575,802 bytes, all RGB-decodable at 1404×1070. Their original paths and byte/RGB fingerprints are retained in ignored `data/metadata/pcb1-preparation.json`. No held-out images or masks were extracted. Canonical official manifest SHA-256: `592491db725af85a71aa357da81b28d0536193676bd6a53963348d24e29ad96b`; derived manifest: `82a6d379a3cbb3604756c01c0d7898f6be8d2524c92659da00a53eef02dda34c`.

Seed 42, SHA-256 ordering of exact decoded-RGB groups and 70/15/remainder allocation produce 632 fitting, 135 validation and 137 calibration normals. Official test records are unchanged. Exact duplicates cannot span these development partitions; none were present. Across 190,399 development cross-partition pairs, 41,137 met the deliberately broad dHash/pHash/RGB candidate rule. Full/crop refinement found zero strong candidates. All 20 deterministically required visual pairs were reviewed without confirming repeated capture. No sample was removed or reassigned. This review does not exhaustively adjudicate every candidate or establish independent physical items/acquisition batches.

## Four category-specific candidates

All use full-frame RGB128, bilinear resizing with antialiasing, float32 /255 input, CPU, four Torch threads, zero workers and seed 42. No new architecture, GPU runtime, augmentation or learned normalization was added.

| Family | Frozen fit / selection | Frozen anomaly score |
|---|---|---|
| Autoencoder | Accepted 37,763-parameter convolutional architecture; fresh initialization; Adam 0.001, batch 16, maximum 12 epochs, patience 3, min-delta 1e-5, 1,800-second cap. Every strict validation minimum saved; ties retain earlier epoch. | Mean RGB squared reconstruction error; channel-mean squared residual map 128. |
| Spatial mean | Sequential, sample-ID ordered float64 average over all 632 fitting images, then float32. | Same reconstruction score/map semantics as AE. |
| Feature memory | Accepted frozen ResNet18 layer2/3 averaged features, 384 dimensions at16×16. Every fitting image contributes `min(16,floor(8192/N_train))=12` deterministically selected patches; 7,584 total. SHA256(42\|ID)-seeded randperm; exact RMS nearest neighbor, chunks 256/512; 600-second cap. | Maximum native patch distance; bilinear map 128. |
| Student–teacher | Accepted 112,128-parameter compact student freshly initialized; frozen 2,782,784-parameter teacher; Adam 0.001, batch 8, 12 epochs maximum, patience 3, strict normal validation minimum, 600-second cap. | Unit-L2 layer2/3 feature discrepancy, equally averaged at16×16; maximum image score and bilinear map 128. |

The official pretrained ResNet18 V1 weight file is reused, SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`. It is never downloaded implicitly. ImageNet mean/std normalization applies only to feature and student models. No Cashew-fitted parameters or bank entries are substituted for PCB1 training. Equal fresh seed 42 initial student hashes across categories are expected; learned checkpoints differ.

Trainable families first passed the predeclared two-update normal-only smoke checks, then full runs restarted from fresh initialization. No PCB1 synthetic anomaly probes were used. Every epoch records training and validation IDs. Training processes enforce the two-GiB peak-memory limit and finite numerical checks.

## Calibration and interpretation

Each model independently uses only the 137 calibration normals: float64 linear 95th percentile, anomaly iff score strictly exceeds threshold. Validation chooses trainable checkpoints but does not set thresholds. No pixel threshold is fitted. Frozen figures use each model's maximum calibration map value as display scale.

| Family | Threshold | Validation flags /135 | Calibration flags /137 | Preloaded batch1 median ms |
|---|---:|---:|---:|---:|
| Autoencoder | 0.012788945995271203 | 8 | 7 | 1.4185 |
| Spatial mean | 0.019585608690977096 | 8 | 7 | 0.4818 |
| Feature memory | 0.07782351225614548 | 9 | 7 | 12.7806 |
| Student–teacher | 0.8243726611137387 | 2 | 7 | 5.3412 |

The 95th percentile is an empirical calibration rule, not a guaranteed future 5% false-positive rate or confidence bound. Validation is also used for checkpoint selection, so its reported false-alarm counts are descriptive. Timings are warm-process measurements (five warmups, 30 repeats), on this workstation with possible concurrent verification activity. Separate warm decode medians are approximately 7.2–7.7ms. Scores are on incompatible scales across families; threshold magnitudes are not rankings.

The future primary endpoint is student pooled pixel average precision over every resized pixel of every official test image. All four methods are reported, with student-minus-each-comparator pixel-AP differences and descriptive descending pixel-AP ranks (ties shared). Secondary outcomes include image AUROC/AP, pooled pixel AUROC, and image confusion counts/precision/recall/F1/normal FPR at frozen thresholds. Positive means anomalous. Mask nonzero classes are binarized before nearest-neighbor128 resizing. Disappearing resized defects are reported and retained; normal masks are zeros.

Rank metrics are null when either class is absent; zero-denominator rates are null. Exact pooled evaluation accepts up to 8,000,000 pixels and 488 images, not the previous 3M-pixel cap. Image-level paired stratified bootstrap uses 1,000 resamples, seed 42, 95% percentile intervals for student-minus-comparator AUROC/AP. These intervals condition on unverified image independence. Pixel confidence intervals, AUPRO and thresholded pixel metrics are deferred. No result-dependent exclusions, threshold changes or automatic model optimization are allowed.

## Seal, operation and deferred opening

Evaluation seal (local evidence, not distributed: `docs/evidence/pcb1-evaluation-seal.json`) and its SHA-256 companion (local evidence, not distributed: `docs/evidence/pcb1-evaluation-seal.sha256`) bind the exact protocol, executable sources, lock, dataset/partition identities, archive, normal files, teacher weights, four category-specific artifacts, calibration records and thresholds. It contains only portable paths. Run directories, images and weights remain ignored. The seal is hash-bound evidence, not a digital signature or OS access-control boundary.

Safe 2E1 verification commands:

```powershell
.\.venv\Scripts\python.exe scripts/check_quality.py
.\.venv\Scripts\python.exe -m pytest integration_tests/test_pcb1_development_real.py --basetemp=.tmp/pytest-pcb1-real -q
.\.venv\Scripts\python.exe scripts/verify_pcb1_seal.py
```

The fitting/calibration commands in VS Code are explicit one-shot operations; existing run/started-marker files are refused. Workspace opening and ordinary pytest discovery neither fit models nor open the official test. DevelopmentReadGuard permits only the normal allowlist inside the development root. This Python audit hook is defense in depth, not protection against arbitrary external/native processes.

**The following procedure is prepared for 2E2 and was NOT executed.**

1. The owner must separately authorize opening the official PCB1 test against the exact final seal SHA. Do not infer this from 2E1 authorization. Only then create ignored `reports/pcb1-owner-approval.json` with `decision: "OWNER_AUTHORIZES_PCB1_HELDOUT_OPENING"`, `milestone: "2E2"`, and that exact `seal_sha256`. The operator must retain the actual owner instruction; the JSON is a bound execution record, not proof of human identity.
2. From the repository root, execute the explicit one-time opening command:

```powershell
$pcb1SealSha = (Get-Content docs/evidence/pcb1-evaluation-seal.sha256).Split()[0]
.\.venv\Scripts\python.exe scripts/pcb1_blinded.py open --seal-sha $pcb1SealSha --approval reports/pcb1-owner-approval.json
```

3. Opening first verifies the frozen inventory and authorization, then exclusively creates `reports/pcb1-opening/started.json`. It extracts only official test image/mask paths into new `data/pcb1-test-quarantine`. No predictions occur. Integrity validation checks decodes, mask alignment, exact duplicates and the frozen cross-partition similarity review. Any corrupt payload, duplicate compromise or unresolved required review blocks evaluation. Every required visual pair must be reviewed; no threshold/model tuning is allowed during this inspection.
4. Retain a review JSON containing the exact `integrity_sha256`, `review_complete: true`, `confirmed_leakage: false`, and the exact complete `required_pairs` list copied to `reviewed_pairs`, plus substantive review notes. Do not create a passing record without completing the review. Exact duplicates block execution even if the review claims otherwise.
5. Only after the gate passes, execute:

```powershell
.\.venv\Scripts\python.exe scripts/pcb1_blinded.py evaluate --seal-sha $pcb1SealSha --approval reports/pcb1-owner-approval.json --review reports/pcb1-opening/review.json
```

6. Evaluation verifies identities again and creates an exclusive new output directory. It loads frozen candidates only, uses stable sample ordering and batch 8, preserves per-image scores, masks/maps in NPZ, all metrics, CPU/wall/latency evidence and content hashes. Figures use the first four anomalous and first two normal IDs and frozen normal-derived scales. A partial or completed evaluation cannot be overwritten. Stop on failures and retain evidence; do not delete markers to conceal a retry. Later model/protocol changes require a separately declared exploratory experiment, not insertion into this confirmation.

## Attribution and limits

VisA: Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang and Onkar Dabeer, *SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation*, ECCV 2022. Data are [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); preserve attribution, official source, license/NOTICE and disclose derivatives (internal partitioning, resizing, binary masks and visualizations). Images and pretrained weights are not redistributed in Git or packages. Public licensing of project code remains an owner decision.

No PCB1 real-anomaly sensitivity, localization accuracy, superiority, factory transfer or deployment readiness is claimed. The official test will contain only one category, its physical independence is not established, resizing can erase small defects, and four comparisons invite multiplicity/selection concerns. Cashew is already observed; its historical results must not be re-used as a fresh test. No additional Cashew evaluation took place in 2E1.
