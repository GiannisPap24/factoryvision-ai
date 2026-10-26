# VisA Cashew preparation and loading â€” Milestone 1B

## Provenance and acquisition

Source: [AWS VisA registry](https://registry.opendata.aws/visa/), with the [official research repository](https://github.com/amazon-science/spot-diff). Downloaded on 24 October 2026, 12:36 UTC (15:36 Europe/Athens), from https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar.

The archive is 1,929,840,640 bytes (1.797 GiB), under the owner's 2.5 GiB limit. Local SHA-256: `2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362`. No authoritative published SHA-256 was found in the reviewed official sources; this locally measured digest does not independently authenticate the archive. Its multipart S3 ETag is not an MD5 checksum.

Metadata is pinned to repository revision `2a692ab575001cbde74d402d897a7286086c6199`. The acquired archive's one-class CSV and dataset license exactly match the separately fetched pinned repository versions. The full acquisition record, including source URLs, byte lengths, hashes and timestamps, is ignored data/raw/acquisition.json.

The archive is retained in data/raw. Extraction inspected all 12,122 members, then copied only 703 regular files totaling 184,175,357 bytes: 600 Cashew images, 100 anomaly masks, category annotations, the dataset license and the one-class split CSV. No other category was extracted. The official archive necessarily contains all categories; only the selected files are available unpacked.

## Reproduction commands

Run from the authoritative repository after the CPU smoke test succeeds. Acquisition is an explicit network operation, not a VS Code folder-open task:

~~~powershell
.\.venv\Scripts\python.exe scripts/smoke_torch.py
.\.venv\Scripts\python.exe scripts/acquire_visa.py --download
.\.venv\Scripts\python.exe scripts/extract_cashew.py
.\.venv\Scripts\python.exe scripts/verify_cashew.py
.\.venv\Scripts\python.exe scripts/debug_dataloader.py
.\.venv\Scripts\python.exe -m pytest integration_tests/test_cashew_real.py --basetemp=.tmp/pytest-real -q
~~~

Acquisition refuses an existing archive and extraction requires a new destination. **Do not rerun those two commands against the already prepared data.** Re-run verify_cashew.py and the explicitly selected real-data tests to verify the existing copy. A partial acquisition/extraction is retained for diagnosis; inspect it before any deliberate cleanup/retry. No automatic destructive cleanup or resume logic is implemented.

The downloader restricts the reviewed URL/size, limits transfer to 2.5 GiB, checks free space and requires the archive path to be Git-ignored. Metadata transfers are limited to 2 MiB each. It refuses redirects to a different archive URL. Initial free space was approximately 490 GiB.

Archive inspection validates every member before writes: portable relative paths only, no traversal, Windows drive/UNC/device names, symlinks, hardlinks, device/FIFO/sparse entries, case-aliased duplicates or file-as-directory conflicts. Only the observed VisA category/CSV/license structure is allowed. Limits are 50,000 members, 32 MiB per file, 3 GiB total expanded bytes and 512 MiB selected bytes. Extraction uses exclusive file creation under a new root, rejects a linked parent, rechecks containment and copies bytes without applying archive permissions. Keep the directory under trusted local control; these checks do not provide race-proof protection against concurrent filesystem mutation.

## Actual inventory and annotation interpretation

| Official partition | Normal | Anomaly | Stored anomaly masks |
|---|---:|---:|---:|
| Train | 450 | 0 | 0 |
| Test | 50 | 100 | 100 |
| Total | 500 | 100 | 100 |

All 600 images decoded as RGB JPEG at width 1274, height 1176. All 100 anomaly masks decoded as single-channel PNG at matching geometry, with nonempty positive regions. Observed class IDs were 0 through 8; the official Cashew taxonomy additionally defines 9 (Other), which was not observed.

The category image_anno.csv contains detailed defect names, sometimes comma-separated, whereas split_csv/1cls.csv contains binary normal/anomaly labels. Direct string equality initially reported 100 differences. Inspection of the [official taxonomy](https://raw.githubusercontent.com/amazon-science/spot-diff/2a692ab575001cbde74d402d897a7286086c6199/utils/id2class.py) confirmed the representation difference. Ingestion now explicitly maps recognized Cashew defect names to anomaly, rejects unknown names, and checks exact image/mask correspondence. This is a documented adapter, not a modification of the source annotations.

Following the [official preparation semantics](https://raw.githubusercontent.com/amazon-science/spot-diff/2a692ab575001cbde74d402d897a7286086c6199/utils/prepare_data.py), nonzero mask class IDs become binary anomaly pixels at load time. Original masks remain unchanged. Declared normal images have no stored masks; their all-zero tensor targets are derived from the normal label and returned with mask_available=false. Missing anomaly masks are errors and are never fabricated.

No missing/unreferenced image or mask files, corrupt decodes, duplicate sample identities, annotation contradictions or geometry mismatches were found after this explicit mapping. Full per-image file/pixel/mask hashes and dimensions are in ignored data/metadata/cashew-audit.json.

## Splits and leakage limits

The audit computes file SHA-256 and a dimension-tagged RGB-pixel SHA-256 for all images. Equal decoded-pixel hashes share a group; cross-partition byte/group matches fail validation. No exact-byte or decoded-pixel duplicate groups were found. These hash groups do not establish physical-item or acquisition independence.

The existing sha256-group-v1 algorithm, seed 42, allocates the 450 permitted normal training groups into 315 fit, 67 validation and 68 calibration samples. The official test remains exactly 50 normals plus 100 anomalies. The parent and derived manifests are data/manifests/cashew-official.json and cashew-seed42.json. Fingerprints and assignment provenance are included in the audit and manifests. Sorting/repeated execution produces identical assignments.

Near-duplicate/perceptual comparisons and acquisition/physical-item provenance were not audited. Before interpreting experiment performance, review this residual leakage risk. No unseen-factory/device generalization is claimed. The test set cannot select hyperparameters, checkpoints, transforms or thresholds. Future preprocessing fitting uses training records only; normal-only calibration estimates a normal false-alarm tail, not anomaly recall or an optimal F1 threshold.

## Dataset/DataLoader interface

~~~python
from pathlib import Path
from factoryvision.data.images import ImageSettings, ManifestDataset, make_loader
from factoryvision.data.manifest import load_manifest
from factoryvision.domain.contracts import Partition

manifest = load_manifest(Path("data/manifests/cashew-seed42.json"))
dataset = ManifestDataset(manifest, Path("data/visa"), Partition.TRAIN, ImageSettings(256, 256))
batch = next(iter(make_loader(dataset, batch_size=4, seed=42)))
~~~

| Key | Batched value |
|---|---|
| image | CPU float32 [B,3,H,W], RGB values in [0,1] |
| mask | CPU float32 [B,1,H,W], binary 0/1 |
| label | int64 [B], normal=0, anomaly=1 |
| sample_id | Stable ordered string IDs |
| partition | Explicit partition strings |
| mask_available | bool [B], whether a stored anomaly mask exists |
| original_size | int64 [B,2], original height/width |

Images use bilinear antialiased resizing; masks use nearest-neighbor resizing. Direct H-by-W resizing can change aspect ratio; the recipe is explicit and must be fixed before experiments. No random augmentation, automatic EXIF rotation, dataset caching, model transforms or learned normalization occurs. Nontrivial EXIF orientation is rejected for review.

Dimensions must be integers in [1,2048], batch size in [1,64], encoded files at most 32 MiB and decoded images at most 20 million pixels. A theoretical image/mask batch above 512 MiB is rejected. Loading uses zero workers, no shuffle/pinning/prefetch and a dedicated seeded CPU generator. This avoids loading the dataset into memory; it is not a performance-optimized multiprocessing pipeline.

A four-image 256x256 batch used 4,194,404 bytes of returned tensors. One standalone run measured 0.0425 seconds for the first batch and 302,125,056 bytes peak process working set; these are smoke measurements, not training/inference benchmarks. Debugger overhead and warm/cold cache state change these values.

## License and attribution

Visual Anomaly (VisA), by Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang and Onkar Dabeer, was accessed on 24 October 2026 from the AWS registry. Cite SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation, ECCV 2022 / arXiv:2207.14315. Data uses [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); preserve applicable notices, attribution, source/license links and descriptions of changes. The research code's license is separate. Original license/NOTICE files are retained locally. This repository distributes neither real images nor source annotation CSVs. Public project licensing remains undecided.

## Subsequent Milestone 2A gate

The original preparation record above describes Milestone 1B. Milestone 2A rechecked all image/mask records against it, then added a documented near-duplicate screen without changing membership. It flagged 7,946 of 114,581 cross-partition pairs and visually reviewed the 24 closest thumbnail pairs. No confirmed duplicate was established in that subset; remaining candidates and physical-item identity are still unresolved. See [training.md](training.md) for screening thresholds, preserved reports and the limited basis for proceeding with training feasibility. The accepted split and source data are unchanged.

## Milestone 2B integrity gate

Before official model scoring, all 600 images and 100 masks were re-audited against accepted identities, exact hashes, original geometry and unchanged official membership. All 7,946 prior perceptual candidates received a predeclared full/crop refinement; zero strong candidates and 20 deterministic visual-review pairs yielded no confirmed repeated acquisition. This is not proof of physical-item independence. No exclusions or reassignments occurred. Full local records and three review sheets remain under ignored reports/leakage-2b-*. See [evaluation-protocol.md](evaluation-protocol.md) and [the completion report](milestone-2b-completion.md) for procedure and limitations.
