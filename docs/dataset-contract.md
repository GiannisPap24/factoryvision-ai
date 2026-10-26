# Dataset contract v1

## Source and provenance

Target: [official VisA registry](https://registry.opendata.aws/visa/). Dataset release identifier: VisA_20220922. The one-class metadata revision is 2a692ab575001cbde74d402d897a7286086c6199 in amazon-science/spot-diff. Milestone 1B acquired the official archive and selectively extracted/verified Cashew. Full acquisition details and measured counts are in [dataset preparation](dataset-preparation.md).

Supported categories are candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4 and pipe_fryum. Start future ingestion with cashew, then pcb1. No manifest is required to contain the full dataset or an invented count.

Source fields are fixed to the reviewed release/revision/URI/license relationship. Supporting a new official release requires reviewing its provenance and updating the contract. This is intentionally narrower than accepting any URL labeled VisA.

## Immutable types

| Type | Fields |
|---|---|
| Source | kind, uri, revision, license_id, attribution |
| DatasetIdentity | dataset_id, version, source |
| Sample | sample_id, dataset_id, dataset_version, category, image_path, label, mask_path, official_split, partition, group_id, content_sha256 |
| SplitProvenance | algorithm, seed, parent_manifest_sha256, assignments_sha256 |
| Manifest | schema_version, dataset, samples, split_provenance |

Accepted SourceKind values are visa and synthetic. Labels are normal and anomaly. OfficialSplit is train or test. Internal Partition is train, validation, calibration or test. Python constructors require enum instances; the strict JSON boundary converts exact enum spellings.

Samples carry references, not decoded image arrays. No dimensions, acquisition timestamps or physical-item IDs are invented.

## Official facts versus internal conventions

| Field | Origin |
|---|---|
| category, image_path, label, mask_path, official_split | Correspond to official CSV object, image, label, mask and split columns |
| dataset version and source revision | Reviewed archive/repository identity |
| sample_id | Internal globally unique stable identifier, assigned during future ingestion; never derive it from partition |
| partition | Internal development assignment; official test always remains test |
| group_id | Internal grouping; the Cashew audit groups equal decoded RGB hashes. Physical-item independence remains unknown |
| content_sha256 | Computed by the real-data audit; null in metadata-only fixtures means not measured |
| schema_version, split algorithm and fingerprints | FactoryVision application conventions |

All keys must be present. Explicit null is allowed only for mask_path, group_id, content_sha256 and split_provenance, subject to their relationship rules. An empty string is never a substitute for null. Empty samples are invalid. Unknown keys and repeated JSON keys are errors. Invalid UTF-8, non-finite JSON numbers and files above the configured 2 MiB bound are rejected.

## Rules and error behavior

Stable diagnostics contain code, field and message. Pure manifest validation returns a sorted tuple of issues; ContractError retains those issues. CLI JSON output is deterministic. Successful validation means the declared metadata is internally consistent, not that it is authentic or independently sampled.

- IDs use 1–128 lowercase ASCII letters/digits, dots, underscores or hyphens; first character is alphanumeric.
- A sample's dataset ID and version must match the manifest source.
- Normal images require a null mask reference. Anomaly images require a PNG mask reference.
- Official one-class training records must be normal. All internal fitting, validation and calibration records must be normal.
- Official test membership cannot be promoted into development or vice versa.
- Sample IDs and image/mask reference paths must be unique. Path uniqueness is case-insensitive for Windows.
- A group cannot cross partitions or categories; group IDs are dataset-wide.
- Equal supplied content hashes cannot cross partitions. The Cashew audit computes file and decoded-pixel hashes; pure metadata validation alone still trusts supplied values.
- Derived partitions need split provenance and a matching assignment hash. Provenance records an asserted parent fingerprint; verifying that parent requires retaining and checking the original manifest.
- VisA paths must match category/Data/Images/Normal or Anomaly/*.JPG and category/Data/Masks/Anomaly/*.png. No filename stem or numeric range is invented.

Representative codes: FIELD_MISSING, FIELD_UNKNOWN, ID_INVALID, CATEGORY_UNSUPPORTED, ENUM_INVALID, MASK_LABEL_MISMATCH, SAMPLE_DATASET_MISMATCH, OFFICIAL_TEST_CHANGED, DUPLICATE_ID, IDENTITY_LEAKAGE, REFERENCE_LEAKAGE, GROUP_LEAKAGE, CONTENT_LEAKAGE and SPLIT_HASH_MISMATCH. Tests verify both accepted records and precise failure codes.

## Path containment

Use forward-slash relative references. Reject POSIX absolute paths, Windows absolute/drive-relative/UNC paths, backslashes, dot/parent traversal, empty components, reserved device names, alternate data streams, controls, wildcard characters and trailing spaces/dots.

Optional filesystem checking resolves the chosen root and every file reference. Missing files, non-directory roots and symlink/junction escapes fail before image contents can be used. Permission and loop errors become stable diagnostics. This validation is not race-proof; future ingestion must control the dataset directory during reads. Hardlinks, alternative file aliases and decoded/near-duplicate image content require the future duplicate audit.

## Synthetic fixtures and licensing

tests/fixtures/synthetic-manifest.json contains ten fabricated normal training records, one fabricated test normal and one fabricated test anomaly. Paths identify nonexistent synthetic images unless a test creates placeholder bytes in its temporary directory. Those placeholders are deliberately not image files.

Synthetic manifests use factoryvision-synthetic / synthetic-v1 and a local URN. LicenseRef-FactoryVision-Synthetic is an internal provenance marker, not a standardized license grant. The owner has not selected a public source-code/fixture license; do not infer redistribution permission from the marker.

VisA data is CC BY 4.0. Preserve attribution, the source/license links, applicable notices and modifications; do not imply endorsement. VisA code has a separate license. Cite the VisA paper by Zou, Jeong, Pemula, Zhang and Dabeer, ECCV 2022. Record actual image acquisition date in the later ingestion record. No real dataset is distributed in this project. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Some tests fabricate VisA-shaped metadata and tiny image pixels solely to exercise the pinned official-source branch and ingestion. Their filenames explicitly state that they are fabricated; they are not claims about real samples. Real dataset checks live separately in integration_tests/ and run only when explicitly selected.
