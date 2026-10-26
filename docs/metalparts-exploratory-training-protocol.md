# MetalParts exploratory normal training — protocol v1

Milestone 3D, 25 October 2026. This is a new owner-authorized exploratory
experiment, not a revision of the 3C integrity protocol or its BLOCKED verdict.
Official independent MetalParts evaluation remains NOT CLEARED.

## Inputs and identities

Use only the pinned AmazonScience/RobustAD revision
`82c642b2d18be60f2dab104f8dd4dea1f4ccfacb`, MetalParts source `train` metadata.
Verify its saved 3C content hash and each selected image hash before decoding.
The actual metadata contains 180 label-0 normals and 51 label-1 anomalies.
Only the 180 normals are eligible. Exclude all 51 anomalies from fitting,
validation and diagnostics; do not open their payloads or masks. Never open test
images or masks. A process audit hook permits only the source-train metadata and
published normal-image paths within the pinned dataset root. No download occurs.

Retain 3C sample IDs: SHA-256 of dataset identity, pinned revision and normalized
full relative path (which includes official source partition), prefixed
`metalparts-`. A repeated basename is not a duplicate full path or proof of
leakage. Preserve the 626 similarity candidates and 145 basename correspondences
in existing evidence without reopening or re-adjudicating their images.

## Development split and limitations

Recompute byte and native RGB pixel hashes only for the 180 source-train normals.
Union exact-byte/pixel groups, and any separately confirmed related-image pairs
if authoritative evidence exists. None is currently established. Do not promote
heuristic candidates to physical groups. No new perceptual all-pairs search.
Order groups by SHA-256 of `42|group-representative-ID`, assign the first ceiling
20% to validation and the rest to fitting. A group's representative is its
lexically smallest sample ID. Keep every patch with its parent image.

Preparation found 180 exact-content groups, with no duplicate content or pixels:
144 fitting parents and 36 validation parents. Split/inventory hash:
`bb87d41b2a8296d91f9f67496128e7f7280c2a7ccb688e78f48c175874874c41`.
The full image-hash/assignment inventory is ignored
`reports/metalparts-3d-v1/development.json`. Physical-item identities remain
unknown. Validation is potentially dependent and measures reconstruction on
normal development data, not independently established generalization.

No independent calibration set is justified. There is no calibration operation,
validated binary threshold, anomaly decision or real-defect metric in 3D.
All artifacts explicitly store `threshold: null`.

## Native patch geometry

All permitted parents are 2681 × 1500. Use native 256 × 256 RGB crops with
configurable stride, fixed at 192 for this pilot. Enumerate row-major grid
positions from zero, appending the exact trailing-edge coordinate where needed.
The complete grid has 112 tiles per parent. Select exactly 12 fixed, spatially
spread grid entries using integer `linspace(0, grid_count-1, 12)`; no content,
defect or mask information selects coordinates. Keep these coordinates identical
across epochs and across fitting/validation preprocessing. Each selected tile is
traceable to parent ID and native x/y. This bounded subset is not full coverage
and cannot support a claim about every possible defect location.

There are 1,728 fitting and 432 validation patches. Convert RGB uint8 to CPU
float32 [0,1], channel-first, without resizing, aspect-ratio distortion,
augmentation, learned normalization or source-image modification. Cache at most
two decoded parents. Shuffle fitting parent order and each parent's patch order
with Torch generator seed `42 + epoch`; keep patches together to bound decode
cost. Validation order is sorted parent ID then y/x. Native diagnostic outputs
use all 112 tiles, uniform overlap averaging, a full-coverage assertion and
native image dimensions, not sparse zero-filled maps.

## Model and fixed optimization

Use the existing `conv-ae-v1` compact convolutional autoencoder at supported
resolution 256, unchanged architecture, 37,763 parameters. Fresh random
initialization for MetalParts; do not load Cashew weights. Seed Python, NumPy and
Torch to 42, enable deterministic algorithms, four Torch CPU threads, no loader
workers, batch size eight, Adam learning rate 0.001, RGB mean squared error.
Reuse the existing finite-gradient `train_step`, `train_epoch` and normal scoring
functions. No additional model family is authorized.

First perform one real single-batch update and a one-epoch smoke on the first
four fitting parents and first two validation parents in stable ID order. Record
finite loss/gradients and changed parameter hash. The smoke retains its own
artifacts; its result must pass before the full pilot. The full pilot uses a new
initialization with the same seed, all 144/36 parents, and **four fixed epochs**
(within the owner ceiling of twelve). This is one full pilot, with no automatic
restart or hyperparameter search. Existing run directories are refused.

Select the earliest strict minimum validation MSE across the four completed
epochs. Each parent has the same 12 patches, so mean patch MSE equally weights
parents. Record untrained validation MSE, every epoch's fitting/validation MSE,
optimizer step count, duration and peak resident memory. Do not use test labels,
images, masks or any official test prediction to make a training decision.

## Resources and artifacts

Smoke training/validation budget: 900 seconds. Full-pilot training/validation
budget: 3,600 seconds. Check deadline and measured process peak working set
before each training and validation batch; stop over 8 GiB. Use zero workers and
bounded batches/cache. Native diagnostics are bounded to one normal validation
parent and at most 4,096 tiles by code (112 here). Record overall wall and process
CPU time, including source checks and diagnostics separately from epoch time.
If smoke is nonfinite, has no real update, cannot reload or exceeds its budget,
stop without the full pilot. Preserve partial artifacts on failure.

Create separate ignored `runs/metalparts-ae-smoke-v1` and
`runs/metalparts-ae-pilot-v1`; refuse either if already present. Record protocol,
execution-source, dataset and split hashes, exact patch inventory, configuration,
fresh initialization parameter hash, selected checkpoint file hash, and read
allowlist log. Save only CPU state_dict tensors. Load with `weights_only=True`,
size/link checks, expected identity/configuration matching and strict finite
tensor key/shape/dtype validation. Never accept a Cashew artifact as MetalParts.

Compare loaded validation scores exactly against those saved when the selected
checkpoint was written. Compare outputs from two independent safe reloads. Save
one normal native reconstruction and aligned continuous mean-squared error map,
with native pixel-mean score and no decision threshold. Synthetic geometry tests
verify overlap reconstruction; pseudo-defects, if used, are functionality probes
only. Normal-only losses are not anomaly-detection accuracy measurements.

## Evidence and prohibited conclusions

Seal this document, the prepared split and execution code before the first real
optimizer update. Retain the seal in `reports/metalparts-3d-v1/seal.json` and each
run. No patch parameter or training setting is selected from test information.
The existing Cashew application, checkpoints, thresholds and historical results
remain unchanged. PCB1 remains BLOCKED; PCB2 remains INCONCLUSIVE. MetalParts
official independent evaluation remains NOT CLEARED, irrespective of training
success. No AUROC, anomaly recall, pixel AP, production reliability, blind-test
or independent-generalization claim is permitted. Do not add an operational
MetalParts UI mode, publish, push, automatically commit 3D or start a later milestone.
