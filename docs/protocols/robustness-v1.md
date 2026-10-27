# Synthetic robustness protocol V1

Milestone 5C-1 freezes **protocol version 1**, identifier
`factoryvision-synthetic-robustness`, metadata **schema version 1**. This is a
design and contract milestone. It authorizes no image transformation, inference,
dataset access, OOD detector, model training or UI integration. Execution requires
the separately reviewed 5C-2 scope and its acceptance checks.

## Scientific scope and admission

Only owner-controlled, procedurally generated synthetic parent images are eligible.
The provenance is `owned_synthetic`; classification is `software_diagnostic`.
These declarations are not proof of origin: a future runner must construct parents
from an approved generator, not trust uploaded images or user-supplied labels.
No private evidence directories, official dataset samples, transformed copies of
those samples, arbitrary image paths or remote URLs are admitted by this protocol.
Synthetic images do not become real Cashew or MetalParts samples by choosing an
adapter category. A plan covers **one existing model**, resolved through the
existing Model Catalog at a later execution gate; category controls budgeting only.
There is no new model registry or checkpoint discovery mechanism.

PCB1 remains **BLOCKED**; PCB2 remains **INCONCLUSIVE**. Cashew's official test has
already been observed. MetalParts remains **exploratory source-domain only**;
RobustAD independent target evaluation is **NOT CLEARED**. Human Review contains
observations, not ground truth. None of these gates changes with a synthetic check.
An anomaly score is not a validated OOD confidence measure. This protocol measures
software behavior, not industrial detection accuracy, semantic invariance or OOD
detection performance. Historical metrics, thresholds and nine source-resource
SHA-256 pins remain immutable.

## Independent conditions

Each listed condition starts from the same original, safely decoded, opaque RGB
parent. Never apply a condition to the output of another condition. V1 excludes
flips, large rotations, object-removing crops and all unlisted transformations.
The identity condition is required exactly once, first in the explicit plan.
Do not automatically expand families, severity combinations or seed combinations.

| Kind | Permitted severity | Frozen interpretation |
|---|---|---|
| `identity` | null | Exact copy of decoded RGB bytes; no codec round-trip |
| `brightness` | 0.90, 1.10, 0.80, 1.20, 0.65, 1.35 | Multiply each RGB channel value |
| `contrast` | 0.90, 1.10, 0.80, 1.20, 0.60, 1.40 | Scale about each original channel's spatial mean |
| `gaussian_blur` | 0.5, 1.0, 2.0 | Gaussian standard deviation in native pixels |
| `gaussian_noise` | 2/255, 5/255, 10/255 | IID Gaussian sigma on normalized RGB, zero mean |
| `jpeg` | 95, 80, 60 | Integer quality; one encode/decode, fixed settings below |
| `rotation` | -1.0, +1.0, -3.0, +3.0, -5.0, +5.0 | Degrees; positive rotates visually counterclockwise |
| `translation` | -0.01, +0.01, -0.02, +0.02, -0.04, +0.04 | Signed fraction of width for x or height for y |

For translation, x increases rightwards, y downwards. Offsets remain subpixel;
they are not rounded to integer coordinates. Every translation specifies exactly
one axis. A noise specification specifies exactly one seed from **42, 1729, 2026**;
every other kind has a null seed. Noise replicates must be explicitly listed as
distinct specifications. Selecting all candidates yields 46 specifications:
identity 1, brightness 6, contrast 6, blur 3, noise 9, JPEG 3, rotation 6,
translation 12. This enumeration describes candidates, not a default execution.

Pixel values are interpreted as encoded RGB channel values, not linear-light
radiometry. Arithmetic uses float64. When needed, normalize by exactly 255;
convert back by multiplying by 255, clipping to [0, 255], rounding to nearest
integer with ties to even, then casting to uint8. No automatic contrast adjustment,
ICC conversion, per-image rescaling or metadata propagation. Decode/admission
must retain the existing safe-image checks (bounded dimensions, single frame,
upright orientation and no transparency); no weakening of upload validation.
Identity must preserve dimensions and bytes, including at range boundaries.

Future Gaussian blur uses a normalized separable Gaussian kernel with radius
`ceil(3 * sigma)`, float64 intermediate values and edge-replication padding. This
explicit definition avoids silently substituting a library's approximate blur.
Noise uses a fresh NumPy `Generator(PCG64(seed))` for each specification, sampled
in H-W-C order; a seed is replayed without Python's randomized hash or filenames.
Same-seed noise is intentionally correlated across parents and severities.
JPEG uses RGB, quality as listed, no chroma subsampling, non-progressive mode and
no optimize pass; omit EXIF/ICC/comment metadata. Codec and numeric dependency
versions must be recorded at future execution acceptance. Replay is defined for
that pinned environment, not promised bit-identical across library versions.

Geometry uses the original fixed canvas, pixel centers at integer (x, y), center
((width - 1)/2, (height - 1)/2), inverse sampling and bilinear interpolation.
Use neutral RGB (128, 128, 128) outside the source; never wrap pixels. In image
coordinates a positive rotation maps centered (x, y) to
(cos(theta)*x + sin(theta)*y, -sin(theta)*x + cos(theta)*y).
Translation maps (x, y) to (x + offset_x, y + offset_y).
An output location is valid only if its contributing interpolation support lies
inside the source. A future implementation must test these conventions against
asymmetric synthetic fixtures before reporting aligned metrics.

## Model-family applicability and limitations

| Existing family | Photometric / codec conditions | Geometry |
|---|---|---|
| Cashew autoencoder and spatial-mean baseline | Software diagnostics applicable; brightness/contrast directly affect reconstruction statistics | Conditional; alignment and resizing can alter correspondence |
| Cashew feature-memory and student-teacher | Software diagnostics applicable; noise/blur/JPEG may vanish or change under 128-square preprocessing | Conditional; coarse feature maps and receptive fields limit alignment |
| MetalParts autoencoder and feature-memory | Software diagnostics applicable on native tiled inputs; clipping and texture statistics matter | Conditional; padding and tile boundaries may dominate the observed change |

Applicable means the software comparison is meaningful, not that predictions
should stay constant. No family has established industrial invariance. Geometry
is conditional for all families until correspondence and valid coverage checks
pass; some metrics may remain `not_applicable`. There is no new operating
threshold for MetalParts and no threshold calibration on these synthetic inputs.
The existing Cashew resize/preprocessing and MetalParts 256-pixel tiles with
stride 192 and edge coverage must remain unchanged. Transform before model
preprocessing. Clipping saturates information; blur and codec alter spatial
statistics; interpolation and resize can suppress or introduce artifacts; padding
can affect receptive fields beyond the visibly masked region. A valid-area mask
does not eliminate these effects. Geometry is not guaranteed object-preserving
even at small angles; no claim of semantic label preservation is made.

## Enforced planning budgets

| Budget | Hard maximum and unit |
|---|---|
| Parents | 4 unique synthetic parents per run |
| Calls | 256 single-image or single-native-tile inference units per run |
| Source pixels | 4,194,304 total width * height pixels **summed across parents**, RGB channels excluded |
| Native tiles | 128 per parent for MetalParts; each parent dimension at least 256 |
| Output | 2,097,152 bytes (2 MiB) for the entire scalar UTF-8 report |

Dimensions are positive integer pixels, at most 4096 per axis. Smaller positive
budgets are allowed, larger ones are rejected. Cashew counts one inference unit
per parent per condition, including the spatial-mean comparator. MetalParts counts
every tile: per axis `1 + ceil((dimension - 256) / 192)`, multiplied across axes.
There is no tensor batching that can disguise multiple units as one call.

Identity is evaluated `baseline_repetitions` times (default 3, minimum 1).
Every other explicitly listed specification is evaluated once for each parent.
Thus planned calls equal the sum of parent inference units multiplied by
`baseline_repetitions + number_of_specifications - 1`. A noise seed replicate
is a separate specification and counts in full. For example, four Cashew parents
with all 46 candidates and three baseline repetitions reserve 192 calls.
Native tiling may make the same condition list inadmissible for MetalParts.

Every attempted call consumes its reservation, including failures; no automatic
retries or replacement samples. A parent that partially fails retains its full
planned reservation. Skipped/not-applicable conditions never free reservations
for extra work. Budgets are checked against the complete explicit plan before
inference, with no silent truncation. Future execution must independently verify
actual decoded dimensions and tile counts, process sequentially with bounded
buffers, enforce the output cap over the entire report (not per row), and stop
with a sanitized failure rather than emit partial success on overflow. A single
metadata record is capped more tightly at 65,536 bytes. The contracts do not
implement image decoding, memory allocation control or a runner.

## Future measurements; not implemented in 5C-1

- Compare each score to the same parent's first identity result: signed difference
  and absolute difference. Show identity repeat variability separately; all
  repetitions consume budget. A missing score remains null, never a zero.
- Count paired crossings only against an existing accepted, frozen model threshold
  using its original comparison rule. Otherwise report `not_applicable`.
- Compare raw anomaly maps using MAE, RMSE and maximum absolute difference.
  Never use display-normalized or colorized heatmaps. For non-geometric conditions
  compare the same raw-map grid; for geometry sample the perturbed map at the
  forward-transformed original locations to return to original coordinates.
  Include the existing model's source-to-map/resize coordinate mapping; do not
  assume raw maps always equal source resolution. Require valid interpolation
  support in both maps and at least **80%** valid reference-area coverage.
  Below that fraction, aligned metrics are `not_applicable`, not zero or success.
- Replay identity and noise seeds in the same pinned CPU environment, evaluation
  and inference modes, sequentially, with the existing bounded thread policy.
  Record exact replay agreement or descriptive numerical deviations; do not
  silently introduce tolerance-based pass claims. Cross-platform bitwise
  equivalence is not assumed.
- Summarize per parent and per condition with counts, signed values, medians and
  ranges, plus failure/skip denominators. Keep same-parent conditions and seeded
  replicas visibly grouped; they are not independent evaluation samples. No
  inferential confidence intervals or population generalization from four parents.

`not_evaluated` means unexecuted; `not_applicable` means a comparison is unsupported;
`failed` means a planned attempt did not produce a usable result. None is success.
Never calculate AUROC, average precision, precision/recall or OOD detection accuracy
without separately permitted ground truth and an independently justified evaluation
design. No automatic pass/fail robustness threshold, threshold retuning, classifier
accuracy claim or validated OOD confidence is introduced here.

## Exact metadata boundary

The stdlib-only contracts in `factoryvision.robustness.contracts` are frozen,
slotted dataclasses with exact enums and canonical sorted-key UTF-8 JSON.
`from_json(raw_bytes, Plan)` and `from_json(raw_bytes, Diagnostic)` are the public
untrusted-input boundaries. Require every field, including explicit nulls; reject
unknown/missing fields at every level, duplicate JSON keys, bool-as-int,
nonfinite/incorrect numeric values and unsupported identifiers. Integer fields
require JSON integers. Severity requires a JSON float except JPEG's integer and
identity's null; blur/rotation integers must therefore be written as `1.0`, etc.
Noise sigma is the binary64 value of the listed fraction (for example the result
of `2 / 255`), preserved by canonical serialization. No approximate matching.

`Plan` contains protocol/schema identity, fixed provenance/classification, category,
parents, explicit perturbations, budgets and baseline repetitions. Parents carry
only `case-001` through `case-004` and dimensions; unique IDs are mandatory.
The first perturbation must be identity; duplicate specifications are rejected.
`Diagnostic` contains the same protocol/provenance boundary, parent ID,
perturbation, applicability and status. `not_applicable` applicability and status
must occur together; applicable/conditional permit only not_evaluated or failed
in this metadata-only version. There is deliberately no success state or measured
value field yet. It cannot represent historical evidence, a label, a model
promotion, an image, a raw map, an arbitrary evidence reference or an OOD score.

JSON export revalidates the record, caps bytes, and has no persistence or file I/O.
Errors at the public parsing/export boundary contain a fixed message, not values,
paths, parse context, filenames or exception details. No user notes, timestamps,
machine identifiers, source digests, private filenames, paths or URLs are accepted.
These metadata contracts cannot establish that a real attempt occurred; the future
runner must bind diagnostics to its admitted plan and execution ledger. Frozen
Python objects are an accidental-mutation guard, not a hostile-code sandbox.
The existing Model Registry and ExperimentEvidence formats remain independent.

## Next gate

5C-2 needs owner approval of its implementation scope, synthetic generator and
fixtures, numeric/geometry replay tests, actual runtime/output budget enforcement
and a reviewed measured-result schema. This document specifies future behavior;
it does not claim that those checks or any robustness experiment have run.
Model weights and real data remain outside this phase. Full CI-equivalent
acceptance and packaging belong to the later milestone acceptance/publication gate.
