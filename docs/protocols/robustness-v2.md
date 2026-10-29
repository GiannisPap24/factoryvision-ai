# Synthetic robustness protocol V2: numerical implementation

Milestone 5C-2 implements pure image transformations and synthetic score/raw-map
diagnostics. Protocol identifier remains `factoryvision-synthetic-robustness`;
**protocol version is 2**. The existing metadata shape is unchanged, so metadata
schema version remains 1. There is no inference runner, model loading, training,
dataset access, persistent experiment tracking or public measured-result export.

## Explicit amendments to frozen V1

The owner approved the following two changes during 5C-2. The
[V1 protocol](robustness-v1.md) and its six-file local 5C-1 checkpoint remain the
historical design record. V2 incorporates all V1 scientific gates, parameter
allowlists, budgets, independent-transform rules and exclusions, with these
explicit replacements:

| Operation | V1 design | V2 implemented definition |
|---|---|---|
| Contrast | Scale about each channel's spatial mean | `255 * ((RGB / 255 - 0.5) * multiplier + 0.5)`; fixed normalized midpoint 0.5 |
| Gaussian noise stream | Fresh `PCG64(seed)` for each specification | `PCG64(SeedSequence([seed, case_ordinal, replicate_ordinal]))` |

Case ordinals are 1, 2, 3, 4 for the four existing `ParentID` values, in order.
Approved seeds 42, 1729, 2026 correspond to replicate ordinals 0, 1, 2 respectively.
No new replicate or severity dimension is introduced. Sample standard normals in
H-W-C order, float64, then use `255 * (RGB / 255 + sigma * noise)`. A fresh generator
is created per call. Same case/seed across sigma levels intentionally shares the
underlying standard-normal draws; replicates and same-parent conditions are not
independent evaluation samples. Global NumPy RNG state and Python `hash()` are unused.

Current `Plan` and `Diagnostic` default to protocol 2 and reject protocol 1 rather
than silently reinterpret it. No historical evidence schema, threshold or resource
pin changes. No V1 experiment was executed or migrated. The metadata-only Diagnostic
statuses remain unchanged; measured numerical results use separate in-memory types.

## Transform API and admission

`perturb(rgb, specification, *, plan, parent)` accepts an existing validated Plan,
Parent and Perturbation. It revalidates all three, requires exact parent/specification
membership in the plan, and checks decoded dimensions against the parent record.
Only exact NumPy arrays of shape H-W-3 and dtype uint8 are accepted. No dtype
coercion, image file decoding, path, URL, sample filename or metadata is accepted.
Positive dimensions are bounded by 4096 per side and 4,194,304 pixels per array;
the Plan also enforces the aggregate parent-pixel budget and other V1 budgets.

Arrays alone cannot prove synthetic provenance or detect that a caller copied a
previous transformed result. The caller must supply an approved original synthetic
parent independently for each condition; this API is not a real-image admission
path. The future runner must enforce origin and the independent-input ledger.
Tests reuse the existing original synthetic demo geometry and small analytical
arrays; no second synthetic-demo framework is introduced.

All eight families are implemented with exactly the V1 allowlisted parameters:

- Identity copies RGB bytes exactly, without codec conversion.
- Brightness multiplies float64 channel values by the specified multiplier.
- Contrast uses the amended fixed-midpoint equation above.
- Gaussian blur uses the V1 normalized separable kernel, radius `ceil(3*sigma)`,
  standard deviation in native pixels, edge-replication padding and float64
  intermediate values. It is not an approximate Pillow blur substitute.
- Gaussian noise uses the amended PCG64 stream definition above.
- JPEG performs one in-memory RGB round trip with integer quality, subsampling 0,
  progressive false, optimize false, empty EXIF and no ICC profile. A fresh
  pixel-only Pillow image prevents source metadata propagation.
- Rotations use the V1 fixed canvas, center, sign convention and bilinear sampling.
- Translations use the V1 signed axis fraction and unrounded subpixel displacement.

All arithmetic results are clipped to [0, 255], rounded to nearest with ties to even
and cast to uint8. Every result preserves dimensions and contains detached,
read-only pixels plus a boolean validity mask and derived coverage. Identity and
non-geometric conditions have an all-valid geometric mask. That mask does not claim
semantic invariance or absence of clipping/blur/codec effects. Frozen records and
NumPy read-only flags prevent ordinary accidental mutation, not hostile Python code.

These encoded-pixel perturbations are **not calibrated physical camera exposure
or industrial image-acquisition simulations**. Numeric and JPEG replay is scoped
to the installed, pinned environment, not promised bit-identical across dependency
versions or platforms. No intensity invariance or score monotonicity is required.

## Geometry and raw-map alignment

`Geometry(width, height, perturbation).points(points, inverse=False)` exposes
bounded N-by-2 coordinate mapping for identity, approved rotations or translations.
Coordinates are float32/float64, finite and bounded in magnitude. Unsupported
transforms are rejected. Pixel centers, positive-angle rotation, forward/inverse
equations, neutral RGB (128, 128, 128) fill and canvas dimensions follow V1.

RGB sampling is bilinear. A sample is valid only when every positive-weight
contributor lies inside the source. Zero-weight neighbors do not invalidate an
exact edge pixel. Masks are never bilinearly blended and rounded or sampled with
a nearest-neighbor shortcut: raw-map validity uses the logical conjunction of
all positive-weight contributors. There is no wrapping or border extrapolation
for map metrics. Invalid regions remain excluded even when rendered pixels exist.

`map_stability` requires explicit source image dimensions, perturbation and grid
conventions for both raw maps:

- `NATIVE`: raw-map centers coincide with source image centers; dimensions must match.
- `HALF_PIXEL`: the caller explicitly establishes an axis-aligned full-image resize
  mapping, `image_x = (map_x + 0.5) * image_width / map_width - 0.5`, likewise for y.
- `UNSUPPORTED`: no established correspondence; return `not_applicable`.

There is no automatic selection by model name, map size or model family. The
six existing models do not necessarily share this correspondence or score logic.
Learned receptive-field centers, arbitrary crops, projective/shear warps, unknown
registration and unsupported resize conventions need separately reviewed adapters.
No such adapters or real model-map evaluation are implemented here.

For geometry, a candidate validity mask on the candidate **raw-map grid** is
mandatory. The RGB mask is usable directly only when that grid is native. A future
adapter must establish conservative mask projection for other grids; silently
resizing the RGB mask is not permitted. The method maps reference grid centers to
source coordinates, applies the forward image transform, then samples the candidate
raw-map grid using the declared inverse grid mapping. It intersects contributor
validity, supplied reference validity and image bounds. Compare only after this
alignment. Unsupported correspondence returns `not_applicable` with no coverage
claim or numerical metric.

For non-geometric conditions on the same declared grid, compare matching map
elements directly with the mask intersection. Do not resample an identity pair:
a coordinate round trip can introduce artificial errors on fractional resize grids.

Coverage is valid reference-grid samples divided by all reference-grid samples,
including invalid/masked samples in the denominator. For these uniform grids it is
the discrete valid-area fraction; coarse grids limit its spatial precision. Require
at least 80%, tested by integer counts, before map metrics. An all-false mask yields
0% and `not_applicable`; a zero-size or wrong-shape mask is malformed and rejected.
Validity masks do not remove receptive-field or interpolation effects.

## Numerical diagnostics and boundaries

`score_stability(reference, candidate, *, threshold=None)` accepts finite Python
floats or explicit None. It reports candidate minus reference, its absolute value,
and signed relative difference divided by `abs(reference)` only when the denominator
is nonzero and the ratio is finite. Otherwise relative difference is None.
Missing scores produce `not_applicable`/`missing_score`, never a numerical zero.

A `FrozenThreshold` must explicitly supply both a finite value and the original
`greater` or `greater_equal` comparison rule. Crossing is up, down or none; with
no threshold, crossing is None. These pure functions cannot authenticate a threshold
against a model: that responsibility belongs to a future reviewed catalog adapter.
There is no threshold fitting, calibration or inference in this phase.

`map_stability` accepts only finite, nonempty 2-D float32/float64 arrays, bounded by
the same side/pixel limits. It rejects colorized/uint8 display heatmaps and malformed
arrays. A 2-D display-normalized float array cannot be recognized by its bytes;
callers must supply raw, unnormalized synthetic values. Metrics are raw-map MAE,
RMSE and maximum absolute difference over the aligned valid region. Scaled reductions
avoid unnecessary overflow when squaring large finite differences. A nonrepresentable
score/map difference is rejected, never emitted as infinity. Constant, zero and
identical maps have defined zero-difference behavior.

`ScoreStability` and `MapStability` are frozen, validated, diagnostic-only in-memory
types with protocol 2 and `software_diagnostic` classification. `measured` means
an arithmetic comparison was computed; it does not mean robustness passed. Missing
or insufficient comparisons have explicit reasons and null measurements. These
types cannot be exported through the existing metadata JSON exporter and never
enter Model Registry or ExperimentEvidence. No persistent store or public export
is implemented. No AUROC, AP, precision, recall, OOD accuracy, model ranking or
production-readiness score is calculated.

## Bounded computation, verification and next gate

Only one condition is computed per API call. Pixel arithmetic, geometry sampling
and map alignment use bounded 64-row blocks; blur retains one full float64
intermediate. No perturbation grid is materialized. Full output arrays and bounded
map-reduction buffers remain necessary, so memory still scales with the enforced
pixel cap. Tests use small synthetic arrays. Whole-run sequencing, inference-call
reservations, runtime limits and the 2 MiB report cap remain future runner obligations;
no report is produced here.

Analytical tests cover pixel equations, rounding/clipping, independent 2-D blur
oracles, seed/case/replicate replay, JPEG parameters and metadata, forward/inverse
geometry, conservative masks, native/coarse raw-grid alignment, exact scalar/map
metrics, 79/80/81% coverage boundaries and memory-only operation. This evidence
establishes numerical software behavior, not real industrial detection performance.

PCB1 remains BLOCKED, PCB2 INCONCLUSIVE, Cashew test already observed, MetalParts
exploratory source-domain only, and RobustAD target NOT CLEARED. Human Review is
not ground truth. All nine scientific source pins remain unchanged. The UI,
training/inference services, model cards and existing evidence are untouched.

5C-3 requires separate owner review of its scope, synthetic admission/runner design,
model-specific correspondence and any result/report schema. Full CI-equivalent,
packaging/privacy and fresh-install checks remain an integrated acceptance gate.
The 5C-2 implementation is local and uncommitted; no publication is authorized.
