# Bounded synthetic runner and scalar reports

Milestone 5C-3 integrates the accepted [protocol V2](robustness-v2.md), its unchanged
transformations and assessment equations. [V1](robustness-v1.md) remains the historical
design record. The runner/report implementation is V1; the scientific protocol is
**version 2**, and the separate scalar report schema is **version 1**. These versions
are deliberately distinct. No model, real dataset, checkpoint, UI or camera is used.

## Provenance and execution boundary

`factoryvision.robustness.runner.run(plan, *, adapter=None, threshold=None, cancel=None)`
accepts the existing strictly validated Plan. It accepts no input array, directory,
filename, image bytes, generator callback or caller-declared trusted image provenance.
Each parent is generated internally, once, after complete preflight. The fixed
`controlled-probes-v1` provider selects an original probe by the four existing,
non-identifying case ordinals:

1. Solid RGB (64, 128, 192).
2. Horizontal red and vertical green gradients, rounded evenly over [0,255]; blue
   is the integer-floor average of those channels. A one-pixel axis starts at zero.
3. Eight-pixel checkerboard using the original demo's dark and teal colors.
4. A centered teal rectangle on the same dark background, bounded by integer quarter
   coordinates, with at least one pixel in each dimension.

Dimensions come only from the admitted Parent. Positive side and total-pixel limits
are checked before allocation. The simple demo colors and geometric concept are
reused directly; its encoded-image decoder is not called during generation. No
generation-time filesystem or network access occurs. These patterns are software
probes, not normal/defective/OOD industrial labels.

The public API accepts only the exact `AnalyticalAdapter` class, identified as
`analytical-rgb-mean-v1`. It computes a raw map of mean RGB channel intensity divided
by 255 and a scalar mean of that map. Tests explicitly substitute its method with
controlled doubles to exercise failures. There is no registration system, arbitrary
callable/plugin loader, model lookup or checkpoint deserialization. Trusted Python
code can monkeypatch a process; this API is not a sandbox against such code.

## Ordering, units and failure accounting

Preflight revalidates the whole Plan, protocol 2, exact adapter type and any explicit
FrozenThreshold. It checks all parent/specification budgets and independently counts
the analytical windows. It computes a conservative byte bound for both report formats
before allocating image buffers. Parameters and transformation applicability are
the unchanged V2 allowlists; this adapter provides native-grid analytical maps.

Execution follows the Plan's parent order. For each parent, the identity condition
runs `baseline_repetitions` times, then each remaining explicit specification runs
once in listed order. Every condition receives the original generated parent,
never the preceding transformed pixels. The first identity output is the fixed
comparison anchor; later repetitions never replace it.

One evaluation unit is **one analytical adapter call**:

- Cashew accounting mode: one full synthetic image per condition/repetition.
- MetalParts accounting mode: one 256x256 synthetic window per call, stride 192,
  with the final window touching each edge. This simulates the protocol's native
  window count, not a real MetalParts model. The synthetic aggregate score is the
  mean of window scores; overlapping native maps are averaged by coverage count.
  If any tile score is unavailable, the aggregate score is unavailable. If any
  tile map is unavailable or lacks native correspondence, no stitched map is used.

Real service tile accounting, score aggregation and model behavior are **not
verified or claimed**. The category only selects a synthetic accounting mode.

The hard limits remain four parents, 256 units, 4,194,304 total source pixels,
128 native windows per parent and 2 MiB final scalar report. Baseline repetitions
and every explicitly seeded noise replicate consume their full reserved window
count. A single cumulative counter is charged **before** every attempted adapter
call, including exceptions, invalid outputs and interrupted calls. Both per-condition
and whole-plan counters prevent extra attempts if a window iterator deviates.

There are no retries, replacement parents, implicit Cartesian expansions or
reallocation of skipped reservations. Failed/unused reservations stay visible in
the report. If the first identity fails, remaining rows for that parent are
`not_evaluated`/`baseline_unavailable`; the next parent may proceed. A later failed
condition does not change the anchor or reset cumulative accounting. Generation
or transformation failures consume no adapter call; any calls already attempted
remain counted. Unknown exception details are replaced by fixed reason codes.

Cancellation uses an optional exact `threading.Event`, checked at condition and
adapter-unit boundaries and after calls. Once observed it is latched for the run.
KeyboardInterrupt at these boundaries also produces cancellation. An interrupted
condition with attempted calls is failed/cancelled; otherwise it is not evaluated.
Every remaining planned row is retained as not evaluated/cancelled. Cancellation
is cooperative, not a hard preemption or wall-clock timeout of a NumPy operation.

Only one original parent, its first baseline outputs and one transformed condition
are retained at a time; scalar rows contain no arrays. V2's bounded block operations
remain intact. Planned parent-pixel visits are bounded by the finite condition/repeat
count, separately from adapter units; native calls always use 256-square windows.
No full perturbation grid is allocated. Numeric arrays and blur/map buffers still
require memory proportional to the admitted pixel cap; no operating-system RAM
reservation or hard wall-clock guarantee is claimed.

## Numerical and status integration

The runner uses the existing `score_stability` and `map_stability` functions without
changing equations, tolerances, masks or the 80% valid-coverage gate. Native geometry
receives the transformed image's conservative mask. Unknown/coarse correspondence
is never guessed; unsupported alignment remains not applicable. Display maps are
not generated or used. Adapter output types, finite values, bounded map shapes and
native correspondence are validated before assessment; maps are detached before
becoming a baseline so adapter buffer reuse cannot mutate the anchor.

Missing score or map components remain explicitly unavailable. A row is `measured`
if at least one applicable numerical component was computed; component statuses
still disclose missing/unsupported metrics. If neither component is measurable,
the row is `not_applicable`/`no_applicable_metrics`. Failed and unexecuted rows have
no metric values. The existing Diagnostic contract validates non-measured metadata;
the existing ScoreStability/MapStability types validate numerical components.

An optional FrozenThreshold is explicit, fixed for the run and included in the
report with its original `greater`/`greater_equal` rule. Here it is a synthetic
analytical reference only, never a newly calibrated or automatically retrieved
FactoryVision model threshold. No accuracy, AUROC, AP, precision, recall, OOD
confidence, model ranking or production-readiness score is computed.

## Reporting and size enforcement

`factoryvision.robustness.reporting` provides `json_bytes`, `csv_bytes`, `from_json`
and `from_csv` for validated in-memory objects/bytes. None writes files, performs
uploads or imports images. JSON contains a fixed schema envelope and flat scalar
rows; CSV contains the same allowlisted scalar columns. Repeated metadata makes
each CSV row self-contained and allows exact Plan/report reconstruction. Cases,
dimensions, explicit conditions, baseline repetitions, budgets, counts, threshold
metadata, metrics and fixed reason codes are the only variable fields.

No pixels, raw-map arrays, thumbnails, source paths/names/hashes, private IDs,
environment variables, Human Review notes or historical evidence are admitted.
There are no free-text fields. Readers reject unknown/missing fields, duplicate
JSON keys/CSV headers, bool-as-integer, nonfinite values, mixed schemas, invalid
status combinations, wrong ordering, missing rows and inconsistent repeated metadata
or aggregate counts. Errors do not contain rejected values or exception details.

JSON uses sorted keys, compact canonical formatting, UTF-8 and disallows NaN/Infinity.
CSV uses a fixed column order, all fields quoted and LF record endings. The existing
`spreadsheet_cell` protection is applied to every CSV value, including negative
numbers. Readers restore numeric values only through typed parsing and verify the
exact protected-cell representation; unprotected formula-like cells are rejected.
Both formats round-trip to the same validated scalar report.

Preflight uses a conservative bound: every column is provisioned as a quoted
64-character ASCII cell plus its actual key/delimiter overhead, for every reserved
row. Validated enum/fixed strings and bounded integers/finite floats fit within
that cell envelope; final CSV protection is included. This can reject a small
custom byte budget even when one particular successful report would fit, because
failure outcomes must also fit. The final serialized UTF-8 length is independently
checked against both the requested budget and 2,097,152 bytes. Both encodings must
pass before `run` returns; no truncated or partial-success report is returned.

Report readers validate structure and internal consistency, not cryptographic
authenticity or proof that execution occurred. Imported reports are not accepted
as image provenance, historical evidence or executable input; passing their Plan
to the runner generates new probes internally. No disk import/export CLI is added.

## Scientific boundaries and next gate

PCB1 remains BLOCKED, PCB2 INCONCLUSIVE, Cashew test already observed, MetalParts
exploratory-only and RobustAD target NOT CLEARED. Human Review observations are not
ground truth. Model Registry, the nine SHA-256 resources, all thresholds and existing
single/batch/review/UI code remain unchanged. Reports never enter historical evidence.

5C-3 is local implementation pending owner review. 5C-4 requires separate scope
authorization. No real models, datasets, push, PR, merge or release are authorized.
Focused distribution and installed-package smoke checks do not replace the later
integrated full CI-equivalent and fresh-install acceptance gate.
