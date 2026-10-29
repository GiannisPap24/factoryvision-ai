# Development admission and service compatibility foundation

Milestone 5C-4B1 implements metadata and an internal synthetic interface exercise.
It does not admit a real sample, verify an artifact, construct an inference service,
load a checkpoint, execute a neural model or run the proposed Cashew pilot.
The accepted 5C-4A review identified Cashew AE as the first candidate because its
single-frame score/map contract is simpler and it needs no pretrained backbone.
That is a planning preference, not sample eligibility or permission to execute.

## Separate authority and states

`development_admission.DevelopmentAdmission` is a frozen, strict proposal record.
Its protocol identifier is `factoryvision-development-admission`, protocol version
1, schema version 1. `numerical_protocol_version: 2` references the unchanged
[V2 equations](robustness-v2.md); it does not extend their synthetic admission scope.
The [V1 historical protocol](robustness-v1.md), synthetic Plan, AnalyticalAdapter,
[5C-3 runner/report schema](robustness-runner-v1.md) and all services are unchanged.

Only this proposed combination is representable: Cashew `autoencoder`, VisA
`VisA_20220922`, metadata revision `2a692ab575001cbde74d402d897a7286086c6199`, official
training normal, internal validation partition. Purpose is exploratory development
diagnostics; prior exposure is explicitly validation used for model selection.
The CC BY 4.0 field records documented source terms, not a new rights clearance.
Other families/categories, official test, calibration/fitting subsets, anomalies,
unknown revisions, independence claims and tuning purposes are rejected.

| Declared proposal review | Derived eligibility | Sample / artifact / execution |
|---|---|---|
| Unreviewed | NEEDS_PROVENANCE | NOT_VERIFIED / NOT_VERIFIED / NOT_AUTHORIZED |
| Withheld | NOT_CLEARED | NOT_VERIFIED / NOT_VERIFIED / NOT_AUTHORIZED |
| Adverse | BLOCKED | NOT_VERIFIED / NOT_VERIFIED / NOT_AUTHORIZED |
| Documented candidate only | ELIGIBLE_IN_PRINCIPLE | NOT_VERIFIED / NOT_VERIFIED / NOT_AUTHORIZED |

These are caller-supplied proposal descriptions, not authenticated review receipts.
No verified/cleared sample state, approved artifact state, execution token, private
sample ID/hash/path or arbitrary source URL exists in this implementation.
`require_execution` always raises `ExecutionDenied`, even after proposal round-trip
or eligibility-in-principle. Metadata cannot authorize an image read or a model load.
Verified individual admission and owner execution authorization are distinct future
gates, deliberately unavailable here. Transformations cannot promote a test parent
to development; this phase exposes no real-parent transformation path at all.

Proposal JSON is an allowlisted, scalar-only metadata envelope, bounded to 16 KiB,
with required exact fields, duplicate-key rejection using the existing strict
JSON utility, exact enum/version types and redundant eligibility consistency checks.
Unknown fields, bool-as-int, nonfinite values and forged clearance are rejected.
There is no persistence, file importer, URL resolver, CSV export or measured
real-data report. Importing a proposal supplies no proof of source authenticity.

## Service interface without execution

`development_boundary.InterfaceSpec` reuses the existing `ModelInfo` type.
It checks the Cashew AE identity, finite declared threshold and strict greater-than
rule. Artifact/catalog digest syntax and output equality are compatibility checks,
not hash verification of any file. Tests deliberately use fabricated hashes and
thresholds; production provenance must eventually come from the existing verified
catalog/services and separately approved private records. There is no second registry.

`validate_inspection` accepts the existing exact `InspectionResult` and compares it
with the declared interface and original decoded RGB8 pixels. It validates identity,
threshold/flag, finite nonnegative scores, finite nonnegative float raw maps of shape
128x128, original-pixel preservation, optional reconstruction and timing types.
Missing score/map components remain missing. Supplied scores are preserved, never
recomputed from spatial or display arrays. Errors contain fixed messages only.
Validated maps are detached and read-only; the result is an unverified numerical
view, not evidence of actual execution. The module imports InspectionResult lazily;
importing the foundation itself does not import Torch or model services.

Actual Cashew AE preprocessing remains RGB, full-frame bilinear antialiased resize
to 128x128, CPU float32 /255. Actual score is the mean raw RGB reconstruction MSE.
The service's raw map is on the **128x128 preprocessing grid**, not source-native
coordinates. The future adapter must pass transformations BEFORE this preprocessing.
No displayed/colorized/normalized heatmap may be a numerical input. A float display
normalization cannot be identified from array bytes alone: a verified service binding
must establish raw provenance. No source-to-map geometric mapping is established here.

`exercise_synthetic_interface` admits only the exact `SyntheticInspectionDouble`.
It has no image, path, generator callback, service or admission-record argument.
It reuses the 5C-3 internal solid RGB provider at 10x10 and the existing V2 perturb
API with a genuinely synthetic Plan. The five fixed calls are identity three times,
brightness 0.90 and brightness 1.10, always from the same immutable original.
The double consumes transformed RGB8 and emits an analytical constant 128x128 map
through InspectionResult. It performs neither image decoding nor a neural forward,
and does not simulate the autoencoder's scientific response or actual resize kernel.
This validates boundary ordering/compatibility only, not real preprocessing equivalence.

The first identity is the anchor. Double failures consume an attempt without retry;
preparation failures consume none. Missing first baseline skips remaining conditions;
later failures never replace the anchor. Generation errors are sanitized. This tiny
fixed exercise is not an isolated worker or a real-model execution ledger.

Stability comparisons reuse existing functions. Map comparison is elementwise on
the 128x128 preprocessing frame for identity/brightness only. Internally NATIVE is
relative to that comparison frame, never to the 10x10 source image. The outer
metadata always says `cashew_preprocessed_128_grid`. No geometric transform,
mask projection, native-coordinate alignment or physical invariance is inferred.

Returned CompatibilityObservation records are scalar diagnostic structures with
`synthetic_service_interface_probe`, `model_execution: not_run`, separate protocol
`factoryvision-synthetic-interface-probe` version/schema 1 and numerical version 2.
They cannot be exported as a 5C-3 SyntheticReport or historical registry evidence.
They contain no source/artifact fingerprints, source paths, pixels, maps or notes.
No real-development measured-report exporter is implemented. A future private schema
must be separately reviewed, retain exposure/denominators, use existing JSON/CSV
allowlists and `spreadsheet_cell` protections, cap final UTF-8 at 2 MiB, and reject
synthetic/historical schema mixing. Never serialize InspectionResult or ModelInfo
wholesale: they include arrays and artifact identities outside the scalar boundary.

## Windows isolation feasibility: not implemented or certified

The 5C-4B1 read-only environment probe found the relevant Windows Job APIs available
and the probe process already assigned to a job. No job, limit, worker, process tree
or Windows security setting was modified. API presence does not prove permission to
create/assign a nested job or prove containment in the eventual app/debugger context.
No stress test or real-model runtime measurement was performed.

Proposed future mechanism, requiring implementation and synthetic acceptance first:

1. A dedicated supervisor creates one hidden worker suspended, assigns it to a
   dedicated Job Object and verifies effective limits before resuming it. Do not
   attach existing VS Code/editor/terminal processes to the job. Do not enable breakaway.
2. Use job/process committed-memory limits and kill-on-job-close. The proposed 2 GiB
   must explicitly mean a committed-memory budget; it is **not a hard 2 GiB RSS/working
   set guarantee**. Measure peak working set separately. Refuse unsupported assignment,
   inherited-limit conflicts or limit-query mismatches before any permitted asset read.
3. A supervisor monotonic watchdog covers the proposed 300-second total from launch,
   including startup, verification and model load. Terminate the dedicated job on
   deadline/cancellation and wait for confirmed worker exit. A cooperative Event or
   waiting on a Python future is insufficient to interrupt a blocked load/forward.
4. Keep the execution ledger in the supervisor and use bounded scalar IPC with explicit
   parsing, not arbitrary pickle input. Allow no UI/image payloads or unsanitized tracebacks.
   Do not share locks/queues whose corruption could block the host during termination.
   A job's resource limits do not themselves sandbox filesystem or network access.
5. Before execution reserve all five units and decoded-source/report budgets. Send one
   command at a time; mark its unit dispatched/consumed before the worker may execute it.
   Success, failure, cancellation or termination never refunds it. On loss of acknowledgement,
   retain the reservation and mark execution outcome unknown, not fabricated zero attempts.
   Unstarted units remain reserved/skipped. Model-load failure before dispatch consumes no
   inference unit but still consumes time/resources; do not retry or replace the sample.

The OS APIs support enforcing committed-memory allocation limits and terminating a
job's process tree. They do not establish an exact wall-clock exit instant or a tested
hard RSS cap. CPU time limits are not a wall-clock watchdog. Windows Process.terminate
alone does not kill descendants and skips finally/exit handlers. Required future tests
include failed job assignment, nested-job restrictions, a noncooperative synthetic
worker, parent death, deadline kill, bounded allocation rejection at a SMALL test cap,
confirmed exit and ledger preservation. Do not allocate 2 GiB to test the proposal.

Official references reviewed for this design:

- [Job objects and kill-on-close](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects).
- [Assignment and inherited limits](https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-assignprocesstojobobject).
- [Committed-memory limits](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information).
- [Job termination](https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-terminatejobobject).
- [Python process termination caveats](https://docs.python.org/3.14/library/multiprocessing.html#multiprocessing.Process.terminate).

**Real pilot execution remains BLOCKED until isolation is implemented and demonstrated
in the intended launch context.** No hard timeout or memory enforcement is claimed by
the current synchronous synthetic exercise. Four CPU threads, batch one, one worker,
2 GiB and 300 seconds are proposed settings, not measured safe operating limits.

## Requirements for a separately authorized next phase

The proposed pilot remains one normal Cashew validation parent, exact membership
sealed before outcomes, existing AE checkpoint and threshold, three identities plus
brightness 0.90/1.10, five total attempts with no hidden warmups. Its intended meaning
is exploratory behavior on validation data already used for selection. Never report
independent held-out accuracy, OOD confidence, population robustness or deployment readiness.

Before any access, the owner must separately approve the precise asset/root allowlist,
sample-level provenance review, source licence/use, private evidence location, runtime
limits and implementation scope. Metadata review and eventual execution approval must
remain separate gates. No access to the original private repository is implied.
Verify accepted manifest identity/membership, original partition, exposure, parent bytes,
decoded pixels and artifact provenance only within that future permission. Hashes are
consistency evidence, not acquisition independence or anonymization. Derived inputs
inherit parent restrictions, including rejection of transformed official-test images.

Before a real adapter can reuse transformations, separately review a provenance-neutral
numeric-kernel boundary with equivalence tests; never create a synthetic Plan around real
pixels. Preserve the current public synthetic entry points and all frozen equations.
Verify lossless in-memory identity transport and unchanged service preprocessing/scoring.
Geometry, alternate models, retraining, tuning and recalibration remain excluded.

Keep private evidence out of Git and wheel/sdist contents; verify exclusion beyond merely
trusting ignore patterns. Strip EXIF/ICC and disclose no paths, sample IDs/hashes, environment,
tokens, Human Review notes, images or heatmaps. No automatic output writes or uploads.
MIT and dataset/pretrained attributions are unchanged. Preserve all nine scientific pins.
PCB1 stays BLOCKED, PCB2 INCONCLUSIVE, Cashew official test observed, MetalParts exploratory
source-domain only, RobustAD target NOT CLEARED; Human Review is not verified ground truth.

5C-4B1 is local, uncommitted foundation awaiting owner review. 5C-4B2 has not started.
The previous 365-test and installed-wheel results belong to accepted 5C-3, not new
5C-4B1 validation. Focused tests here use fabricated metadata and analytical doubles only.
