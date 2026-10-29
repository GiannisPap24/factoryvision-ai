# FactoryVision AI roadmap

Milestone 4E completed source-only publication on 2026-10-26 after owner approval
and the final security/privacy review. The [initial hosted CI run](https://github.com/GiannisPap24/factoryvision-ai/actions/runs/40376270944)
passed. Bounded Milestone 4F launch work is now authorized; external announcements
are pending platform access and destination verification, not claimed published.
Completion of an engineering milestone does not reverse a scientific integrity gate.

## Completed research and engineering

| Milestone | Outcome |
|---|---|
| 0–1B | Hardware feasibility, dataset contracts and CPU PyTorch foundation |
| 2A–2B | Cashew autoencoder and spatial baseline; one historical held-out evaluation, now observed |
| 2C–2D | Frozen pretrained feature memory and trained student–teacher prototypes |
| 2E1–2E2-R | PCB1 preparation and leakage incident audit; **BLOCKED** |
| 2F0–2F0-A | PCB2 integrity screening and adjudication workflow; **INCONCLUSIVE** |
| 3A–3C | Local application and RobustAD integration; independent target evaluation **NOT CLEARED** |
| 3D–3F | MetalParts normal-source training and comparator; exploratory source-domain diagnostics only |
| 4A | Product usability, reliability and research demo |
| 4B1–4B2 | Bounded sequential batch inspection and validated CSV/JSON research reports |
| 4C | Six-method catalog, synthetic demo and source-only CI configuration |
| 4D and remediation | Technical/privacy/licensing audit; Python 3.14.10 adoption; MIT for owned source |
| 4D-UX | Eight-workspace interface with navy, blue, teal and warm scientific maps |
| 4E | Independent source-only publication, final security/privacy review and successful hosted CI |

The [model cards and research chronology](reproduction.md) record limitations and
provenance. No model is validated for consequential industrial quality decisions.

## Separately gated next work

Source publication and successful hosted CI are complete. For the authorized 4F
launch, GitHub private vulnerability reporting has been enabled and its public entry
point verified. Safe issue templates and a [feedback process](community-feedback.md)
support the initial community round. Community announcements remain pending;
no external-user acquisition or industrial validation is claimed.

Milestone **5A — Human Review V1** was merged in [PR #1](https://github.com/GiannisPap24/factoryvision-ai/pull/1)
at `648091dbe2301ee2429b0e4c19576f64479efc88`; its
[hosted CI](https://github.com/GiannisPap24/factoryvision-ai/actions/runs/40404981472)
passed. See the [implementation record](milestone-5a-human-review.md).
Human observations do not change predictions, thresholds or research status.

Milestone **5B — Model Registry & Experiment Tracking V1** extends the existing
Model Catalog with content-addressed artifact versions, recorded provenance and
an offline historical evidence index. See the [5B acceptance record](milestone-5b-model-registry.md).
It was merged in [PR #2](https://github.com/GiannisPap24/factoryvision-ai/pull/2)
at `2e59101b011b74b5fc0658c79d46cb2d7fb5f002`. The accepted
[main CI run](https://github.com/GiannisPap24/factoryvision-ai/actions/runs/40565890977)
passed with **840 tests and 98.24% coverage**. The acceptance record retains its
historical local results; those are not the later hosted counts.

Milestone **5C-1** is limited to the [synthetic robustness protocol](protocols/robustness-v1.md),
strict metadata contracts and focused offline tests. It implements no perturbation
algorithms, inference runner, OOD detector or UI. Its six-file local checkpoint was
approved and verified before 5C-2 implementation. This work preserves PCB1 BLOCKED,
PCB2 INCONCLUSIVE, observed Cashew test status, exploratory-only MetalParts and
RobustAD target NOT CLEARED. Human Review observations remain distinct from ground
truth. Synthetic diagnostics do not establish industrial accuracy or OOD confidence.

Milestone **5C-2** implements [protocol V2 numerical diagnostics](protocols/robustness-v2.md):
deterministic synthetic-array transformations and model-independent score/raw-map
metrics. The owner approved explicit contrast and noise-stream amendments to V1.
This remains local work awaiting review, with no neural-network inference, dataset
access, runner, UI or publication. Its nine-file local checkpoint was verified
before the separately authorized 5C-3 work.

Milestone **5C-3** adds a [bounded synthetic runner and scalar reports](protocols/robustness-runner-v1.md),
with internally generated probes, analytical-only adapters, cumulative unit
accounting and validated in-memory JSON/CSV. Protocol V2 remains unchanged.
This local implementation awaits owner review; it makes no real-model or industrial
performance claim. Milestone 5C-4 requires separate authorization.

Milestone **5C-4A** accepted the seven-file 5C-3 local checkpoint and reviewed the
six inference interfaces without real-data/model access. It selected Cashew AE as
a conditional planning candidate, not an approved sample or execution.
Milestone **5C-4B1** adds the [development admission foundation](protocols/development-admission-v1.md)
and an internal synthetic service-interface probe. Sample/artifact admission stays
unverified and real execution remains denied. Windows isolation is a documented,
unverified prerequisite; the real pilot remains blocked pending its acceptance and
separate owner authorization. The foundation was checkpointed locally for 5C-4B2A.

Milestone **5C-4B2A** adds [synthetic Windows isolation and supervisor accounting](protocols/windows-isolation-v1.md).
Actual nested-job assignment, suspended launch, small committed-memory caps,
short watchdogs, cancellation and owned-handle cleanup were tested. The implementation
was checkpointed locally for 5C-4B2B. Proposed real-model limits and actual
development execution remain unvalidated and unauthorized; the Cashew pilot is blocked.

Milestone **5C-4B2B** verifies [supervisor crash containment](protocols/windows-crash-containment-v1.md)
with disposable synthetic processes, explicit handle ownership and fail-closed
interruption accounting. The worker exits after actual supervisor termination while
the controller remains outside the dedicated job. These test-only additions were
checkpointed locally for 5C-4C1; real integration and execution need separate approval.

Milestone **5C-4C1** adds [Cashew integration preparation](protocols/cashew-development-preparation-v1.md)
with internal synthetic pixels, lossless transport, existing interface/diagnostic
validators and abort-only ledger accounting. The lifecycle is explicitly a test double;
the Windows runner and real services were unchanged in that phase. Its four-file source-only
checkpoint was reviewed and created locally before 5C-4C2A.
No sample/artifact is cleared, no real worker is enabled and the Cashew pilot remains blocked.

Milestone **5C-4C2A** adds [isolated synthetic Cashew IPC](protocols/cashew-isolated-ipc-v1.md):
one fixed analytical worker, two explicitly inherited pipe endpoints, bounded frames,
five supervisor-charged requests and scalar-only diagnostics. Actual Windows subprocess,
failure and cleanup tests pass locally. The numerical double and owned-job cleanup are
reused through small extractions. This implementation remains uncommitted for owner review;
real models/data remain unauthorized and the Cashew pilot remains blocked.

## Owner-agreed strategic direction

These milestones are **proposed and separately gated**. Their listing is neither
execution authorization nor evidence of achieved industrial performance, customer
adoption or deployment approval.

| Milestone | Proposed direction |
|---|---|
| 5C | Robustness & OOD-oriented Testing; synthetic-first methodology and separate evaluation authorization |
| 5D | Performance Optimization, ONNX and AMD GPU feasibility; preserve numerical and scientific validity |
| 5E | Camera Inspection; separately approved capture, privacy and inspection scope |
| 5F | Industrial Pilot Discovery; identify use cases and evidence requirements before pilot claims |
| 6A | Local Inference API |
| 6B | Industrial Integrations |
| 6C | Commercial Readiness |

Historical acceptance records remain intact. Later documentation maintenance should
reconcile the 5B record's pre-publication wording, older CI counts where presented
as current, the dataset-contract licensing decision wording and the inference
guide's historical future-work paragraph; these are outside 5C-1's implementation.

See [publication record](publication.md), [contributing](../CONTRIBUTING.md)
and [licensing](licensing.md).
