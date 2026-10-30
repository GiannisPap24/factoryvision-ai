# FactoryVision AI roadmap

Milestone 4E completed source-only publication on 2026-10-26 after owner approval
and the final security/privacy review. The [initial hosted CI run](https://github.com/GiannisPap24/factoryvision-ai/actions/runs/40376270944)
passed. Bounded Milestone 4F launch work is now authorized; external announcements
are pending platform access and destination verification, not claimed published.
Completion of an engineering milestone does not reverse a scientific integrity gate.

**Current public baseline:** Milestone **5C** plus the source-only Reddit-readiness
improvements merged through [PR #4](https://github.com/GiannisPap24/factoryvision-ai/pull/4)
at `8a303e9b9641a6d01379a7d20e02239315acbd2d`.
Historical [5C main CI](https://github.com/GiannisPap24/factoryvision-ai/actions/runs/40795664976)
recorded **1,448 passed, 97.61% combined coverage**; those counts do not describe
the later readiness release or this candidate.
**5D is a local, uncommitted integration candidate**, pending final acceptance and
owner publication approval. The [5D summary](milestone-5d-integration.md) records
CPU diagnostics, exact display allocation reduction and synthetic ONNX/DirectML
feasibility. The GPU feasibility experiment is closed as successful synthetic
research; production GPU integration is deferred. No trained-model speedup is established.

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

The following 5C chronology records each phase's authorization and evidence at
the time. Local checkpoints and pending approvals mentioned below are historical;
the source-only implementation was subsequently merged in PR #3. That merge did
not authorize real data/model execution or establish validated OOD detection.

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
At that checkpoint this was local work awaiting review, with no neural-network inference, dataset
access, runner, UI or publication. Its nine-file local checkpoint was verified
before the separately authorized 5C-3 work.

Milestone **5C-3** adds a [bounded synthetic runner and scalar reports](protocols/robustness-runner-v1.md),
with internally generated probes, analytical-only adapters, cumulative unit
accounting and validated in-memory JSON/CSV. Protocol V2 remains unchanged.
That local implementation awaited owner review; it makes no real-model or industrial
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
reused through small extractions. Its eight-file local checkpoint is
`c2148cba82d59ecff9684db40d0a0ea14aa79082`; real models/data remain unauthorized
and the Cashew pilot remains blocked.

Milestone **5C-RC1** integrates the source-only acceptance review, packaging smoke
and [runnable robustness guide](protocols/robustness-source-only.md). This was local
release preparation; publication then required a separate owner decision. Earlier
paragraphs and protocol reports retain the scope and evidence of their own phases.

## Owner-agreed strategic direction

These milestones are **proposed and separately gated**. Their listing is neither
execution authorization nor evidence of achieved industrial performance, customer
adoption or deployment approval.

| Milestone | Proposed direction |
|---|---|
| 5D | Local integration candidate; synthetic GPU feasibility closed, production GPU integration deferred |
| 5E | Camera Inspection; separately approved capture, privacy and inspection scope |
| 5F-A / 5F-B / 5F-C | Dataset eligibility, independent model validation, then supervised pilot discovery |
| 6A | Local Inference API |
| 6B | Industrial Integrations |
| 6C | Commercial Readiness |

## Unresolved Problems, Remediation Roadmap and Acceptance Gates

This is a planning map, **not authorization to execute future milestones**.
Software test success is not scientific validation or commercial readiness.
Evidence and permissions must be reviewed for the specific use case. No row
authorizes threshold tuning against the already observed Cashew test, automatic
training from Human Review, autonomous product rejection or machine control.

| Target milestone / problem | Current status | Proposed remediation | Completion evidence | Required authorization | Stop / blocked condition |
|---|---|---|---|---|---|
| **5D — Performance & Runtime Compatibility** | Local CPU/display/random-weight ONNX candidate; DirectML synthetic execution and five-case parity verified, experiment closed. Production GPU integration deferred; real-model performance unverified. P2 deprecated pathlib usage remains. | Profile CPU time and memory, optimize display allocations, verify ONNX export and numerical parity, investigate AMD compatibility and reliable CPU fallback; runtime isolation, resource monitoring and reproducible benchmark methodology. Replace deprecated pathlib usage before future Python compatibility changes. | Repeated workload-matched measurements with uncertainty, equivalence/parity tests, supported-runtime tests, bounded failure/resource evidence and no deprecation regressions. Separate synthetic evidence from trained-model benchmarks. | Separate owner approval for trained models/data, GPU/runtime installation and each implementation scope. | Unclear artifact eligibility, changed scientific semantics, unsupported runtime or need to weaken Windows protections. No speedup claim from synthetic evidence alone. |
| **5E — Camera & Input Reliability** | Camera workflow and validated OOD detection absent. Category is manual; wrong-category inputs are not reliably detected. P2 accessibility/responsive work remains. | Camera acquisition, focus and quality checks, illumination variation, motion blur/invalid frames, safe input limits and unsupported-image rejection; clear uncertainty/abstention, keyboard/focus/readability/responsive usability and human confirmation before consequential decisions. | Authorized capture tests across acquisition conditions; bounded input rejection; keyboard and narrow-viewport acceptance; independent OOD evaluation before any OOD accuracy claim. | Capture/privacy permissions and scoped UI/input implementation approval. | No consent, unsupported acquisition conditions or uncertainty presented as a reliable defect decision. Basic quality checks are not validated OOD detection. |
| **5F-A — Dataset Integrity & Scientific Eligibility** | PCB1 **BLOCKED** by source overlap; PCB2 **INCONCLUSIVE**, 56 similarity pairs await independent adjudication; RobustAD target **NOT CLEARED**. | Establish train/validation/test and source/acquisition-group independence; exact/near-duplicate screening; provenance/licensing review, independently acquired evaluation data where necessary and independent human label adjudication. | Documented group-aware split audit, resolved similarity evidence, licensing/provenance records and independent adjudication accepted before evaluation. | Explicit data acquisition/access and independent-adjudication approval. | Any unresolved overlap, missing rights or non-independent labels. Historical observed tests never become new blind evidence. Frozen gates cannot be overridden. |
| **5F-B — Independent Model Validation** | Cashew AE historical recall **0.52**, **48/100 anomalies missed** at its frozen threshold; official test already observed. MetalParts remains exploratory-only. | False-negative/false-positive analysis; predeclared model selection and evaluation; separate development/calibration, independent held-out testing, pixel localization where applicable, real acquisition robustness, independent OOD evaluation, defensible uncertainty and use-case acceptance thresholds. | Eligible independent data from 5F-A, preregistered protocol, error/localization analyses with uncertainty and independently reviewed acceptance results. | New model/evaluation scope and trained-artifact access approved in advance. | Tuning against the observed Cashew official test, unresolved eligibility or presenting exploratory scores as industrial accuracy. |
| **5F-C — Supervised Industrial Pilot Discovery** | No validated industrial pilot or customer acquisition claimed; real Cashew development pilot remains **BLOCKED**. | Establish customer/use-case requirements, defect definitions, missed-defect costs, false-alarm tolerance, camera conditions/throughput, data/privacy permissions, human-supervised design, escalation and pilot stop criteria. | Documented use case, appropriate scientific eligibility and independent validation, signed-off permissions and supervised pilot protocol. Evidence collection is separately authorized. | **New owner decision** plus data/customer permissions; discovery alone does not approve a pilot. | Missing eligibility, unacceptable error costs, absent supervision or unclear stop/escalation rules. |
| **6A — Local Inference API Reliability** | API not implemented; current source-only diagnostics use bounded workers and an in-memory execution ledger with restart limitations. Windows Job Objects are **not a full security sandbox**. | Bounded inputs, request validation and authenticated local access where appropriate; resource limits, timeouts, cancellation/failures, execution accounting, replay prevention/idempotency and supervisor-failure recovery. Explicit unknown/interrupted states; no automatic retry after uncertain execution. | Failure-injection, restart/accounting and replay tests; resource-limit evidence and reviewed access design. | Scoped API and security-design approval; separate model execution permission. | Lost ledger state, uncertain prior execution or unverifiable isolation. Never silently retry an unknown outcome. |
| **6B — Industrial Integration Safety** | No industrial camera/controller actuation authorized. | Connection-loss/interruption handling, operator control/manual fallback, auditable commands/outcomes, duplicate-action prevention and separation of research predictions from operational decisions. | Approved integration specification, simulated failure/duplicate-command tests and human-supervised safety acceptance. | Separate owner and equipment-operator approval for each integration. | Unknown command outcome, absent manual fallback or autonomous rejection/machine control outside explicit scope. |
| **6C — Commercial Readiness** | Not commercially/industrially validated. Third-party redistribution rights are separate from MIT source licensing. | Security/privacy review, runtime/deployment maintenance, monitoring/drift, independent use-case validation, dataset/weight/runtime licensing and redistribution permissions, support/documentation, incident response and operational fallback. | Reviewed technical/scientific/legal evidence and final documented **GO/NO-GO** for the exact deployment/use case. | Separate commercial/deployment approval and required rights/permissions. | Missing independent performance evidence, unresolved licensing/security or inability to support safe operations. Passing software tests is insufficient. |
| **4F — Community feedback (cross-cutting)** | Initial presentation pending; confirmed external testers are **not yet observed**, not assumed zero. | Track synthetic-demo testers, installation issues, reproducible bugs, usability/documentation feedback and GitHub issue triage using non-sensitive voluntary evidence. | Deduplicated confirmed reports and reproducible resolutions under the [feedback process](community-feedback.md). | Destination/rules review and explicit owner approval before posting or outreach. | Sensitive data, unsupported capability claims or treating stars/views as users or industrial evidence. |
| **Maintenance — Privacy, documentation and UI** | Known historical public Git author email exposure; historical 5B acceptance, dataset-contract licensing and inference-guide future-work wording needs contextual maintenance. P2 accessibility and pathlib findings remain tracked above. | Review privacy-maintenance options without repeating the address; label historical docs, reconcile current guides; consider a separately scoped industrial-research-workbench UI polish (navy/light surfaces, blue/teal, clear hierarchy/spacing, useful image areas and compact diagnostics). Keep one Streamlit frontend. | Owner-reviewed privacy decision, consistent current docs, accessibility/browser evidence and no invented models, charts, confidence values, histories, accounts or GPU monitoring. | Separate approval for any public-history rewrite or broader redesign; never automatic. | Destructive history change without approval, loss of historical evidence, fabricated features or unnecessary frontend rewrite. |

This launch-readiness checkpoint addresses the P2 first-run action and contextual
navigation warning only. Cleanup remains intentional: workspace/category changes
clear current images, results and reviews; only a completed batch and saved reviews
survive Batch Inspection ↔ Reports. The warning is not a confirmation dialog or
persistence mechanism. Broader accessibility and visual redesign require their own
scope. Historical reports, thresholds and nine pinned scientific resources stay intact.

See [publication record](publication.md), [contributing](../CONTRIBUTING.md)
and [licensing](licensing.md).
