# Research reproduction and boundaries

The application runs frozen artifacts; starting it never starts an experiment.
Reproducing research requires licensed data, verified inventories, split/gate
records and the specific frozen protocol/source snapshot. A public source checkout
alone does not include these assets. Do not remove one-shot guards or manufacture
an integrity approval to make a command run.

## Recorded entry points

The command syntax below is checked against the current scripts. They are research
instructions for a separately prepared, approved fresh workspace, **not commands
to rerun in the completed workspace**. No training was run during Milestone 4A.

| Experiment | Explicit entry point | Required preparation and behavior |
|---|---|---|
| Cashew AE and spatial baseline | `.venv-31410/Scripts/python.exe scripts/train_cashew.py --run-id <fresh-id>` | Verified VisA manifest, leakage review/screen and bytes; [training](training.md). Fits 315 normal images, validates 67, calibrates on 68. Existing IDs refused. The historical byte-integrity gate hashes all manifest members; it is not a test-inference step. |
| Cashew feature bank | `.venv-31410/Scripts/python.exe scripts/feature_experiment.py fit` | Approved normal-only manifest, backbone and frozen source seal; [feature recipe](feature-memory.md). Fixed one-shot run ID; refuses completed output. No backbone gradient updates. |
| Cashew student | `.venv-31410/Scripts/python.exe scripts/student_experiment.py train` | Approved teacher, splits and sealed recipe; [student training](student-training.md). Fixed one-shot run ID. The smoke option also trains and is not a default installation check. |
| MetalParts AE | `.venv-31410/Scripts/python.exe scripts/metalparts_experiment.py prepare`, then `smoke`, then `train` | Pinned source verification inventory, normal-only allowlist, 3D protocol/code seal; [exploratory protocol](metalparts-exploratory-training-protocol.md). Prepare creates a reviewed split; smoke/train are explicit gradient-training actions. Existing runs refused. |
| MetalParts feature bank | `.venv-31410/Scripts/python.exe scripts/metalparts_feature_experiment.py seal`, then `fit` | Approved 3D normal split, original ResNet18 backbone and 3F source seal; [feature protocol](metalparts-feature-memory-protocol.md). Fits memory without gradient training; fixed run guards. |

Read the linked protocol and script prerequisites before executing. These commands
have no automatic dataset or backbone download. Obtain data only through the
original source and after reviewing its terms. Local historical manifests and
human integrity decisions cannot be regenerated from a package install alone.
Fresh runs need distinct protocols/identities and a separate review to register
artifacts; application registries pin exact historical bytes. Training does not
promise bit-identical hashes on a different system. Do not change frozen thresholds
to improve observed defect results.

## Scientific chronology

- [2A training](milestone-2a-completion.md), [2B recorded Cashew evaluation](milestone-2b-completion.md).
- [2C frozen feature memory](milestone-2c-completion.md), [2D student training](milestone-2d-completion.md).
- [PCB1 blocked evaluation](milestone-2e2-completion.md): source overlap invalidated the intended confirmation.
- [3B dataset strategy](milestone-3b-dataset-strategy.md), [3C MetalParts integrity](milestone-3c-robustad-integration.md).
- [3D source-normal training](milestone-3d-metalparts-training.md),
  [3E observed source diagnostics](milestone-3e-metalparts-development-evaluation.md),
  [3F feature comparator](milestone-3f-metalparts-feature-memory.md).

Do not run historical evaluation entry points on already observed official data
and label them fresh held-out confirmation. PCB1 is BLOCKED, PCB2 INCONCLUSIVE;
RobustAD independent target evaluation is NOT CLEARED. A future independent study
requires separate authorization and resolution of source/acquisition independence.

## Safe developer verification

First follow [isolated setup](ci.md); the original environment remains offline-only.

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path (Get-Location) '.venv-31410'
.tools/uv-0.13.5/uv.exe sync --locked --extra app --python '.venv-31410/Scripts/python.exe'
.venv-31410/Scripts/python.exe scripts/check_quality.py --report-dir reports/local-quality
```

The default suite uses fabricated data and blocks network access. It does not run
`integration_tests/` or real experiments. Use the synthetic inference debugger
configurations, not the training configurations, for a portfolio demonstration.

## Inspect historical provenance without rerunning (5B)

Open **Model Catalog** in a source-only installation. Registry schema 1 links the
existing model identity to its exact artifact SHA-256, dataset/split provenance,
recorded run ID, recipe/protocol digests where recorded, and accepted evidence
sources. No dataset, checkpoint or pretrained download is needed.

The evidence index is discovery, not a rerun or an experiment scheduler. Historical
Cashew Python 3.12.15 provenance is not rewritten to the application's Python
3.14.10 runtime. Execution timestamps and canonical configuration fingerprints
remain unavailable; a recipe-file digest is explicitly a different identity.
Source references are portable package names, not owner directories.

New historical entries require reviewed source/evidence changes and adapter tests.
There is no runtime metadata import or append-only store in V1: the existing package
evidence remains authoritative. See [5B](milestone-5b-model-registry.md).

## Current scientific evidence baseline

The integrated public baseline is commit
[`5d28861c9a2ed3c67e3e976b0b2935ab16926ed2`](https://github.com/GiannisPap24/factoryvision-ai/commit/5d28861c9a2ed3c67e3e976b0b2935ab16926ed2),
tree `1d91bc1fec581693faabad6202d68c07867a804d`. Its
[main CI](https://github.com/GiannisPap24/factoryvision-ai/actions/runs/42331567116)
passed 1,803 tests, 13 optional isolated-ONNX skips and exact combined coverage
96.84001981%, including packaging/privacy and fresh installation. These are
software results for that source, not new training, inference or independent
scientific confirmation. [Current roadmap](roadmap.md) and [CI chronology](ci.md)
retain the integration and earlier acceptance records.

| Area | Implemented or recorded evidence | Current interpretation |
|---|---|---|
| 5C protocol/numerics | Frozen [V1 design](protocols/robustness-v1.md), implemented [V2 transformations/diagnostics](protocols/robustness-v2.md); deterministic explicit conditions and seeds | VERIFIED integrated software; numerical stability is not real-model robustness. V2 supersedes the documented V1 contrast/noise definitions, not historical results. |
| 5C execution/reporting | [Bounded analytical runner](protocols/robustness-runner-v1.md), scalar reports, [ledger/Windows containment](protocols/windows-isolation-v1.md), [fixed synthetic IPC](protocols/cashew-isolated-ipc-v1.md) | VERIFIED synthetic infrastructure. Reports do not authenticate execution; in-memory accounting and Job Objects are not durable recovery or a complete security sandbox. |
| 5C real admission | [Proposal-only development foundation](protocols/development-admission-v1.md), no real-worker binding | BLOCKED real Cashew pilot. Integrated synthetic robustness infrastructure does not establish real trained-model robustness or permit real-model execution. |
| 5D CPU/display | [Synthetic CPU baseline](performance-baseline.md), [exact display allocation/equivalence work](performance-display-5d2.md), [public integration addendum](milestone-5d-integration.md#current-integration-status-addendum) | VERIFIED allocation reduction; a general latency improvement is not established. Historical timings retain their original source/environment. |
| 5D runtime feasibility | [Optional synthetic ONNX CPU tooling](onnx-synthetic-feasibility.md), historical DirectML experiment in the [5D summary](milestone-5d-integration.md) | HISTORICAL random-weight parity/feasibility, not trained-model runtime parity, industrial throughput or production GPU readiness. Production DirectML integration deferred. |
| Six registered methods | [Cashew AE/spatial mean](model-card.md), [feature memory](feature-model-card.md), [student–teacher](student-model-card.md), [MetalParts AE](metalparts-model-card.md), [MetalParts feature memory](metalparts-feature-model-card.md) | HISTORICAL trained/development evidence with method-specific scope; raw scores are not calibrated defect probabilities and incompatible datasets/protocols cannot rank models. |
| Industrial Precision UI/viewer | [PR #17](https://github.com/GiannisPap24/factoryvision-ai/pull/17), including F5/F6/F7 ancestor [PR #16](https://github.com/GiannisPap24/factoryvision-ai/pull/16); [interface](interface.md) | VERIFIED integration. Current NVDA speech, genuine Chrome 200% zoom and physical touch remain UNVERIFIED, deferred for research-stage integration only. Four eager encodings per viewer remain documented; no WCAG certification or industrial readiness. |

The [5B evidence index](milestone-5b-model-registry.md) has six method histories
and two integrity summaries, not eight trained models. Missing execution timestamps
and canonical configuration fingerprints stay null; recipe/protocol digests are
not interchangeable with those fields. A source-only package need not distribute
restricted data or weights. Their absence is distinct from missing independent
scientific evidence, and neither is repaired by inventing provenance.

Historical cards/protocols remain unchanged. The feature/student cards' future-work
pointers describe their original phases: see the later [2D completion](milestone-2d-completion.md)
and [PCB1 blocked 2E2 outcome](milestone-2e2-completion.md), not a current approval.
The MetalParts card's unopened-target wording is specific to its source-development
phase: [3C](milestone-3c-robustad-integration.md) already decoded target payloads for
integrity; [3E](milestone-3e-metalparts-development-evaluation.md) opened none during
that diagnostic run. This distinction does not imply accepted target-model evaluation.
Earlier undecided-project-license wording records prepublication history; the later
[owned-source MIT decision](licensing.md) does not relicense datasets or weights.

## Dataset eligibility dossier

This is a sanitized prerequisite record, **not an executable authorization token**,
new validator, registry or clearance decision. No dataset, weight or restricted
record was opened to fill it. Classifications describe evidence, not permissions:
VERIFIED (checked public source/declaration), HISTORICAL (accepted earlier record),
INFERRED (identified interpretation), UNVERIFIED (not established), BLOCKED,
INCONCLUSIVE, NOT CLEARED and NOT APPLICABLE. A VERIFIED declaration is not proof
of its underlying data, rights or independent sampling.

Supplied metadata is not authenticated evidence; checkpoint existence is not
permission to load it; dataset availability is not rights clearance; a clean
screen is not proof of acquisition independence. Missing metrics remain missing,
not zero. Historical observations cannot become new blind confirmation. Human
Review remains observations, not verified ground truth. Future custody/access
locations, reviewer identities and approvals are intentionally not supplied here.

**No eligible independent confirmation dataset is established by the currently
available evidence.** Exploratory reuse requires its own declared limitations;
independent confirmation additionally requires defensible independent data and a
predeclared protocol. Public/common research use alone establishes neither.

### Cashew exploratory development pilot

| Required field | Evidence and prerequisite |
|---|---|
| Intended scientific question | Can the unchanged Cashew AE produce bounded, interpretable development stability diagnostics on explicitly admitted validation material? This is exploratory, not independent accuracy. |
| Dataset/source and release identity | VisA_20220922; one-class metadata revision `2a692ab575001cbde74d402d897a7286086c6199`, as recorded in [dataset preparation](dataset-preparation.md). |
| Declared rights/licensing evidence | HISTORICAL source record declares VisA CC BY 4.0 and attribution. Current purpose-specific access/redistribution permission is UNVERIFIED here; public availability is not clearance. [Licensing boundaries](licensing.md). |
| Permitted activity | Only proposal/synthetic infrastructure is integrated. No real sample or artifact is admitted by this dossier; `require_execution` always denies in the [existing admission contract](protocols/development-admission-v1.md). |
| Acquisition/source grouping and independence | Exact-pixel groups exist historically; physical-item/acquisition identities remain UNVERIFIED. Validation was already used for model selection. |
| Historical exposure chronology | Normal fitting/validation/calibration were accessed historically; the official test is OBSERVED. [2B chronology](evaluation-protocol.md) remains authoritative. |
| Training/validation/calibration/test splits | 315 fitting, 67 validation, 68 calibration normals; 150 official test images retain test membership. The proposed pilot is limited in principle to internal validation, not test/calibration/fitting substitutes. |
| Label/mask reliability and adjudication | Historical normal labels and manifest identities require authorized verification for each proposed sample. Human Review notes are not verified ground truth; no independent defect labels are created. |
| Exact-duplicate screening provenance | Historical file/decoded-pixel audit found no exact cross-partition duplicates; no sample bytes are checked here. [Preparation and addenda](dataset-preparation.md#milestone-2b-integrity-gate). |
| Near-duplicate/similarity provenance | 7,946 historical candidates refined, 20 deterministic pairs reviewed, no confirmed repeated acquisition in that reviewed evidence; not proof of independence. [Frozen evaluation explanation](evaluation-protocol.md). |
| Artifact/model identity | Existing `autoencoder` identity, checkpoint/calibration/metadata pins; see [inference prerequisites](inference.md#architecture-and-frozen-identities) and [Cashew card](model-card.md). Exact future permitted locations and bytes remain UNVERIFIED. |
| Preprocessing identity | RGB full-frame bilinear antialiased resize to 128×128, CPU float32 /255; transformations must precede unchanged preprocessing. Real raw-map/mask correspondence requires review, not an assumed native grid. |
| Calibration and frozen threshold | Frozen AE threshold 0.0031531278509646654, strict score > threshold, historical 68-normal linear q95. No refit or probability calibration. |
| Existing evaluation evidence | HISTORICAL AE official-test image AUROC 0.937600 and recall 0.52; synthetic 5C interface results are not AE predictions. Neither clears this pilot. |
| Missing prerequisites | Exact owner-approved sample/artifact scope, rights/provenance/exposure and byte checks; real service binding/transport/map semantics; validated limits and abort/unknown recovery. |
| Required independent review | Scientific eligibility/provenance and numerical-boundary review; security/resource review of any real worker. No reviewer identity or approval is supplied. |
| Required owner authorization | Separate named metadata/data/artifact access, implementation and bounded real execution decisions. The private development repository is never an implicit source. |
| Stop condition | Unverified identity/rights/membership; official-test or other unapproved sample; changed threshold/preprocessing; uncertain prior execution/cleanup; any missing execution approval. |
| Current eligibility classification | **BLOCKED** for actual real-model pilot. Proposal eligibility-in-principle is not authenticated sample clearance or execution authority. |

### Cashew genuinely independent confirmation

| Required field | Evidence and prerequisite |
|---|---|
| Intended scientific question | What real-defect accuracy/localization and uncertainty hold on genuinely independent, predeclared data for specified frozen methods? |
| Dataset/source and release identity | No eligible independent dataset/release has been established. Existing VisA Cashew identity is recorded above solely to identify the OBSERVED official test; no new dataset is invented. |
| Declared rights/licensing evidence | **UNVERIFIED** for a new independent source and intended use. Historical VisA terms do not grant access or rights to hypothetical new acquisitions. |
| Permitted activity | Planning/evidence review only under a separate scope; no acquisition, model selection, test opening or evaluation is authorized by this record. |
| Acquisition/source grouping and independence | Independent acquisition/physical-item groups and separation from all development/exposed data must be established; current evidence cannot supply them. |
| Historical exposure chronology | AE/mean official test already observed; later feature/student research direction was informed by earlier results. A new model or seed cannot restore blindness. |
| Training/validation/calibration/test splits | Existing 315/67/68 and official 150 remain frozen historical roles. A genuinely new study needs separately accepted development/calibration/held-out partitions; none are assigned here. |
| Label/mask reliability and adjudication | Independent defect definitions, labels/masks and adjudication reliability required for the new study; UNVERIFIED. Human Review is not a substitute. |
| Exact-duplicate screening provenance | Existing historical Cashew audit applies to that dataset only. New authorized source and cross-corpus file/pixel screening evidence is missing. |
| Near-duplicate/similarity provenance | Historical similarity review does not prove independence of a prospective source. New group-aware review and prior-exposure comparison are required. |
| Artifact/model identity | Prospective method/version selection is UNVERIFIED. Existing four Cashew identities and frozen artifacts remain historical authorities; see [model evidence](#current-scientific-evidence-baseline). |
| Preprocessing identity | Prospective study preprocessing must be predeclared and fixed before confirmation; no changes to historical RGB128 or existing method recipes are authorized. |
| Calibration and frozen threshold | Existing thresholds stay frozen. Any new calibration/selection protocol must use independent permitted development material before confirmation; no test-driven tuning. |
| Existing evaluation evidence | HISTORICAL 2B paired AE/mean metrics only. Feature/student artificial-probe results are not independent real-defect metrics. [Model cards](#current-scientific-evidence-baseline). |
| Missing prerequisites | Eligible independent source, rights/custody, acquisition groups, labels, fixed question/model selection/calibration/metrics/uncertainty and one-shot protocol. |
| Required independent review | Independent data-integrity/label review and scientific protocol acceptance before any confirmation; review evidence is UNVERIFIED. |
| Required owner authorization | Separate acquisition/access, study design/freeze and evaluation authorization for exact assets/protocol. No command or permission is provided here. |
| Stop condition | Observed official test reused as fresh confirmation; unresolved overlap/labels/rights; post-test model or threshold selection; absent independent-source evidence. |
| Current eligibility classification | **BLOCKED** for new independent confirmation with the currently established evidence; prospective independent dataset is **UNVERIFIED**. |

### PCB1 integrity blocker

| Required field | Evidence and prerequisite |
|---|---|
| Intended scientific question | Can any future defensible study be designed after confirmed cross-partition repeated acquisition? The original intended blind confirmation cannot simply resume. |
| Dataset/source and release identity | VisA_20220922; one-class metadata revision `2a692ab575001cbde74d402d897a7286086c6199`, as recorded in [dataset preparation](dataset-preparation.md). |
| Declared rights/licensing evidence | HISTORICAL source record declares VisA CC BY 4.0 and attribution. Current purpose-specific access/redistribution permission is UNVERIFIED here; public availability is not clearance. [Licensing boundaries](licensing.md). |
| Permitted activity | HISTORICAL normal preparation and integrity opening only; no new access or rerun. [2E2 blocked outcome](milestone-2e2-completion.md) supersedes conditional 2E1 readiness. |
| Acquisition/source grouping and independence | Confirmed repeated/near-duplicate acquisition crosses intended fitting/test partitions; exact grouping alone did not establish independence. |
| Historical exposure chronology | 200 official test images and 100 masks opened for integrity; zero official model predictions/metrics accepted. Earlier source/selection exposure remains historical. |
| Training/validation/calibration/test splits | Preserve original official/internal membership and sealed four-family preparation. Do not silently remove, relabel, reassign or replace a pair to manufacture independence. |
| Label/mask reliability and adjudication | Historical image/mask integrity checks do not cure source dependence. Any new independent labeling/review requirements need an approved study. |
| Exact-duplicate screening provenance | Historical byte/decoded-pixel checks and seal identities in [2E2](milestone-2e2-completion.md); no new byte review here. No exact-identity shortcut can clear a confirmed nonexact relation. |
| Near-duplicate/similarity provenance | Frozen similarity/refinement and reviewed repeated-acquisition evidence caused the block; [PCB1 v3](protocols/pcb1-confirmatory-v3.json) and incident record remain unchanged. |
| Artifact/model identity | Four historical PCB1-specific prepared families in v3; not additional app registrations. Actual permitted artifacts/locations not accessed or newly verified. |
| Preprocessing identity | Frozen v3 full-frame RGB128 bilinear antialias /255; ImageNet normalization for feature/student methods only. Not a new implementation scope. |
| Calibration and frozen threshold | Frozen v3 normal-only linear q95, strict greater-than; per-artifact calibrated values stay with the original seal. No new values inferred and no retuning. |
| Existing evaluation evidence | HISTORICAL preparation/incident evidence, not accepted official-test accuracy. Missing performance metrics remain absent, never zero. |
| Missing prerequisites | A separate defensible data/design proposal with independence, rights and exposure accounted for; original compromised confirmation is not repaired here. |
| Required independent review | Independent scientific/integrity review of any new proposal; no alteration of the original incident or protocol to obtain clearance. |
| Required owner authorization | Separate narrowly scoped evidence/access decision, then any new design/implementation/execution approval. Historical authorization does not carry forward. |
| Stop condition | Confirmed material cross-partition relation, unresolved provenance, silent exclusions/reassignment or attempt to label this previously opened set blind. |
| Current eligibility classification | **BLOCKED**; no official evaluation clearance. |

### PCB2 unresolved adjudication

| Required field | Evidence and prerequisite |
|---|---|
| Intended scientific question | Can the 56 frozen similarity pairs be resolved by genuine independent evidence without a material cross-partition relation? |
| Dataset/source and release identity | VisA_20220922; one-class metadata revision `2a692ab575001cbde74d402d897a7286086c6199`, as recorded in [dataset preparation](dataset-preparation.md). |
| Declared rights/licensing evidence | HISTORICAL source record declares VisA CC BY 4.0 and attribution. Current purpose-specific access/redistribution permission is UNVERIFIED here; public availability is not clearance. [Licensing boundaries](licensing.md). |
| Permitted activity | HISTORICAL integrity screening and review-material preparation only. No independent reviewer is established by this dossier; no new viewing/access or model activity. |
| Acquisition/source grouping and independence | Physical/acquisition identities unknown. Differing hashes, layout, illumination or damage alone do not prove distinct items/captures. |
| Historical exposure chronology | Integrity-only mechanical image decoding and later pair-material preparation occurred; generation is not independent human review. Preserve exposure records; [adjudication protocol](protocols/pcb2-adjudication-v1.json). |
| Training/validation/calibration/test splits | Official train/test membership preserved; no approved fitting/calibration/model-evaluation stage. 18 of 56 required pairs are cross-partition. |
| Label/mask reliability and adjudication | Labels used for inventory consistency, not similarity selection; mask payload/model evaluation outside the original integrity scope. All required nonexact decisions need the frozen independent criteria. |
| Exact-duplicate screening provenance | Historical byte/decoded-pixel identity checks in [integrity v1](protocols/pcb2-integrity-v1.json); exact relations are mechanical evidence only, not general physical independence. |
| Near-duplicate/similarity provenance | 56 unresolved required pairs; technical SSIM/translation evidence is correlated image evidence, not independent adjudication. Preserve frozen queue, criteria and unknowns. |
| Artifact/model identity | **NOT APPLICABLE** to integrity adjudication: no PCB2 fitted model or approved evaluation artifact is established. |
| Preprocessing identity | Frozen integrity fingerprint/refinement and adjudication image methods only; no model preprocessing is established or authorized. |
| Calibration and frozen threshold | **NOT APPLICABLE**: no fitted detector calibration/threshold; similarity criteria are not anomaly decision thresholds. |
| Existing evaluation evidence | **INCONCLUSIVE** integrity outcome; no model-accuracy metrics. The 56/18 counts are review inventory, not predictions. |
| Missing prerequisites | Authorized custody/access and genuine independent reviewer/provenance evidence for all mandatory pairs under the frozen criteria; no invented identity/date/attestation. |
| Required independent review | Independent blinded adjudication with custodian verification as specified by v1; metadata validation alone cannot authenticate a human reviewer. |
| Required owner authorization | Separate approval for exact review materials, access and actual independent adjudication. Any later training/evaluation requires a new scientific gate. |
| Stop condition | Any confirmed material train/test relation yields BLOCKED; unresolved pair, identity/resource limitation or absent independent evidence retains INCONCLUSIVE. |
| Current eligibility classification | **INCONCLUSIVE**; not eligible for model evaluation on this evidence. |

### MetalParts exploratory source-domain evidence

| Required field | Evidence and prerequisite |
|---|---|
| Intended scientific question | What limited, explicitly exploratory insight is supported by the existing source-normal methods and observed source comparison? |
| Dataset/source and release identity | AmazonScience/RobustAD, MetalParts, revision `82c642b2d18be60f2dab104f8dd4dea1f4ccfacb`; [3C integrity record](milestone-3c-robustad-integration.md). |
| Declared rights/licensing evidence | HISTORICAL 3C/model-card record declares CC BY 4.0. Rights for a future specific use and pretrained/derived assets need review; no new access or redistribution permission is established. [Licensing](licensing.md). |
| Permitted activity | HISTORICAL approved source-normal training and source diagnostics are recorded; no new data/artifact access or execution granted here. [3D](milestone-3d-metalparts-training.md), [3E](milestone-3e-metalparts-development-evaluation.md), [3F](milestone-3f-metalparts-feature-memory.md). |
| Acquisition/source grouping and independence | Exact source-content groups distinct; physical/capture identities and train/target independence remain unresolved. Source validation is already used for selection. |
| Historical exposure chronology | 144 fit and 36 validation normals used; 51 source anomalies and 36 validation normals evaluated diagnostically. 3C decoded target payloads for integrity; later 3D/3E/3F excluded target-model evaluation. |
| Training/validation/calibration/test splits | 180 source normals split 144/36; no independent calibration partition. Observed 87-image diagnostic set is not a held-out independent test; official target memberships unchanged. |
| Label/mask reliability and adjudication | Historical aligned native anomaly-mask checks and publisher labels; no new independent label adjudication. Small positive prevalence and observed labels limit interpretation. |
| Exact-duplicate screening provenance | Historical 3C byte/pixel checks and source grouping; no new verification. Zero exact duplicates do not establish target independence. |
| Near-duplicate/similarity provenance | Unresolved source/target similarity and capture relations in 3C persist. Existing source comparison cannot resolve them. |
| Artifact/model identity | `metalparts_autoencoder` checkpoint and `metalparts_feature_memory` bank/backbone/metadata pins remain authoritative; [AE card](metalparts-model-card.md), [feature card](metalparts-feature-model-card.md). No assets opened. |
| Preprocessing identity | Native 256×256 tiles/stride 192/trailing-edge coverage; AE overlap-average reconstruction/native MSE; feature tiles resized to 128×128 for frozen ResNet18, expanded/averaged maps. Recipes remain fixed. |
| Calibration and frozen threshold | **NOT APPLICABLE** to accepted recipes: no independent binary calibration or validated threshold. Threshold/decision remain null, not zero. |
| Existing evaluation evidence | HISTORICAL exploratory same 87-image source comparison; native pixel ranking uses 65,536-bin quantization. Better feature aggregate ranks are not independent superiority or comparable to resized Cashew metrics. |
| Missing prerequisites | For new exploratory work: exact question, permitted assets/rights/exposure and fixed protocol. For independent claims: target-data eligibility and an independent design remain missing. |
| Required independent review | Scientific review distinguishing exploratory reuse from confirmation; provenance/rights review; independent label/acquisition evidence as required by a new question. |
| Required owner authorization | Separate authorization for any data/artifact access, changed study or execution. No threshold fitting, target evaluation or deployment follows from source results. |
| Stop condition | Exploratory scores presented as independent accuracy/probability; reuse for tuning then confirmation; unresolved identity/rights or unauthorized target access. |
| Current eligibility classification | **HISTORICAL** exploratory-only evidence. Future activity eligibility is **UNVERIFIED**; independent target evaluation remains **NOT CLEARED**. |

### RobustAD independent target evaluation

| Required field | Evidence and prerequisite |
|---|---|
| Intended scientific question | Can a specific frozen method be evaluated independently on MetalParts target domains without material source/acquisition dependence? |
| Dataset/source and release identity | AmazonScience/RobustAD, MetalParts, revision `82c642b2d18be60f2dab104f8dd4dea1f4ccfacb`; [3C integrity record](milestone-3c-robustad-integration.md). |
| Declared rights/licensing evidence | HISTORICAL 3C/model-card record declares CC BY 4.0. Rights for a future specific use and pretrained/derived assets need review; no new access or redistribution permission is established. [Licensing](licensing.md). |
| Permitted activity | HISTORICAL 3C target access was for inventory/alignment/mask/hash/fingerprint integrity only. Independent target-model evaluation is NOT CLEARED; no new activity permitted here. |
| Acquisition/source grouping and independence | No adequate publisher acquisition/physical-item grouping established. 145 train/test0 basename correspondences are unresolved hints, not proof of sameness or independence. |
| Historical exposure chronology | 3C decoded 1,402 images and 502 masks including target payloads for integrity; do not call them never observed. No accepted target-model evaluation; later source experiments excluded target inference. |
| Training/validation/calibration/test splits | Original source train and target test0–test6 remain unchanged. Source-normal 144/36 split does not independently clear target groups or a calibration/test design. |
| Label/mask reliability and adjudication | Historical native shape/alignment/nonempty-mask checks; independent intended-use label/adjudication evidence remains UNVERIFIED. |
| Exact-duplicate screening provenance | Historical source hashes (LFS SHA-256 or Git blob SHA-1 plus local SHA-256) and native pixel identity checks; zero exact duplicates recorded, not independence proof. |
| Near-duplicate/similarity provenance | 626 train/target similarity candidates, 475 involving eligible normals; unresolved high-risk correspondence. [3C record](milestone-3c-robustad-integration.md) and [frozen protocol](protocols/robustad-metalparts-integrity-v1.json). |
| Artifact/model identity | No target study model/version is approved. Existing MetalParts artifact identities remain exploratory source methods, not target-cleared models. |
| Preprocessing identity | Prospective target-study recipe must be explicitly accepted before evaluation. Frozen 3C 512-tile planning was a proposal, distinct from later implemented 256-tile source methods; no automatic recipe transfer. |
| Calibration and frozen threshold | Existing MetalParts thresholds remain null. No target-data calibration or tuning authorized; a prospective calibration/selection plan needs separate legitimate development data. |
| Existing evaluation evidence | HISTORICAL integrity findings and exploratory source results only; independent target accuracy/robustness metrics are UNVERIFIED, not zero. |
| Missing prerequisites | Defensible acquisition/group independence, rights/custody, exposure/label review, study-specific artifact/preprocessing and predeclared metric/uncertainty protocol. |
| Required independent review | Independent provenance/scientific eligibility review; resolve high-risk relationships through new defensible evidence, not a repeated similarity loop or unsupported clearance. |
| Required owner authorization | Separate narrowly defined metadata/data access decision; only after eligibility, separate study design, artifact access and evaluation authorization. |
| Stop condition | Unresolved high-risk source/target relation, absent rights/groups, changed official membership, target-driven tuning or misrepresenting prior integrity exposure as blindness. |
| Current eligibility classification | **NOT CLEARED** for independent target evaluation. |

### Next scientific decision

First choose a precise question and authorize only the named, sanitized evidence
needed to assess its prerequisites. Prefer a bounded provenance/rights/acquisition
metadata review before any image or weight access. No such access is approved by
this dossier. If independent acquisition evidence or adjudication cannot be
obtained, retain the block; do not substitute another synthetic benchmark.

Any later data/artifact access, independent human review, real-worker implementation,
model execution, protocol freeze or evaluation needs its own owner decision and
scientific/security gate. A documentation or publication approval does not grant
those permissions. Preserve one-shot restrictions, all original official memberships,
model/threshold identities and the settled UI. Camera/API work, acceleration and
operational deployment remain separately gated in the [roadmap](roadmap.md).
