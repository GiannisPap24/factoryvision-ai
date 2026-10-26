# Milestone 3B: Dataset selection and migration feasibility

Date: 2026-10-25. Scope: research, preservation checks and integration design.

**READY FOR DATASET ACQUISITION PLANNING**

Recommend **MVTec AD 2, Can**, for a bounded, non-commercial research pilot.
Acquisition planning is ready; acquisition, training and external submission are
not authorized by this milestone. Download sizes, category-specific distribution
and eligibility for a category-only server submission remain explicit gates.
The recommendation does not depend on getting a high benchmark score.

Throughout this report, **verified** means established from local checks or the
linked primary source; **estimate/proposal** means our planning judgment;
**unresolved** means evidence or owner action is still required. Source checks were
performed on the date above. No dataset images were acquired or inspected in 3B.

## A. Verified Milestone 3A checkpoint

The accepted application was reviewed and committed locally on `main`:

```text
bdaa52908869fb2300daced8d6aee11a38d06af3
feat: add local visual anomaly inspection application
```

Its parent is `bb820b874b98a014ba654b12c817c89e56773c39`. The approved repository-local
GitHub noreply identity was used. The commit contains the expected 24 files,
3,631 insertions and three deletions. The working tree was clean immediately
afterward. Phase 2 began only after verifying this checkpoint.

Staged review covered credentials, personal information, absolute paths and
generated artifacts. No credentials or unintended generated artifacts were staged.
Existing intentional local workspace instructions were retained; no new personal
paths were introduced by the application checkpoint. Data, checkpoints, caches,
environments and reports remain excluded. The exact committed file list is recorded
in checkpoint evidence (local evidence, not distributed: `reports/dataset-strategy-3b-v1/checkpoint.json`).
No remote is configured and nothing was pushed or published.

The older [3A completion report](milestone-3a-completion.md) describes its own
pre-commit state. It remains unchanged; this section records the later checkpoint.

## B. Current application preservation

Verified SHA-256 comparison: **2,492 baseline files checked, zero changed, zero
missing**. This includes the preserved historical evidence and application files
selected in the baseline inventory. See
preservation result (local evidence, not distributed: `reports/dataset-strategy-3b-v1/preservation-final.json`)
and baseline inventory (local evidence, not distributed: `reports/dataset-strategy-3b-v1/preservation-before.json`).

The four frozen Cashew methods remain available: autoencoder, spatial mean,
feature memory and student-teacher. Their artifact catalog, weights, preprocessing,
score rules and thresholds are unchanged. Each method passed actual CPU inference
on a fabricated RGB image, both in the workspace and in an isolated wheel install;
each returned a finite score and a 128 x 128 map. These checks exercised artifact
hash verification without reading real dataset images. Streamlit at
`http://127.0.0.1:8501` returned `ok` from its health endpoint. Offline AppTest
regressions passed as part of the full suite.

The browser and debugger executions documented in 3A remain historical evidence;
they are not represented as fresh 3B browser/debugger sessions. This documentation
change did not require another breakpoint run or a new browser connection claim.

Historical status is preserved: Cashew is an already-observed research benchmark;
PCB1 remains **BLOCKED** by confirmed source overlap; PCB2 remains **INCONCLUSIVE**
with 56 unresolved similarity candidates and no independent human adjudication.
Neither PCB investigation nor held-out model selection was reopened.

## C. Candidate comparison

The numerical facts below come from the linked dataset sources; feasibility and
integration ratings are our judgments. Image counts are not counts of independent
physical objects.

| Criterion | MVTec AD 2 | MVTec AD | Real-IAD | Existing VisA |
|---|---|---|---|---|
| Provenance and diversity | MVTec industrial acquisition; eight scenarios | MVTec; ten object and five texture categories | Industrial multi-camera acquisition; 30 products | Amazon research release; 12 classes, three domains |
| Usable normal data | 2,528 training and 302 validation images, summed from the category table | 3,629 normal training images; 60–391 per category | 99,721 normal images in the dataset, not all training or independent objects; use official normal-only split JSON | 9,621 normal images dataset-wide; preserve the existing Cashew fit/validation/calibration assignments |
| Anomalies and masks | Public annotated tests; hidden private annotations | 1,258 defective and 467 normal test images; pixel masks | 51,329 anomalous images; pixel annotations; five camera views per sample | 1,200 anomalous images; image and pixel annotations |
| Independence evidence | Published split design; paired private lighting variants; physical-unit ledger not established | Official train/test membership; physical-unit ledger not established | Capture/sample keys support grouping views; persistent physical identity across recapture remains unverified | Local investigations establish specific limitations; published membership alone was insufficient |
| Exact/near-duplicate screening | Feasible for one category; include all supplied image subsets | Feasible per category | Exact hashing feasible; group-aware, chunked near-neighbor work necessary at full scale | Existing tools/evidence retained; no new PCB review |
| Capture groups | Scene/lighting relationship must be represented | Need archive metadata or publisher evidence | Five views must remain together; view count must not inflate sample size | Missing provenance cannot be invented from filenames |
| Independent evaluation | External scoring service | Local scoring utilities | Local ADEval/sample-level evaluation | Local evaluation; previously observed Cashew tests |
| Hidden ground truth | Yes, two private test subsets | No hidden service identified in reviewed official sources | No hidden service identified in reviewed official sources | No hidden service identified in reviewed official sources |
| Access friction | Official form; later MVLogin for server | Official form | Hugging Face login, information sharing and terms acceptance | Public S3; no AWS account needed; already local |
| Storage | Exact release/category archive sizes unresolved; pilot budget in F | Current archive sizes unresolved; reserve 20 GiB provisionally for one-category work | Project reports about 53 GB at 1024 resolution, about 507 GB raw | Existing archive approximately 1.80 GiB; no new acquisition needed |
| CPU feasibility | One-category bounded pilot plausible; tiny defects make resolution costly | Strongest low-cost new baseline option | One category possible; full multi-view benchmark disproportionate now | Established CPU feasibility for the frozen small Cashew recipes |
| Resolution risk | High; native Can is wide, so square resizing is particularly lossy | Original image sides 700–1,024 pixels; resizing can still erase defects | 1,024 release preserves more detail than 128; five views multiply work | Existing 128-pixel limitations remain |
| Python/PyTorch integration | Official loader utilities; new split/hidden-label contract required | Simple image/mask adapter; normal validation split must be derived | Adapter plus view grouping and split-JSON interpretation | Already implemented; lowest migration cost |
| Dataset licence | CC BY-NC-SA 4.0 | CC BY-NC-SA 4.0 | Distribution specifies CC BY-NC-SA 4.0 and research-only use | CC BY 4.0 |
| GitHub portfolio | Code and synthetic demo preferred; real images/weights need rights review | Same | Same; website/code licences do not replace dataset terms | Attribution-compatible image use is possible, subject to other rights and owner publication approval |
| Future commercial product | Separate permission or different authorized training data required | Same | Same | Licence permits commercial use, but engineering validity and other rights still require assessment |

Sources: [AD 2 overview][ad2], [AD 2 category counts][ad2table],
[MVTec AD paper][adpaper], [MVTec AD access/licence][ad],
[Real-IAD paper][realpaper], [Real-IAD project][real],
[Real-IAD distribution][realhf], [VisA registry][visa].

Real-IAD supplies normal-only UIAD and separate contaminated-training FUIAD
settings. Its official JSON membership, per-category usable counts and sample keys
would have to be verified after authorized access; a random image split would be
invalid for multi-view evaluation. Real acquired defects can be deliberately
introduced during collection: none of these benchmark descriptions establishes
representative factory defect prevalence. [Real-IAD collection/protocol][realpaper].

## D. Authoritative references and evidence boundaries

The comparison links the dataset owners, original research and licence texts,
rather than third-party download mirrors. Further sources used:

- [AD 2 research paper][ad2paper]: scene pairing, acquisition, resolution and evaluation design.
- [MVTec benchmark][server] and its [public application bundle][serverjs]: current visible rule text, file formats and metric names. Only unauthenticated public HTML/JavaScript was fetched; no account or authenticated API was used.
- [CC BY-NC-SA 4.0][nc] and [CC BY 4.0][by]: permitted uses and obligations.
- [AMD versioned Windows matrix][amd]: evidence of a supported hardware/software combination, with an explicit version limitation described in G.

Public page snapshots are retained locally under
`reports/dataset-strategy-3b-v1/public-source-notes/`. Some web-reader requests failed;
the AD 2 statistics table and benchmark bundle were successfully read through
ordinary unauthenticated HTTPS. This does not establish access to dataset files.
No gated file was accessed, and no form, terms acceptance or newsletter request
was submitted. Missing archive information is reported as unresolved.

## E. Licensing, portfolio and commercial strategy

**Verified:** both MVTec datasets explicitly prohibit commercial dataset use under
CC BY-NC-SA 4.0. Real-IAD's authoritative distribution also declares that licence
and research-only use. Its project website's CC BY-SA footer concerns the website;
the linked implementation's licence does not license the dataset.
[MVTec AD 2][ad2], [MVTec AD][ad], [Real-IAD distribution][realhf].

CC BY-NC-SA allows sharing/adaptation subject to attribution, non-commercial use
and share-alike for shared adaptations. Preserve supplied notices, credit creators,
link the licence/source and identify modifications. Do not add restrictions that
remove granted rights. VisA's CC BY permits commercial sharing/adaptation with
attribution and modification notices. Neither licence supplies every possible
privacy, trademark or other third-party permission. [NC-SA terms][nc], [BY terms][by].

Our proposed publication policy is to release original engineering code,
methodology, aggregate measurements and synthetic demonstrations first. A public
GitHub portfolio is not automatically a determination that every image use is
non-commercial. Before including NC dataset images, overlays or promotional demo
screenshots, the owner must assess the intended use and seek permission when
uncertain. Link to official acquisition instead of redistributing archives.

Training-derived weights need a separate provenance and licensing review. This
report does not resolve whether particular weights are adaptations, or assert that
share-alike automatically applies to every model or to unrelated code. Keep NC
research artifacts segregated and exclude them from a commercial release pending
appropriate rights clearance. A future MIT licence on our own code would not
override third-party data, utility-code or pretrained-backbone obligations.

For a product, obtain independently authorized customer/owned acquisition data or
written commercial permissions, establish unit/lot/time provenance, and train a
new, separately identified model. Do not silently promote this research checkpoint
into a commercial model. Publication of data, weights or images remains a separate
owner action.

## F. Access, distribution and storage planning

**Verified access:** MVTec's current dataset pages lead to a form. The AD 2 form
requests first name, last name and email, and acknowledgement of non-commercial
use; position and newsletter fields are optional. The owner must handle personal
information and acceptance. Dataset access and later benchmark MVLogin access are
separate steps. Official AD 2 code utilities include a loader, output-structure
checks and runtime/memory snippets; their downloadable archive is linked by the
owner's page. Its code licence must be inspected before incorporating code.
[Official AD 2 access/utilities][ad2].

**Unresolved:** full and Can-only compressed byte counts, whether category-only
archives are offered after the form, expanded size, release checksums and any
additional distribution conditions. These facts were not inferred from mirrors or
guessed direct URLs. MVTec AD's current archive sizes/category download options
are likewise unverified. This is enough to plan owner acquisition, not enough to
start a downloader.

Real-IAD's project reports approximately 53 GB for the 1,024-resolution release
and 507 GB for raw images. The distribution currently displays 622 GB total across
its repository, which is not the size of the desired single release. It requires
login and agreement to share account information; an automatically approved
request still requires that consent. [Project download description][real],
[distribution gate][realhf]. A provisional full 1,024-release workspace would
reserve 120–160 GiB for archive, extraction and working files; this is an estimate,
not a verified compression ratio. Do not clone/download the entire repository.

**Can planning estimate:** reserve **30 GiB** initially, with a hard acquisition
gate based on `archive + expanded files + bounded outputs + 25% headroom`. At its
published dimensions, 1,262 RGB images occupy approximately 8.1 GiB as uncompressed
8-bit pixel payload alone. This is neither archive size nor actual on-disk size.
A complete native float32 score-map set for 642 private images is approximately
5.5 GiB per method before compression. Stream outputs and retain only planned
artifacts. If the real inventory exceeds the allowance, stop for a revised plan;
do not escalate silently to all categories. Counts/dimensions: [Table 3][ad2table].

Download time is unresolved because archive size and measured throughput are
unknown. For planning, `minutes = bytes * 8 / bits_per_second / 60`; 10 decimal GB
at 100 Mbps is about 13.3 minutes before overhead. Owner acquisition must record
the actual offered filename, size, release, terms and checksum before transfer.

VisA needs no download. Preserve its existing archive and selected extraction;
the official distribution remains [public S3 without an AWS account][visa].

## G. Hardware and preprocessing feasibility

Verified locally: Ryzen 7 9700X, eight cores/16 logical processors; Windows 11
Home; about 31.61 GiB OS-visible RAM and 20.11 GiB free at inspection; about
484.56 GiB disk free before isolated-install work. Python 3.12.15 and
PyTorch 2.13.0+cpu remain the active runtime. The RX 9060 XT is detected.
Approximately 16 GiB VRAM remains the established hardware baseline; the current
32-bit WMI AdapterRAM field is unsuitable for confirming that capacity.
Hardware snapshot (local evidence, not distributed: `reports/dataset-strategy-3b-v1/hardware.json`).

**Proposal:** use CPU float32, four Torch threads, `num_workers=0`, batch size two
initially, and one decoded image/tile batch at a time. Cap the experiment process
at 8 GiB resident memory through monitoring/abort controls; keep at least 8 GiB
system RAM available. Do not predecode the dataset into RAM. Record decode time
separately from model time, peak process memory, cold/warm latency and total fit
time. Existing 128-pixel timing is evidence of functionality, not a prediction of
high-resolution throughput.

Pixel-work scaling alone is 4x at 256 versus 128 and 16x at 512 versus 128;
activations, decoding, nearest-neighbor search and tiling prevent direct runtime
extrapolation. A Can RGB float32 image is about 26 MiB before model activations.
A 128-pixel full-width reduction maps a five-pixel native feature to about 0.29
pixels; even a 256-wide resize maps it to about 0.57 pixels. No claim of tiny-defect
preservation is defensible at those full-image sizes.

Use a 256-pixel, aspect-preserving, padded full-image recipe for the first
engineering baseline, without center cropping. Keep original dimensions and
padding transforms so maps can be restored. Treat this as a deliberately limited
resolution baseline. Design a separate native 512 x 512 tile recipe with 128-pixel
overlap, border coverage, deterministic weighted stitching and no tile-level
split assignment. For Can this is approximately 18 tiles per image; record the
actual edge policy. Test transform/stitch correctness using synthetic defects
before real evaluation. Select one recipe using only fit/validation resource
and normal-image evidence, then freeze it. Do not select resolution on test masks.

Bound a feature bank to at most 10,000 deterministic fit-only descriptors and
chunk distance queries, initially 256 descriptors per chunk. A 10,000 x 384
float32 bank is about 14.6 MiB; a 256 x 10,000 distance matrix about 9.8 MiB,
excluding activations and library workspaces. Do not materialize every training
patch embedding or an all-pairs pixel distance matrix.

Planning budgets, not measured predictions: a normal-only decode/forward smoke
should finish within five minutes; allow at most 30 minutes per full-image fit
and two hours per tiled fit before stopping to reassess. Actual high-resolution
training durations remain unknown. No real training benchmark was run in 3B.

Future AMD acceleration is plausible, but the current CPU wheel cannot use it.
AMD's **7.2.1** matrix explicitly includes RX 9060 XT/Windows 11 with PyTorch 2.9
and Python 3.12; that documentation now redirects readers toward unified newer
documentation. This is version-specific eligibility, not validation of our
PyTorch 2.13 environment or every backward operator. A future separate environment
must verify the then-current OS/driver/framework matrix, forward/backward tests,
numerical behavior and memory before adoption. No GPU software was installed.
[AMD versioned support matrix][amd].

## H. Evaluation integrity strategy

This is a proposed protocol, to be serialized and sealed before any acquisition
inspection that could affect selection. The private server reduces label-driven
tuning; it does not prove image or physical-object independence.

1. **Inventory before training.** Preserve official membership and filenames in
   a read-only source inventory. Verify archive digest, safe extraction paths,
   expected counts, decodability, dimensions, channels, mask alignment and mask
   availability. Record release and acquisition terms. Hidden labels/masks must
   be unknown, never fabricated as normal or represented by zero masks.
2. **Group before partitioning.** Use publisher-provided physical-unit, scene,
   capture, lot and lighting keys when available; record which are absent.
   AD 2's private regular and mixed subsets depict the same scenes under lighting
   variants. Keep those paired within the held-out family and do not count them
   as independent replication. [Dataset design][ad2paper].
3. **Objective screen.** Hash every file and decoded canonical pixels; compare
   fitting, validation/calibration and all test subsets. Predeclare image
   canonicalization and near-duplicate candidate rules. Initial proposed rule:
   64-bit perceptual-hash Hamming distance <= 4, plus each image's nearest
   cross-partition neighbor under a fixed, versioned image descriptor. Use
   deterministic ordering and chunked comparisons. These are candidate detectors,
   not proof of duplicate identity or proof of completeness. Test their behavior
   on synthetic resize/re-encode/lighting controls before sealing the rules.
4. **Proportionate disposition.** Confirmed cross-development/test identical
   content or shared acquisition unit is a hard stop. Expected paired private
   lighting views are documented, not falsely treated as leakage into training.
   Missing metadata is an explicit limitation, not an invented independence pass.
   Any high-risk unresolved cross-partition candidate blocks a confirmatory claim.
   At most one bounded desk review of the first 20 ranked candidates is allowed
   to assess evidence needs; the agent is not an independent human reviewer.
   More candidates, or unresolved physical identity, triggers a stop and a request
   for new publisher/group evidence. Do not keep changing cutoffs or repeatedly
   generating contact sheets. There is no automatic waiver because a reviewer
   is unavailable. Mechanical checks can pass with provenance uncertainty only
   as an explicitly qualified research feasibility result.
5. **Exposure control.** Integrity tooling may process test pixels solely for
   hash/group checks, with a separate exposure log and no training features,
   statistics, normalizations or anomaly-score feedback. Avoid displaying test
   images. If visual inspection becomes necessary, record the affected IDs and
   downgrade any claim of image-blind evaluation. Hidden-label status alone is
   not image blindness.

### Partitions and freezing

Proposed Can assignment uses all 412 official normal training images for fitting.
If the 46 validation images are independent single-image groups, deterministically
assign 14 to model validation and 32 to calibration using a sealed seed/hash rule.
With multi-image groups, preserve groups and report actual counts; never split a
group to force these numbers. Keep public and both private test memberships intact.
There must be at least 19 independent calibration units for the proposed finite
5% calibration rule; otherwise stop and revise the protocol before test exposure.
Published counts: [AD 2 Table 3][ad2table].

Validation chooses a checkpoint using a predetermined normal-only criterion and
bounded training budget. Calibration is not used to choose architecture,
preprocessing, epoch count or memory-bank size. Neither public test labels nor
private test pixels enter fitting or model selection. Hold the public annotated
subset for one final local evaluation, even though the publisher permits local
development use. This is our stricter experimental choice.

For each model, define the image score as the maximum of its final valid-pixel
anomaly map after the frozen restore/stitch operation. Define
`k = ceil((n + 1) * 0.95)` and use the kth ascending calibration image score;
with n=32, this is the maximum. Flag only `score > threshold`; use the same scalar
to binarize pixel maps. This yields an image-alarm calibration rule, **not** a
pixel-level false-positive guarantee. Exchangeability is an assumption and can
fail under lighting shift. Report calibration size, ties and test uncertainty;
do not infer zero production risk from no observed calibration alarms.

Freeze artifact/manifest/code/environment hashes, seeds, transforms, model family,
fit IDs, score aggregation, calibration values and all metric settings before
opening test labels or producing an external submission. Per-image display
normalization is not the quantitative anomaly map. No test-dependent min/max
normalization or threshold adjustment is allowed.

### Metrics and submission limits

Predeclare local public-test image AUROC/AP, confusion counts, precision, recall,
F1 and normal-image false-positive rate at the fixed threshold. Pixel measures:
AUROC/AP, AU-PRO integrated to false-positive rate 0.05, and fixed-threshold pixel
F1. Report denominators, undefined metrics, connectivity, interpolation and
implementation version. Treat masks consistently; normal images have no positive
regions. Use image/scene-cluster bootstrap intervals, never independent-pixel
bootstrap. Separate regular and changed-light conditions where metadata permits.

**Verified server distinction:** the public application reports AU-PRO_0.05,
classification F1 and segmentation F1 for regular and mixed private sets. Its FAQ
does not offer AUROC evaluation. Local AUROC/AP must not be presented as private
server results. Current submission instructions specify a `.tar.gz`, float16 TIFF
continuous maps and optional complete PNG binary maps; model-sized maps can be
bilinearly enlarged by the server. Confirm current checker behavior before export.
[Benchmark application][server], [public rule bundle][serverjs].

The current server allows **two uploads within seven days**, counting upload starts;
uploading starts evaluation. The documented layout includes every category and
requires all files. Category-only acceptance is therefore **unverified**, not
assumed. Registration uses MVLogin. [Public server instructions][serverjs].

Our proposed feedback budget is stricter: one frozen submission per method,
two methods total for this protocol, with no tuning after either result. A failed
upload still enters our attempt log; any replacement needs an explicitly recorded
reason and owner authorization. The seven-day allowance is not permission for
repeated model selection. Any stricter current server rule prevails.

External submission is a later separately authorized action, including account
creation, upload and any public leaderboard choice. If all-category completeness
is required, keep the one-category pilot local and defer server scoring until a
separately approved wider benchmark. Never fabricate missing-category predictions
or expand acquisition silently. A local public-test pilot is not an official
private-server benchmark result.

## I. Recommended target dataset

**MVTec AD 2** best addresses this project's next research need: a fresh benchmark
with a route to independently held labels and lighting-shift evaluation. It is
selected for evaluation design and a bounded category scope, not because it is
larger or guaranteed duplicate-free. Its external evaluation route is conditional
on the access and completeness gates in F/H.

MVTec AD is an easier CPU/integration alternative but offers less improvement in
label isolation. Real-IAD's multi-view data is valuable, but adds group handling,
storage and acquisition burden without a verified hidden evaluator. VisA remains
valuable historical evidence and the working application backend; it cannot
provide a fresh confirmation by rerunning already observed results.

This recommendation serves non-commercial research. None of the NC candidates
is recommended as the product's automatically deployable training source.

## J. Recommended initial category and pilot

**Can** is a single RGB object with a useful normal training count. Compared with
bulk-object or transparent-object scenarios, it avoids those particular modeling
complications. Reflections and low-contrast/small defects remain difficult; this
is not a claim that Can is the easiest accuracy benchmark. Its acquisition uses
diffuse illumination and additional test spotlights. [Acquisition description][ad2paper].

Verified category inventory: **412 train, 46 validation, 162 public test
(72 normal/90 anomalous), 321 private regular and 321 private mixed**. Dimensions:
**2,232 x 1,024**. Private class counts are published in aggregate but individual
labels remain withheld. The two private sets are paired scenes; 642 images are
not 642 independent physical samples. [Category table][ad2table].

Proposed first two families are a **compact convolutional autoencoder** and
**frozen pretrained feature memory**. They reuse existing reconstruction and
descriptor infrastructure while testing different error mechanisms. Fit only
normal training data. Use new Can artifacts and new preprocessing recipes; never
fine-tune or reinterpret the frozen Cashew weights. Record the pretrained
backbone source/hash/licence and acknowledge unverified pretraining overlap.
Student-teacher remains a possible later experiment, not a third pilot model.

Reproducibility requires source inventory/digests, group and partition manifests,
fixed seeds and deterministic sampling, environment lock, code commit, model
configuration, training log, stopping reason, resource measurements, thresholds
and result hashes. A successful pilot means a reproducible bounded execution and
an honestly reported held-out result; no minimum accuracy is invented here.
Integrity failure, unavailable rights, or resource-budget failure is a valid
negative feasibility outcome.

## K. Minimal non-destructive integration architecture

**Design only; no implementation in 3B.** The existing v1 domain contracts allow
VisA/synthetic providers, train/test source splits and known normal/anomaly labels.
They validate pinned VisA provenance. AD 2 cannot safely be inserted by renaming it
VisA or labelling its private images normal.

Add a narrow versioned provider boundary: inventory source items and map them to
canonical image/mask/sample records. Retain v1 parsing, canonical serialization,
hashes and validation unchanged. Introduce a v2 manifest/envelope only for new
providers, with explicit release, provider, category, original split, label
availability, mask availability, group/scene/lighting IDs, source digest and
licence/provenance reference. Distinguish an unavailable mask from a verified
normal empty mask. Preserve exact official split names alongside internal
fit/validation/calibration/test roles.

Reuse contained-path resolution, decoding, split-integrity routines and model
components. A small adapter feeds provider-neutral training/evaluation inputs;
do not fork a second copy of the training pipeline. The present fixed-resolution
checks must remain strict for historical recipes. Add explicit new resolution
recipes and shape/round-trip tests for Can, including mask-nearest-neighbor
transforms, score-map interpolation, padding exclusion and tile stitching.

Bind every new model to provider + release + category + manifest hash + recipe
version + fit-ID digest + calibration digest. Reject cross-category/backbone
mismatches before inference. Proposed ignored paths are
`data/raw/mvtec-ad2/<release>/can/` and
`runs/mvtec-ad2/<release>/can/<recipe>/<run-id>/`; refuse overwrites. Keep original
archives immutable and provenance/licence notices attached to each release.

The existing Streamlit Cashew catalog remains the default. Add a new model entry
only after a later accepted experiment supplies a verified artifact and supported
preprocessing adapter. No automatic model detection, replacement of Cashew
thresholds, or new upload/export mechanism is part of this milestone.

## L. Future implementation sequence

1. Owner confirms non-commercial research scope and handles the MVTec access
   form/terms. Inspect the offered release and category inventory without starting
   a transfer; settle sizes, checksum availability and the acquisition budget.
2. Separately authorize the exact acquisition. Download only the agreed release
   scope; record hashes and extraction inventory in a new ignored directory.
3. Implement/test the minimal provider contract and sealed integrity protocol
   using synthetic fixtures first. Run the objective integrity gate on acquired
   files; document provenance uncertainty and stop conditions.
4. Separately authorize the bounded two-model experiment. Run normal-only resource
   probes, freeze a recipe, fit, validate and calibrate with disjoint groups.
5. Seal the experiment, run one public-test evaluation and report both methods.
   Resolve server completeness requirements before requesting any external
   evaluation authorization. Do not automatically proceed through these steps.

## M. Actual verification results

| Check | Result |
|---|---|
| Full offline pytest | **496 passed, one existing Windows symlink-privilege skip**, 43.02 seconds |
| Coverage | **98.577929% combined**, 98.939513% statement, 97.628866% branch; unchanged from 3A at reported precision |
| Ruff lint | Passed |
| Ruff formatting | Passed; 111 files already formatted |
| `uv lock --check --offline` | Passed |
| Workspace dependency compatibility | Passed; 61 packages |
| Offline wheel and sdist build | Passed, under `dist/3b/` |
| Archive inspection | Wheel 56 members, sdist 159; catalog present; no datasets, weights, reports, environments or caches |
| Fresh isolated runtime/app install | Passed offline; wheel installed non-editably; 49 packages compatible |
| Isolated `python -I` inference | Imported from site-packages; all four actual frozen models passed synthetic-image inference |
| Streamlit regression | Full offline AppTest suite passed; running loopback service health `ok` |
| VS Code workspace | Four JSON configurations valid; 36 tasks, nine debug configurations; referenced executable/script paths exist; no folder-open startup task |
| Official Codex extension | Installed `openai.chatgpt@26.1021.22027`; Python, Pylance, debugpy and Ruff also verified installed |
| Preservation | 2,492 inventoried files unchanged; none missing |

Evidence: quality commands/results (local evidence, not distributed: `reports/dataset-strategy-3b-v1/baseline-quality.json`),
coverage (local evidence, not distributed: `reports/dataset-strategy-3b-v1/coverage.json`),
archive inspection (local evidence, not distributed: `reports/dataset-strategy-3b-v1/archive-inspection.json`),
isolated install (local evidence, not distributed: `reports/dataset-strategy-3b-v1/isolated-install.json`),
workspace checks (local evidence, not distributed: `reports/dataset-strategy-3b-v1/workspace-check.json`).

Verification harness corrections were needed: the first isolated sync lacked the
project-local Python path; an initial archive filter incorrectly treated the
Python `factoryvision/data` source package as a dataset; an initial inference
probe iterated catalog metadata instead of model IDs. These were corrected in
temporary verification commands, with successful checks above. No production
code, tests, dependency lock or historical evidence was changed to obtain a pass.
The baseline suite was not pointlessly repeated after this documentation-only
addition; the tested application sources remain byte-identical.

## N. Exact files created or modified

**3B deliverable:** created `docs/milestone-3b-dataset-strategy.md` (this file).
No existing tracked project file was modified by 3B.

The Phase 1 commit contains the already accepted 3A changes; its exact 24-path list
is in A's checkpoint evidence. Those files were not subsequently edited.

Local ignored verification artifacts were created under
`reports/dataset-strategy-3b-v1/`; an exact file/digest inventory is retained as
`evidence-index.json` there (excluding that index itself). Package artifacts are
`dist/3b/factoryvision_ai-0.1.0-py3-none-any.whl` and
`dist/3b/factoryvision_ai-0.1.0.tar.gz`, plus uv's ignored directory marker.
The disposable installed environment is `.tmp/wheel-env-3b/`. Existing ignored
test scratch/coverage/cache files were refreshed by standard checks. No generated
artifact is proposed for Git publication. This strategy document is a repository
document; no package manifest was changed merely to include it in the sdist.

## O. Git status

At completion, `main` points to the full checkpoint hash in A. The index is empty;
the only new non-ignored path is `docs/milestone-3b-dataset-strategy.md`. The 3B
document is left uncommitted. Existing tracked files have no diff. No push,
publication or remote configuration occurred.

## P. Remaining risks and owner decisions

- Accept a **non-commercial research** pilot; commercial training/deployment
  requires different rights or data. Real-image portfolio use and weight release
  require a separate decision.
- Handle registration/personal information and confirm the offered release,
  category-only availability, sizes, checksums and terms before authorizing a
  download. The proposed 30 GiB allowance is not a measured archive size.
- Establish available physical/capture identity evidence after access. This
  strategy does not claim a clean dataset in advance or replace independent review.
- Accept that 46 validation images provide limited normal calibration evidence,
  and that changed illumination can invalidate exchangeability assumptions.
- Confirm whether category-only external evaluation is accepted. If not, accept
  a local-only pilot initially; broader benchmarking needs a new scope decision.
- Reassess CPU budgets from normal-only measurements. High-resolution fit time,
  tiny-defect performance and industrial usefulness remain unmeasured.

These are acquisition/experiment gates, not reasons to restart PCB adjudication or
to claim that the dataset choice is unresolved. No pending permission question
was needed to finish the authorized research and report.

## Q. Next milestone acceptance criteria

The next milestone should be **one-category acquisition and integrity preparation**,
with explicit scope before any transfer. It is accepted only when:

1. Owner-approved source, terms, release, exact acquisition scope and byte/storage
   budget are recorded; no automatic account creation or external submission.
2. Authorized files have verified local digests, safe extraction, inventory,
   expected split counts and preserved source provenance.
3. Provider/label/mask/group contracts handle hidden labels correctly and retain
   all v1 VisA fixtures and historical hashes unchanged.
4. The sealed objective integrity gate produces an explicit pass-with-limitations,
   inconclusive or blocked disposition. No unresolved high-risk overlap is waived.
5. A reproducible fit/validation/calibration assignment and CPU recipe proposal
   are prepared without opening test labels for model selection.
6. All four Cashew methods and the application still pass existing checks; no
   training, threshold changes or server uploads occur without their later scope.

**Milestone 3B stops here. READY FOR DATASET ACQUISITION PLANNING.**

[ad2]: https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2
[ad2paper]: https://link.springer.com/article/10.1007/s11263-026-02743-0
[ad2table]: https://link.springer.com/article/10.1007/s11263-026-02743-0/tables/3
[ad]: https://www.mvtec.com/research-teaching/datasets/mvtec-ad
[adpaper]: https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/05_research_teaching/datasets/mvtec_ad.pdf
[real]: https://realiad4ad.github.io/Real-IAD/
[realpaper]: https://arxiv.org/html/2403.12580v1
[realhf]: https://huggingface.co/datasets/Real-IAD/Real-IAD
[visa]: https://registry.opendata.aws/visa/
[server]: https://benchmark.mvtec.com/
[serverjs]: https://benchmark.mvtec.com/assets/index-But0_l4L.js
[nc]: https://creativecommons.org/licenses/by-nc-sa/4.0/
[by]: https://creativecommons.org/licenses/by/4.0/
[amd]: https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/compatibility/compatibilityrad/windows/windows_compatibility.html
