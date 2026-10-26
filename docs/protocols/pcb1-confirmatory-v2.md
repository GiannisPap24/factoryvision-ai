# pcb1 confirmatory design v2 — local preregistration, 24 October 2026

Planning only. No pcb1 download, extraction, decoding, inspection or evaluation is authorized by this document. The archive retained since Milestone 1B contains category bytes but pcb1 remains unextracted/unobserved. This dated local design is not a public, independently time-stamped registration.

## Amendment and enrolled families

Preserve [v1](pcb1-confirmatory-v1.md), SHA-256 `5e2a40897a880fcb2741d09ed8cc701d187d9001d15a0d3ecb130151fcdb0c1a`, unchanged. This pre-access amendment enrolls exactly three families: the compact student–teacher prototype, ResNet18 feature memory, and spatial mean reference. The accepted Cashew autoencoder is historical and is not enrolled or retrained. No additional architectures, layer/seed sweeps, ensembles or test-based winner selection are permitted.

The student recipe is [student-development-v1.json](student-development-v1.json), SHA-256 `fb5fd19e04aaa737de16b0483ca3f218af50aaf47707549dbc7c0345a42b7779`: teacher ImageNet V1 ResNet18 raw layer2/layer3; compact 112,128-parameter convolutional student; RGB128; channel unit-L2 normalization epsilon1e-6; mean squared unit-feature discrepancy; maximum16x16 image score and bilinear128 map. Teacher weights stay frozen, including BatchNorm. Student initialization is independent, seed42, fresh per category. No learned feature normalization is fitted.

Fresh pcb1 student training uses only its permitted normal training partition: Adam0.001, betas(0.9,0.999), epsilon1e-8, zero weight decay, batch8, four CPU threads, zero workers, deterministic seeded shuffled order. Maximum12 epochs, patience3, strict minimum normal validation loss; ties retain earlier checkpoint, min_delta0. Calibration is unavailable to optimization/selection. Stop on nonfinite losses/gradients, changed teacher, failed provenance, 600-second training budget or observed2GiB peak RAM; record an incomplete run rather than silently changing the recipe. Resource feasibility may be amended using normal data only, with a new dated seal before test access.

Feature memory and spatial mean use v1's recipes. Feature memory draws k=min(16,floor(8192/N_train)) deterministic patches from every training image; abort if k<1. This category-cap implementation and all model adapters require tests before execution. Fit each category-specific model/reference from scratch; do not reuse Cashew-fitted students, bank, mean or thresholds. The generic teacher alone is reused.

## Development partitions and calibration

Follow v1's deterministic seed42 sha256-group-v1 development assignment, 70/15/remainder of official normal training groups. Use verified physical/acquisition groups where available; otherwise exact-content groups with stable IDs are provisional. Preserve actual counts/fingerprints and the complete official test membership. Normal validation selects only the student checkpoint; it does not select architectures, layers, resolution, feature-memory settings or score rules. No pseudo-defect AP selects checkpoints or families.

Calibrate each selected model separately on disjoint normal calibration images: float64 linear q95, strict score > threshold. Higher image/map values indicate anomaly; class1 is anomalous. No per-image min-max normalization, test-derived calibration, pixel threshold, smoothing, ensemble or hyperparameter search. Small-sample exceedances do not guarantee an operational5% false-positive rate.

## Endpoints and uncertainty

This amendment explicitly replaces v1's feature-memory primary endpoint with **student–teacher pooled pixel average precision at128x128**. Report all three models regardless of outcome. Predeclared descriptive contrasts are student minus feature-memory pixel AP and student minus spatial-mean pixel AP. These are not post-test selection criteria or formal claims of superiority across factories.

Secondary endpoints for every family: image AUROC/AP, pooled pixel AUROC, and frozen-image-threshold TN/FP/FN/TP, precision, recall, F1 and normal FPR. Use scikit-learn non-interpolated AP. If either class is absent, AUROC/AP are null; zero-denominator rates are null. Report class counts/prevalence. Preserve exact scores/maps and all official samples.

RGB is full-frame bilinear antialiased resize128. Original nonzero mask IDs become binary, resized nearest-neighbor; normal masks are derived zeros. Include every pixel of every official image without subsampling. Report vanished masks and resizing limitations; do not adapt resolution or drop records after seeing them. AUPRO, Dice/IoU and pixel confusion remain deferred because no independently calibrated pixel operating point is specified.

Use v1's seed42,1,000 paired stratified image bootstrap resamples for image AUROC/AP contrasts, reported with95% percentile intervals. A validated paired image/group bootstrap for pixel AP may be included only if implemented and frozen before test opening; otherwise explicitly omit pixel intervals. Never bootstrap individual correlated pixels. With unknown physical groups, intervals are conditional on unverified independence. No p-value, repeated-testing selection or deployment acceptance claim is preregistered.

## Exact gate for opening official test assets

All of the following must be satisfied and recorded before the first official pcb1 test image/mask is opened:

1. Separate owner authorization for the category experiment; verified existing archive checksum and licensing provenance. Initially extract only normal training assets, retaining test assets unread in quarantine.
2. Verified normal-only membership, hashes, corruption/geometry checks, exact-content grouping and development split integrity; no unresolved confirmed cross-development leakage. Record acquisition-identity uncertainty.
3. A dated, hashed final protocol and implementation passing offline tests, permitted-normal checks, model/source/lock compatibility and resource feasibility. The historical3-million-pixel evaluator needs tested expansion to v1's exact8-million-pixel/observed4GiB ceiling; no silent subsampling. Any needed resource-plan amendment must precede test access.
4. All three category-specific fits complete without numerical/resource failures; student checkpoints selected only by normal validation; all normal calibration thresholds numerically frozen. Seal all fit IDs, model/bank/reference/teacher hashes, selected epoch, thresholds, configuration, source snapshots, dependency lock, preprocessing, metric and uncertainty code.
5. A persisted permission-to-open record verifies items1–4 and the protocol SHA. No test prediction, mask inspection, visual model selection or test-based enrollment decision has occurred.

Opening assets after that gate is **only for the predeclared integrity audit initially**, not immediate scoring. Apply v1's exact byte/decoded-pixel cross-partition duplicate checks and its full deterministic dHash/pHash/RGB candidate/refinement/visual-review rules unchanged. Use no model scores or defect masks to choose duplicate-review examples. Confirmed leakage or incomplete adjudication stops before official metrics; preserve all official samples and failures. A transparent amendment cannot turn a compromised test into clean independent confirmation.

Only when that integrity gate also passes may the one-shot evaluator run. Predetermined illustrations are the first four anomalous and first two normal test IDs, with display scales sealed from normal calibration. Record all outputs, hashes, batch settings and latency. Mark pcb1 observed afterwards and do not recycle it as an untouched selection benchmark.

## Interpretation and scope

This tests **fresh category-specific fitting on pcb1**, not zero-shot transfer from Cashew and not cross-factory generalization. Prior Cashew official results remain historical; new normal/synthetic diagnostics are exploratory. Unverified physical independence, dataset artifacts, domain shift, coarse feature maps and absent operational costs limit all conclusions. No deployment readiness threshold has been supplied.

The next milestone may implement and validate these gates using synthetic metadata and permitted normal data. It must not open pcb1 test assets merely because this design exists. Additional authorization and the recorded gates remain necessary.
