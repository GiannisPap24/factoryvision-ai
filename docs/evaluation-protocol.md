# Frozen Cashew evaluation

Milestone 2B evaluated the accepted autoencoder and one spatial mean baseline exactly once under [cashew-evaluation-v1.json](protocols/cashew-evaluation-v1.json). This explanatory document may evolve; the JSON and its companion digest are immutable. Actual results are in [the completion report](milestone-2b-completion.md) and [Model Card](model-card.md). Historical 2A reports describe the state before this evaluation.

## Identity and chronology

The local 2A commit is `e493c943f45f6d6053b43ef4fadeb872f2ea3f18`. The accepted epoch-12 `runs/cashew-ae-128-seed42/best.pt` has SHA-256 `da7e0caed8e365a08a3d301ec7e88536bc5df8e3f8b59f27635853e15ab4cdbc`. Its architecture, weights, transform, scoring and calibration were not changed.

Protocol SHA-256: `3b89b8447f00e478ab5d7793a50ca4cb92f3d0804bd6bee9fada564618adaa81`. Protocol freeze preceded the integrity gate, baseline fit/calibration and artifact freeze; all preceded the first test prediction. Exact UTC timestamps and source/artifact hashes are preserved in `runs/cashew-eval-v1/`. A one-shot marker prevents accidentally repeating official inference. Do not remove it to reuse this test set as untouched data.

## Membership and integrity

Official one-class VisA Cashew test membership remains 150 images: 50 normal and 100 anomalous. The seed-42 derived manifest retains 315 training, 67 validation and 68 calibration normals. Normal groups use `sha256-group-v1`; absent physical identities use stable sample identities provisionally. Dataset, parent and assignment fingerprints are in the frozen protocol.

The full 600-image/100-mask audit rechecked bytes, decoded pixels, alignment, mask identities and exact cross-partition duplicate detection against accepted records. All 7,946 earlier perceptual candidates received the predeclared full-image/crop refinement. The deterministic visual-review union contained 20 pairs and there were zero strong candidates. No duplicate acquisition was confirmed. Nothing was discarded, reassigned or relabeled. Similarity screening cannot establish physical-object/acquisition independence; unreviewed pairs remain a limitation. See the completion report for thresholds and counts.

## Models, transform and scores

RGB bytes become float32 [0,1] and resize to 128x128 using bilinear antialiasing. No augmentation, fitted normalization, cropping or test-driven preprocessing is used for inference. Original masks align with source geometry; nonzero mask classes become binary, then nearest-neighbor resize preserves binary values. Normal masks are derived zeros. Zero anomaly masks disappeared at the frozen resolution. Resizing can still alter small-defect boundaries and aspect ratio.

Both models use raw channel-mean squared RGB residuals as their spatial map and the spatial mean as their image score. Higher means more anomalous; class 1 is anomaly/defective pixel. There is no smoothing or per-image normalization. The AE reconstructs the input with the fixed network. The baseline predicts one spatial reference: sequential float64 accumulation of 315 sample-ID-ordered training normals, divided by count and cast to float32. No test/validation/calibration image contributes to that reference.

Each model uses its own 68-normal calibration scores, the 95th percentile with NumPy linear interpolation, and strict `score > threshold`. Equality is normal. AE threshold: **0.0031531278509646654**; mean-reference threshold: **0.017185416119173163**. The first was inherited from 2A; the second was frozen before any test prediction. Calibration is a coarse empirical false-alarm target, not a guarantee of 5% operational FPR.

## Metrics and uncertainty

Image AUROC uses scikit-learn 1.9.1 [roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html); AP uses [average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html). AP is the non-interpolated precision-recall summary, not trapezoidal PR area. If either binary class is absent, both ranking metrics are null. Confusion matrices are `[[TN, FP], [FN, TP]]`; precision, recall, F1 and normal FPR are null only when their respective denominators are zero.

Pixel AUROC/AP pool all 2,457,600 pixels across the same 150 images, including normal-image zeros. There are 27,672 defective pixels (about 1.126%). Computation is exact without subsampling and rejects more than 3,000,000 pixels. The process peak working set was about 602 MiB. Pixels are spatially correlated; pixel count does not establish statistical independence. No per-pixel confidence interval is claimed.

AUPRO was explicitly deferred before testing because a validated region-level implementation was outside this milestone. Dice/IoU and thresholded pixel confusion were not computed because no independent pixel threshold exists. The image threshold is not a pixel threshold.

Predeclared uncertainty uses 1,000 paired stratified image bootstrap resamples, seed 42, for AE-minus-mean image AUROC/AP with percentile 95% intervals. These intervals condition on image independence, which physical-item metadata has not verified. They do not account for acquisition clusters, seed variability or factory shift.

## Outputs and measurement

Test batches use sorted sample IDs, CPU, four Torch threads, batch size 16 and zero workers. `per-image.json` retains all IDs, labels, scores, predictions and mask-pixel counts. `spatial-maps.npz` preserves both raw maps and binary masks. `metrics.json`, `latency.json`, frozen artifacts, source snapshots and SHA-256 records make results replayable without new fitting or test inference.

Batch-1 latency uses the first normal validation image, preloaded, five warmups and 100 serial repetitions per model, timing forward plus score and validation overhead. Separate 30-repeat warm-cache timing covers decode/resize and derived normal mask. These measurements exclude startup and debugger overhead, and do not measure a camera or industrial pipeline.

Six examples were selected before predictions: the first four anomalous and first two normal test IDs. Each shows resized RGB, AE reconstruction, raw error displayed on a fixed blue-to-red [0,0.05] scale, and binary ground truth. Display clipping never changes metric scores. These are illustrative examples, not independent proof. Figures and original images remain Git-ignored.

## Boundary for further work

The official test set is now observed. Do not select models, epochs, thresholds or preprocessing on these results and call subsequent scores untouched confirmation. Any authorized follow-up needs an explicit exploratory/confirmatory distinction and independent acquisition/group confirmation. This milestone makes no unseen-factory, camera-shift, safety-critical or industrial readiness claim.
