# MetalParts feature-memory exploratory comparator — v1

Milestone 3F. **EXPLORATORY SOURCE-DOMAIN COMPARISON — NOT INDEPENDENT TEST PERFORMANCE**.
The 51 source anomalies and their autoencoder results have already been observed.
This is a single exploratory candidate, not a blind confirmation or parameter search.

## Frozen design before fitting

Reuse the existing trusted ResNet18 extractor, ImageNet normalization, layer2/3
384-dimensional descriptors, 3×3 local average pooling, bilinear feature alignment,
RMS Euclidean 1-nearest-neighbor distances and maximum descriptor image score.
Pretrained file: `data/pretrained/resnet18-f37072fd.pth`, 46,830,571 bytes, SHA-256
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
No downloads, backbone changes, gradients, fitted feature normalization, whitening,
projection, coreset search, Gaussian smoothing or score reweighting.

This is PatchCore-inspired, not exact PatchCore: the paper motivates a nominal
patch-feature memory, whereas this candidate keeps the existing smaller ResNet18
and deterministic sampling, not a greedy coreset. Reference:
[Roth et al., CVPR 2022](https://arxiv.org/abs/2106.08265).

Use the accepted native 256-pixel grid and stride 192, appending exact trailing
edges. Each native crop is bilinearly antialiased resized to 128×128 using the
existing Cashew preprocessing, retaining its frozen extractor geometry. No entire
source parent is squeezed to 128×128. At 2681×1500 there are 112 full-grid tiles.
Fitting uses the same 12 spatially spread native tile coordinates per parent as
the accepted 3D normal fit. Parent and native x/y remain attached to every tile.

Use exactly the accepted 144 fitting normals, stable ID order; verify the 3D split
and source hashes. No validation/anomaly/test image enters fitting. Each of 1,728
tiles yields 256 candidate descriptors. Select four per tile: first four indices
of a Torch randperm(256) seeded from the first eight big-endian SHA-256 bytes of
`42|parent-ID-yNNNNN-xNNNNN`, then sorted. This preserves the existing deterministic
sampling mechanism. Four follows from floor(8192/(144*12)), so all parents and
selected tiles contribute equally within the existing 8,192-descriptor cap.
The planned bank is 6,912×384 float32 (10,616,832 tensor bytes). No Cashew bank
entry is reused. Fit batch one, four CPU threads, zero workers, deterministic
algorithms, all backbone parameters and BatchNorm buffers frozen. Re-extract the
same selected fitting features once for exact determinism verification; this is
verification of the one bank, not a second candidate or selection run.

Inference uses every native grid tile, batch one. For each tile, the existing
FeatureMemory produces a 16×16 descriptor-distance map, bilinearly upsamples it
to 128×128; then bilinear align_corners=False interpolation returns it to 256×256.
Uniformly average native overlapping distance maps and assert complete coverage.
The image score is the maximum raw descriptor distance across all full-grid tiles,
before interpolation/overlap averaging, consistent with the existing feature
method's maximum rule. This deliberately differs from the autoencoder's mean MSE;
never compare their raw values as common units. No threshold or binary decision.
Minimum upload geometry remains 256 pixels per side; existing upload caps apply.

Nearest-neighbor blocks reuse 256 query × 512 bank entries, hard-bounded by the
existing tested implementation. A single float32 pairwise block is at most 0.5 MiB
(the existing permitted maximum 512×512 is 1 MiB). Never allocate the full
all-image/all-bank distance matrix. Each phase has an 8 GiB peak working-set limit;
fit including determinism replay at most 900 seconds, normal validation at most
900 seconds, full comparison at most 1,800 seconds. Check before each tile; stop
and preserve partial evidence on exhaustion, without reducing coverage or tuning.

## Provenance, normal diagnostics and one comparison

Seal this document, new code, existing reused code, 3D split, 3E source inventory,
3E score/map hashes, exact fitting IDs/tile identities and pretrained weights
before fitting. Save a separate project-local tensor bank and metadata with
category, revision, split/protocol hashes, selections, backbone hash, null
threshold and source-only status. Never overwrite a fit/evaluation directory.
Bounded local weights_only loading verifies hash, dtype, dimensions, finite values,
exact category/source/recipe/membership and provenance. The accepted AE checkpoint,
all old protocols, maps, metrics and model cards remain byte-preserved.

After fitting, score all 36 permitted validation normals with the frozen pipeline;
verify finite native maps, cached/independent-reload equality on the first normal,
score distribution, timings and process peak memory. Do not calibrate on these
already-used normals. Cache these exact maps and scores for the comparison rather
than rerunning them. This normal phase cannot change the fixed model or settings.

Then execute one comparison on the identical 87 IDs in the 3E order: cached 36
normal feature results plus 51 newly scored source anomalies. Verify source masks
as native-aligned 0/255, binarize >0, and derive zero masks for normals. No mask
interpolation for metrics. Reuse original AE maps/scores and metrics; no AE forward
pass or retraining is needed. Report exact image AUROC and non-interpolated AP,
all-native-pixel AUROC/AP, class prevalence, latency, artifact sizes and memory.

Pixel quantization preserves EXACTLY the 3E rule and bin width:
`floor(float64(score)*65535)`. AE MSE is in [0,1]; feature distance can exceed 1.
Extend the histogram's nonnegative integer index range without clipping, scaling,
normalizing or changing width. Allocate only up to observed maximum bin, bounded
by a predeclared 1,048,576-bin resource ceiling (16 MiB for two int64 histograms).
Exceeding that budget is an explicit resource failure, never clipping or retuning.
Metric tie handling and definitions are identical to 3E and verified against its
histogram on [0,1] and scikit-learn on synthetic distances above 1. Metrics remain
approximate for unquantized rankings; same bin width does not imply common score
units or equal tie frequency. Report the used bins and tied-score limitation.

Keep the four 3E preselected example IDs for direct local visual comparison.
Additional best/worst examples identified from cached AE maps are labelled post-hoc
illustrations and never used to choose feature settings. A fixed descriptive
16-pixel outer-frame band can measure boundary concentration in old maps. Describe
its post-hoc status; do not change inference borders in response.

The constant-score AP reference is image/pixel positive prevalence, not a
finite-sample guarantee for every random ordering. Pixels and related captures
are dependent; validation normals were used in AE selection. No independent
confidence intervals, target-domain winner, production readiness or calibrated
false-alarm/recall claim. PCB1 BLOCKED, PCB2 INCONCLUSIVE and official MetalParts
independent evaluation NOT CLEARED remain. No test0–test6 payload access.
