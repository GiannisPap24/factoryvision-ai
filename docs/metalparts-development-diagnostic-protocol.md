# MetalParts frozen source-development diagnostic — v1

Milestone 3E. EXPLORATORY SOURCE-DOMAIN DIAGNOSTICS — NOT INDEPENDENT TEST PERFORMANCE.

Freeze this protocol, source audit, selected IDs, execution code and checkpoint
before the first source-anomaly forward pass. No retraining or recalibration.
Checkpoint: `runs/metalparts-ae-pilot-v1/best.pt`, SHA-256
`c23543a8d22a305193e95a36985331c57b86ecb8b71e86c58b29a82fb1ab8651`.
Architecture: unchanged conv-ae-v1, 37,763 parameters, CPU float32, eval mode,
inference_mode, deterministic algorithms and four threads. Batch size one.

Only pinned RobustAD revision `82c642b2d18be60f2dab104f8dd4dea1f4ccfacb`
source-training metadata/images/masks may be read. Verify bytes against saved 3C
source hashes. Audit all 180 normal and 51 anomaly source records. Preserve the
exact 3D 144 fitting / 36 validation normal assignments. Score only those 36
validation normals and all 51 anomalies, in stable sample-ID order. All 51 were
excluded from 3D fitting and checkpoint selection. No test0–test6 payload access.
Known source/test relationships are retained as unresolved metadata evidence;
they are not dismissed or converted into confirmed physical groups.

Use native 256 × 256 crops, stride 192, row-major order, appending trailing-edge
coordinates. Use the complete grid (112 tiles at 2681 × 1500). No whole-image
resize, padding, augmentation, learned normalization or content-selected crops.
RGB uint8 is converted to float32 [0,1]. Average reconstructed RGB uniformly over
overlapping tiles; assert full coverage. THEN subtract original RGB, square and
average channels. The image score is the float32 spatial mean of this full native
map, identical to the accepted 3D native diagnostic. Never average patch scores
as a substitute. No smoothing, threshold, probability conversion or per-image
normalization enters numerical scoring.

Masks must be native-aligned, single-channel, nonempty for anomalies and contain
only 0/255; binarize by >0. Normals have derived all-zero masks. No interpolation
is required or permitted for metric masks. Display thumbnails may resize images
bilinearly and masks with nearest-neighbor, without altering metric inputs.

Image AUROC: exact tie-aware ranking, half credit for ties. Image AP: grouped,
non-interpolated precision weighted by increases in recall. Report counts,
prevalence and per-class min/quartiles/median/max. Pixel metrics pool EVERY native
pixel across the 87 images, including normal pixels. To bound memory, predeclare
65,536 fixed score bins with floor(float64(MSE)*65535), bounded to [0,1]. Pixel
AUROC/AP are exact for these quantized scores and explicitly approximate the
unquantized ranking. No pixel sampling or data-selected bin edges. Report bin
width 1/65535, total pixels, positives and prevalence. Return null with a reason
when either class is absent. Verify tie and quantization calculations against
scikit-learn using synthetic arrays before real scoring.

Save full float32 native maps, per-image records, pixel histograms and metrics in
ignored `runs/metalparts-source-diagnostic-v1`. Refuse an existing run/started
marker; retain failures. Seal exact input and execution hashes in ignored
`reports/metalparts-3e-v1/diagnostic-manifest.json`. Before seeing performance,
select the first normal validation ID and first three anomaly IDs lexically for
local-only previews. Save original/reconstruction/map/overlay and native mask
thumbnails for these four; never choose examples based on favorable scores.
Visual heatmaps use relative display color; brightness is not comparable across
images. Record per-image inference time, total elapsed/CPU time and process peak
working set; batch-one execution and streaming histograms bound memory.

The 36 normals already influenced checkpoint selection. Source anomalies and
normals may depict related physical items; no reliable capture grouping exists.
Pixels within images and images across captures are dependent. No independent
confidence intervals, accuracy, binary recall, false-alarm guarantee, shifted
domain performance or production claim follows from these descriptive results.
No metric result authorizes changing the pipeline in this milestone. Existing
blur and patch seams may dominate residuals. Independent evaluation remains NOT
CLEARED; 3C BLOCKED, PCB1 BLOCKED and PCB2 INCONCLUSIVE remain unchanged.
