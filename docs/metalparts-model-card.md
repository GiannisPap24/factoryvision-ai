# Model Card: experimental MetalParts autoencoder

Milestone 3E, 25 October 2026. Local research only.

**Experimental MetalParts model — trained on normal source-domain images.
Independent defect detection performance has not been established.**

## Identity and intended use

Unchanged conv-ae-v1, 37,763 parameters, CPU float32. Checkpoint
`runs/metalparts-ae-pilot-v1/best.pt`, SHA-256
`c23543a8d22a305193e95a36985331c57b86ecb8b71e86c58b29a82fb1ab8651`.
Use to inspect reconstruction and continuous error on permitted MetalParts
source-development or owner-selected non-test images. No production acceptance,
defect absence, safety or independent-generalization claim is supported.

Source: [AmazonScience/RobustAD](https://huggingface.co/datasets/AmazonScience/RobustAD/tree/82c642b2d18be60f2dab104f8dd4dea1f4ccfacb/MetalParts),
pinned revision `82c642b2d18be60f2dab104f8dd4dea1f4ccfacb`, CC BY 4.0.
Attribution: Amazon Science; Latha Pemula, Dongqing Zhang, Onkar Dabeer;
*Robust AD*, CVPRW 2025. Third-party images and trained weights remain local and
Git-ignored. The original acquisition and license evidence are preserved in 3C.

## Training and selection

Fresh initialization, 144 source-training normal parents, 36 normal validation
parents; 12 fixed native 256-pixel patches per parent, stride 192. Four epochs,
864 updates, Adam 0.001, batch eight, seed 42, four CPU threads. The minimum normal
validation loss selected epoch 4. No anomaly entered fitting or selection.
Initial validation MSE 0.0743444142; final fitting MSE 0.0054579481; final validation
MSE 0.0040141833. These are normal reconstruction losses, not detection accuracy.
No training or calibration occurred in 3E.

## Frozen inference

Complete native-image grid, batch one, 256-pixel crops, stride 192 and exact
trailing-edge coverage. RGB uint8 becomes float32 [0,1]. Uniformly average
overlapping reconstructed RGB, then compute channel-mean squared residual.
Image score is the native spatial mean. No resizing, padding, smoothing or fitted
normalization enters scoring. At source dimensions 2681 × 1500, all 112 tiles
cover each parent. Uploads require at least 256 pixels per side. Higher MSE means
more reconstruction mismatch; it is not a probability. **Threshold: none.**

## Source-development observations

**EXPLORATORY SOURCE-DOMAIN DIAGNOSTICS — NOT INDEPENDENT TEST PERFORMANCE**

| Level | AUROC | Average precision | Positive / total |
|---|---:|---:|---:|
| Image | 0.5969498911 | 0.7142665461 | 51 / 87 |
| Native pixel, fixed quantization | 0.8671747847 | 0.0215414150 | 84,903 / 349,870,500 |

Comparison uses all 51 source-training anomalies and the existing 36 validation
normals. Pixel metrics include all native pixels, with fixed 65,536 bins and
`floor(float64(MSE)*65535)`: exact on quantized scores, approximate for the
unquantized ranking. Native masks align without interpolation. Image anomaly
prevalence is 58.62%; pixel anomaly prevalence is 0.024267%.

The normals already influenced checkpoint selection. Physical-item and capture
relationships remain unknown, including unresolved source/target similarity and
basename correspondences. Pixels and images are dependent; no independent
confidence intervals are justified. All official test0–test6 payloads remained
unopened. Independent MetalParts evaluation remains **NOT CLEARED**; the original
3C **BLOCKED** verdict is unchanged. Source ranking cannot establish performance
under target-domain shifts. Raw scores are not comparable with Cashew methods.

## Failure modes and measured resources

Normal and anomalous image-score distributions overlap substantially. Pixel AP
is low despite a higher pixel AUROC; background imbalance matters. The model
blurs details, leaves patch seams and responds to object edges. A defect may be
reconstructed well or contribute too few pixels to change mean image error.
Lighting, pose, surface texture and unseen categories may alter scores.

Native forward/stitch/map latency in this local run: median 358.63 ms, p95
363.30 ms; total diagnostic 39.94 seconds; peak process working set 765.43 MiB.
These measurements are host-specific, include four CPU threads and are not a
controlled deployment benchmark. The app reports upload-specific timings.

Artifact bytes and identity are verified before CPU `weights_only` loading;
only the pinned local model is supported. Bounded uploads are decoded in memory,
metadata stripped, and known official test byte/pixel hashes rejected. Hash
blocking does not recognize every transformed copy. Users must explicitly select
a permitted category/image and request inference. No images are persisted by
the application. See the [frozen protocol](metalparts-development-diagnostic-protocol.md)
and [3E report](milestone-3e-metalparts-development-evaluation.md).
