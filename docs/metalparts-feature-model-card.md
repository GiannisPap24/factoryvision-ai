# Experimental MetalParts feature memory — Milestone 3F

**EXPLORATORY SOURCE-DOMAIN COMPARISON — NOT INDEPENDENT TEST PERFORMANCE**

This is a local research comparator for RobustAD MetalParts. It is not a classifier,
probability estimator, independently validated detector or production inspection system.
No binary threshold exists. Official target-domain evaluation remains **NOT CLEARED**.

## Model and provenance

Frozen ImageNet ResNet18 layer2/3 features; native 256-pixel tiles at stride 192,
antialiased resizing to 128, pooled 384-dimensional descriptors on a 16×16 grid.
The accepted 144 normal fitting parents contribute 12 native tiles each and four
deterministically sampled descriptors per tile: 6,912×384 float32 values. No
validation normal, anomaly, target image or Cashew bank entry enters memory.
This is memory fitting without neural-network gradient training.

Image score: maximum raw descriptor RMS nearest-neighbor distance across tiles.
Native map: bilinear tile-map expansion and uniform overlap averaging. Higher means
more unlike stored nominal descriptors, not necessarily defective. Scores differ
in units and aggregation from the accepted AE's mean native RGB reconstruction MSE.
Heatmap display scales separately for each method/image; metrics use raw maps.

This is PatchCore-inspired, not an exact reproduction: ResNet18 and deterministic
sampling replace the published approach's backbone/coreset design. See the
[frozen protocol](metalparts-feature-memory-protocol.md) and
[Roth et al., CVPR 2022](https://arxiv.org/abs/2106.08265).

Dataset: [AmazonScience/RobustAD](https://huggingface.co/datasets/AmazonScience/RobustAD),
CC BY 4.0; Latha Pemula, Dongqing Zhang, Onkar Dabeer, CVPR Workshops 2025.
Revision `82c642b2d18be60f2dab104f8dd4dea1f4ccfacb`.
Fitting membership is the accepted 3D normal split, verified by source SHA-256.
Pretrained SHA-256:
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
Memory SHA-256:
`38d89de7500b8c9c9ae1bb5e2bbdff7a400318f072fbcfba9441aa7e8a713b10`.
Memory artifact 10,618,409 bytes; trusted full pretrained file 46,830,571 bytes.
Package contains pinned metadata, never weights or third-party images.

## Observed comparison and limits

The same 51 observed source anomalies and 36 previously used validation normals:

| Method | Image AUROC | Image AP | Pixel AUROC | Pixel AP |
|---|---:|---:|---:|---:|
| Frozen AE | 0.59694989 | 0.71426655 | 0.86717478 | 0.02154142 |
| Feature memory | 0.80174292 | 0.88750534 | 0.96350088 | 0.30512141 |

All 349,870,500 native pixels, 84,903 positives; identical fixed quantization
`floor(float64(score)*65535)`, no normalization/clipping. Results are exact for
quantized rankings and approximate for unquantized rankings. Neither score scale
nor quantization tie frequency is shared across models. Constant-score AP equals
58.6207% image prevalence and 0.024267% pixel prevalence.

Feature AP is lower on 10/51 anomalous images; both maxima miss the mask on 22/51.
Memory descriptors can dilute tiny defects, preserve irrelevant geometry or lighting
responses and omit rare normal appearances. Broader context, native tiling and
interpolation constrain localization. Stronger aggregate observed rankings do not
establish independent superiority. No settings changed after results were observed.

Related physical captures may cross internal partitions; capture-group IDs are
unavailable. Normals influenced AE selection; anomalies and earlier results were
already observed. Unresolved official train/test similarity candidates preclude
an independent target-generalization claim. No official test0–test6 payload was
opened for fitting, inference or this comparison. No confidence intervals implying
independent pixels/captures are reported.

Fit: 10.12 seconds; including exact extraction replay, 20.01 seconds; peak 475 MB.
Normal verification peak 538 MB; comparison peak 556 MB. Recorded feature native
inference median 1.64 seconds, p95 2.10; historical AE median 0.36. Runs were separate
and comparison overlapped verification work, so this is not a controlled speed race.
Appropriate for deliberate local inspection, not evidence of live line throughput.

Local upload safeguards, hash-pinned artifacts and weights-only tensor loading
apply. Reproduction requires separately retained local assets. See the
[3F report](milestone-3f-metalparts-feature-memory.md) for evidence and limitations.
