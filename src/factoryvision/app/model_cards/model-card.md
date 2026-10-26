# Model Card: FactoryVision Cashew AE and spatial mean

Version: Milestone 2B, 24 October 2026. Local research artifacts; no deployed service.

## Intended use and scope

Reproducible research on image anomaly ranking and pixel-level localization in the official VisA Cashew one-class benchmark. Intended users are researchers reviewing this experiment. Neither model is approved for production inspection, autonomous product acceptance/rejection, food-safety decisions or safety-critical use. Camera, lighting, pose, factory and defect shifts have not been validated.

## Models and data

The compact RGB convolutional autoencoder has 37,763 trainable parameters, was trained for 12 CPU epochs on 315 normal images and selected at epoch 12 using 67 normal validation images. Its unchanged checkpoint SHA-256 is `da7e0caed8e365a08a3d301ec7e88536bc5df8e3f8b59f27635853e15ab4cdbc`. The mean baseline is a 3x128x128 reference fitted only on the same 315 training normals: 49,152 fitted values, zero gradient-trained parameters. Its reference artifact SHA-256 is `04dc72dd50c54214d7bf874f650da58bf53eb80c803d550f8a75c7368bfd4c63`.

Both use 128x128 float32 RGB in [0,1], bilinear antialiased resizing and raw mean squared RGB residuals. The channel mean is the spatial anomaly map; its spatial mean is the image score. Higher scores mean more anomalous. Original nonzero mask classes become binary and resize with nearest-neighbor interpolation. There is no test-driven postprocessing or threshold adjustment.

Each threshold uses 68 separate normal calibration images, linear 95th-percentile interpolation and strict greater-than classification. AE: **0.0031531278509646654**. Spatial mean: **0.017185416119173163**. These empirical thresholds do not guarantee a production false-alarm rate.

VisA is attributed to its creators through the [official dataset repository](https://github.com/amazon-science/spot-diff) and [dataset registry](https://registry.opendata.aws/visa/). Source data is CC BY 4.0; local license/provenance records are retained. Images are resized/binarized for evaluation as described above. Images, weights and derived examples remain local and Git-ignored pending redistribution and release-scope review. Source-code licensing remains an owner decision.

## Frozen evaluation

One official held-out run: 150 Cashew images, 100 anomalous and 50 normal. Both models evaluated identical membership. The immutable protocol (source distribution: `docs/protocols/cashew-evaluation-v1.json`), integrity gate and model/threshold freeze preceded predictions. No AE retraining or recalibration occurred in 2B. The test set is now observed.

| Metric | Autoencoder | Spatial mean |
|---|---:|---:|
| Image AUROC | 0.937600 | 0.736800 |
| Image Average Precision | 0.943219 | 0.841216 |
| TN / FP / FN / TP | 47 / 3 / 48 / 52 | 50 / 0 / 85 / 15 |
| Precision | 0.945455 | 1.000000 |
| Recall | 0.520000 | 0.150000 |
| F1 | 0.670968 | 0.260870 |
| Normal false-positive rate | 0.060000 | 0.000000 |
| Pixel AUROC | 0.818119 | 0.895301 |
| Pixel Average Precision | 0.080936 | 0.267302 |

Pixel metrics pool 2,457,600 pixels, including normal images, with 27,672 positive pixels. AP is non-interpolated average precision, not trapezoidal PR area. No AUPRO or thresholded pixel metrics are claimed. Image prevalence is 66.7% anomalous; AP/precision do not transfer directly to a low-defect production line.

AE-minus-mean image AUROC difference is 0.2008 (conditional paired-bootstrap 95% interval 0.110195 to 0.291805); AP difference is 0.102003 (0.032783 to 0.167381). The 1,000 image-stratified resamples assume image independence, which is not established at the physical-item/acquisition level.

## Resources

On an AMD Ryzen 7 9700X, Windows AMD64, Python 3.12.15, torch 2.13.0+cpu and four Torch threads, preloaded batch-1 forward-plus-score median/p95 latency was 0.8091/1.09755 ms for AE and 0.3818/0.499685 ms for mean (five warmups, 100 serial repeats). Separate warm-cache decode/resize median/p95 was 7.37465/7.97943 ms (30 repeats). This is not end-to-end camera latency. Serialized files are 156,083 and 198,185 bytes respectively; the mean reference tensor itself occupies 196,608 bytes. The evaluation process peaked at 631,361,536 bytes working set.

## Limitations and decision

The AE improves image ranking and frozen-threshold recall/F1, but its pixel ranking is worse and it generates three false positives. It misses 48% of anomalous test images. The mean misses 85%; zero observed false positives among 50 normals does not establish zero risk. The evidence does not establish overall neural-model superiority.

Exact duplicates and corrupt images were not found. All 7,946 earlier perceptual candidates were refined under a predeclared screen; 20 deterministic pairs were visually inspected without confirming duplication. Remaining near-duplicate uncertainty and missing physical-object/acquisition identities prevent an independence guarantee.

MSE can emphasize object boundaries, pose, texture and background; averaging can dilute small defects, and reconstruction can also reproduce anomalous content. The inspected fixed examples show edge/background residuals and blurred reconstructions. Fixed square resizing changes geometry and may underrepresent small defects even though no anomaly mask disappeared entirely. Pixel correlation and class imbalance limit interpretation of AUROC; pixel AP is low for the AE.

One category, one seed, a small normal calibration set and one test set do not establish robustness or industrial readiness. No operational cost target, independent defect development set, camera validation or physical-group confirmation was available. Do not adapt this experiment to the observed test set and present repeat results as new held-out evidence. A separately authorized next milestone should first resolve evaluation governance and acquisition independence, then design a predeclared CPU benchmark with independent confirmation data.

## Milestone 3A application addendum (25 October 2026)

The local Cashew application now exposes these same frozen artifacts and thresholds
through explicit CPU inference. This interface does not establish new accuracy,
recalibrate thresholds or change the historical findings above. It also supports
the existing feature-memory and student-teacher artifacts; their official Cashew
defect metrics remain unestablished. PCB1 remains BLOCKED by source overlap; PCB2
remains INCONCLUSIVE with 56 unresolved pairs and no independent human adjudication.
See inference design (source distribution: `docs/inference.md`) for image restrictions, relative heatmap
normalization, exact-test-image exclusion limitations and local privacy behavior.
