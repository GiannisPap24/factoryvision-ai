# PCB1 frozen candidates — blocked confirmatory evaluation

Status on 2026-10-24: **BLOCKED — OFFICIAL EVALUATION NOT COMPLETED**. See the [2E2 completion report](milestone-2e2-completion.md) and [unchanged frozen methods](pcb1-confirmatory.md).

The four category-specific research candidates are a convolutional autoencoder, spatial RGB mean reference, frozen ResNet18 feature-memory detector, and trainable student with frozen ResNet18 teacher. They were prepared in 2E1 using 632 fitting, 135 validation and 137 calibration PCB1 normal images. All checkpoints, preprocessing, score definitions and thresholds remain frozen. No retraining or recalibration occurred in 2E2.

The first authorized test opening decoded all 100 normal and 100 anomalous official images and their 100 anomaly masks for integrity. The required review found near-duplicate/repeated-acquisition appearance between `visa-pcb1-normal-0187` (test) and `visa-pcb1-normal-0188` (fitting). Both original images show matching pose, mesh and solder details and background features, despite differing byte/pixel hashes. This violates the preregistered cross-partition integrity gate. The failed review is preserved in ignored `reports/pcb1-opening/review.json`; no official model predictions, anomaly maps, metrics, confidence intervals or performance ranking exist.

The test set is partially unblinded, not untouched. No sample was excluded or reassigned. Image-content review cannot establish independently verified physical-object or production-batch identities. No passing review may replace this finding merely to enable execution.

All models take full-frame RGB resized to 128×128, which distorts the source aspect ratio and can suppress small defects. Feature/student methods apply frozen ImageNet normalization and interpolate coarse spatial maps. Frozen 95th-percentile normal calibration thresholds do not guarantee test false-positive rate or defect recall. Model scores use different units. Pre-opening latency measurements describe one local CPU and one normal image, not production throughput or detection quality.

Historical Cashew results motivated feature-based localization and student-teacher research after reconstruction localization proved weak. That is a separate exploratory research history; it supplies no PCB1 test result. PCB1-specific fitting also means this experiment was not a zero-shot transfer study. These candidates establish neither industrial deployment readiness nor factory-level generalization, production reliability, shift tolerance or safety performance.

Further work requires explicit review of the integrity finding and a new, prospectively frozen evaluation design with defensible acquisition grouping or independent confirmation data. No remedy, replacement evaluation or later milestone was started.
