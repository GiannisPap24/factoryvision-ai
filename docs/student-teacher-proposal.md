# Proposed student-teacher development experiment

Planning only, prepared during Milestone 2C. No student network was implemented or trained. A separately authorized Milestone 2D would investigate whether distillation offers a useful CPU memory/latency tradeoff. The current synthetic probe results do not establish that it will improve real anomaly detection.

The [Student-Teacher Feature Pyramid Matching paper](https://arxiv.org/abs/2103.04257) motivates matching a fixed pretrained feature hierarchy with a student learned from normal images. The proposal below is a small, explicitly specified adaptation, not a claim of exact paper reproduction.

## Proposed recipe to approve and freeze before training

Use the same approved ResNet18 ImageNet V1 weights for a frozen teacher truncated through layer3. Initialize a separate matching student randomly with seed 42; never copy teacher weights into it. Teacher parameters and BatchNorm statistics remain fixed. Student parameters and training BatchNorm statistics are learned only from the 315 permitted Cashew training normals.

Input remains full-frame 128x128 RGB with the declared ImageNet normalization. No augmentation or defect synthesis enters optimization. Extract layer2 and layer3 features before the feature-memory pooling operation. Normalize each spatial feature vector by its channel L2 norm with epsilon 1e-6. For each level, sum squared teacher-student channel differences, average over samples/spatial locations, then average the two level losses equally. Train with Adam learning rate 0.001, batch eight, zero workers, four CPU threads, at most 20 epochs and a 600-second wall budget. No hyperparameter/seed sweep is proposed.

Choose the strict minimum normal validation matching loss on the 67 validation normals, with ties retaining the earlier epoch. Resource-budget interruption produces an incomplete development record, not a silently substituted final recipe. The 68 normal calibration images must remain out of optimization and checkpoint selection. Save selected student weights, teacher hash, configuration, manifest/split fingerprints, source hashes, epoch and resource measurements.

After selecting the checkpoint, compute channel-summed squared distance between unit-normalized teacher/student features at each level. Bilinearly align layer3's distance map to layer2's 16x16 grid (align_corners=False), average levels equally, use the raw maximum grid value as the image score, and bilinearly resize to128 for display/localization. No test-driven normalization, smoothing or score changes. Set the provisional image threshold with the 68 calibration normals, linear q95 and strict greater-than. No pixel threshold or industrial operating point is implied.

## Permitted development evidence

Reuse the strict development adapter/file-read guard. Record normal validation distributions, in-sample calibration exceedances, frozen teacher-state equality, optimizer updates/gradients, checkpoint reload equivalence, CPU time/RAM and inference latency. The same predeclared 16 checkerboards may test software response and geometry but remain artificial diagnostic probes. Do not tune to their AP or compare their scores with historical official-test accuracy. Preserve failed runs and report numerical/training instability.

Default tests must use fabricated inputs and no downloads. Verify teacher immutability, student weight updates, finite losses/gradients, partition rejection before image access, checkpoint selection based solely on normal validation, calibration isolation, score/map geometry and artifact provenance. Do not run the observed Cashew test set.

## Confirmatory boundary

Student generalization to anomalies could suppress the intended discrepancy; normal matching loss is an imperfect anomaly proxy. Same-camera normal checks and high-contrast checkerboards cannot resolve this. If implementation or scoring changes during development, freeze the resulting recipe explicitly before any new category test access.

The existing [pcb1 pre-registration](protocols/pcb1-confirmatory-v1.md) initially enrolls feature memory and spatial mean only. Adding this student-teacher model requires a versioned amendment and a frozen implementation before accessing pcb1 test images/labels/scores; declare all enrolled models and metrics, with no post-test model selection. Train the student afresh on pcb1 normal development images and calibrate there; do not describe that as zero-shot Cashew transfer. This confirmatory work would be a later, separately authorized milestone.
