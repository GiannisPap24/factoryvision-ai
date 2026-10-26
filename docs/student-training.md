# Student–teacher training and reproducibility

Milestone 2D completed one compact architecture and one full CPU experiment. The [Model Card](student-model-card.md), [completion report](milestone-2d-completion.md) and [frozen recipe](protocols/student-development-v1.json) separate actual learning from artificial diagnostics and future confirmation. The observed Cashew official test was not re-evaluated.

## Exact data and optimization semantics

Reuse accepted manifest `80bac86ae2fc1b0ad7e0e046597fb4a3a1e1f4f92ce556d5dddbfdd37d51f84c`, its original parent/assignment identities and local source hashes. The metadata manifest includes historical official membership; the runner filters permitted development records and never uses test labels, masks, pixels or prediction scores to select or train the model. A Python open audit hook allows only the 450 approved normal development image paths. Preflight hashes all 450; only the 315 training normals enter optimizer updates. Saved epoch lists prove complete, unique membership and seeded shuffled order. Validation and calibration are separately ID-ordered. No augmentation, fitted feature transform or pseudo-defect enters optimization.

The teacher is the existing verified ResNet18, not a new download. SpatialTeacher exposes raw layer2/layer3 features without the feature-memory pooling/concatenation. `requires_grad=False`, forced eval mode and no_grad prevent teacher parameter/buffer updates; state hashes include buffers. CompactStudent initializes under a local Torch RNG seed 42 independently of the teacher. Four convolutional stages and two heads produce the same spatial/channel shapes; biases are enabled. Student parameters alone are passed to Adam.

The Model Card specifies normalization, discrepancy and scoring exactly. A spatial vector uses its own channel norm; no cross-image statistics are estimated. Aligned map averaging precedes batch/spatial loss averaging. Epoch averages weight the final three-image training batch correctly: 39 batches of eight plus one of three, 40 updates per epoch. Validation contains eight full batches plus one of three. Float32 arithmetic is used for network/loss; calibration is float64.

Seed Python, NumPy and Torch with 42, enable deterministic Torch algorithms, use four CPU threads, zero DataLoader workers and an independent seed-42 shuffle generator. Adam uses learning rate 0.001, betas (0.9,0.999), epsilon 1e-8 and zero weight decay. Maximum 12 epochs, patience 3, min_delta 0. Every strict normal-validation minimum is saved; ties retain the earlier checkpoint. No calibration loss or synthetic metric selects a checkpoint.

Each step rejects nonfinite losses, missing/nonfinite student gradients, any teacher gradient and nonfinite updated parameters. Epoch membership checks reject repeated, omitted or unexpected IDs and wrong partitions before forward/update for that batch. The 600-second deadline is checked at batch boundaries; observed peak memory is checked before batches and at completion against 2 GiB. These are process observations, not OS-enforced instantaneous limits. Failed/partial runs remain and cannot be silently reused.

## Experiments and artifacts

- `runs/cashew-student-smoke-v1`: first 16 training and first 8 validation normals, one epoch/two updates; empirical feasibility check.
- `runs/cashew-student-v1`: fresh seed, model and optimizer, 315/67 normals, 12 epochs/480 updates; followed once by calibration and development diagnostics.
- `runs/cashew-student-debug-v1`: separately explicit two-update smoke of the same architecture under VS Code. Its deliberate pauses are excluded from resource benchmarks.

No accepted autoencoder/feature-memory implementation or artifact was changed. Historical output hashes were compared without replaying official-test predictions/metrics.

`experiment.json` binds teacher/student state fingerprints, versions, configuration, fitting/validation IDs, manifest, lock and execution-source hashes. `history.json` stores every epoch's complete ID order, losses, updates and elapsed time. `best.pt` contains only the student tensor state_dict; `checkpoint.json` binds architecture, teacher weight hash, manifest, recipe, fit IDs, epoch, loss and file digest. `best-probe.pt` records scores/maps on the first validation batch at the selected checkpoint. `result.json` records changes, immutable teacher, reload error and measured resources. Metadata does not include an optimizer/RNG snapshot, so exact mid-run resume is unsupported.

Reload rejects directory links/junctions, linked/missing/oversized files, incompatible metadata, bad hashes, keys, tensor shapes/dtypes and nonfinite state. It uses weights_only=True and CPU mapping, then strict loading and eval mode. Score/map tolerance is atol 1e-6, rtol 1e-5; observed maximum errors were both zero. The explicit real check also recomputed the selected validation loss over all 67 normals, within 1e-6. Local hashes establish consistency, not signed authenticity; trusted local filesystem control is assumed.

## Calibration and illustrative diagnostics

Only after checkpoint selection and reload, the 68 normal calibration images set linear q95 and strict greater-than. Normal validation/calibration distributions and raw scores are retained separately. No pixel decision rule is calibrated. The same fixed first 16 validation checkerboards as 2C provide deterministic artificial masks; no new defects or locations were chosen after inspecting results.

An inherited figure helper initially printed the feature-memory label “Feature RMS map” over the correct student discrepancy arrays. Original exports and their hashes are preserved. `scripts/render_student_examples.py` creates correctly labelled panels under `runs/cashew-student-v1/illustrations/` from the saved maps and four permitted normal images, with **no model inference, retraining, changed scores, threshold or display scale**. Its provenance records source-map and rendering-source hashes. Use those corrected panels for illustration; all four were visually inspected. The fixed [0, image-calibration threshold] color scale clips display values only and is not a pixel threshold.

## Explicit commands

```powershell
.venv/Scripts/python.exe -m pytest tests/test_student_teacher.py -q
.venv/Scripts/python.exe -m pytest integration_tests/test_student_real.py --basetemp=.tmp/pytest-student-real -q
.venv/Scripts/python.exe scripts/check_quality.py
```

The completed first-run commands were `scripts/student_experiment.py smoke`, `train` and `evaluate`. Preserve their directories and start marker; do not rerun fitting/evaluation to replace evidence. A separately requested smoke accepts `smoke --run-id NEW_ID`; custom full-run IDs are refused. Default tests fabricate images/teacher weights and require no VisA files or network. Do not invoke the unrestricted historical integration suite during development-only work.

VS Code exposes explicit student unit-test, CPU smoke, bounded training, development evaluation and permitted-data verification tasks. The training debugger launches only a two-update smoke, uses the project interpreter and stops at `training/student_engine.py:51` after backward and before optimizer.step. The completed debug ID is preserved. No task runs when the workspace opens; no global Windows/VS Code setting changes are required. The installed official Codex extension was verified separately from Python/debugger operation; no authentication or billable request was used to test it.

## Scope of scientific evidence

Parameter updates, finite gradients, exact memberships, teacher equality and reload demonstrate genuine reproducible learning on this host. Lower normal validation loss does not prove anomaly sensitivity: a student may generalize too well to defects or learn background features. Validation was used for checkpoint selection, so its false alarms are not an independent estimate. Artificial defect ranking/localization cannot be compared to historical real-test metrics as proof of superiority. One seed/category/recipe, coarse features, resizing, ImageNet shift, small calibration tails and unknown object/acquisition independence remain limitations.

The [pcb1 v2 plan](protocols/pcb1-confirmatory-v2.md) defines future independent category-specific fitting and pre-test gates. It is not zero-shot transfer or authorization to open pcb1 assets. Work stops at 2D.
