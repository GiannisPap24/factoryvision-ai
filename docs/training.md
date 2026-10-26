# First real CPU autoencoder experiment - Milestone 2A

The completed experiment is `runs/cashew-ae-128-seed42/`, retained locally and ignored by Git. It demonstrates real parameter learning on Cashew normal images and safe checkpoint reload. It does not establish useful anomaly detection. **At the 2A checkpoint, official test metrics were not yet evaluated.** See the [2B report](milestone-2b-completion.md) for the subsequently authorized frozen evaluation.

## Data gate

The accepted 600-image, 100-mask inventory was decoded again and compared against the frozen parent and per-image/mask audit. The derived manifest still contains 315 training, 67 validation and 68 calibration normals, plus the unchanged 150 official test images. Image byte hashes are checked again before the experiment. No partition was moved, excluded or relabeled.

The pretraining screening script compares every cross-partition pair using horizontal 64-bit dHash, 63-bit AC DCT pHash, and 32x32 RGB root-mean-square difference. The documented flag rule is dHash Hamming distance <=4 **or** pHash distance <=4 **or** RGB RMS <=0.02. Of 114,581 pairs, 7,946 were flagged. None had both hashes identical; none had RGB RMS <=0.02. Similar centered cashews against the fixed background produce many coarse-fingerprint matches.

The 24 flagged pairs with lowest RGB RMS were visually reviewed in contact sheets. They show shape, surface or position differences; this subset did not establish a confirmed duplicate. All candidates remain in `reports/cashew-leakage-screen.json`; the selection, findings and unresolved risks are in `reports/cashew-leakage-review.json`. This is a limited screen, not an exhaustive review or physical-item independence certificate. The remaining 7,922 candidates and acquisition identity require attention before interpreting held-out results. Test images were read for integrity/similarity screening only, never passed through the model.

Canonical dataset fingerprint: `80bac86ae2fc1b0ad7e0e046597fb4a3a1e1f4f92ce556d5dddbfdd37d51f84c`. Parent fingerprint: `9d8e56d672589ef1234a9382cc61a52efd29804364b1b265d0cfeb478ce56372`. The experiment records these, assignment and gate hashes, execution source hashes and seed.

## Architecture and fixed recipe

`ConvAutoencoder`, architecture ID `conv-ae-v1`, has **37,763 trainable parameters**. All four encoder convolutions and decoder transposed convolutions use kernel 4, stride 2, padding 1 and bias. ReLU follows encoder and hidden decoder layers; sigmoid follows the RGB output. There are no skip connections, pretrained weights, batch normalization or dropout.

| Stage | Channels x height x width at resolution 128 |
|---|---|
| RGB input | 3 x 128 x 128 |
| Encoder 1 | 8 x 64 x 64 |
| Encoder 2 | 16 x 32 x 32 |
| Encoder 3 | 32 x 16 x 16 |
| Latent | 16 x 8 x 8 (1,024 values) |
| Decoder 1 | 32 x 16 x 16 |
| Decoder 2 | 16 x 32 x 32 |
| Decoder 3 | 8 x 64 x 64 |
| RGB reconstruction | 3 x 128 x 128 |

Supported square resolutions are integer multiples of 16 from 32 through 256. Inputs must be nonempty float32 CPU RGB tensors in [0,1]. Preprocessing ID `rgb-float01-bilinear-aa-square-v1` means bilinear antialiased square resizing and division by 255, with no fitted statistics or augmentation. Square resizing changes the original aspect ratio; this is a frozen first-baseline choice.

The loss is mean squared reconstruction error over RGB channels and spatial positions, averaged over samples. MSE gives a simple differentiable baseline and matches the preliminary image score, while emphasizing larger residuals. Background texture and normal object edges also contribute; it is not a defect-specific objective. Epoch losses are sample-weighted, including the final 11-image training batch.

## Configuration and execution boundaries

CPU only, batch 16, Adam learning rate 0.001, seed 42, four CPU threads, zero DataLoader workers, 12 maximum epochs, patience 3, minimum significant validation improvement 0.00001. Adam uses its installed default betas (0.9,0.999), epsilon 1e-8 and zero weight decay. Training order is shuffled with a dedicated seeded generator; validation and calibration use stable sample-ID order. Python, NumPy and torch seeds are fixed and deterministic algorithms are enabled. No GPU is selected or installed.

Every strict validation minimum is checkpointed; patience resets only for an improvement greater than min_delta relative to the preceding best. Calibration is unavailable to checkpoint selection. Each batch is checked for the expected normal development partition; non-finite losses, gradients and parameters abort execution. The script enforces a 1,800-second batch-boundary training deadline and checks a 2 GiB process peak limit after each epoch. These are observed checks, not OS-enforced instantaneous process limits.

The real smoke used 32 training and 16 validation images for two Adam steps. A fresh seed/model/optimizer then started the full experiment. It completed all 12 epochs without early stopping, with 20 updates per epoch. Best epoch was 12. Total training/validation/checkpoint time was 38.7994 seconds, process CPU time 152.9844 seconds, and total including reload and calibration was 39.8685 seconds. Peak working set was 365,449,216 bytes (348.52 MiB). Initial untrained normal validation MSE was 0.1256391322; final training MSE was 0.0024057827 and best validation MSE was 0.0023677690. See the completion report for every epoch.

CPU time divided by elapsed time is about 3.94 logical CPU cores on average; it is not system-wide CPU utilization. Timing excludes gate setup and imports, includes normal validation and best-checkpoint writes, and is a single local warm-data run. Debugger timings include deliberate pauses and are excluded from benchmarking. The measured machine is the previously audited Ryzen 7 9700X / 32 GiB host.

The full experiment was run once. Its 240 updates changed the initial parameter fingerprint. All losses remained finite. Reload produced exactly equal scores for all 67 normal validation images. Fixed seeds and this local replay check do not establish cross-hardware or cross-release reproducibility; [PyTorch documents these limits](https://docs.pytorch.org/docs/2.14/notes/randomness.html). The consulted current docs are 2.14; the installed runtime remains the unchanged, directly tested 2.13.0+cpu.

## Scoring and provisional calibration

For input and reconstruction shaped [B,3,H,W], the spatial error map is `(input-reconstruction)^2` averaged over RGB, shape [B,H,W]. The image score is the spatial mean, shape [B]. Values use raw squared [0,1] intensity units. No image-wise min-max normalization, learned score normalization, smoothing or pixel threshold is applied.

After checkpoint selection, the 68 normal calibration images established the predeclared 0.95 quantile with linear interpolation in float64. Threshold: **0.0031531278509646654**; the decision rule is strictly `score > threshold`. Four calibration scores exceed it (4/68, 5.88%). That is an in-sample empirical exceedance count, not a validated unseen-data FPR or a guaranteed 5% rate. Small-sample tail uncertainty, possible dependence and unknown anomalies prevent recall/F1 or detector-quality claims.

`validation-reconstructions.png` shows the first four ordered normal validation images, reconstructions and error maps. The display clips the raw error range [0,0.05] to grayscale; display scaling does not change stored scores. Reconstructions are visibly blurred, and error concentrates on ordinary boundaries/texture. This is expected evidence of a small reconstruction model, not proof of localization quality. No test image was used for these visuals.

## Checkpoint format and artifact safety

`best.pt` contains a plain CPU tensor state_dict, not a Python model object or optimizer object. `checkpoint.json` separately records schema, architecture, resolution, preprocessing ID, manifest hash, best epoch, validation loss and checkpoint SHA-256. `experiment.json` records configuration, parameter count, versions, source hashes and provenance. `history.json`, initial/best validation scores, calibration scores/rule and `result.json` preserve measured evidence. Exported experiment metadata has no absolute machine paths or personal identity.

Writes use an exclusive temporary file, flush/fsync and `os.replace`; JSON rejects non-finite values. The weights and metadata are individually atomic, not one filesystem transaction. If interrupted between writes, the mismatch fails closed on reload. A new portable run ID is required; existing runs are refused. Directory symlinks/junctions and linked artifact targets are rejected. This assumes trusted local filesystem control and is not race-proof against concurrent malicious mutation.

Reload validates metadata compatibility, size limits and the checkpoint digest, then uses `torch.load(..., map_location="cpu", weights_only=True)`, strict keys/shapes/dtypes and finite tensor checks before switching to evaluation mode. The [PyTorch serialization guidance](https://docs.pytorch.org/docs/2.14/notes/serialization.html) recommends state dictionaries and describes restricted loading. A local hash is integrity evidence, not an authenticity signature; do not load untrusted models. Optimizer/RNG state is not saved, so the checkpoint supports inference, not exact mid-run training resume.

Checkpoint SHA-256: `da7e0caed8e365a08a3d301ec7e88536bc5df8e3f8b59f27635853e15ab4cdbc`.

## Explicit commands and VS Code

Run from the existing project root with the existing environment. Verification commands do not train:

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/test_training.py -q
.\.venv\Scripts\python.exe scripts/check_quality.py
.\.venv\Scripts\python.exe -m pytest integration_tests/test_training_real.py --basetemp=.tmp/pytest-training-real -q
~~~

For a separately intended new experiment, use a fresh run ID; never delete or reuse the completed evidence just to rerun:

~~~powershell
.\.venv\Scripts\python.exe scripts/train_cashew.py --smoke --run-id cashew-ae-smoke-new
.\.venv\Scripts\python.exe scripts/train_cashew.py --run-id cashew-ae-128-new
~~~

The scripts require the accepted local manifest, prepared images and reviewed integrity/similarity gate. `scripts/screen_cashew.py` creates the deterministic screening report but does not fabricate a review decision; preserve or deliberately update the review record if the inputs change. No command downloads data, uploads telemetry or scores official test images.

VS Code tasks provide focused model tests, explicit CPU smoke and explicit real training, with a prompt for a new run ID. The training debug launch uses `--smoke --run-id cashew-ae-debug-42`, already executed once. Choose a new ID in launch.json for a deliberate subsequent debug run. A real breakpoint at `training/engine.py:51` stopped after backward and gradient checks, then stepping executed Adam; continuing completed the smoke and reload. The verified breakpoint was left disabled. There is no folder-open training task or global editor change.

## Historical 2A next-milestone boundary

Milestone 2B requires separate approval. First address or explicitly preserve the candidate-review/physical-identity limitations, freeze the current checkpoint/preprocessing/scoring/threshold, and predeclare image-level and pixel-level evaluation methods and uncertainty. Only then perform authorized held-out evaluation. No new models, categories, seeds, GPU setup, threshold tuning on test results, or final test metrics were added in 2A.

## Milestone 2B follow-up

The separately authorized frozen evaluation is complete; see [the 2B report](milestone-2b-completion.md). All accepted training weights, calibration and scoring code were preserved. Historical statements above about unevaluated test data describe 2A only. The official test set is now observed. No new AE training occurred in 2B.
