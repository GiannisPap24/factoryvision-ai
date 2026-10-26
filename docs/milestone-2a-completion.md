# Milestone 1B checkpoint and Milestone 2A completion â€” 24 October 2026

Milestone 2A completed the first genuine CPU model training on real VisA Cashew normals. One full bounded experiment trained a 37,763-parameter convolutional autoencoder for 12 epochs and 240 optimizer updates. The best checkpoint reload reproduced all normal validation scores exactly. **Official held-out image-level and pixel-level anomaly metrics are NOT YET EVALUATED.** Decreasing reconstruction loss does not establish detector accuracy.

## A. Reviewed Milestone 1B checkpoint

Commit on `main`: **`e95423eb92be93a4954c6b9ddd62084324ff340b`**.

Message: `feat: add verified VisA dataset loading and PyTorch CPU foundation`.

The authorized repository-local GitHub noreply identity was used. Its parent is the original foundation commit `9ddb01872be1bac45e8cb089841f4fd342979b01`. The exact 15 created and 11 modified Milestone 1B files matched the saved inventory and source hashes. Baseline tests passed (194 passed, one Windows privilege skip), with Ruff, lock and compatibility checks. Staged contents were restricted to the reviewed inventory and passed whitespace checking; the working tree was clean immediately after the checkpoint, before 2A implementation.

Real data, archives, outputs, temporary files and weights remain ignored. Credential-pattern review found no matches. Intentional local paths in AGENTS/README, historical evidence and synthetic path tests were retained for this local-only checkpoint. Nothing was pushed or published, and no remote was created.

## B. Exact Milestone 2A file inventory

Created (13 files):

- `docs/training.md`
- `docs/milestone-2a-completion.md`
- `docs/evidence/milestone-2a-verification.json`
- `integration_tests/test_training_real.py`
- `scripts/screen_cashew.py`
- `scripts/train_cashew.py`
- `src/factoryvision/data/leakage.py`
- `src/factoryvision/evaluation/reconstruction.py`
- `src/factoryvision/models/autoencoder.py`
- `src/factoryvision/training/checkpoint.py`
- `src/factoryvision/training/config.py`
- `src/factoryvision/training/engine.py`
- `tests/test_training.py`

Modified (12 files):

- `.vscode/launch.json`
- `.vscode/tasks.json`
- `AGENTS.md`
- `README.md`
- `docs/architecture.md`
- `docs/dataset-preparation.md`
- `docs/development.md`
- `docs/evaluation-protocol.md`
- `pyproject.toml` (source-distribution documentation inclusion only)
- `src/factoryvision/evaluation/__init__.py` (module description)
- `src/factoryvision/models/__init__.py` (module description)
- `src/factoryvision/training/__init__.py` (module description)

The lockfile and all existing tests/data contracts/image transforms remain unchanged. Ignored generated artifacts include three deliberately separate run directories (standalone smoke, one full experiment, debugger smoke), leakage reports/contact sheets, quality/build evidence, a fresh `.tmp/wheel-env-2a` installed environment and `dist/` archives. The full experiment's exact artifact inventory and hashes are recorded in the machine evidence.

## C. Integrity and leakage screening

All 600 image and 100 mask records were decoded and compared to the accepted audit, including byte/pixel/mask identities. The accepted frozen parent and deterministic split were reproduced without rewriting or moving samples. The split remains **315 training / 67 validation / 68 calibration / 150 official test**. All three development partitions contain only normal images; the 50 normal and 100 anomalous official test records remain intact. No exact-byte or decoded-pixel duplicate overlap was found. The experiment rechecked every image's byte hash before fitting.

Manifest SHA-256: `80bac86ae2fc1b0ad7e0e046597fb4a3a1e1f4f92ce556d5dddbfdd37d51f84c`.

Parent SHA-256: `9d8e56d672589ef1234a9382cc61a52efd29804364b1b265d0cfeb478ce56372`.

Assignment SHA-256: `6ddb3acf4172bad9c4bdd667606f0fcd12941f570884eccb5ee298d2f8470aa0`.

Near-duplicate screening examined **114,581 cross-partition pairs**, flagging **7,946** under the predeclared rule: 64-bit dHash distance <=4 OR 63-bit DCT pHash distance <=4 OR 32x32 RGB RMS <=0.02. None had both hashes identical, and none met the RGB RMS threshold. All flags remain recorded. The 24 pairs with the lowest RGB RMS were visually reviewed; visible shape, surface or position differences prevented confirmation of duplication in that subset. Coarse hashes are strongly affected by the common background and similar centered silhouettes.

The remaining 7,922 candidates were not individually reviewed. Physical-item and acquisition independence remain unverified. The gate allowed a bounded learning-feasibility experiment because exact integrity passed and no leakage was confirmed; it does **not** certify that all leakage is absent. No suspicious match was silently discarded or used to change the official test. See [training.md](training.md) for the recipe, retained reports and limitations.

## D. Autoencoder

Architecture ID `conv-ae-v1`, **37,763 trainable parameters**. RGB 128x128 input passes through four stride-2 convolution/ReLU stages: channels 3â†’8â†’16â†’32â†’16. The latent tensor is 16x8x8 (1,024 values). Four transposed-convolution stages reconstruct channels 16â†’32â†’16â†’8â†’3, with hidden ReLUs and final sigmoid. Every convolution uses kernel 4, stride 2, padding 1 and bias. There is no pretrained backbone, skip connection, batch normalization or dropout.

The configurable square resolution must be a multiple of 16 from 32 to 256. Explicit bilinear antialiased RGB resize and division by 255 are unchanged from the prepared loading contract. MSE is a simple differentiable reconstruction baseline consistent with the image score; it penalizes large residuals but does not isolate defects from normal texture/background.

## E. Actual configuration

| Setting | Executed value |
|---|---|
| Device / runtime | CPU; Python 3.12.15; torch 2.13.0+cpu |
| Resolution / batch | 128x128 RGB / 16 |
| Optimizer | Adam; learning rate 0.001; betas 0.9/0.999; epsilon 1e-8; weight decay 0 |
| Seeds | Python, NumPy, torch and loader generator: 42 |
| CPU threads / workers | 4 / 0 |
| Epoch bound | 12 |
| Early stopping | Normal validation MSE; patience 3; min_delta 0.00001 |
| Checkpoint rule | Every strict lowest normal validation MSE |
| Training order | Seeded shuffle; each epoch covers all 315 records once |
| Validation/calibration order | Stable sample-ID order |
| Resource bounds | 1,800-second batch-boundary deadline; 2 GiB process-peak check after each epoch |
| Calibration rule declared before fitting | Normal q=0.95, linear interpolation, strict score > threshold |

The real standalone smoke used 32 training normals and 16 validation normals for two updates. The full run began with a fresh model/optimizer and reset seeds; smoke weights were not reused. No test or calibration result selected the architecture, learning rate, checkpoint or stopping decision. No dependency was added or upgraded. The full experiment was launched once and was not retried or extended.

## F. Actual epoch losses

Losses are mean squared RGB intensity errors. Training loss is averaged over the evolving model's batches during the epoch; validation evaluates the end-of-epoch model. Both are weighted by sample count, including the short last batch. Initial untrained validation MSE: **0.1256391322**.

| Epoch | Training MSE | Normal validation MSE | Epoch seconds |
|---:|---:|---:|---:|
| 1 | 0.1225690050 | 0.1073225275 | 3.156 |
| 2 | 0.0830974184 | 0.0587098566 | 3.153 |
| 3 | 0.0412956160 | 0.0286241869 | 3.194 |
| 4 | 0.0224712266 | 0.0196154405 | 3.158 |
| 5 | 0.0184464197 | 0.0173739130 | 3.135 |
| 6 | 0.0165898091 | 0.0156090968 | 3.145 |
| 7 | 0.0146936205 | 0.0137702570 | 3.121 |
| 8 | 0.0126663606 | 0.0106406620 | 3.130 |
| 9 | 0.0071519096 | 0.0043847288 | 3.142 |
| 10 | 0.0034870038 | 0.0029590990 | 3.150 |
| 11 | 0.0027024196 | 0.0025716150 | 3.198 |
| 12 | 0.0024057827 | **0.0023677690** | 3.133 |

Every epoch used 315 fitting and 67 validation records, with 20 optimizer steps. All losses, gradients and parameters remained finite. Best epoch is 12; early stopping was not triggered. The initial and final parameter fingerprints differ. Four saved normal validation reconstructions show learned object structure with blurred detail; error maps highlight ordinary edges and texture as well. This is evidence of learning, not measured defect discrimination.

## G. Duration and resources

Measured training/validation/checkpoint time: **38.7994 seconds**. Process CPU time: **152.9844 seconds**, approximately 3.94 logical CPU cores consumed on average. Total including checkpoint reload and normal calibration: **39.8685 seconds**. Peak process working set: **365,449,216 bytes (348.52 MiB)** on the previously audited Ryzen 7 9700X / 32 GiB Windows host. No GPU was used.

The training timer excludes imports and dataset gate setup, includes initial validation and best-checkpoint writes, and is a single local warm-data measurement. The standalone two-update smoke took 0.5589 seconds before reload and peaked at 354,582,528 bytes including reload. The separate debugger smoke took 47.63 seconds with deliberate pauses; that is not performance evidence. Resource ceilings are periodic software checks, not OS hard limits. No runtime or memory failure occurred.

## H. Checkpoint and reload evidence

Preserved checkpoint: **`runs/cashew-ae-128-seed42/best.pt`**, 156,083 bytes.

SHA-256: **`da7e0caed8e365a08a3d301ec7e88536bc5df8e3f8b59f27635853e15ab4cdbc`**.

The plain state_dict is separate from architecture/configuration metadata. Loading uses CPU `weights_only=True`, metadata compatibility, file-size/hash checks and strict tensor key/shape/dtype/finiteness validation. Atomic temporary writes prevent partially written individual artifacts; a weights/metadata pair mismatch fails closed after an interruption. Paths/run IDs and linked directories/targets are checked; trusted local filesystem control is still assumed. Existing run directories cannot be reused. No trained weights or images are staged or packaged.

Reloaded best-checkpoint scores exactly equal the recorded scores for all **67 normal validation images**, in the development environment and a fresh installed-wheel environment. Validation image scores range from 0.0019654115 to 0.0032929021. Separate real integration tests verify completion, parameter changes, train-only membership, checkpoint hash, exact reload results, error-map shape and normal calibration membership.

The run preserves execution source snapshots matching its recorded source hashes. After the full experiment, final review updated package descriptions, rejected even tiny negative min_delta configuration values, and made failure JSON use a generic message so an exception cannot export a local absolute path. The recorded valid configuration and model/optimizer/scoring computations are unchanged; original executed sources remain preserved in the snapshot. Exported JSON contains relative paths and no personal identity or absolute local machine path. The checkpoint is for inference, not exact mid-run resume: optimizer and RNG states are not stored.

## I. Preliminary scores and calibration

Spatial maps are the mean squared RGB residual at each pixel, shape [B,H,W]. Image scores average those maps spatially, shape [B]. Both use raw [0,1] intensity-squared units. No per-image score normalization, pixel threshold, smoothing or mask-driven tuning was performed.

After checkpoint selection, 68 normal calibration records produced a mean image score of 0.0023358125. The predeclared float64 linear 95th percentile is **0.0031531278509646654**. Strict `score > threshold` produces **4/68 (5.88%)** calibration exceedances. This is an in-sample count, not an unseen-data FPR estimate, guarantee, anomaly recall or F1-optimal threshold. It cannot be used as a pixel threshold. With only 68 possibly dependent normals, extreme-tail claims would be unsupported.

## J. Tests and coverage

Default offline tests: **258 passed, 1 skipped**. All 194 baseline passing tests are preserved; **64 new synthetic tests** pass. The inherited skip is Windows symlink creation privilege (WinError 1314). Actual junction containment/rejection tests pass. Default tests use tiny synthetic images/toy tensors and block in-process network connections; no VisA download is required.

Explicit real-data tests: **6 passed** (existing full inventory/splits/DataLoader checks) and **4 passed** (new completed-training/checkpoint/calibration checks). These checks do not train or score official test images through the model.

| Coverage scope | Statements | Branches |
|---|---:|---:|
| Entire package | 926/937 (98.83%) | 358/366 (97.81%) |
| New fingerprint screen | 30/30 | 8/8 |
| Autoencoder | 24/24 | 10/10 |
| Training configuration | 28/28 | 10/10 |
| Training/validation/early stopping | 77/77 | 24/24 |
| Checkpoint and artifact functions | 76/76 | 28/28 |
| Reconstruction/calibration | 14/14 | 6/6 |

Combined package coverage: **98.54%**, passing the unchanged 95% gate. New listed modules have 100% measured statement/branch coverage, not proof of correctness for every possible input. The remaining uncovered lines/branches belong to the pre-existing archive defenses and module entry wrapper. Scripts, including orchestration and Windows resource querying, are outside this coverage denominator and were exercised by explicit real execution.

## K. Ruff, lock and packaging

Ruff lint and formatting pass. Offline `uv lock --check` and installed dependency compatibility pass; `uv.lock` is byte-for-byte unchanged from the checkpoint. Existing Python/torch/torchvision/NumPy/Pillow versions and all development pins remain intact.

Source distribution and wheel-from-sdist builds pass offline. Actual archive-member inspection confirms that original images, masks, CSVs, trained weights, runs, reports, caches, temporary environments and tools are absent. uv's warning that its cache sits inside the source root was checked against the actual archives; no cache entries were packaged.

A fresh `.tmp/wheel-env-2a` runtime environment installed the built wheel from the lock/cache without development dependencies. Isolated `-I` import resolved to its site-packages. A synthetic Adam update changed model parameters; the real checkpoint loaded and reproduced all 67 normal validation scores. Its 14 distributions (project plus runtime dependencies) pass compatibility checks, with no pytest/Ruff/Hatchling installed. Exact build hashes and member lists are in the verification evidence.

## L. VS Code verification

The existing workspace and Python 3.12.15 environment were used, with no global editor changes or duplicate repository. Added tasks are model unit tests, explicitly invoked CPU training smoke and explicitly invoked full CPU training. The training tasks prompt for new run IDs and have no folder-open trigger. The focused model-unit task was actually executed in VS Code: **63 passed** at that point; the final CLI suite includes one additional negative-min_delta regression case (64 focused tests total).

The new debug configuration launched the real two-batch smoke. An actual breakpoint at `src/factoryvision/training/engine.py:51` stopped after backward and finite-gradient checks, with real loss (~0.1263), reconstructed tensors, batch, model and Adam optimizer visible. Stepping executed `optimizer.step()`; continuing completed two updates, checkpoint save/load and matching normal validation scores. The verified breakpoint was left disabled, and the debugger run is preserved separately.

Ruff/Python/debugpy/Pylance extensions remain installed. Official Codex `openai.chatgpt@26.1021.22027` is installed and its current task panel displayed the ongoing responses. No separate authenticated or billable Codex service probe was run. Python test discovery was refreshed after adding the training module; task execution and pytest results are independently verified. Browser/frontend testing is outside this milestone.

## M. Limitations, failures and unverified assumptions

- The heuristic near-duplicate screen is broad and incomplete as a manual review. Its 7,946 flags are not confirmed duplicates, but neither are the unreviewed pairs proven independent. Physical objects and acquisition batches are unknown.
- One category, one full seed/run and a small normal calibration set cannot establish production generalization, robust tail behavior or seed variability. No anomaly quality, final image metrics or pixel metrics were evaluated.
- Normal validation MSE selects reconstruction quality, not anomaly discrimination. Reconstructions are blurred; ordinary texture/boundaries contribute residuals. Square resizing distorts aspect ratio.
- Deterministic seeds and local exact reload equality do not guarantee cross-hardware or cross-version reproducibility. No GPU acceleration or driver work occurred.
- Checkpoint file hashes are local integrity checks, not authenticity signatures. Filesystem checks assume trusted local control and are not race-proof. The two checkpoint files are not a single atomic transaction; mismatches are rejected. Exact optimizer resume is unsupported.
- One inherited Windows symlink privilege test remains skipped. Actual junction tests pass; native CPU torch and real optimization worked without any security-policy change.
- A formatting-stage long-line Ruff failure was fixed before training; final review also tightened negative min_delta rejection and failure-message privacy, with an additional regression case. All final lint/tests pass. No real training attempt failed. Version-specific 2.13 documentation URLs were unavailable through browsing, so current official 2.14 guidance was consulted and installed 2.13 APIs were verified directly; dependencies were not upgraded.
- Historical repository documents include local paths. Exported experiment metadata was separately checked to exclude them. No telemetry, cloud tracking or paid external service was introduced.

## N. Git state and proposed Milestone 2B

HEAD remains `e95423eb92be93a4954c6b9ddd62084324ff340b` on `main`, with exactly two local commits total. The 13 new and 12 modified Milestone 2A files remain **uncommitted and unstaged**. No remote, push or publication exists. Data and model artifacts remain ignored. Final diff whitespace and evidence-file integrity checks pass.

Recommended Milestone 2B, only after separate authorization: address or explicitly retain the remaining leakage/physical-identity uncertainty; freeze the checkpoint, preprocessing, image/spatial score semantics and normal threshold; predeclare held-out image AUROC/AP, frozen-threshold confusion/FPR metrics and appropriate pixel metrics with geometry and uncertainty rules. Do not repurpose the image threshold as a pixel threshold or tune on test outcomes. Additional training seeds, models, categories, GPU setup and publication are separate decisions.

**Milestone 2A stops here. No final held-out evaluation or Milestone 2B work has started.**

Evidence: milestone-2a-verification.json (local evidence, not distributed: `docs/evidence/milestone-2a-verification.json`). Detailed method and commands: [training.md](training.md).
