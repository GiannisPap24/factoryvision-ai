# Feature-memory development experiment - Milestone 2C

This is a CPU-first, PatchCore-inspired prototype, not a reproduction of published PatchCore. The [PatchCore paper](https://arxiv.org/abs/2106.08265) motivates nominal patch-feature memories. This implementation uses a smaller backbone, fixed deterministic sampling rather than a greedy coreset, RMS 1-nearest-neighbor distances, an unweighted maximum image score and no Gaussian smoothing. It has no pretrained feature search, learned projection, FAISS dependency or ensemble.

The immutable [development recipe](protocols/feature-memory-development-v1.json) was saved before fitting or inspecting development scores. Its SHA-256 is `62ebced7c38f87a73a31099895de9120dc2320a0906bf3ed7263523f13b656fa`. One recipe, one successful fit and one development evaluation were executed; there was no score-driven retuning. The previously observed Cashew test experiment remains historical and was not re-evaluated.

## Backbone and preprocessing

Installed torch 2.13.0+cpu and torchvision 0.28.0+cpu are reused. Full-frame RGB is bilinear antialiased resized to 128x128 and divided by 255, then normalized with ImageNet mean [0.485,0.456,0.406] and std [0.229,0.224,0.225]. This deliberately differs from the classification preset's resize256/center-crop224 to bound CPU work and retain the full localization field. It was declared in advance and is not claimed to be optimal.

The frozen ResNet18 stem/layer1 feeds layer2 [128,16,16] and layer3 [256,8,8]. Each map receives 3x3 stride-one average pooling, padding one including zeros. Layer3 is bilinearly resized to 16x16 with align_corners=False and concatenated with layer2 into 384-dimensional patch descriptors. No per-layer whitening, feature normalization or learned projection is fitted. Unused layer4/classifier modules are not retained by the extractor. Its 2,782,784 parameters are frozen, all BatchNorm modules stay in eval mode, and inference runs without gradients. Before/after hashes of parameters and buffers are identical.

## Pretrained provenance and terms

The installed enum and [versioned official source](https://raw.githubusercontent.com/pytorch/vision/v0.28.0/torchvision/models/resnet.py) identify `ResNet18_Weights.IMAGENET1K_V1` at `https://download.pytorch.org/models/resnet18-f37072fd.pth`. The official TLS endpoint delivered **46,830,571 bytes**, within the authorized 100 MB maximum. Default certificate validation was enabled; redirects were refused; no account credential was needed. Published filename hash prefix `f37072fd` matched. Full local SHA-256: `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`. That full digest is a locally recorded integrity value, not a separately published signature.

Weights and acquisition metadata live under ignored `data/pretrained/`. The loader never downloads implicitly, verifies size/hash, validates state tensor shapes/types/finiteness and uses weights_only loading. The first strict custom key check rejected this legacy official file because it omits 20 BatchNorm num_batches_tracked counters. PyTorch's own strict loader accepted it. The compatibility rule now permits precisely the expected full state or that full state without counters; missing learned weights remain errors. A synthetic regression test verifies equivalent restored state.

Torchvision code has the [BSD 3-Clause license](https://raw.githubusercontent.com/pytorch/vision/main/LICENSE). [Torchvision's weight guidance](https://docs.pytorch.org/vision/stable/models.html) warns that pretrained models can carry dataset-derived terms. [ImageNet's access page](https://image-net.org/download.php) points to dataset access terms; its separate accessagreement endpoint could not be retrieved during this review. These sources do not establish blanket commercial/redistribution permission for the weights. This milestone uses them locally for research and does not redistribute them or the underlying ImageNet data. Reassess rights before any later release or commercial use. No legal clearance is implied by the software license alone.

## Isolation, fitting and score

The accepted manifest/split fingerprints remain unchanged. The DevelopmentDataset refuses official test partitions and accepts only official-training normal records without stored masks. The real runner installs an allowlist-based Python file-open audit hook before image access. All 450 permitted normal files are hash-verified; its logs show exactly those files and zero official-test image/mask reads. Metadata from the accepted manifest/audit is used only for identity and membership checks, not test-driven selection. The hook is a defense for this Python/Pillow pipeline, not a general operating-system sandbox for arbitrary native software.

Fitting uses exactly 315 training normals, in sample-ID order, batch eight, zero workers, four CPU threads. Each image yields 256 candidate patches. For each ID, seed the local Torch generator from the first eight big-endian SHA-256 bytes of `42|sample_id`; select the first 16 entries of randperm(256), then sort those indices. The saved bank contains 5,040 x 384 float32 values (7,741,440 bytes), selected from 80,640 candidate patches. Saved patch-index records make sampling auditable. The cap is 8,192 patches (12,582,912 tensor bytes); exceeding it is an error, not an implicit sample discard.

For each query patch, compute RMS Euclidean distance to its closest bank patch: sqrt(max(sum(q^2)+sum(b^2)-2q.b,0)/384). Query/bank blocks are 256/512 by default, each side hard-capped at 512; a single pairwise matrix therefore uses at most 1 MiB. This bounds individual distance matrices, not total process RAM. The image score is the maximum raw 16x16 patch distance. Bilinear interpolation (align_corners=False) produces a 128x128 continuous spatial map. Higher scores mean more anomalous. Float32 distance arithmetic may incur small rounding differences; near-zero negative squared distances are clamped to zero.

Calibration uses only 68 separate normal calibration scores with float64 95th-percentile linear interpolation, strict score > threshold. The threshold is **0.08742312490940093**. It is an image threshold; no pixel threshold was calibrated. Validation normals do not enter the bank or quantile calculation.

## Measured development evidence

| Normal partition | Count | Min | Median | 95th percentile | Max | Above threshold |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 67 | 0.06195803 | 0.07229325 | 0.08581289 | 0.10171185 | 2 (2.985%) |
| Calibration | 68 | 0.06304479 | 0.07153935 | 0.08742312 | 0.10850630 | 4 (5.882%) |

Calibration exceedances are an in-sample description, not an independent FPR estimate. The 67 validation normals provide only a small, same-category check; neither partition provides real anomaly recall.

Sixteen synthetic probes use the first 16 normal validation IDs, each modified with one 16x16 magenta/green checkerboard of 4x4 cells. Top-left coordinates are y=40+SHA256(`pseudo-v1|ID`)[0]%33 and x=40+digest[1]%33. The binary mask is exactly the overwritten square. These locations and colors were not optimized against results. Original pixels are unchanged outside the square and input images are cloned.

All 16 scores increased (median paired increase 0.15725876), all exceeded the image threshold, and all spatial maxima were inside the injected masks. Pooled **synthetic-only** pixel AUROC is **0.99798822**, AP **0.88751292**, over 262,144 pixels including 4,096 artificial positives. These obvious high-contrast squares are unlike subtle industrial defects; there is no claim that these figures predict real defect accuracy or outperform the historical autoencoder. Four labelled figures in `runs/cashew-feature-v1/synthetic-example-00.png` through `03.png` show broad responses around the squares. Their display range [0, image-calibration threshold] is fixed after normal calibration and before probe scoring; clipping is display-only, not a pixel decision rule.

## CPU resources

AMD Ryzen 7 9700X, Windows AMD64, Python 3.12.15, CPU-only Torch with four threads. Fitting/feature extraction plus deterministic bank construction took **3.1513 seconds**, with **478,789,632 bytes (456.61 MiB)** peak working set. Development scoring/reporting took **3.1321 seconds** excluding imports/preflight, with **468,221,952 bytes (446.53 MiB)** peak. Bank serialization is 7,743,017 bytes; pretrained source download is 46,830,571 bytes, although only the truncated feature extractor is retained in model memory. There are zero trainable parameters in this experiment.

| Batch-one operation | Median ms | p95 ms |
|---|---:|---:|
| Feature extraction | 3.40775 | 4.930205 |
| Nearest-neighbor search | 4.71600 | 5.163330 |
| Full preloaded feature+search+map | 9.44965 | 11.029735 |
| Decode/resize, warm cache | 7.59400 | 8.073910 |

The first normal validation image was used for all timing, with five warmups and 30 serial repetitions per inference component; decoding used 30 repeats after the file was already warm. Search timing excludes reshape/upsampling; total includes validation checks, reshape and map interpolation. Components were measured separately and their medians are not additive. Startup, disk cold-cache, camera I/O and debugger work are excluded. No industrial throughput or latency guarantee is made.

## Reproducibility commands

The successful run is preserved under ignored `runs/cashew-feature-v1/`. Fitting refuses an existing run; evaluation refuses an existing start marker. Never remove these guards to overwrite evidence.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_feature_memory.py -q
.venv/Scripts/python.exe -m pytest integration_tests/test_feature_development_real.py --basetemp=.tmp/pytest-feature-real -q
.venv/Scripts/python.exe scripts/debug_features.py
```

The explicit first-run commands were `scripts/feature_experiment.py fit` then `scripts/feature_experiment.py evaluate`. Both stages verify recipe/manifest identity and restrict reads; evaluation also checks the fitted source snapshot. The artifact loader verifies bank hashes, recipe, pretrained identity, expected fitting IDs and tensor bounds. No original 2A/2B source or experiment artifact was rewritten.
