# Milestone 5D: local integration candidate

This candidate combines accepted public main `8a303e9b9641a6d01379a7d20e02239315acbd2d`
with the unchanged 5D checkpoint ancestry ending at
`f6762f9a4aeab3d2d46044415faa26a9b0121a37`. Merge preparation is local and uncommitted;
publication and a final integration commit require owner approval.

## CPU baseline and display allocation

[5D-1](performance-baseline.md) measures generated 128- and 512-pixel images through
validation, preprocessing, analytical diagnostics, display and PNG preparation in
bounded Windows workers. It performs no neural model inference. Three processes
per size use five warmups and thirty timed calls. Historical pipeline p50 values
were 1.962 ms and 18.751 ms respectively; these are synthetic software diagnostics.

[5D-2](performance-display-5d2.md) rounds and converts each interpolated channel to
uint8 before stacking RGB. At 512 pixels this changes the stacked allocation from
6 MiB to 0.75 MiB and avoids the full rounded float64 RGB temporary. Historical
mean peak process commitment decreased about 7.96 MiB. Independent oracle tests
check exact pixels, PNG bytes, geometry, palette boundaries, auxiliary panels and
unchanged inputs/scores across all six method identifiers. Latency improvement
remains **INCONCLUSIVE**; this is an allocation reduction, not a validated speedup.

The checkpoints are `ceb06ba3e973a140d954d645831d954a7f581a46` (5D-1),
`4e8f3c525ccccbe7d8c87555f850218ab2cff675` (5D-2), and
`f6762f9a4aeab3d2d46044415faa26a9b0121a37` (5D-3).

## ONNX and DirectML synthetic evidence

[5D-3](onnx-synthetic-feasibility.md) exported a deterministically initialized,
random-weight ConvAutoencoder: FP32 `[1,3,128,128]`, opset 18, IR 10. Five generated
inputs passed original PyTorch versus ONNX CPU parity at fixed `atol=1e-6`,
`rtol=1e-5`; maximum absolute error was 5.96e-8. Production validation, scoring,
preprocessing and thresholds were unchanged. Optional ONNX tests skip when their
isolated dependencies are absent. No ONNX runtime is an application dependency.

5D-4A/4A2 verified the official isolated DirectML runtime and corrected the provider
diagnostic to recognize legitimate fusion. 5D-4B verified all five cases against
the original hash-checked PyTorch references and same-runtime CPU outputs.
Maximum absolute error remained 5.96e-8. Five profiled fused graph executions were
assigned to DirectML with CPU node fallback disabled and runtime fallback disabled.
Raw traces, graphs, arrays, wheels and experimental runtimes remain ignored locally.
They are not source-distribution or wheel content.

The bounded comparison used the same ONNX Runtime DirectML 1.24.4 distribution
for CPUExecutionProvider and DmlExecutionProvider on an AMD Radeon RX 9060 XT,
Ryzen 7 9700X and Windows 11 build 26300. One fixed random input, four intra-op
threads, one inter-op thread, sequential execution and disabled graph optimization
were used. Three independent processes/provider each performed five warmups and
thirty timed calls. First inference was warmup one: 210 benchmark calls plus ten
parity calls, no retries. Synchronous CPU-input/CPU-output `session.run` timing
included transfers and completion; profiling was outside timed samples.

| Historical measurement | CPU | DirectML |
|---|---:|---:|
| Warm p50 | 0.23735 ms | 0.17990 ms |
| Warm p95 | 0.29167 ms | 0.25448 ms |
| Warm mean / sample SD | 0.24512 / 0.02283 ms | 0.19443 / 0.04867 ms |
| Session creation | 6.8-7.6 ms | 80.5-82.6 ms |
| First call | 0.65-0.71 ms | 3.30-3.70 ms |
| Peak process commitment | 41.1-41.4 MiB | 87.6-87.9 MiB |
| Worst timed sample | 0.3604 ms | 0.5102 ms |

Warm median latency was 24.2% lower, an absolute difference of only 0.05745 ms.
All three worker-median comparisons favored GPU with separated median ranges and
pooled interquartile ranges. GPU tails were more variable and overlapped CPU;
one paired GPU p95 was slower. New-session startup and host memory were worse.
Observed GPU memory was approximately 8.68 MiB local and 5.625 MiB nonlocal;
these are snapshots, not peak VRAM or a hard GPU-memory cap.

## Owner decision and integration boundaries

**Close the current GPU feasibility experiment as successful synthetic research.**
Production DirectML integration is deferred. No end-to-end application or trained-model
speedup, industrial throughput, industrial accuracy or general GPU operator support
has been demonstrated. The tests use random weights and synthetic images only.
No new GPU benchmark or inference is part of this integration acceptance.

The candidate retains public main's pandas 3.0.6 pin and uv.lock, Synthetic Demo
first-run path, Model Catalog availability messaging, contextual navigation warning,
Human Review, five README screenshots and Windows security safeguards. Only the
shared display allocation changes the existing application path. ONNX/DirectML,
driver and Windows policy changes are absent. Existing MIT licensing and third-party
attribution remain applicable; experimental runtime redistribution is not proposed.

All nine scientific resources must retain their pinned bytes and SHA-256 values:
PCB1 **BLOCKED**, PCB2 **INCONCLUSIVE**, historical Cashew official test **already
observed**, MetalParts **exploratory-only**, RobustAD target **NOT CLEARED**,
Human Review **not verified ground truth**, and the real Cashew pilot **BLOCKED**.

Remaining work is final offline integration acceptance, packaging/privacy and fresh
installed-app verification, followed by owner review for any commit or publication.
Later real-model parity/performance needs separately approved artifacts, inputs,
protocol and eligibility. No UI redesign, camera work or Milestone 5E begins here.
