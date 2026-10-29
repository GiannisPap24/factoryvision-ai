# 5D-3: isolated synthetic ONNX CPU feasibility

Decision: **GO for the narrowly scoped synthetic operator proof of concept**.
Owner review is still required before any further milestone or publication.
This is not trained-model parity, production performance evidence, an anomaly
detection evaluation, or GPU feasibility evidence. The real Cashew pilot is BLOCKED.

## Scope and repository

Only the authorized public staging checkout was accessed. Branch:
`feat/performance-5d-v1`. Origin: the owner's public FactoryVision repository.
Remote main was checked with `ls-remote` and remained
`4c3987e669360e36930ffe8d4317dd040bc0f47c` at preflight.
No private development checkout was accessed and no editor session was changed.

The eight directly reviewed 5D-2 files were checkpointed locally as
`4e8f3c525ccccbe7d8c87555f850218ab2cff675`, with parent
`ceb06ba3e973a140d954d645831d954a7f581a46`. Author and committer both used the
verified GitHub noreply identity. Its exact eight-file inventory was verified;
the working tree was clean immediately afterward.

Unchanged 5D-2 evidence was reused: 339 focused tests, exact pixels and PNG bytes,
reduced intermediate float64 allocation and approximately 7.96 MiB lower observed
mean peak process commit. Latency remains **INCONCLUSIVE**.

The complete 5D-3 source inventory is three new, untracked files:

- `scripts/onnx_synthetic_probe.py`
- `tests/test_onnx_synthetic_contracts.py`
- `docs/onnx-synthetic-feasibility.md`

Nothing from 5D-3 is staged or committed. No push, PR, merge or release occurred.

## Mandatory capability preflight

The actual callable terminal/file-edit tools, tool catalog and installed skill
catalog were inspected. Relevant installed instructions were read for
`plugin-management:plugin-management` and `security-best-practices`. Existing
Python 3.14.10, CPU PyTorch, uv, pytest and Ruff were reused. Official package and
API sources were checked using the available web tools. No new Codex plugin or
skill was needed or installed; no account connection or elevation was requested.

Browser and computer-use skills and the callable CUA entry point are available
in the session. This non-UI milestone did not exercise them; browser operation
and Chrome connection are **unverified**, not claimed successful or missing.
There is no missing capability blocking the synthetic CPU experiment.
Python packages below are experiment dependencies, not Codex plugins.

## Dependency isolation and provenance

The disposable environment is `.tmp/onnx-synthetic-5d3/env`. Official PyPI JSON
metadata and wheel filenames were reviewed before installation. Downloads came
from `files.pythonhosted.org`; their SHA-256 digests matched PyPI. Wheel members
were checked for path traversal, startup `.pth` files and executable script
installation payloads. Only compatible wheels were used; no source builds.
Installation used offline local wheels, exact hashes and `--no-deps` after all
required dependency constraints were reviewed. All constraints passed afterward.

| Installed package | Version | Wheel / Python compatibility | License |
|---|---|---|---|
| onnx | 1.23.2 | cp312-abi3-win_amd64, compatible with CPython 3.14 | Apache-2.0 |
| onnxscript | 0.7.2 | py3-none-any | MIT |
| onnxruntime | 1.31.1 | cp314-cp314-win_amd64, CPU distribution | MIT |
| ml_dtypes | 0.6.0 | cp314-cp314-win_amd64 | Apache-2.0 |
| onnx_ir | 1.1.0 | py3-none-any | Apache-2.0 |
| flatbuffers | 25.12.19 | py2.py3-none-any | Apache-2.0 |

Exact wheel SHA-256 values, in the same order:

```text
9b382ba898a7c142a0801d03cf04ecabced96c1543c7b643a86f0928143802de
d0e7121c6a1eefd608058928e111cbdb76709f70d269ff0d07aee493bd1d13c9
0a3998e18cb667ef54287556bd62291a788ed235723dd90c83b2294b24e71d81
d574c2b28921dc72e869df248f1a278f6eee176a1f237c8642e1a71eb15f3977
76cc313a788c073b75271c1b6abb172aaf90fcc6335ed40f9f3238b2064a2fa3
7634f50c427838bb021c2d66a3d1168e9d199b0607e6329399f04846d42e20b4
```

The ignored directory retains the wheelhouse and `dependency-provenance.json`,
including dependency constraints and official project URLs. Third-party license
metadata remains in those wheels; no third-party distribution is being published.
The initial uv installation did not explicitly pin its cache directory, so its
normal per-user cache may also have been used. That cache was not inspected or
cleaned; this report does not claim all installer cache activity was confined to
the disposable environment. Future setup should use `--no-cache` or explicitly
set the cache to a project-local ignored directory.

Existing primary site-packages are added read-only to the experiment's import
path; they are not copied or installed into the disposable environment. Shared
versions: torch 2.13.0+cpu, numpy 2.5.4, protobuf 7.36.2,
typing_extensions 4.16.0, packaging 26.3, sympy 1.14.0. They were rechecked.
All six new packages remain absent from the primary environment.
`pyproject.toml` and `uv.lock` are byte-identical to the checkpoint.

Native ONNX, ml_dtypes and ONNX Runtime imports succeeded under the current
Windows policy. No Application Control bypass or security-policy change was
needed. Runtime advertises Azure and CPU providers; the session explicitly uses
only `CPUExecutionProvider`, with fallback disabled and telemetry disabled.
No GPU runtime, driver, model download or external account was installed/used.

Official sources:
[ONNX](https://pypi.org/project/onnx/1.23.2/),
[ONNX Script](https://pypi.org/project/onnxscript/0.7.2/),
[ONNX Runtime](https://pypi.org/project/onnxruntime/1.31.1/),
[ml_dtypes](https://pypi.org/project/ml-dtypes/0.6.0/),
[ONNX IR](https://pypi.org/project/onnx-ir/1.1.0/),
[FlatBuffers](https://pypi.org/project/flatbuffers/25.12.19/),
[CPU installation](https://onnxruntime.ai/docs/install/),
[provider and session API](https://onnxruntime.ai/docs/api/python/api_summary.html).
The online PyTorch stable documentation redirected to a newer release; the
installed 2.13 source and signature were used to verify actual exporter behavior.

## Architecture, export and safety boundary

Reviewed original autoencoder, inference service, reconstruction scoring,
checkpoint/provenance contracts and existing validation tests. No production
file was changed. Original `forward` validates shape, batch, dtype, CPU device,
finite values and [0,1] range. Decisions using `isfinite(...).all()`, `min() < 0`
and `max() > 1` are Python control flow unsuitable for the tensor-only graph
contract. Production checks
remain active; tests exercise their rejection behavior at resolution 128.

`TensorOnly` reuses the **same original encoder and decoder objects**, with no
replacement learned operations. It is an export adapter, not an application
validation API. The exact class and resolution are checked. CPU initialization
uses seed 20261029 and restores the caller's RNG; synthetic random inputs use
20261030. Evaluation and `torch.inference_mode()` are used throughout comparison.
Parameter classification: `randomly_initialized_synthetic_weights`.
Parameter digest: `1fff2cf23381f1d1fc9bfafc45374adcdf847635d1a9ef4d5cb0b56d068c967b`.

Actual export succeeded using PyTorch 2.13.0+cpu, `dynamo=True`, static shape,
opset 18, IR version 10 and no external data. The installed exporter implements
torchlib opset 18. ONNX full structural validation passed.

- Input `image`: float32 `[1,3,128,128]`.
- Output `reconstruction`: float32 `[1,3,128,128]`.
- Operators: Conv x4, ConvTranspose x4, Relu x7, Sigmoid x1.
- Initializers: 16 float32 tensors from random initialization.
- Graph size: **154,831 bytes**.
- Graph SHA-256: `ac952909f36717a6fcdc45d40d766c1a46188a7edcf4c257092ebb2c63f13abc`.
- 35 nonsemantic annotation fields were removed before validation and saving.
- External-data references, custom domains/functions and executable custom ops: zero.

The exporter graph is built in memory. The validator rejects unexpected shape,
dtype, operator counts, domains, functions, training graphs, external tensors,
metadata, path-bearing strings and unsupported attribute types before any
runtime session. Only known integer attributes and `auto_pad=NOTSET` strings are
allowed. Runtime consumes these validated bytes, never a supplied model path.
Only then is the random-weight graph saved inside the ignored experiment run.

The launcher has no input/model/path arguments. It uses a fresh child, a fixed
small workload, four intra-op threads, one inter-op thread, sequential ORT
execution and a 60-second process timeout. Graph optimization is disabled for
this operator comparison. Each launch creates a fresh ignored run directory;
linked output roots are refused. A Python audit hook rejects networking except
the local hostname query needed by imports, subprocesses, pickle class loading,
weight-file access and covered filesystem operations outside fixed approved
read roots / the run's write root. Link and metadata mutations are rejected.

This is **not** an OS security boundary for hostile native code or arbitrary
models. Python audit hooks do not observe every native DLL filesystem/network
operation. There was no observed unexpected side effect, but no comprehensive
native network trace was collected and no hard process-memory cap is claimed.
No 5C IPC interface or production ONNX loading path was introduced.

## Frozen numerical method and results

Before the first parity comparison, tolerances were set in source to
`atol=1e-6`, `rtol=1e-5` using elementwise
`abs(runtime-reference) <= atol + rtol*abs(reference)`. Float32 epsilon is about
1.19e-7; separate convolution implementations can accumulate rounding differences.
The absolute term covers values near zero; relative error is reported only where
`abs(reference) >= 1e-3`. These are compatibility tolerances for this bounded
random-weight experiment, not calibrated tolerances for trained research models.
They were never changed after observing results.

Errors below are computed in float64 from float32 outputs. Five internal RGB
inputs were used: all-zero, all-one, horizontal ramp, checkerboard, fixed random.

| Input | Maximum absolute error | Mean absolute error | Maximum relative error |
|---|---:|---:|---:|
| zeros | 5.96046448e-8 | 9.88196310e-9 | 1.17671631e-7 |
| ones | 5.96046448e-8 | 9.39811192e-9 | 1.17669651e-7 |
| ramp | 5.96046448e-8 | 9.78009969e-9 | 1.17668474e-7 |
| checker | 5.96046448e-8 | 9.80677820e-9 | 1.17671798e-7 |
| random | 5.96046448e-8 | 9.95957331e-9 | 1.17672351e-7 |

All five pass the fixed tolerance. Original versus wrapper is bit-exact; repeated
PyTorch and repeated ORT outputs are bit-exact within each backend. Shapes,
float32 dtype and finiteness pass. A subsequent safety-hook refinement was
reverified with the actual runner: graph digest and numerical results were
unchanged. No scoring replacement, thresholds, labels or research metrics were
exported or evaluated. The existing host scoring implementation was read only.

## Failures, warnings and tests

Two development attempts stopped before runtime comparison:

1. The exporter's progress checkmark failed under the child process's Windows
   cp1252 stdout encoding. Explicit UTF-8 streams and `verbose=False` fixed this.
2. The initial strict graph attribute allowlist omitted the standard convolution
   string attribute `auto_pad=NOTSET`. It was narrowed to permit precisely this
   value; arbitrary strings and tensor/graph attributes remain rejected.

Neither failure required production changes, dependency substitution, security
policy changes or relaxed numerical tolerance. Export emits one upstream
`FutureWarning` about deprecated `LeafSpec` checks. Full local diagnostics remain
ignored; only sanitized warning descriptions are in this document.

Final verification:

- Primary environment: **42 passed, 13 optional skipped**, 0.95 seconds.
- Isolated ONNX environment: **55 passed**, 3.18 seconds; one upstream warning.
- Directly affected regressions: **101 passed**, 4.24 seconds, covering original
  model shape/backprop/validation, inference, registry and checkout contracts.
  Existing `PurePath.is_reserved()` deprecation emitted 39 warnings.
- The bounded actual probe passed on all five inputs with the final safety hook.
- Ruff lint/format and whitespace checks passed for the new source files.

Optional tests validate the actual generated graph, CPU runtime parity and 12
unsafe graph mutations. Offline tests cover same-module identity, deterministic
parameters, exact wrapper equivalence, invalid original inputs, blocked I/O and
deliberately malformed or inaccurate runtime outputs. Test guards disallow
checkpoint loading, state-dict loading and model download helpers.
Normal imports and CI do not require ONNX; no CI or dependency configuration changed.

To rerun in the already reviewed isolated environment, from the repository root:

```powershell
.tmp/onnx-synthetic-5d3/env/Scripts/python.exe -I -B scripts/onnx_synthetic_probe.py
.venv-31410/Scripts/python.exe -B -m pytest tests/test_onnx_synthetic_contracts.py -q
```

The environment intentionally reuses the primary CPU packages read-only; it is
disposable, not a self-contained deployment or a fresh-install acceptance claim.
Full CI-equivalent tests, packaging scans and fresh installation remain deferred
to the integrated 5D acceptance gate, as requested.

## Final security/privacy and scientific audit

All three new files were reviewed. No secrets, personal email addresses, machine
paths, private sample identifiers, research weights or binary graphs are in the
source changes. Package MIT licensing is unchanged. Temporary model files,
environment, provenance and process logs remain under ignored `.tmp/`; `*.onnx`
is also ignored. Git has no staged files or tracked ONNX files. The source
distribution's explicit inclusion list excludes `.tmp/`, and the wheel includes
only `src/factoryvision`. No generated model is eligible under those rules.

All nine pinned scientific resources were compared against both their accepted
SHA-256 digests and accepted main bytes: **9/9 unchanged**. Architecture, inference
service, reconstruction scoring, model registry, thresholds, preprocessing,
historical evidence and 5C protocols are unchanged. No real dataset, checkpoint,
pretrained model or historical research-output directory was opened.

- PCB1: **BLOCKED**.
- PCB2: **INCONCLUSIVE**.
- Cashew official test: **already observed**.
- MetalParts: **exploratory-only**.
- RobustAD target evaluation: **NOT CLEARED**.
- Human Review: **not verified ground truth**.
- Real Cashew development pilot: **BLOCKED**.

Next prerequisites, requiring separate owner review: resolve the real-data and
trained-artifact admission gates before trained-model parity; define an approved
host-scoring comparison protocol and tolerances; establish any GPU platform,
provider, driver, security-policy and license compatibility independently.
No GPU speedup, production reliability or industrial deployment readiness follows
from this experiment. No 5D-4 work was started.
