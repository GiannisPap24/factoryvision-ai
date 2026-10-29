# Source-only robustness diagnostics

Milestone 5C implements software testing infrastructure: strict plans, deterministic
image perturbations, numerical score/raw-map stability, a bounded analytical runner,
scalar reports, proposal-only development admission and Windows process containment.
The probes generate original synthetic pixels internally. They do not load a dataset,
checkpoint or neural network. There is no new robustness UI or validated OOD detector.

## Run from an installed source checkout

Complete the [Windows installation](../local-demo.md) first. From the repository root,
run this PowerShell example; it writes only scalar JSON to the terminal:

```powershell
@'
from factoryvision.robustness.contracts import Category, Kind, Parent, ParentID, Perturbation, Plan
from factoryvision.robustness.runner import run
from factoryvision.robustness.reporting import json_bytes

plan = Plan(
    Category.CASHEW,
    (Parent(ParentID.FIRST, 10, 10),),
    (Perturbation(Kind.IDENTITY), Perturbation(Kind.BRIGHTNESS, 1.1)),
)
print(json_bytes(run(plan)).decode())
'@ | .venv-31410/Scripts/python.exe -
```

This performs three identity evaluations and one brightness evaluation using RGB
means. `Category.CASHEW` selects a software contract, not a real sample or model.
The public runner accepts fixed metadata and an exact analytical adapter, not paths,
uploaded images, arbitrary callbacks or model identifiers. Reports remain in memory
unless the caller explicitly saves the returned bytes.

To exercise the fixed isolated Windows worker:

```powershell
@'
from factoryvision.robustness.cashew_isolated_ipc import IsolatedCashewProbe, json_bytes
print(json_bytes(IsolatedCashewProbe().run()).decode())
'@ | .venv-31410/Scripts/python.exe -
```

Expect `kind=synthetic_service_interface_probe`, `model_execution=not_run` and,
when all checks succeed, `run_status=complete`. Any aborted or unknown result is
not successful execution. One single-use session contains five charged analytical
attempts; there are no automatic retries. Numeric values are synthetic diagnostics,
never actual neural-network measurements.

## Contracts and limits

- [Numerical protocol V2](robustness-v2.md) explicitly supersedes V1 contrast and
  noise-stream rules. Report schemas and IPC transport remain separately versioned
  at 1; those numbers do not mean the numerical protocol reverted to V1.
- [The in-process runner](robustness-runner-v1.md) bounds parents, dimensions,
  analytical calls and report bytes. Its cancellation is cooperative; it is not
  an OS memory limit or hard deadline. RGB-mean and constant-map interface doubles
  are deliberately different analytical fixtures, not competing model results.
- [The IPC worker](cashew-isolated-ipc-v1.md) accepts only its reproducible 10 by 8
  RGB fixture: 4 KiB JSON and 4 KiB PNG frames, strict sequencing, bounded PNG
  decompression and scalar replies. This PNG cap is **not a validated transport
  contract for full-size real development images**. Maps stay inside the worker.
- The tested local environment is Windows AMD64 build 26300, Python 3.14.10 and
  uv 0.13.5. Nested jobs, suspension, handle inheritance, deadlines and crash
  containment were exercised in that environment. Other hosts need their own
  evidence; successful local tests do not guarantee a hosted runner will pass.
- IPC defaults are 256 MiB process/job committed-memory limits, one active process
  and a five-second execution deadline. These are synthetic fixture limits, not
  validated real-model budgets. Working-set observations differ from committed
  memory and are not a complete system-memory bound. OS scheduling and synchronous
  setup calls prevent a hard real-time guarantee. Cleanup permits three separate
  waits of up to two seconds, plus up to two seconds to join IPC I/O.
- Windows Job Objects provide process/resource containment, **not a complete
  filesystem/network security sandbox**. The fixed worker has additional Python
  I/O guards after initialization; native or hostile code is outside that claim.
- Process creation is suspended but creation and job assignment are **not atomic**.
  A surviving supervisor verifies assignment before resume and fails closed on
  errors. A crash before assignment can leave a suspended child outside the
  dedicated job; kill-on-close protection applies only after assignment.
- The supervisor-owned ledger is in memory and **does not survive a supervisor
  crash**. After uncertain execution/cleanup the result is unknown, without refunds,
  restart or automatic retry. Crash tests verify owned child containment after
  assignment, not durable execution accounting.

## Scientific boundaries

[Development admission](development-admission-v1.md) has no admitted sample or
execution-grant state. `require_execution` always denies real access, and no real
worker binding exists. Verified artifacts alone do not enable this robustness
workflow; a future real-model pilot needs a separately reviewed implementation and
owner authorization. Existing application artifact prerequisites are unchanged.

PCB1 remains **BLOCKED**, PCB2 **INCONCLUSIVE**, Cashew official test **already
observed**, MetalParts **exploratory-only**, and RobustAD independent target
evaluation **NOT CLEARED**. Human Review is not verified ground truth. Historical
thresholds, metrics, cards and manifests remain unchanged.

Perturbation stability and OOD-oriented probes establish neither OOD detection
accuracy nor independent real-model robustness. They provide no certified
industrial operating limits, production reliability or authorization for
consequential automated decisions. Historical milestone test counts describe their
own checkpoints; use the [complete local gate](../ci.md) for a new candidate.

No additional dependency is required. The wheel contains the robustness modules;
the sdist also contains these protocols, tests and smoke scripts. Neither includes
datasets, weights or private acceptance reports. Future 5D through 6C work remains
separately authorized in the [roadmap](../roadmap.md).
