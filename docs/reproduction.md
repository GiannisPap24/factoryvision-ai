# Research reproduction and boundaries

The application runs frozen artifacts; starting it never starts an experiment.
Reproducing research requires licensed data, verified inventories, split/gate
records and the specific frozen protocol/source snapshot. A public source checkout
alone does not include these assets. Do not remove one-shot guards or manufacture
an integrity approval to make a command run.

## Recorded entry points

The command syntax below is checked against the current scripts. They are research
instructions for a separately prepared, approved fresh workspace, **not commands
to rerun in the completed workspace**. No training was run during Milestone 4A.

| Experiment | Explicit entry point | Required preparation and behavior |
|---|---|---|
| Cashew AE and spatial baseline | `.venv-31410/Scripts/python.exe scripts/train_cashew.py --run-id <fresh-id>` | Verified VisA manifest, leakage review/screen and bytes; [training](training.md). Fits 315 normal images, validates 67, calibrates on 68. Existing IDs refused. The historical byte-integrity gate hashes all manifest members; it is not a test-inference step. |
| Cashew feature bank | `.venv-31410/Scripts/python.exe scripts/feature_experiment.py fit` | Approved normal-only manifest, backbone and frozen source seal; [feature recipe](feature-memory.md). Fixed one-shot run ID; refuses completed output. No backbone gradient updates. |
| Cashew student | `.venv-31410/Scripts/python.exe scripts/student_experiment.py train` | Approved teacher, splits and sealed recipe; [student training](student-training.md). Fixed one-shot run ID. The smoke option also trains and is not a default installation check. |
| MetalParts AE | `.venv-31410/Scripts/python.exe scripts/metalparts_experiment.py prepare`, then `smoke`, then `train` | Pinned source verification inventory, normal-only allowlist, 3D protocol/code seal; [exploratory protocol](metalparts-exploratory-training-protocol.md). Prepare creates a reviewed split; smoke/train are explicit gradient-training actions. Existing runs refused. |
| MetalParts feature bank | `.venv-31410/Scripts/python.exe scripts/metalparts_feature_experiment.py seal`, then `fit` | Approved 3D normal split, original ResNet18 backbone and 3F source seal; [feature protocol](metalparts-feature-memory-protocol.md). Fits memory without gradient training; fixed run guards. |

Read the linked protocol and script prerequisites before executing. These commands
have no automatic dataset or backbone download. Obtain data only through the
original source and after reviewing its terms. Local historical manifests and
human integrity decisions cannot be regenerated from a package install alone.
Fresh runs need distinct protocols/identities and a separate review to register
artifacts; application registries pin exact historical bytes. Training does not
promise bit-identical hashes on a different system. Do not change frozen thresholds
to improve observed defect results.

## Scientific chronology

- [2A training](milestone-2a-completion.md), [2B recorded Cashew evaluation](milestone-2b-completion.md).
- [2C frozen feature memory](milestone-2c-completion.md), [2D student training](milestone-2d-completion.md).
- [PCB1 blocked evaluation](milestone-2e2-completion.md): source overlap invalidated the intended confirmation.
- [3B dataset strategy](milestone-3b-dataset-strategy.md), [3C MetalParts integrity](milestone-3c-robustad-integration.md).
- [3D source-normal training](milestone-3d-metalparts-training.md),
  [3E observed source diagnostics](milestone-3e-metalparts-development-evaluation.md),
  [3F feature comparator](milestone-3f-metalparts-feature-memory.md).

Do not run historical evaluation entry points on already observed official data
and label them fresh held-out confirmation. PCB1 is BLOCKED, PCB2 INCONCLUSIVE;
RobustAD independent target evaluation is NOT CLEARED. A future independent study
requires separate authorization and resolution of source/acquisition independence.

## Safe developer verification

First follow [isolated setup](ci.md); the original environment remains offline-only.

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path (Get-Location) '.venv-31410'
.tools/uv-0.13.5/uv.exe sync --locked --extra app --python '.venv-31410/Scripts/python.exe'
.venv-31410/Scripts/python.exe scripts/check_quality.py --report-dir reports/local-quality
```

The default suite uses fabricated data and blocks network access. It does not run
`integration_tests/` or real experiments. Use the synthetic inference debugger
configurations, not the training configurations, for a portfolio demonstration.
