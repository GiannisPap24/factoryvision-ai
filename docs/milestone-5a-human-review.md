# Milestone 5A — Human Review & Feedback V1

**COMPLETE — HUMAN REVIEW V1 VERIFIED** (local verification, 2026-10-26).

Human Review V1 adds local observations to existing inspection results. It does
not change a score, research flag, threshold, model configuration or historical
evaluation. This feature is implemented on a local feature branch; publication
requires separate diff and security review. No source changes have been pushed.

## Using the workflow

1. Run a permitted image in **Inspect**, **Compare** or **Batch Inspection**.
   Real inference still requires verified local research artifacts.
2. Under **Human review**, choose the safe image/model pair to review. Batch
   failures, rejected inputs and skipped rows remain in the research report but
   cannot receive a review of a prediction that does not exist.
3. Choose a decision, optionally add a note, and press **Save human review**.
4. Review the pending/reviewed counts and the current decision, revision and UTC
   time for each result. Saving an identical decision/note is a no-op.
5. Download **human review JSON** or **human review CSV** explicitly to retain
   observations. These are separate from the original research report exports.

| Decision | Meaning |
|---|---|
| `AGREES_WITH_RESULT` | The reviewer agrees with the displayed evidence |
| `DISAGREES_WITH_RESULT` | The reviewer disagrees with the displayed evidence |
| `NEEDS_FURTHER_REVIEW` | A human follow-up is needed |
| `NOT_REVIEWED` | Pending; the current note must be empty |

Agreement concerns the displayed evidence, not product acceptance or a verified
defect label. MetalParts retains continuous scores with **no binary flag**.

Every saved change appends an immutable observation. Resetting a decision to
`NOT_REVIEWED` with an empty note returns it to pending but keeps previous
revisions in the session and export. A record allows at most 20 revisions;
export before clearing if the history is needed. Unsaved form edits are not
exported and may be lost when switching the selected result.

## Session lifecycle

The review bundle is bound to the completed result object, not a filename or
the current row number alone. Each execution receives an opaque random
`fvi1-` identifier, stable across review edits, UI reruns and exports. Image IDs
are safe ordinal identifiers (`Image 001` etc.); compare-mode records share an
image ID but have distinct model identities. A fresh execution gets a new ID.

Changing the input, model selection, category, consent or batch limit/order
invalidates the associated result and review. Clear actions remove the current
bundle and review widgets. Navigating between **Batch Inspection** and
**Reports** preserves the completed batch and reviews without inference.
Other workspace transitions clear them. Refresh/session expiry or a server
restart may discard all session state. Exports cannot be imported in V1.

The **Demo** workspace supports separate synthetic review practice without any
weights. Its records carry category `synthetic`, origin `synthetic_demo`, safe
`Demo` IDs and null model, score, threshold, research flag and artifact fields.
Regenerating the demo starts a fresh review run. Its rejected example is not
reviewable. Synthetic and research records cannot be mixed in an export.

## Evidence and observation are different

| Concept | What it establishes |
|---|---|
| Model prediction | The frozen method's score and, only where defined, its existing research flag |
| Human observation | A local user's opinion about that evidence, with a note and time |
| Verified ground truth | A separately established reference label under a documented verification process; V1 does not create this |
| Independent model evaluation | A prespecified assessment on appropriately independent data; session feedback does not meet this requirement |

Cashew historical results remain historical. **PCB1 stays BLOCKED**, **PCB2
INCONCLUSIVE**. MetalParts evidence remains exploratory source-domain evidence;
independent target-domain reliability is not established. Human observations
never trigger fitting, training, threshold adjustment or benchmark access.

## Contract and export format

The typed frozen `Evidence`, `Observation` and `ReviewRecord` objects live in
`factoryvision.inference.review`. Schema name: `human-review-v1`; per-record
`schema_version`: integer `1`. Provenance is explicitly
`local_user_observation_not_verified_ground_truth`.

Evidence contains the opaque inspection ID, safe image ID, category, model ID,
original score, frozen threshold, original research flag, registered artifact
SHA-256 and origin. It contains no image hash, pixels, previews, EXIF, filename,
upload location, identity of the reviewer or machine path. Registered artifact
identity is checked against installed metadata, with no weight loading.

JSON contains immutable evidence, the full ordered observation history and
current decision/comment/time/revision. Never-reviewed records have an empty
history, `NOT_REVIEWED`, empty comment, null time and revision zero. UTC times
use `YYYY-MM-DDTHH:MM:SS.ffffffZ`; revisions cannot move backwards in time.

CSV contains one row per historical revision, or one pending row for an
unreviewed item. It repeats the version, evidence and provenance on every row;
null values are empty fields. All cells are quoted and formula-like strings
receive the existing `[text] ` prefix. JSON preserves the original accepted
note; consumers must not paste raw JSON strings into spreadsheet formulas.
Exports are byte-for-byte deterministic for unchanged records. Generation does
not refresh timestamps, run inference or create a new ID.

## Privacy and limitations

There is no database, background worker, feedback API, automatic upload or
automatic persistence. The app holds notes and download buffers in session
memory. Only an explicit browser download writes an export at the browser's
chosen destination. Existing localhost binding, CORS/XSRF protection and
disabled application telemetry remain unchanged.

Notes are optional, limited to 500 characters and validated before being saved.
Control characters, slash/backslash path separators, common email addresses
and credential patterns are rejected, with Unicode compatibility normalization
used for screening. Errors are fixed messages that do not echo rejected text.
Invalid edits leave the previous accepted observation and export unchanged.
The note guard is **not comprehensive sensitive-data detection**: names,
proprietary text, unusual credential formats and other private information
cannot reliably be recognized. Enter no private information. Exports include
every saved note revision, even after resetting to pending. Browser downloads
are outside session clearing; users control their retention and sharing.

Clearing drops application references; it is not secure memory erasure. There
is no authentication, reviewer identity proof, tamper-proof audit trail,
signature, verified labelling protocol or multi-user review coordination.
UTC timestamps are local system-clock readings. Use one trusted local session.

## Verification and reproduction

Development baseline: public `main` commit
`a8402a9059289a7b999b841f04be338530240e76` and
[successful hosted CI](https://github.com/GiannisPap24/factoryvision-ai/actions/runs/40389298430).
Work is on local branch `feat/human-review-v1`. Only the clean public-source
repository is used; no private history, datasets or research weights were copied.

On Windows AMD64, using project-local verified Python 3.14.10, uv 0.13.5 and
unchanged locked dependencies:

```powershell
.venv-31410/Scripts/python.exe scripts/ci_check.py --uv .tools/uv-0.13.5/uv.exe
```

Baseline: 710 tests passed, one symlink-privilege skip; combined statement and
branch coverage 98.09%. The first baseline attempt lacked the configured `.tmp`
parent; creating that project-local directory resolved the environment issue.
Existing tests remain; three download-count assertions now separately verify
the original and newly added export controls.

New offline tests cover six-model evidence preservation, single/compare/batch
isolation, decisions, bounds, revisions, no-op edits, resets, missing artifacts,
category/model/image/consent changes, invalid cached state, sanitized errors,
deterministic exports, formula protection and synthetic separation. Network
connections remain blocked by the existing test fixture, except the documented
same-process Windows asyncio socket pair.

The VS Code configuration **FactoryVision: debug human review** runs
`scripts/debug_review.py`, a synthetic observation probe with no file writes
or model inference. Set a breakpoint at its evidence-preservation assertion,
inspect `original` and `changed`, then continue to the PASS message.

### Local verification results (2026-10-26)

- Complete offline suite: **797 passed, 1 skipped**, 798 collected. The skip is
  the existing Windows symbolic-link privilege check; junction containment tests
  pass. Existing warnings remain (12,597); no warning filters or security guards
  were weakened.
- Combined statement/branch coverage: **98.18%**, up from 98.09%. Statement
  coverage: 98.55%; branch-only coverage: 97.06%. Both new review modules have
  100% statement and branch coverage. The existing 95% gate is unchanged.
- Ruff lint and formatting: pass across 166 Python files. Offline lock check,
  installed dependency compatibility, wheel/sdist build, archive privacy/link
  inspection and isolated installed-app smoke: pass. Installed smoke now saves
  a synthetic review and verifies export determinism and category reset.
- The first archive check flagged two deliberately fabricated negative test
  literals. The test path now uses a neutral fixture location and the synthetic
  credential shape is constructed in the test. The distribution scanner and
  security tests were not disabled or relaxed. Final archives have no findings.
- Final wheel: 91 members; sdist: 257 members. Neither contains test screenshots,
  datasets, weights, local environments, browser harness or evidence logs.
- `git diff --check`: pass. Privacy scan of 260 candidate source files: no
  private-user paths or credential-pattern findings. Comparison with the 254-file
  baseline found only the 12 intended modified files plus six new files.
  Scientific metadata, model/training/evaluation/data code, lockfile and
  Streamlit security/theme configuration are unchanged.

### Browser verification

The callable in-app browser was verified by loading and operating the app in this
session. No Chrome connection was exposed in the current browser inventory;
Chrome was not needed or claimed tested. Browser installation state was not
changed. Native VS Code was opened on the public-source repository.

Two localhost servers were used: the actual source-only app, and an ignored QA
harness prominently labelled **synthetic test doubles and fabricated outputs**.
The latter uses only original generated geometric images and existing tiny test
models. It is not part of the product or distribution. No research checkpoint or
third-party dataset was accessed, and this verification makes no new ML claim.

Verified in the actual browser:

- Source-only missing-checkpoint state disables inference; the synthetic demo
  still generates, accepts observations, revises decisions and downloads JSON/CSV.
- Downloaded demo JSON retains both revisions and a separate pending record;
  predictions and model identity stay null. A rejected fixture path note does not
  appear in the export; the previous accepted observation remains unchanged.
- Synthetic single-image review saves and exports the original score/flag;
  removing its model invalidates the result and review.
- A three-input synthetic batch produces two reviewable rows and one rejected
  row. Decisions on the two valid rows remain independent, persist into Reports,
  and export correctly. Switching category clears the completed report/reviews.
- Four Cashew methods appear as independent compare-mode review records. Two
  MetalParts methods produce review records with null threshold and research
  flag; downloaded MetalParts JSON confirms both remain null after feedback.
- Desktop and 390-pixel narrow layouts support the review form and exports in
  the existing navy/blue/teal design. Wide tables scroll within their container.
  No browser console warnings/errors were captured during the workflows.

Local evidence is in `reports/human-review-5a/evidence/`: CI/coverage/audit records,
actual downloaded synthetic exports, browser DOM snapshots and screenshots.
These artifacts are ignored and not published. Browser fixtures demonstrate
software behavior only, not performance of the unavailable research weights.

### VS Code debugger status

The owner explicitly approved workspace trust for the public-source folder.
On resuming, Manage Workspace Trust already showed that exact folder trusted;
no parent-folder trust was added for this verification.

The **FactoryVision: debug human review** configuration ran in the actual VS Code
debugger and stopped on the breakpoint at `scripts/debug_review.py:16`.
The live Debug Console confirmed Python 3.14.10 and the expression
`(original.history, len(changed.history), changed.evidence is original.evidence,
changed.evidence.score, changed.evidence.research_flag, changed.decision.value)`
returned `((), 1, True, None, None, 'NEEDS_FURTHER_REVIEW')`.
This verifies a new observation revision while preserving the original evidence
object and the synthetic null score/flag. Continuing completed all assertions
and printed `PASS: synthetic review revision; immutable null-score evidence;
no writes.` The debug session then ended normally.

Ignored local evidence includes `vscode-breakpoint.png`, `vscode-breakpoint.txt`,
`vscode-debug-complete.png` and `vscode-debug-complete.txt` under the evidence
directory above. These native debugger captures are local verification records,
not public release assets.

## Change inventory

- Contract: `src/factoryvision/inference/review.py`.
- Review UI: `src/factoryvision/app/review_ui.py`.
- Integration: `src/factoryvision/app/ui.py`, `metalparts_ui.py`, `batch_ui.py`,
  `demo_ui.py`, `presentation.py`.
- Tests: `tests/test_review.py`, `tests/test_review_app.py`, `tests/test_app.py`.
- Debug/installed checks: `scripts/debug_review.py`, `scripts/smoke_installed.py`,
  `.vscode/launch.json`.
- Packaging/docs: `pyproject.toml`, `README.md`, `docs/README.md`,
  `docs/roadmap.md`, this document.

Ignored local verification logs, screenshots, original geometric image fixtures
and a conspicuously labelled synthetic browser harness live under
`reports/human-review-5a/`. They are excluded from source distributions and are
not scientific evidence or public demo assets.

The recommended next scope is **5B — Model Registry**, after reviewing 5A's
diff/security results and separately deciding whether to publish it. No later
milestone, camera integration or community message is initiated here.
