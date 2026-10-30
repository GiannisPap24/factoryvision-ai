# FactoryVision AI working rules

FactoryVision AI is a source-only, local visual anomaly inspection research
application. Work in the independent public staging repository and the worktree
approved for the current milestone. The main agent owns implementation; specialist
reviewers inspect independently and return findings without editing files.

## Find the relevant context

- Start with [the documentation index](docs/README.md), then read only the guides
  relevant to the task. [Architecture](docs/architecture.md) maps `src/factoryvision/`:
  `domain/` holds contracts, `inference/` holds services and provenance, and `app/`
  is the native Streamlit interface. Preserve the eight workspaces and session
  lifecycle unless the approved milestone explicitly changes them.
- Use [development](docs/development.md) and [CI](docs/ci.md) for the pinned
  environment and checks, [reproduction](docs/reproduction.md) for scientific
  boundaries, and [SECURITY.md](SECURITY.md) for the local-only threat model.
  Historical counts and status paragraphs are evidence for their named commits,
  not proof of the current checkout.

## Preserve the source and scientific boundaries

- Never access the original private development repository or import its history,
  datasets, checkpoints, private reports, credentials or personal uploads. Use
  fabricated inputs and original synthetic images. No real-model execution,
  training, production GPU integration or new external services in this workflow.
- Preserve scientific identities, frozen thresholds, splits, one-shot guards,
  historical metrics and the nine pinned manifests/model cards. Consult the
  registry pins for their inventory; compare bytes/hashes against the approved
  base. Unexpected scientific changes stop the checkpoint.
- Keep unavailable models unavailable; never invent inference results. Synthetic
  validation and human observations are not research inference or ground truth.
  Scientific status must retain the limitations documented in reproduction.
- Do not add dependencies, external UI assets or network behavior outside explicit
  milestone scope. Keep local evidence ignored and inspect its contents before
  sharing; ignore rules alone do not establish privacy.

## Work, verify and stop at the right boundary

- Confirm repository/remote, approved base, branch, worktree and existing changes
  before editing. Preserve other worktrees and user changes. Stop on unexpected
  Git state, a security/privacy blocker or an unapproved scientific change.
- Routine local edits and focused checks within an approved milestone need no
  repeated approval. Validate documentation/configuration syntax and links for
  documentation-only changes; use focused tests for affected behavior. For UI
  changes, reuse existing UI/browser checks with synthetic inputs on desktop and
  mobile, including keyboard access and unavailable-model behavior.
- Full acceptance, when the milestone or publication gate requires it, uses
  `scripts/ci_check.py` as documented in `docs/development.md`; do not duplicate it.
  The positive test allowlist is exactly `tests/` and
  `integration_tests/test_performance_isolation.py`. Never collect all integration
  tests. Preserve the exact combined statement/branch coverage floor of 95% and
  packaging/privacy/fresh-install checks. Avoid repeating a full run without a
  material change or an identified evidence gap.
- Commit, push, PR creation, merge, release, external messages and destructive Git
  or worktree actions require explicit task authorization. Honor authorization
  already given for that task. Stage only individually reviewed paths; use the
  verified GitHub noreply identity when committing. No force-push or history rewrite.
  The historical direct-main example in `docs/publication.md` is not permission
  to publish; follow the owner's current branch/PR instructions.
- Finish with a short checkpoint: scope and changed paths, actual checks/results,
  privacy/scientific integrity, Git state, limitations and the next approval
  boundary. Classify `READY FOR OWNER REVIEW`, `NEEDS DECISION` or `BLOCKED`.
  Separate verified results from assumptions and pending checks.
