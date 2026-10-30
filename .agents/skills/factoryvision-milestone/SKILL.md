---
name: factoryvision-milestone
description: Carry out an owner-approved FactoryVision AI source-only development milestone or checkpoint in its designated public staging worktree, with proportionate validation, independent reviews when useful, and a concise handoff. Use for scoped implementation, acceptance preparation or workflow configuration; this skill does not authorize research runs or publication.
---

# FactoryVision milestone

## Establish the checkpoint

1. Identify the current owner-approved scope, base SHA, branch, worktree, acceptance
   criteria and stop conditions from the conversation. Read the applicable
   `AGENTS.md` chain. Existing task authorization remains valid; ask only for
   missing decisions that block safe progress, not routine local edits.
2. Verify the public staging repository and remote, current HEAD and Git status.
   Inventory and preserve existing edits before changing anything. Use the
   designated worktree; create a new one only when authorized and its path/branch
   are unused. Never access the original private repository or alter other
   worktrees. Stop on an unexpected state rather than resetting or stashing it.
3. Discover tools and skills relevant to this checkpoint, including the actual
   callable browser and subagent tools when needed. Reuse installed capabilities;
   do not install plugins, change global settings or enable services to fill gaps.
   Read relevant Security Best Practices guidance for security-sensitive work.
4. Start at `docs/README.md`; read only the relevant architecture, interface,
   development, CI, reproduction or publication sections and existing checkpoint
   evidence. Inspect the actual implementation and tests. Historical reports
   apply to their recorded SHA and must not substitute for current verification.

## Implement and review

5. Make a short plan sized to the task, then complete the approved scope with one
   implementation owner. Preserve scientific contracts, session semantics and
   source-only restrictions from `AGENTS.md`. Do not start another milestone.
6. Delegate an independent read-only review when it adds useful coverage:
   `ui-quality-reviewer` for UI/Streamlit/mobile/accessibility/browser changes;
   `risk-reviewer` for privacy, scientific boundaries, dependencies, scope or
   publication risk. Use only relevant reviewers, not both automatically. Give
   each the exact worktree, base, changed paths, acceptance criteria and existing
   evidence. Reviewers return findings; the main agent makes any authorized edits.
   If custom agents are unavailable, perform the review in the main session and
   report the limitation. Do not create substitute permanent configurations.

## Validate proportionately

7. Run focused checks first. Documentation/configuration-only work needs metadata,
   syntax, references, discovery and privacy checks, not the application suite.
   For behavior changes reuse existing tests for affected paths. Test selection
   must stay within `tests/` and `integration_tests/test_performance_isolation.py`;
   never run bare pytest collection or the whole integration directory.
8. For UI changes, use synthetic-only workflows and the approved visual reference.
   Unless the task specifies other dimensions, compare actual desktop 1440x900
   and mobile 390x844 screens. Check navigation, upload, large preview, disabled
   inference, clear/reset, focus, sidebar and responsive flow. Inspect current
   browser tools and use the installed browser skill; report installation,
   callable availability and Chrome connection separately. Do not claim browser
   validation without an actual successful operation or use brittle DOM hacks.
9. Run full acceptance only when required by the milestone/publication gate or
   when broad behavior changes leave a material validation gap. After focused
   checks pass, reuse `scripts/ci_check.py` with the pinned existing environment
   and documented offline command in `docs/development.md`. Preserve its exact
   positive allowlist, combined coverage comparison `100 * covered >= 95 * possible`,
   lint/format, lock/dependencies, packaging/privacy and fresh installed-app smoke.
   Do not add a second runner, weaken gates or repeat an unchanged full run.
   Explain the need before any additional full run. Respect task-specific limits.
10. Review every changed file for credentials, personal/workstation paths, private
    assets, external dependencies and unintended scope. Check the nine scientific
    resource hashes against the approved base and ensure no scientific code or
    protocol drift. Keep generated reports/screenshots out of staged changes.
    Stop and report P0/P1 findings or unexpected scientific/security changes.

## Handoff and authorized external actions

11. Complete the local reviewable result before seeking any still-needed approval.
    Do not commit, publish, send messages, merge, release or change remote settings
    unless the current task explicitly authorizes that action. When authorized,
    follow its precise repository, branch, identity and CI gates; inspect real
    hosted logs and report pending/failing states accurately. Never infer current
    permission from an old report or publication example.
12. Report briefly: changed paths and behavior; focused/full checks actually run
    (passes, skips, exact coverage when measured); privacy/scientific findings;
    Git state; remaining limitations and next approval boundary. Use
    `READY FOR OWNER REVIEW`, `NEEDS DECISION` or `BLOCKED`. Keep detailed local
    evidence in approved ignored reports when useful and authorized, without
    private content. Stop at the requested checkpoint.

## Discovery

The repository skill is at `.agents/skills/factoryvision-milestone/SKILL.md`;
invoke it as `$factoryvision-milestone` from the approved worktree. Reviewer
definitions live in `.codex/agents/`. Start a fresh Codex session rooted in that
worktree if new instructions, skill metadata or reviewer names are not loaded.
Parsing the files does not prove runtime discovery. Read-only reviewer instructions
are behavioral constraints, not a sandbox; parent permissions and tool restrictions
still apply. No model, reasoning effort or tool permissions are set here.
