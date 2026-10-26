# Publication preparation

**Local candidate only. Publication requires a separate explicit owner decision.**
No GitHub repository, remote, push, public screenshot or binary release is created
by this milestone. Hosted GitHub Actions: **NOT RUN — REQUIRES REMOTE REPOSITORY**.

## Proposed metadata

- Repository: `GiannisPap24/factoryvision-ai`
- Description: Local visual anomaly inspection research with PyTorch, Streamlit, model comparison and a dataset-free synthetic demo.
- Topics: `computer-vision`, `anomaly-detection`, `pytorch`, `streamlit`, `visual-inspection`, `research`, `python`, `model-evaluation`
- Licence: MIT for FactoryVision-owned source, Copyright (c) 2026 GiannisPap24.

The [README](../README.md) links [quick start](local-demo.md), [roadmap](roadmap.md),
[contributing](../CONTRIBUTING.md), [licensing](licensing.md) and [security](../SECURITY.md).
Historical Cashew results, PCB1 BLOCKED, PCB2 INCONCLUSIVE and exploratory
MetalParts results must retain their qualifications. Synthetic examples demonstrate
workflow only. No production-ready quality-control claim is supported.

## Owner review checklist

- Review the private 4E acceptance report, complete file inventory and exact root SHA.
- Confirm the candidate has one root commit on `main`, a clean working tree, no
  remotes/tags/alternate object store and no inherited development commits.
- Review source-only setup, package scans, licensing and all eight private UI captures.
- Keep datasets, weights, private logs and screenshots out of this source release.
- Approve the exact source commit, account, repository name and visibility separately.
- Establish a private security-reporting channel before community launch.
- Expect and review the first hosted workflow run; do not claim it has already passed.

## Exact later publication procedure

These are instructions for a later, separately authorized action, not commands
executed during 4E. Use only the independent candidate directory. Never use the
development repository or copy its `.git` directory.

1. The owner approves the exact candidate root SHA recorded in the local 4E report
   and the public destination `GiannisPap24/factoryvision-ai`. Re-review any changes.
2. In that candidate, run `git status --porcelain`, `git rev-list --count HEAD`,
   `git rev-list --parents -n 1 HEAD`, `git branch --list`, `git tag --list`,
   `git remote -v` and `git fsck --full`. Require a clean tree, one parentless
   commit on main, no tags/remotes and a healthy object database. Compare `git
   rev-parse HEAD` with the approved SHA; verify the private inventory again.
3. Create an **empty** public GitHub repository using the approved name and
   description. Do not initialize a README, licence or `.gitignore` on GitHub.
4. Only after steps 1–3, from the candidate, execute:

   ```powershell
   git remote add origin https://github.com/GiannisPap24/factoryvision-ai.git
   git push -u origin main
   ```

5. Confirm the GitHub default branch is main and its HEAD is the approved SHA.
   Set the reviewed description and topics. Observe the actual Actions run;
   investigate failures and report its real status. Do not silently weaken checks.
6. Verify the public tree, rendered README and links. Do not upload distributions,
   screenshots or research assets automatically. Tags, releases, social promotion
   and 4F community launch remain separately gated.

Recommended future screenshots are source-only Overview, Catalog and generated
Demo. Use original geometry and label it synthetic; review all visible details
before separate publication approval. Private model-enabled screenshots are
review evidence only and are not included in the source candidate.
