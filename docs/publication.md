# Source publication

**Published on 2026-10-26 after owner approval and a successful independent final
security/privacy/supply-chain/file-content audit.**

- Public repository: [GiannisPap24/factoryvision-ai](https://github.com/GiannisPap24/factoryvision-ai).
- Initial root commit: `691507dd409177219b4ba39afedfaaedfe440c06`.
- Initial inventory: 245 reviewed source/configuration/documentation files.
- Initial [hosted CI run 40376270944](https://github.com/GiannisPap24/factoryvision-ai/actions/runs/40376270944):
  **SUCCESS**, 711 tests passed, 98.09% coverage on Windows AMD64 / Python 3.14.10.
- Licence: MIT for FactoryVision-owned source, Copyright (c) 2026 GiannisPap24.

The independent staging repository was published to an empty public repository
with one parentless commit on `main`. The remote tree and commit matched the
reviewed source. No development history, datasets, weights, private reports,
screenshots or release binaries were uploaded. Later documentation/source updates
extend this clean history; the initial SHA above is the publication baseline,
not a claim that the branch will always contain only one commit.

The [README](../README.md) links [quick start](local-demo.md), [roadmap](roadmap.md),
[contributing](../CONTRIBUTING.md), [licensing](licensing.md) and [security](../SECURITY.md).
Historical Cashew results, PCB1 BLOCKED, PCB2 INCONCLUSIVE and exploratory
MetalParts results retain their qualifications. Synthetic examples demonstrate
workflow only. No production-ready quality-control claim is supported.

## Verification evidence

The final local audit compared all 245 working files and Git blobs with the reviewed
inventory and inspected source/configuration, dependencies and distribution contents.
Fresh wheel and sdist installations and synthetic application checks passed.
The local suite had 710 passed and one Windows symlink-permission skip; the hosted
runner passed all 711 tests. Private audit records are not distributed here.

The linked hosted run also passed lint/format, lock/dependency checks, wheel/sdist
scans and an installed-application smoke test. See [CI details](ci.md) and
[current Actions runs](https://github.com/GiannisPap24/factoryvision-ai/actions)
for later commit results. A passing audit or CI run is not a guarantee that every
future vulnerability is absent.

## Future source updates

1. Use only the independent public-source repository. Never copy the development
   repository's `.git`, private assets or historical commits into it.
2. Confirm the intended account, repository, branch and remote HEAD. Review the
   entire proposed diff and file inventory for secrets, personal information,
   workstation paths, unwanted binaries and scientific or licensing changes.
3. Run checks appropriate to the change, including distribution/content checks
   when packaged documentation or source changes. Stage only reviewed files.
4. For an owner-authorized update, create a normal commit and push only `main` to
   `GiannisPap24/factoryvision-ai`. Do not force-push, mirror or push unrelated refs.
5. Verify the remote SHA, public diff and the actual hosted CI result. Report
   pending or failed checks accurately; do not silently weaken them.

## Separate release boundaries

Datasets, weights, private logs, screenshots and generated distributions remain
outside the source-release scope. Tags, binary releases, social promotion and
the wider 4F community launch require separate review and authorization.
A private security-reporting channel must be established before community launch.

Future portfolio captures should use source-only Overview, Catalog or generated
Demo with original geometry and explicit synthetic labels. Review visible details
before separate publication approval. Private model-enabled screenshots remain
local review evidence.
