# Security and privacy

FactoryVision AI is intended for one trusted local user on loopback, not an
internet-facing or multi-tenant service. The launcher binds `127.0.0.1:8501`, keeps
CORS/XSRF protections, disables telemetry and static serving, and validates uploads.
These controls do not establish production security or industrial reliability.

Images stay in process/session memory; clearing them does not securely erase RAM.
Count and aggregate upload limits apply after reception and are not operating-system
memory quotas. Do not expose the server publicly or upload sensitive images.
Datasets, weights, secrets, private audit logs and local environments are excluded
from the source release. Dependency installation uses separately licensed upstream
packages; no dependency binaries are bundled here.

Use the pinned runtime and lockfile, and review updates explicitly. A clean local
check does not prove every dependency vulnerability is absent. The
[initial hosted CI run](https://github.com/GiannisPap24/factoryvision-ai/actions/runs/40376270944)
passed on 2026-10-26; passing CI is not a security guarantee.

Do not post secrets or exploit details publicly. The source repository is public,
but a private security-reporting channel has not yet been established. The owner
must establish one before community launch. No response-time or supported-version
commitment is implied.
