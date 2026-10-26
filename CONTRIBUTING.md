# Contributing

FactoryVision AI is a local visual anomaly inspection research application.
Read the [README](README.md), [roadmap](docs/roadmap.md) and
[scientific boundaries](docs/reproduction.md) before proposing a change.

Use Windows AMD64, Python 3.14.10 and uv 0.13.5 with the committed lockfile.
Follow [local setup](docs/local-demo.md#first-installation), including development
dependencies, then run the [offline CI equivalent](docs/ci.md).
Default tests use fabricated inputs and require no datasets or model weights.

Keep changes focused and explain the problem, resulting behavior and checks run.
For UI changes, inspect desktop and narrow screens with synthetic images. Never
attach personal uploads, dataset samples, trained weights or private audit logs.
Review every staged file; ignored files are not a substitute for a privacy check.

Do not silently change frozen model definitions, thresholds, source inventories,
split decisions or historical metrics. New research requires a new protocol and
identity; existing one-shot guards must remain intact. PCB1 is BLOCKED, PCB2 is
INCONCLUSIVE, and MetalParts results remain exploratory source-domain evidence.

Owned code, documentation and generated synthetic content are licensed under
[MIT](LICENSE), Copyright (c) 2026 GiannisPap24. Submit only material you have
rights to contribute under those terms. Dependencies and research assets retain
their own licences; see [attribution](docs/licensing.md).

This is a local release candidate. No hosted contribution channel or CI run is
claimed yet. Public repository creation and publication need a separate owner decision.
For security concerns, read [SECURITY.md](SECURITY.md).
