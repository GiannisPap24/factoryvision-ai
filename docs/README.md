# FactoryVision AI documentation

Start with the guide that matches your goal. The public source runs the catalog
and synthetic workflow without datasets or weights. Real inference requires
separately prepared, verified research artifacts.

## Run and explore the application

| Goal | Guide |
|---|---|
| Install on Windows and run the first demo | [Local setup and troubleshooting](local-demo.md) |
| Understand the eight workspaces and relative heatmaps | [Interface](interface.md) |
| Explore methods and download validation-only examples | [Model Catalog and synthetic demo](model-catalog-demo.md) |
| Understand real inference, upload validation and scores | [Inference contracts](inference.md) |
| Inspect a completed batch and interpret CSV/JSON | [Research reporting](reporting.md) |

## Understand and contribute to the code

| Goal | Guide |
|---|---|
| Follow data through the application | [Architecture](architecture.md) |
| Run offline checks or use VS Code | [Development](development.md) |
| Understand hosted checks and coverage | [CI](ci.md) |
| Propose a focused change | [Contributing](../CONTRIBUTING.md) |
| Understand local security and privacy boundaries | [Security](../SECURITY.md) |
| Review licence and asset redistribution boundaries | [Licensing and attribution](licensing.md) |
| Review completed milestones and future scope | [Roadmap](roadmap.md) |
| Review the publication baseline and update procedure | [Publication record](publication.md) |

## Research methods and evidence

Read [reproduction prerequisites](reproduction.md) before running research scripts.
The [Cashew autoencoder and spatial baseline](model-card.md),
[feature memory](feature-model-card.md), [student–teacher](student-model-card.md),
[MetalParts autoencoder](metalparts-model-card.md) and
[MetalParts feature memory](metalparts-feature-model-card.md) cards describe
training, scoring, provenance and limitations.

Milestone reports, frozen protocols and recorded hardware/runtime measurements
are historical evidence. They describe the state when each experiment occurred;
they are not current installation instructions. Their private development commit
identifiers are provenance references, not commits available in the public history.
Do not rewrite historical results or remove one-shot guards to rerun a benchmark.
PCB1 remains **BLOCKED**, PCB2 **INCONCLUSIVE**, and MetalParts results remain
exploratory source-domain evidence with independent target evaluation **NOT CLEARED**.
