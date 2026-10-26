"""Read-only frozen inventory verification; never opens held-out image payloads."""

from pathlib import Path

from factoryvision.evaluation.confirmatory import verified_seal

root = Path(__file__).resolve().parents[1]
seal = root / "docs/evidence/pcb1-evaluation-seal.json"
sha = (root / "docs/evidence/pcb1-evaluation-seal.sha256").read_text().split()[0]
verified_seal(root, seal, sha)
print("Frozen PCB1 inventory verified; no held-out payload access.")
