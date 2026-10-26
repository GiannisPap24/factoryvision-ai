"""Asset-free VS Code breakpoint probe; never invokes a research model."""

import json

from factoryvision.demo import build_demo, csv_bytes, json_bytes
from factoryvision.inference.model_catalog import definitions

definitions_snapshot = definitions()
demo = build_demo()
document = json.loads(json_bytes(demo.report))
csv_export = csv_bytes(demo.report)
assert len(definitions_snapshot) == 6
assert document["inference_executed"] is False
assert all(row["score"] is None for row in document["rows"])
print("Catalog/demo probe passed: six definitions, three rows, no inference.")
