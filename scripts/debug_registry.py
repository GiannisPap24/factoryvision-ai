"""Offline registry breakpoint; no weights, training, private files or network."""

from pathlib import Path

from factoryvision.inference.batch import BatchService
from factoryvision.inference.metalparts import MetalPartsService
from factoryvision.inference.metalparts_feature import MetalPartsFeatureService
from factoryvision.inference.model_registry import experiment_index, registry_records
from factoryvision.inference.service import InspectionService

root = Path(__file__).resolve().parents[1]
engine = BatchService(
    InspectionService(root), MetalPartsService(root), MetalPartsFeatureService(root)
)
records = registry_records(engine)
evidence = experiment_index()
assert len(records) == 6 and len(evidence) == 8  # Set breakpoint here.
assert evidence[-2].status == "BLOCKED" and evidence[-1].status == "INCONCLUSIVE"
print("PASS: six registry records and eight evidence summaries; no model loaded.")
