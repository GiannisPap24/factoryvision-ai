"""Explicit scalar-only synthetic report debugger; no models, datasets or output files."""

from factoryvision.inference.batch import BatchReport, BatchRow
from factoryvision.inference.reporting import create_report, csv_bytes, json_bytes, model_info

model = model_info("cashew", "autoencoder")
snapshot = BatchReport(
    "b" * 64,
    model.category,
    model.model_id,
    (
        BatchRow(
            "Image 001",
            "a" * 64,
            "inspected",
            "",
            model.name,
            model.score_label,
            0.01,
            model.frozen_threshold,
            True,
            2.0,
            3.0,
            model.registered_artifact_sha256,
        ),
    ),
    4.0,
)
report = create_report(snapshot)
csv_export, json_export = csv_bytes(report), json_bytes(report)
print(f"Synthetic report verified: {report.report_id}; {report.summary}")
print(f"CSV {len(csv_export)} bytes; JSON {len(json_export)} bytes; no inference executed.")
