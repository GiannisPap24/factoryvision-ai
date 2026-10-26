"""Deterministic validation-only example, deliberately outside research report schemas."""

import csv
import hashlib
import io
import json
from dataclasses import asdict, dataclass

from PIL import Image, ImageDraw

from factoryvision.inference.reporting import spreadsheet_cell
from factoryvision.inference.uploads import InputRejected, decode_upload

NOTICE = "Synthetic workflow demonstration — not real model inference or defect detection."
SCHEMA = "synthetic-workflow-v1"


@dataclass(frozen=True)
class DemoRow:
    image_id: str
    validation_status: str
    width: int | None
    height: int | None
    score: None = None
    threshold: None = None
    research_anomaly_flag: None = None


@dataclass(frozen=True)
class DemoReport:
    rows: tuple[DemoRow, ...]

    def validate(self):
        if type(self.rows) is not tuple or len(self.rows) != 3:
            raise ValueError("Invalid synthetic demonstration.")
        for index, row in enumerate(self.rows, 1):
            if type(row) is not DemoRow or row.image_id != f"Demo {index:03d}":
                raise ValueError("Invalid synthetic demonstration.")
            valid = index < 3
            if (
                row.validation_status != ("demo_valid" if valid else "demo_rejected")
                or (row.width, row.height) != ((384, 320) if valid else (None, None))
                or any(v is not None for v in (row.score, row.threshold, row.research_anomaly_flag))
            ):
                raise ValueError("Invalid synthetic demonstration.")

    def document(self):
        self.validate()
        return {
            "schema": SCHEMA,
            "notice": NOTICE,
            "kind": "synthetic_validation_only",
            "model_id": None,
            "artifact_sha256": None,
            "inference_executed": False,
            "rows": [asdict(row) for row in self.rows],
        }

    @property
    def demo_id(self):
        return "demo1-" + hashlib.sha256(encoded(self.document())).hexdigest()


@dataclass(frozen=True)
class Demo:
    report: DemoReport
    images: tuple[bytes, ...]


def encoded(document):
    return (json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode(
        "utf8"
    )


def build_demo() -> Demo:
    """Generate two original geometric PNGs and a deliberately invalid byte fixture."""
    images = []
    for index in range(2):
        im = Image.new("RGB", (384, 320), (18, 32, 48))
        draw = ImageDraw.Draw(im)
        if index == 0:
            draw.rectangle((80, 65, 304, 255), fill=(45, 212, 191))
        else:
            draw.ellipse((96, 64, 288, 256), fill=(245, 198, 55))
        buffer = io.BytesIO()
        im.save(buffer, format="PNG")
        images.append(buffer.getvalue())
    rows = []
    for index, payload in enumerate((*images, b"deliberately invalid synthetic image"), 1):
        try:
            image = decode_upload(payload)
            status, width, height = "demo_valid", image.width, image.height
        except InputRejected:
            status, width, height = "demo_rejected", None, None
        rows.append(DemoRow(f"Demo {index:03d}", status, width, height))
    report = DemoReport(tuple(rows))
    report.validate()
    return Demo(report, tuple(images))


def json_bytes(report: DemoReport) -> bytes:
    return encoded({"demo_id": report.demo_id, **report.document()})


def csv_bytes(report: DemoReport) -> bytes:
    doc = report.document()
    output = io.StringIO(newline="")
    fields = (
        "schema",
        "notice",
        "demo_id",
        "inference_executed",
        "model_id",
        "artifact_sha256",
        "image_id",
        "validation_status",
        "width",
        "height",
        "score",
        "threshold",
        "research_anomaly_flag",
    )
    writer = csv.writer(output, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
    writer.writerow(fields)
    for row in doc["rows"]:
        record = {**doc, **row, "demo_id": report.demo_id}
        values = []
        for key in fields:
            value = record[key]
            value = "null" if value is None else "false" if value is False else str(value)
            values.append(spreadsheet_cell(value))
        writer.writerow(values)
    return output.getvalue().encode("utf8")
