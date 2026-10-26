"""Bounded human observations over immutable scalar evidence; no inference or storage."""

import csv
import io
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from factoryvision.inference.reporting import ReportInvalid, model_info, spreadsheet_cell

SCHEMA = "human-review-v1"
PROVENANCE = "local_user_observation_not_verified_ground_truth"
MAX_COMMENT = 500
MAX_REVISIONS = 20
NOTICE = "Human observation, not verified ground truth or independent model evaluation."


class ReviewInvalid(ValueError):
    def __init__(self):
        super().__init__(
            "Review rejected. Check the decision, UTC time and note: at most 500 characters, "
            "no control characters, paths, contact details or credentials. Maximum 20 revisions."
        )


def require(condition):
    if not condition:
        raise ReviewInvalid()


class Decision(StrEnum):
    NOT_REVIEWED = "NOT_REVIEWED"
    AGREES_WITH_RESULT = "AGREES_WITH_RESULT"
    DISAGREES_WITH_RESULT = "DISAGREES_WITH_RESULT"
    NEEDS_FURTHER_REVIEW = "NEEDS_FURTHER_REVIEW"


def comment_valid(value):
    require(type(value) is str and len(value) <= MAX_COMMENT)
    normalized = unicodedata.normalize("NFKC", value)
    require(not any(unicodedata.category(c).startswith("C") for c in normalized))
    require("/" not in normalized and "\\" not in normalized)
    require(
        re.search(
            r"(?:[a-z]:[\\/]|\\\\|(?<!\w)/(?:\w|~)|file:|[\w.+-]+@[\w.-]+\.\w+|"
            r"\b(?:ghp_|github_pat_|sk-)[\w-]+|\bAKIA[A-Z0-9]{16}|"
            r"\b(?:password|secret|token|api[_ -]?key)\s*[:=]|\bbearer\s+|PRIVATE KEY)",
            normalized,
            re.I,
        )
        is None
    )


def timestamp_valid(value):
    require(type(value) is str)
    require(re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z", value) is not None)
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise ReviewInvalid() from exc


def utc_now():
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def run_id():
    # Opaque execution identity, deliberately not a hash of uploaded pixels or filenames.
    return "fvi1-" + uuid4().hex


@dataclass(frozen=True)
class Evidence:
    inspection_id: str
    image_id: str
    category: str
    model_id: str | None
    score: float | None
    threshold: float | None
    research_flag: bool | None
    artifact_sha256: str | None
    origin: str = "research_inference"

    def __post_init__(self):
        self.validate()

    def validate(self):
        require(type(self.inspection_id) is str)
        require(re.fullmatch(r"fvi1-[0-9a-f]{32}", self.inspection_id) is not None)
        require(type(self.image_id) is str)
        prefix = "Demo" if self.origin == "synthetic_demo" else "Image"
        require(re.fullmatch(prefix + r" 0(?:0[1-9]|1[0-9]|20)", self.image_id) is not None)
        if self.origin == "synthetic_demo":
            require(self.category == "synthetic")
            require(
                all(
                    v is None
                    for v in (
                        self.model_id,
                        self.score,
                        self.threshold,
                        self.research_flag,
                        self.artifact_sha256,
                    )
                )
            )
            return
        require(self.origin == "research_inference")
        try:
            info = model_info(self.category, self.model_id)
            require(
                type(self.score) in (float, int) and math.isfinite(self.score) and self.score >= 0
            )
            require(self.threshold == info.frozen_threshold)
            if self.threshold is None:
                require(self.research_flag is None)
            else:
                require(type(self.threshold) in (float, int))
                require(type(self.research_flag) is bool)
                require(self.research_flag == (self.score > self.threshold))
            require(self.artifact_sha256 == info.registered_artifact_sha256)
        except (ReportInvalid, OverflowError) as exc:
            raise ReviewInvalid() from exc


@dataclass(frozen=True)
class Observation:
    decision: Decision
    comment: str
    reviewed_at_utc: str

    def __post_init__(self):
        require(type(self.decision) is Decision)
        comment_valid(self.comment)
        timestamp_valid(self.reviewed_at_utc)
        require(self.decision != Decision.NOT_REVIEWED or self.comment == "")


@dataclass(frozen=True)
class ReviewRecord:
    evidence: Evidence
    history: tuple[Observation, ...] = ()
    schema_version: int = 1
    reviewer_provenance: str = PROVENANCE

    def __post_init__(self):
        self.validate()

    def validate(self):
        require(type(self.evidence) is Evidence)
        self.evidence.validate()
        require(type(self.schema_version) is int and self.schema_version == 1)
        require(self.reviewer_provenance == PROVENANCE)
        require(type(self.history) is tuple and len(self.history) <= MAX_REVISIONS)
        previous = None
        for event in self.history:
            require(type(event) is Observation)
            event.__post_init__()
            time = timestamp_valid(event.reviewed_at_utc)
            require(previous is None or time >= previous)
            previous = time

    @property
    def decision(self):
        return self.history[-1].decision if self.history else Decision.NOT_REVIEWED

    @property
    def comment(self):
        return self.history[-1].comment if self.history else ""

    @property
    def reviewed_at_utc(self):
        return self.history[-1].reviewed_at_utc if self.history else None


def revise(record, decision, comment, timestamp):
    require(type(record) is ReviewRecord)
    record.validate()
    event = Observation(decision, comment, timestamp)
    if record.decision == decision and record.comment == comment:
        return record
    require(len(record.history) < MAX_REVISIONS)
    return replace(record, history=record.history + (event,))


def single_records(results, category):
    require(type(results) in (tuple, list) and 1 <= len(results) <= 6)
    identity = run_id()
    return tuple(
        ReviewRecord(
            Evidence(
                identity,
                "Image 001",
                category,
                r.model_id,
                r.score,
                r.threshold,
                r.above_threshold,
                r.artifact_sha256,
            )
        )
        for r in results
    )


def batch_records(report):
    from factoryvision.inference.reporting import ResearchReport

    require(type(report) is ResearchReport)
    try:
        report.validate()
    except ReportInvalid as exc:
        raise ReviewInvalid() from exc
    identity = run_id()
    return tuple(
        ReviewRecord(
            Evidence(
                identity,
                r.image_id,
                report.model.category,
                report.model.model_id,
                r.score,
                r.threshold,
                r.research_anomaly_flag,
                r.artifact_sha256,
            )
        )
        for r in report.rows
        if r.status == "inspected"
    )


def demo_records(report):
    from factoryvision.demo import DemoReport

    require(type(report) is DemoReport)
    report.validate()
    identity = run_id()
    return tuple(
        ReviewRecord(
            Evidence(
                identity, r.image_id, "synthetic", None, None, None, None, None, "synthetic_demo"
            )
        )
        for r in report.rows
        if r.validation_status == "demo_valid"
    )


def document(records):
    require(type(records) is tuple and 1 <= len(records) <= 20)
    require(all(type(r) is ReviewRecord for r in records))
    for record in records:
        record.validate()
    require(len({r.evidence.inspection_id for r in records}) == 1)
    require(len({r.evidence.origin for r in records}) == 1)
    require(len({r.evidence.category for r in records}) == 1)
    require(len({(r.evidence.image_id, r.evidence.model_id) for r in records}) == len(records))
    return {
        "schema": SCHEMA,
        "notice": NOTICE,
        "records": [
            {
                **asdict(r),
                "decision": r.decision,
                "comment": r.comment,
                "reviewed_at_utc": r.reviewed_at_utc,
                "revision": len(r.history),
            }
            for r in records
        ],
    }


def json_bytes(records):
    return (
        json.dumps(document(records), ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    ).encode("utf8")


def csv_bytes(records):
    rows = document(records)["records"]
    output = io.StringIO(newline="")
    writer = csv.writer(output, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
    writer.writerow(
        [
            "schema",
            "schema_version",
            "inspection_id",
            "image_id",
            "category",
            "model_id",
            "score",
            "threshold",
            "research_flag",
            "artifact_sha256",
            "origin",
            "reviewer_provenance",
            "revision",
            "decision",
            "comment",
            "reviewed_at_utc",
        ]
    )
    for row in rows:
        # One line per revision; a never-reviewed item has one null-time pending line.
        events = row["history"] or [
            {"decision": Decision.NOT_REVIEWED, "comment": "", "reviewed_at_utc": None}
        ]
        for index, event in enumerate(events, 1):
            values = [
                SCHEMA,
                row["schema_version"],
                *row["evidence"].values(),
                row["reviewer_provenance"],
                index if row["history"] else 0,
                event["decision"],
                event["comment"],
                event["reviewed_at_utc"],
            ]
            writer.writerow([spreadsheet_cell("" if v is None else str(v)) for v in values])
    return output.getvalue().encode("utf8")
