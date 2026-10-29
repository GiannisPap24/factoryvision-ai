"""Proposal metadata only. No file/sample verification or execution grants exist in V1."""

import json
from dataclasses import asdict, dataclass, fields
from enum import StrEnum

from factoryvision.robustness.contracts import _unique

PROTOCOL_ID = "factoryvision-development-admission"
MAX_BYTES = 16384


class AdmissionInvalid(ValueError):
    def __init__(self):
        super().__init__("Invalid development admission metadata.")


class ExecutionDenied(ValueError):
    def __init__(self):
        super().__init__("Real sample access and model execution are not authorized.")


def require(condition):
    if not condition:
        raise AdmissionInvalid()


class Eligibility(StrEnum):
    NOT_CLEARED = "not_cleared"
    NEEDS_PROVENANCE = "needs_provenance"
    BLOCKED = "blocked"
    ELIGIBLE_IN_PRINCIPLE = "eligible_in_principle"


class Review(StrEnum):
    UNREVIEWED = "unreviewed"
    WITHHELD = "withheld"
    ADVERSE = "adverse"
    DOCUMENTED_CANDIDATE = "documented_candidate_only"


class Admission(StrEnum):
    NOT_VERIFIED = "not_verified"


class Authorization(StrEnum):
    NOT_AUTHORIZED = "not_authorized"


@dataclass(frozen=True, slots=True)
class DevelopmentAdmission:
    """The sole proposed candidate; caller declarations never authenticate private evidence.

    There is deliberately no admitted/cleared sample state, actual sample identity,
    artifact identity, path, signature or execution token in this phase.
    """

    review: Review = Review.UNREVIEWED
    model_id: str = "autoencoder"
    category: str = "cashew"
    dataset: str = "visa"
    release: str = "VisA_20220922"
    revision: str = "2a692ab575001cbde74d402d897a7286086c6199"
    licence_record: str = "documented_cc_by_4.0"
    official_split: str = "train"
    partition: str = "validation"
    label: str = "normal"
    purpose: str = "exploratory_development_diagnostic"
    prior_exposure: str = "validation_used_for_model_selection"
    sample_admission: Admission = Admission.NOT_VERIFIED
    artifact_admission: Admission = Admission.NOT_VERIFIED
    execution: Authorization = Authorization.NOT_AUTHORIZED
    kind: str = "development_admission_proposal"
    protocol_id: str = PROTOCOL_ID
    protocol_version: int = 1
    numerical_protocol_version: int = 2
    schema_version: int = 1

    def __post_init__(self):
        fixed = {
            "model_id": "autoencoder",
            "category": "cashew",
            "dataset": "visa",
            "release": "VisA_20220922",
            "revision": "2a692ab575001cbde74d402d897a7286086c6199",
            "licence_record": "documented_cc_by_4.0",
            "official_split": "train",
            "partition": "validation",
            "label": "normal",
            "purpose": "exploratory_development_diagnostic",
            "prior_exposure": "validation_used_for_model_selection",
            "kind": "development_admission_proposal",
            "protocol_id": PROTOCOL_ID,
        }
        for key, value in fixed.items():
            require(type(getattr(self, key)) is str and getattr(self, key) == value)
        for key, value in (
            ("schema_version", 1),
            ("protocol_version", 1),
            ("numerical_protocol_version", 2),
        ):
            require(type(getattr(self, key)) is int and getattr(self, key) == value)
        require(type(self.review) is Review)
        require(type(self.sample_admission) is Admission)
        require(type(self.artifact_admission) is Admission)
        require(type(self.execution) is Authorization)

    @property
    def eligibility(self) -> Eligibility:
        self.__post_init__()
        return {
            Review.UNREVIEWED: Eligibility.NEEDS_PROVENANCE,
            Review.WITHHELD: Eligibility.NOT_CLEARED,
            Review.ADVERSE: Eligibility.BLOCKED,
            Review.DOCUMENTED_CANDIDATE: Eligibility.ELIGIBLE_IN_PRINCIPLE,
        }[self.review]


def require_execution(admission: DevelopmentAdmission) -> None:
    """Unconditional fail-closed gate, before any path resolution, decode or model load."""
    require(type(admission) is DevelopmentAdmission)
    admission.__post_init__()
    raise ExecutionDenied()


def json_bytes(admission: DevelopmentAdmission) -> bytes:
    """Serialize public proposal metadata, never sample admission or a measured report."""
    require(type(admission) is DevelopmentAdmission)
    admission.__post_init__()
    raw = json.dumps(
        {**asdict(admission), "eligibility": admission.eligibility},
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    require(len(raw) <= MAX_BYTES)
    return raw


def from_json(raw: bytes) -> DevelopmentAdmission:
    try:
        require(type(raw) is bytes and 0 < len(raw) <= MAX_BYTES)
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique)
        require(
            type(data) is dict
            and set(data) == {f.name for f in fields(DevelopmentAdmission)} | {"eligibility"}
        )
        eligibility = data.pop("eligibility")
        require(type(eligibility) is str)
        for key, enum in (
            ("review", Review),
            ("sample_admission", Admission),
            ("artifact_admission", Admission),
            ("execution", Authorization),
        ):
            require(type(data[key]) is str)
            data[key] = enum(data[key])
        result = DevelopmentAdmission(**data)
        require(eligibility == result.eligibility.value)
        return result
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
        raise AdmissionInvalid() from None
