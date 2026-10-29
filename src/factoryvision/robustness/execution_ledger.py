"""Supervisor-owned synthetic reservations. Not real-development authorization."""

from dataclasses import dataclass, replace
from enum import StrEnum


class LedgerInvalid(ValueError):
    def __init__(self):
        super().__init__("Invalid synthetic execution accounting.")


class Case(StrEnum):
    FIRST = "synthetic_1"
    SECOND = "synthetic_2"
    THIRD = "synthetic_3"
    FOURTH = "synthetic_4"
    FIFTH = "synthetic_5"


class Outcome(StrEnum):
    RESERVED = "reserved"
    DISPATCHED = "dispatched"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    NOT_STARTED = "not_started"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Entry:
    case: Case
    outcome: Outcome = Outcome.RESERVED
    charged: int = 0

    def __post_init__(self):
        if (
            type(self.case) is not Case
            or type(self.outcome) is not Outcome
            or type(self.charged) is not int
            or self.charged not in (0, 1)
            or (self.outcome in (Outcome.RESERVED, Outcome.NOT_STARTED) and self.charged != 0)
            or (
                self.outcome
                in (Outcome.DISPATCHED, Outcome.SUCCEEDED, Outcome.FAILED, Outcome.UNKNOWN)
                and self.charged != 1
            )
        ):
            raise LedgerInvalid()


class ExecutionLedger:
    """Single-supervisor ledger; all reservations consume capacity permanently.

    Charged means dispatch may have executed, not a claim of completed computation.
    No worker counts, reset, refunds, retry, external serialization or execution grants.
    The supervisor serializes access; this is not a tamper-proof cross-process journal.
    """

    def __init__(self, budget=5):
        if type(budget) is not int or not 1 <= budget <= 5:
            raise LedgerInvalid()
        self._budget = budget
        self._entries = ()

    @property
    def entries(self):
        return self._entries

    @property
    def reserved(self):
        return len(self._entries)

    @property
    def charged(self):
        return sum(e.charged for e in self._entries)

    def reserve(self, case):
        if (
            type(case) is not Case
            or self.reserved >= self._budget
            or any(e.case is case for e in self._entries)
            or any(e.outcome in (Outcome.RESERVED, Outcome.DISPATCHED) for e in self._entries)
        ):
            raise LedgerInvalid()
        self._entries += (Entry(case),)

    def dispatch(self):
        if not self._entries or self._entries[-1].outcome is not Outcome.RESERVED:
            raise LedgerInvalid()
        self._entries = self._entries[:-1] + (
            replace(self._entries[-1], outcome=Outcome.DISPATCHED, charged=1),
        )

    def finish(self, outcome):
        if not self._entries or type(outcome) is not Outcome:
            raise LedgerInvalid()
        current = self._entries[-1]
        allowed = (
            (Outcome.NOT_STARTED, Outcome.CANCELLED, Outcome.TIMED_OUT)
            if current.outcome is Outcome.RESERVED
            else (
                Outcome.SUCCEEDED,
                Outcome.FAILED,
                Outcome.TIMED_OUT,
                Outcome.CANCELLED,
                Outcome.UNKNOWN,
            )
            if current.outcome is Outcome.DISPATCHED
            else ()
        )
        if outcome not in allowed:
            raise LedgerInvalid()
        self._entries = self._entries[:-1] + (replace(current, outcome=outcome),)
