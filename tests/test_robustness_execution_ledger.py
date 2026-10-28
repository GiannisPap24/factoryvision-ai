"""Pure synthetic supervisor accounting; no subprocess or model work."""

from dataclasses import FrozenInstanceError

import pytest

from factoryvision.robustness.execution_ledger import (
    Case,
    Entry,
    ExecutionLedger,
    LedgerInvalid,
    Outcome,
)


@pytest.mark.parametrize("budget", [0, 6, True, 1.0, None, -1])
def test_invalid_budget(budget):
    with pytest.raises(LedgerInvalid):
        ExecutionLedger(budget)


def test_exact_boundary_and_failures_never_refund_or_reset():
    ledger = ExecutionLedger()
    outcomes = [
        Outcome.SUCCEEDED,
        Outcome.FAILED,
        Outcome.TIMED_OUT,
        Outcome.CANCELLED,
        Outcome.UNKNOWN,
    ]
    for i, (case, outcome) in enumerate(zip(Case, outcomes, strict=True), 1):
        ledger.reserve(case)
        assert ledger.reserved == i and ledger.charged == i - 1
        ledger.dispatch()
        assert ledger.charged == i
        ledger.finish(outcome)
        assert ledger.reserved == ledger.charged == i
    with pytest.raises(LedgerInvalid):
        ledger.reserve(Case.FIRST)
    assert tuple(e.outcome for e in ledger.entries) == tuple(outcomes)


@pytest.mark.parametrize("outcome", [Outcome.NOT_STARTED, Outcome.CANCELLED, Outcome.TIMED_OUT])
def test_unstarted_still_consumes_capacity(outcome):
    ledger = ExecutionLedger(1)
    ledger.reserve(Case.FIRST)
    ledger.finish(outcome)
    assert ledger.reserved == 1 and ledger.charged == 0
    with pytest.raises(LedgerInvalid):
        ledger.reserve(Case.SECOND)


def test_illegal_transitions_duplicates_and_immutable_snapshots():
    ledger = ExecutionLedger()
    for operation in (ledger.dispatch, lambda: ledger.finish(Outcome.SUCCEEDED)):
        with pytest.raises(LedgerInvalid):
            operation()
    with pytest.raises(LedgerInvalid):
        ledger.reserve("synthetic_1")
    ledger.reserve(Case.FIRST)
    snapshot = ledger.entries
    for operation in (
        lambda: ledger.reserve(Case.SECOND),
        lambda: ledger.finish(Outcome.SUCCEEDED),
        lambda: ledger.finish("not_started"),
    ):
        with pytest.raises(LedgerInvalid):
            operation()
    ledger.dispatch()
    with pytest.raises(LedgerInvalid):
        ledger.dispatch()
    with pytest.raises(LedgerInvalid):
        ledger.finish(Outcome.NOT_STARTED)
    ledger.finish(Outcome.UNKNOWN)
    for operation in (
        lambda: ledger.reserve(Case.FIRST),
        lambda: ledger.finish(Outcome.SUCCEEDED),
        ledger.dispatch,
    ):
        with pytest.raises(LedgerInvalid):
            operation()
    assert snapshot[0].outcome is Outcome.RESERVED
    with pytest.raises(FrozenInstanceError):
        snapshot[0].charged = 1


@pytest.mark.parametrize(
    "args",
    [
        ("synthetic_1", Outcome.RESERVED, 0),
        (Case.FIRST, "reserved", 0),
        (Case.FIRST, Outcome.RESERVED, True),
        (Case.FIRST, Outcome.NOT_STARTED, 1),
        (Case.FIRST, Outcome.SUCCEEDED, 0),
        (Case.FIRST, Outcome.UNKNOWN, 0),
        (Case.FIRST, Outcome.FAILED, 2),
    ],
)
def test_invalid_scalar_records(args):
    with pytest.raises(LedgerInvalid):
        Entry(*args)
