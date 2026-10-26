"""Stable, machine-readable contract diagnostics."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, order=True)
class Issue:
    code: str
    field: str
    message: str


class ContractError(ValueError):
    """One or more sorted contract violations; input is never silently repaired."""

    def __init__(self, issues: tuple[Issue, ...]) -> None:
        self.issues = tuple(sorted(issues))
        super().__init__("; ".join(f"{i.code} [{i.field}]: {i.message}" for i in self.issues))


def reject(code: str, field: str, message: str) -> None:
    raise ContractError((Issue(code, field, message),))
