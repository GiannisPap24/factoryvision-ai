"""Breakpoint target using original synthetic geometry; no inference or file writes."""

from factoryvision.demo import build_demo
from factoryvision.inference.review import Decision, demo_records, json_bytes, revise


def main():
    records = demo_records(build_demo().report)
    original = records[0]
    changed = revise(
        original,
        Decision.NEEDS_FURTHER_REVIEW,
        "Synthetic debugger observation",
        "2026-01-01T00:00:00.000000Z",
    )
    assert original.history == () and changed.evidence is original.evidence
    assert changed.evidence.score is None
    assert b"not_verified_ground_truth" in json_bytes((changed, records[1]))
    print("PASS: synthetic review revision; immutable null-score evidence; no writes.")


if __name__ == "__main__":
    main()
