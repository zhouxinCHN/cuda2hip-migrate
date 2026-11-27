"""Shared records for deterministic source transformations."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RuleChange:
    """One deterministic source edit, with enough context to audit it."""

    rule_id: str
    line: int
    before: str
    after: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
