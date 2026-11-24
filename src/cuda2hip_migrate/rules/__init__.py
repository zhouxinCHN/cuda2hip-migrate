"""Deterministic post-HIPIFY rule registry."""

from __future__ import annotations

from collections.abc import Callable

from . import cublas, hostmalloc
from .base import RuleChange

Rule = Callable[[str], tuple[str, list[RuleChange]]]

RULES: tuple[Rule, ...] = (cublas.apply, hostmalloc.apply)

__all__ = ["RULES", "RuleChange", "apply_post_hipify_rules"]


def apply_post_hipify_rules(text: str) -> tuple[str, list[RuleChange]]:
    """Apply enabled rules in registry order and return all recorded changes."""
    changes: list[RuleChange] = []
    for rule in RULES:
        text, rule_changes = rule(text)
        changes.extend(rule_changes)
    return text, changes
