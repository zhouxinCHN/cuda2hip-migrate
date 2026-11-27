"""Deterministic post-HIPIFY rule registry.

Rules run in registry order. Every rule is a pure ``str -> (str, changes)``
function, so applying the registry twice yields no further changes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from . import cublas, hostmalloc, profiler
from .base import RuleChange

Rule = Callable[[str], tuple[str, list[RuleChange]]]


@dataclass(frozen=True)
class FeatureRequirement:
    """Build-side requirements introduced when a rule changes source text.

    A source rewrite is not complete until the headers and libraries it relies
    on are visible to the consuming build. Keeping this next to the rule avoids
    the sample-spec-only dependency declarations the generic path cannot use.
    """

    feature: str
    libraries: tuple[str, ...] = ()
    cmake_targets: tuple[str, ...] = ()


RULES: tuple[Rule, ...] = (
    cublas.apply,
    hostmalloc.apply,
    profiler.apply,
)

RULE_IDS: tuple[str, ...] = (cublas.RULE_ID, hostmalloc.RULE_ID, profiler.RULE_ID)

#: Rule id -> requirements. The hipblas entry exists because the include was
#: rewritten to a libhipblas header; a project that already links hipblas gets
#: a duplicate link entry, which linkers tolerate. Reporting the requirement
#: lets the generated CMake file stay complete without sample-specific specs.
RULE_REQUIREMENTS: dict[str, FeatureRequirement] = {
    cublas.RULE_ID: FeatureRequirement("hipblas-subdirectory", libraries=("hipblas",)),
    hostmalloc.RULE_ID: FeatureRequirement("hip-runtime"),
    profiler.RULE_ID: FeatureRequirement(
        "profiler-roctx",
        libraries=("rocprofiler-sdk-roctx",),
        cmake_targets=("ROCmCudaMigrate::profiler",),
    ),
}

__all__ = [
    "RULES",
    "RULE_IDS",
    "RULE_REQUIREMENTS",
    "FeatureRequirement",
    "Rule",
    "RuleChange",
    "apply_post_hipify_rules",
    "rule_hits",
]


def apply_post_hipify_rules(text: str) -> tuple[str, list[RuleChange]]:
    """Apply enabled rules in registry order and return all recorded changes."""
    changes: list[RuleChange] = []
    for rule in RULES:
        text, rule_changes = rule(text)
        changes.extend(rule_changes)
    return text, changes


def rule_hits(changes: list[RuleChange]) -> dict[str, int]:
    """Count rule hits per rule id for evidence output."""
    hits: dict[str, int] = {}
    for change in changes:
        hits[change.rule_id] = hits.get(change.rule_id, 0) + 1
    return hits
