"""Rule HIP-HOSTMALLOC-ARITY-001."""

from __future__ import annotations

from .base import RuleChange
from .cpp import find_calls

RULE_ID = "HIP-HOSTMALLOC-ARITY-001"


def apply(text: str) -> tuple[str, list[RuleChange]]:
    """Remove the redundant ``0`` argument from HIPIFY's 4-argument form.

    HIPIFY can emit:
        hipHostMalloc(ptr, size, 0, hipHostMallocDefault)
    while HIP exposes a three-argument overload. Only the exact form with a
    third argument ``0`` and fourth argument ``hipHostMallocDefault`` is
    changed.
    """
    changes: list[RuleChange] = []
    for start, _end, close_index, raw_arguments in reversed(
        find_calls(text, "hipHostMalloc")
    ):
        arguments = [argument.strip() for argument in raw_arguments]
        if len(arguments) != 4:
            continue
        if arguments[2] != "0" or arguments[3] != "hipHostMallocDefault":
            continue

        replacement = "hipHostMalloc(" + ", ".join(
            [arguments[0], arguments[1], arguments[3]]
        ) + ")"
        before = text[start : close_index + 1]
        line = text.count("\n", 0, start) + 1
        text = text[:start] + replacement + text[close_index + 1 :]
        changes.append(
            RuleChange(
                rule_id=RULE_ID,
                line=line,
                before=before,
                after=replacement,
            )
        )
    changes.reverse()
    return text, changes
