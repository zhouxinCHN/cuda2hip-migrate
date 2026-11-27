"""Rule HIP-HOSTMALLOC-ARITY-001."""

from __future__ import annotations

from .base import RuleChange
from .cpp import find_calls

RULE_ID = "HIP-HOSTMALLOC-ARITY-001"

# HIP 7.2.4 declares only ``hipError_t hipHostMalloc(void**, size_t, unsigned)``.
# HIPIFY emits the CUDA four-argument form, which does not compile.
EXPECTED_ARGUMENTS = 4
FLAGS_INDEX = 3


def apply(text: str) -> tuple[str, list[RuleChange]]:
    """Drop the redundant ``0`` argument from HIPIFY's four-argument form.

    Only the exact shape ``hipHostMalloc(a, b, 0, <flag>)`` is rewritten; any
    other four-argument call is left untouched so that an unexpected HIP
    overload is not silently mangled.
    """
    changes: list[RuleChange] = []
    for start, end, close_index, raw_arguments in reversed(
        find_calls(text, "hipHostMalloc")
    ):
        arguments = [argument.strip() for argument in raw_arguments]
        if len(arguments) != EXPECTED_ARGUMENTS:
            continue
        if arguments[2] != "0":
            continue
        rewritten = ", ".join(arguments[:2] + [arguments[FLAGS_INDEX]])
        call_start = start
        line = text.count("\n", 0, call_start) + 1
        before = text[start : close_index + 1]
        after = f"hipHostMalloc({rewritten})"
        text = text[:start] + after + text[close_index + 1 :]
        changes.append(RuleChange(rule_id=RULE_ID, line=line, before=before, after=after))

    changes.reverse()
    return text, changes
