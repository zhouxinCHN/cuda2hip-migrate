"""Rule HIP-PROFILER-ROCTX-001.

HIP does not implement ``hipProfilerStart`` / ``hipProfilerStop``. Both are
declared ``HIP_DEPRECATED("use roctracer/rocTX instead")`` in
``hip/hip_runtime_api.h`` and documented to return ``hipErrorNotSupported``; the
exported symbols survive only as ``hip_4.2`` ABI stubs. Measured on ROCm 10.0.0
(HIP 7.15.26333) they return 801 in every situation tried, including under an
attached ``rocprofv3``, so the return value does not depend on whether a
profiler is listening.

A sample that brackets its compute with these calls therefore dies before doing
any work: ``checkCudaErrors`` prints ``code=801`` and exits.

ROCm's replacement is rocTX. ``roctxProfilerResume(0)`` / ``roctxProfilerPause(0)``
pause and resume profiling for the whole process, which is the same contract as
``cudaProfilerStart`` / ``cudaProfilerStop``. Measured on ROCm 10.0.0: both
return 0 when no profiling tool is attached and also when one is already in the
requested state, matching the CUDA behaviour of succeeding without a profiler.

Scope of the rewrite:

- only a zero-argument call is rewritten. Any other shape is left untouched, so
  an unexpected overload is never silently mangled;
- calls inside comments, strings and literals are skipped, and so are member and
  qualified calls (``x.hipProfilerStart()``, ``ns::hipProfilerStart()``);
- ``hipProfilerStartExt`` and friends are different identifiers and never match.

The rewritten call is qualified (``rocm10_compat::hipProfilerStart``), which also
makes the rule idempotent: the identifier matcher rejects a ``::`` prefix, so a
second pass cannot re-match.
"""

from __future__ import annotations

from .base import RuleChange
from .cpp import find_calls, find_directives

RULE_ID = "HIP-PROFILER-ROCTX-001"

COMPAT_INCLUDE = '#include "rocm10_compat/hip_profiler.h"'

#: Runtime headers that declare hipProfilerStart(); the compat include goes
#: directly after the first one of them.
ANCHOR_INCLUDES = (
    "#include <hip/hip_runtime.h>",
    "#include <hip/hip_runtime_api.h>",
)

MAPPING = (
    ("hipProfilerStart", "rocm10_compat::hipProfilerStart"),
    ("hipProfilerStop", "rocm10_compat::hipProfilerStop"),
)

ZERO_ARGUMENTS = [""]


def _rewrite_calls(text: str, name: str, replacement: str) -> tuple[str, list[RuleChange]]:
    changes: list[RuleChange] = []
    for start, _end, close_index, raw_arguments in reversed(find_calls(text, name)):
        if [argument.strip() for argument in raw_arguments] != ZERO_ARGUMENTS:
            continue
        line = text.count("\n", 0, start) + 1
        before = text[start : close_index + 1]
        after = f"{replacement}()"
        text = text[:start] + after + text[close_index + 1 :]
        changes.append(RuleChange(rule_id=RULE_ID, line=line, before=before, after=after))
    changes.reverse()
    return text, changes


def _inject_include(text: str) -> tuple[str, list[RuleChange]]:
    """Add the compat include once, after the HIP runtime include."""
    if COMPAT_INCLUDE in text:
        return text, []

    insert_at = 0
    for start, end, directive in find_directives(text):
        if directive.strip() in ANCHOR_INCLUDES:
            insert_at = end
            break

    line = text.count("\n", 0, insert_at) + 1
    text = text[:insert_at] + "\n" + COMPAT_INCLUDE + text[insert_at:]
    change = RuleChange(rule_id=RULE_ID, line=line, before="", after=COMPAT_INCLUDE)
    return text, [change]


def apply(text: str) -> tuple[str, list[RuleChange]]:
    """Route hipProfilerStart/Stop to the rocTX-based compatibility wrapper."""
    changes: list[RuleChange] = []
    for name, replacement in MAPPING:
        text, call_changes = _rewrite_calls(text, name, replacement)
        changes.extend(call_changes)

    if changes:
        text, include_changes = _inject_include(text)
        changes.extend(include_changes)
    return text, changes
