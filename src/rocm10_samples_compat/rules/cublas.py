"""Rule HIP-CUBLAS-INCLUDE-001."""

from __future__ import annotations

from .base import RuleChange
from .cpp import find_directives

RULE_ID = "HIP-CUBLAS-INCLUDE-001"
HIPIFY_INCLUDE = "#include <hipblas.h>"
ROCM_INCLUDE = "#include <hipblas/hipblas.h>"


def apply(text: str) -> tuple[str, list[RuleChange]]:
    """Point HIPIFY's ``hipblas.h`` include at its ROCm subdirectory.

    HIPIFY maps ``#include <cublas_v2.h>`` to ``#include <hipblas.h>``. ROCm
    7.2.4 installs the header as ``hipblas/hipblas.h``, so HIPIFY's output does
    not compile. Only the exact directive is rewritten; comments, strings and
    already-correct includes are left alone.
    """
    changes: list[RuleChange] = []
    for start, end, directive in reversed(find_directives(text)):
        stripped = directive.rstrip()
        if stripped != HIPIFY_INCLUDE:
            continue
        line = text.count("\n", 0, start) + 1
        text = text[:start] + ROCM_INCLUDE + text[end:]
        changes.append(
            RuleChange(rule_id=RULE_ID, line=line, before=stripped, after=ROCM_INCLUDE)
        )

    changes.reverse()
    return text, changes
