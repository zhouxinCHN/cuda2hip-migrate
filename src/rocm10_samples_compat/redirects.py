"""Deterministic pre-HIPIFY include routing.

HIPIFY rewrites ``#include <helper_cuda.h>`` to ``#include
<hip/hip_runtime_api.h>``. The helper definitions the sample calls then vanish,
and no post-HIPIFY rule can restore them: the rewritten include is
indistinguishable from a legitimate direct include of the same HIP header.

Routing must therefore happen before HIPIFY. A redirect is a declared mapping
from an exact include name to a compatibility header shipped by the overlay.
A declared redirect that matches nothing is a configuration failure: it means
either the sample changed or the overlay is not wired up, and both must stop the
run rather than silently produce a build that looks adapted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .rules.base import RuleChange
from .rules.cpp import find_directives

RULE_ID = "PRE-HIPIFY-INCLUDE-REDIRECT"

_DIRECTIVE = re.compile(
    r"^(?P<prefix>#\s*include\s*)"
    r'(?:<(?P<angle>[^<>"\s]+)>|"(?P<quoted>[^<>"\s]+)")'
    r"(?P<tail>\s*(?://.*)?)$"
)


@dataclass(frozen=True)
class Redirect:
    """One declared include redirect."""

    header: str
    replacement: str
    reason: str
    optional: bool = False


#: Default routing for CUDA Samples v12.9 Common CUDA-coupled helpers.
DEFAULT_REDIRECTS: tuple[Redirect, ...] = (
    Redirect(
        header="helper_cuda.h",
        replacement="rocm10_compat/helper_cuda.h",
        reason=(
            "HIPIFY maps helper_cuda.h to hip/hip_runtime_api.h and drops "
            "checkCudaErrors/findCudaDevice definitions"
        ),
    ),
    Redirect(
        header="helper_cuda_drvapi.h",
        replacement="rocm10_compat/helper_cuda_drvapi.h",
        reason=(
            "driver-API helper header has the same HIPIFY mapping problem"
        ),
        optional=True,
    ),
)


def matched_header(directive: str) -> str | None:
    """Return the header named by an include directive, or ``None``."""
    match = _DIRECTIVE.match(directive.rstrip())
    if match is None:
        return None
    return match.group("angle") or match.group("quoted")


def apply_redirects(
    text: str, redirects: tuple[Redirect, ...]
) -> tuple[str, list[RuleChange]]:
    """Rewrite matching include directives, preserving quotes and comments.

    Only ``#`` directives outside comments and literals are considered.
    """
    if not redirects:
        return text, []
    mapping = {redirect.header: redirect.replacement for redirect in redirects}
    changes: list[RuleChange] = []
    for start, end, directive in reversed(find_directives(text)):
        stripped = directive.rstrip()
        header = matched_header(stripped)
        if header is None or header not in mapping:
            continue
        replacement = mapping[header]
        prefix = _DIRECTIVE.match(stripped).group("prefix")  # type: ignore[union-attr]
        tail = _DIRECTIVE.match(stripped).group("tail")  # type: ignore[union-attr]
        after = f'{prefix}"{replacement}"{tail}'
        line = text.count("\n", 0, start) + 1
        text = text[:start] + after + text[end:]
        changes.append(RuleChange(rule_id=RULE_ID, line=line, before=stripped, after=after))
    changes.reverse()
    return text, changes


def unmatched_redirects(
    texts: list[str], redirects: tuple[Redirect, ...]
) -> tuple[Redirect, ...]:
    """Return declared, non-optional redirects that no file matched."""
    seen: set[str] = set()
    for text in texts:
        for _start, _end, directive in find_directives(text):
            header = matched_header(directive)
            if header is not None:
                seen.add(header)
    return tuple(
        redirect
        for redirect in redirects
        if not redirect.optional and redirect.header not in seen
    )
