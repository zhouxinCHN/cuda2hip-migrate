"""Deterministic pre-HIPIFY source transformations.

These run on the copied source before ``hipify-clang`` so that a sample can be
routed to a compatibility header without touching the original checkout. They
are deliberately narrow: an include redirect matches one exact directive and
never fires inside comments, strings or character literals.

Redirects are applied before HIPIFY, not after: HIPIFY rewrites
``<helper_cuda.h>`` to a HIP runtime header, and a post-HIPIFY rule could not
distinguish that output from a legitimate direct include of the same header.
"""

from __future__ import annotations

import re

from .rules.base import RuleChange
from .rules.cpp import find_directives

RULE_ID = "PRE-HIPIFY-INCLUDE-REDIRECT"

_DIRECTIVE = re.compile(
    r"^(?P<prefix>#\s*include\s*)<(?P<header>[^<>\"\s]+)>(?P<tail>\s*(?://.*)?)$"
)


def matched_headers(directive: str) -> str | None:
    """Return the header of an ``#include <...>`` line, ignoring a trailing comment."""
    match = _DIRECTIVE.match(directive.rstrip())
    return match.group("header") if match else None


def apply_include_redirects(
    text: str, redirects: tuple[tuple[str, str], ...]
) -> tuple[str, list[RuleChange]]:
    """Rewrite ``#include <header>`` to ``#include "replacement"`` per mapping.

    A trailing line comment is preserved. Redirects that never match are the
    caller's business: the runner reports them so a silently unused overlay
    header cannot be mistaken for a successful redirect.
    """
    if not redirects:
        return text, []

    mapping = dict(redirects)
    changes: list[RuleChange] = []
    for start, end, directive in reversed(find_directives(text)):
        stripped = directive.rstrip()
        match = _DIRECTIVE.match(stripped)
        if match is None:
            continue
        replacement = mapping.get(match.group("header"))
        if replacement is None:
            continue

        target = f'{match.group("prefix")}"{replacement}"{match.group("tail")}'
        core_end = start + len(stripped)
        line = text.count("\n", 0, start) + 1
        text = text[:start] + target + text[core_end:]
        changes.append(
            RuleChange(
                rule_id=RULE_ID,
                line=line,
                before=stripped,
                after=target,
            )
        )

    changes.reverse()
    return text, changes


def unmatched_redirects(
    text: str, redirects: tuple[tuple[str, str], ...]
) -> tuple[str, ...]:
    """Return declared headers that no directive in ``text`` refers to."""
    present = {
        header
        for _start, _end, directive in find_directives(text)
        if (header := matched_headers(directive)) is not None
    }
    return tuple(header for header, _target in redirects if header not in present)
