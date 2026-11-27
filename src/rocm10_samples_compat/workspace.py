"""Tool-owned workspace paths.

Every temporary file the tool can control belongs under the caller-supplied
output root.  ``temp_root`` and ``temp_environment`` are shared by the generic
converter and the sample acceptance pipeline so that a location policy cannot
drift between the two code paths.
"""

from __future__ import annotations

import re
from pathlib import Path

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_FALLBACK_NAME = "project"
_MAX_NAME_LENGTH = 64


class WorkspaceError(RuntimeError):
    """A tool-owned path cannot be created inside the output root."""


def unit_slug(name: str, *, fallback: str = _FALLBACK_NAME) -> str:
    """Return a safe, deterministic directory name for one project or sample."""
    slug = _SAFE_NAME.sub("-", name).strip("._-")
    slug = slug[:_MAX_NAME_LENGTH].rstrip("._-")
    return slug or fallback


def temp_root(output_root: Path, unit_name: str) -> Path:
    """Create and return ``<output>/.tmp/<unit-name>``.

    The output directory is user-selected and is the only place where the tool
    creates temporary files.  Resolving the candidate catches a ``.tmp``
    symlink that would otherwise redirect writes outside the output root.
    """
    output = Path(output_root).expanduser().resolve()
    candidate = output / ".tmp" / unit_slug(unit_name)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(output)
    except ValueError as error:
        raise WorkspaceError(
            f"temporary directory escapes the output root: {candidate} -> {resolved}"
        ) from error
    try:
        resolved.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise WorkspaceError(
            f"cannot create temporary directory {resolved}: {error}"
        ) from error
    return resolved


def temp_environment(directory: Path) -> dict[str, str]:
    """Return environment overrides that keep child processes inside *directory*.

    ``TMPDIR`` is what clang/hipify-clang use on Linux.  ``TMP`` and ``TEMP``
    are set too so that a tool that uses the portable spellings follows the same
    policy.  Callers merge these over the inherited environment; they do not
    replace ``PATH`` or other build settings.
    """
    resolved = Path(directory).expanduser().resolve()
    try:
        resolved.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise WorkspaceError(
            f"cannot create temporary directory {resolved}: {error}"
        ) from error
    value = str(resolved)
    return {"TMPDIR": value, "TMP": value, "TEMP": value}
