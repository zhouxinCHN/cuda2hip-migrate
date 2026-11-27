"""Process helpers shared by the pipeline stages."""

from __future__ import annotations

import difflib
import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


class ToolError(RuntimeError):
    """A required external tool is missing or failed."""


@dataclass(frozen=True)
class ToolResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return (self.stdout + self.stderr).strip()

    @property
    def tail(self) -> str:
        return "\n".join(self.output.splitlines()[-40:])


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise ToolError(f"required tool not found on PATH: {name}")
    return path


def run_logged(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 900,
    env: Mapping[str, str] | None = None,
) -> ToolResult:
    """Run a tool and capture its output.

    ``env`` contains additions or overrides for the inherited environment.  It
    is intentionally merged rather than replacing the environment: callers set
    temporary-directory variables without dropping ``PATH`` or the user's
    compiler configuration.
    """
    child_env = None
    if env is not None:
        child_env = os.environ.copy()
        child_env.update(env)
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env=child_env,
    )
    return ToolResult(
        command=tuple(command),
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def unified_diff(before: str, after: str, before_name: str, after_name: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=before_name,
            tofile=after_name,
        )
    )
