"""Subprocess, tool discovery and diff helpers."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class ToolError(RuntimeError):
    """A required external tool is unavailable or failed to start."""


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    log_path: Path


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise ToolError(f"required tool not found on PATH: {name}")
    return path


def run_logged(command: list[str], log_path: Path) -> CommandResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.flush()
        completed = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return CommandResult(tuple(command), completed.returncode, log_path)


def log_tail(path: Path, lines: int = 40) -> str:
    if not path.is_file():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def detect_arch(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    from_env = os.environ.get("HIP_ARCH")
    if from_env:
        return from_env
    rocminfo = shutil.which("rocminfo")
    if not rocminfo:
        raise ToolError("rocminfo not found; pass --arch or set HIP_ARCH")
    completed = subprocess.run(
        [rocminfo], capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise ToolError("rocminfo failed; pass --arch or set HIP_ARCH")
    match = re.search(r"^\s*Name:\s+(gfx[0-9a-zA-Z]+)\s*$", completed.stdout, re.M)
    if not match:
        raise ToolError("unable to detect gfx architecture; pass --arch or set HIP_ARCH")
    return match.group(1)


def unified_diff(left: Path, right: Path) -> str:
    """Return a stable unified diff for audit artifacts."""
    completed = subprocess.run(
        [
            "diff",
            "-u",
            "--label",
            f"a/{left.name}",
            "--label",
            f"b/{right.name}",
            str(left),
            str(right),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode not in (0, 1):
        raise ToolError(f"diff failed with exit code {completed.returncode}")
    return completed.stdout
