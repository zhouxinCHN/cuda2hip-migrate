"""Collect ROCm and CUDA Samples environment facts on the target machine."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

EXPECTED_ROCM = "10.0.0"
EXPECTED_CUDA_TOOLKIT = "12.9"
EXPECTED_CUDA_SAMPLES_TAG = "v12.9"
EXPECTED_CUDA_SAMPLES_COMMIT = "cab7c66b4fcf35b09ea8fbc4c3c1a9272e92d997"
REQUIRED_TOOLS = ("hipcc", "hipify-clang", "rocminfo", "cmake", "ninja", "git")


def _run(command: list[str]) -> tuple[int, str]:
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode, output.strip()


def _rocm_version() -> str | None:
    info = Path("/opt/rocm/.info/version")
    if info.is_file():
        return info.read_text(encoding="utf-8").strip()
    if shutil.which("dpkg-query"):
        code, output = _run(["dpkg-query", "-W", "-f=${Version}", "rocm-core"])
        if code == 0 and output:
            return output.split()[-1]
    return None


def _gfx_architectures() -> tuple[str, ...]:
    if not shutil.which("rocminfo"):
        return ()
    code, output = _run(["rocminfo"])
    if code != 0:
        return ()
    # Preserve rocminfo's agent order: the discrete GPU is enumerated before
    # the iGPU on the supported target. Sorting made gfx1036 win over gfx1201.
    return tuple(dict.fromkeys(re.findall(r"^\s*Name:\s+(gfx[0-9a-zA-Z]+)\s*$", output, re.M)))


def _tool_version(name: str) -> str | None:
    if not shutil.which(name):
        return None
    if name == "hipify-clang":
        code, output = _run([name, "--version"])
    else:
        code, output = _run([name, "--version"])
    if code != 0 or not output:
        return None
    return output.splitlines()[0]


def collect_environment() -> dict[str, Any]:
    tools = {name: _tool_version(name) for name in REQUIRED_TOOLS}
    hipify_versions = None
    if shutil.which("hipify-clang"):
        code, output = _run(["hipify-clang", "--versions"])
        if code == 0:
            hipify_versions = output
    return {
        "expected": {
            "rocm": EXPECTED_ROCM,
            "cuda_toolkit": EXPECTED_CUDA_TOOLKIT,
            "cuda_samples_tag": EXPECTED_CUDA_SAMPLES_TAG,
            "cuda_samples_commit": EXPECTED_CUDA_SAMPLES_COMMIT,
        },
        "detected": {
            "rocm_version": _rocm_version(),
            "gfx_architectures": list(_gfx_architectures()),
            "tools": tools,
            "hipify_versions": hipify_versions,
        },
        "missing_tools": [name for name, version in tools.items() if version is None],
    }


def rocm_include_dir() -> Path | None:
    """Return the ROCm HIP include directory, if this machine has one.

    hipify-clang must be able to parse ``#include <hip/hip_runtime.h>`` because
    the pre-HIPIFY overlay introduces it before translation.
    """
    for candidate in (Path(os.environ.get("ROCM_PATH", "/opt/rocm")), Path("/opt/rocm")):
        include = candidate / "include"
        if (include / "hip" / "hip_runtime.h").is_file():
            return include
    return None


def detected_rocm_version() -> str | None:
    """Return the ROCm version installed on this machine, if any."""
    return _rocm_version()


def primary_gfx_architecture() -> str | None:
    """Return the first gfx architecture reported by ``rocminfo``.

    rocminfo lists the boot/display device first on the target host. This is
    deliberately not a sorted choice: sorting preferred the iGPU ``gfx1036``
    over the discrete ``gfx1201``.
    """
    architectures = _gfx_architectures()
    return architectures[0] if architectures else None


def evidence_environment() -> dict[str, Any]:
    """Return the environment facts that every evidence record must carry."""
    environment = collect_environment()
    detected = environment["detected"]
    rocm_version = detected["rocm_version"]
    return {
        "target_rocm": EXPECTED_ROCM,
        "detected_rocm": rocm_version,
        "rocm_matches_target": bool(
            rocm_version and rocm_version.startswith(EXPECTED_ROCM)
        ),
        "cuda_toolkit_baseline": EXPECTED_CUDA_TOOLKIT,
        "gfx_architectures": detected["gfx_architectures"],
        "tools": detected["tools"],
        "hipcc_version": detected["tools"].get("hipcc"),
        "hipify_version": detected["tools"].get("hipify-clang"),
    }


def strict_environment_errors(environment: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for name in environment["missing_tools"]:
        errors.append(f"required tool not found: {name}")
    detected = environment["detected"]
    rocm_version = detected["rocm_version"]
    if not rocm_version:
        errors.append("cannot determine ROCm version")
    elif not rocm_version.startswith(EXPECTED_ROCM):
        errors.append(f"ROCm {EXPECTED_ROCM} required, found {rocm_version}")
    if not detected["gfx_architectures"]:
        errors.append("rocminfo reported no gfx architecture")
    return errors
