"""Run one packaged CUDA probe through HIPIFY, rules, compilers and checks."""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from cuda2hip_migrate.process import (
    ToolError,
    detect_arch,
    log_tail,
    require_tool,
    run_logged,
    unified_diff,
)
from cuda2hip_migrate.rules import RuleChange, apply_post_hipify_rules

logger = logging.getLogger(__name__)

PROBE_PACKAGE = "cuda2hip_migrate.probes"
PROBE_SUFFIX = ".cu"


class ProbeError(RuntimeError):
    """A probe failed at a named stage."""

    def __init__(self, stage: str, message: str, *, tail: str = "", artifacts: Path | None = None):
        super().__init__(message)
        self.stage = stage
        self.tail = tail
        self.artifacts = artifacts


@dataclass(frozen=True)
class RunResult:
    probe: str
    arch: str
    out_dir: Path
    changes: tuple[RuleChange, ...]


def probe_names() -> tuple[str, ...]:
    root = files(PROBE_PACKAGE)
    return tuple(
        sorted(
            entry.name[: -len(PROBE_SUFFIX)]
            for entry in root.iterdir()
            if entry.is_file() and entry.name.endswith(PROBE_SUFFIX)
        )
    )


def probe_source_text(name: str) -> str:
    if name not in probe_names():
        available = ", ".join(probe_names()) or "<none>"
        raise ProbeError("probe", f"unknown probe {name!r}; available: {available}")
    return files(PROBE_PACKAGE).joinpath(name + PROBE_SUFFIX).read_text(encoding="utf-8")


def materialize_probe(name: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(probe_source_text(name), encoding="utf-8")
    return destination


def run_probe(name: str, *, out_dir: Path | None = None, arch: str | None = None) -> RunResult:
    source_text = probe_source_text(name)
    hipify = require_tool("hipify-clang")
    hipcc = require_tool("hipcc")
    try:
        detected_arch = detect_arch(arch)
    except ToolError as error:
        raise ProbeError("environment", str(error)) from error

    artifacts = (out_dir or Path(tempfile.mkdtemp(prefix=f"{name}-"))).resolve()
    source_dir = artifacts / "source"
    hipified_dir = artifacts / "hipified"
    adapted_dir = artifacts / "adapted"
    for directory in (source_dir, hipified_dir, adapted_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source = source_dir / f"{name}.cu"
    source.write_text(source_text, encoding="utf-8")
    hipified = hipified_dir / f"{name}.hip.cu"
    adapted = adapted_dir / f"{name}.hip.cu"
    binary = artifacts / name

    logger.info("probe=%s arch=%s out=%s", name, detected_arch, artifacts)

    hipify_log = artifacts / "hipify.log"
    result = run_logged([hipify, str(source), "-o", str(hipified)], hipify_log)
    if result.returncode != 0 or not hipified.is_file():
        raise ProbeError(
            "hipify",
            "HIPIFY failed",
            tail=log_tail(hipify_log),
            artifacts=artifacts,
        )

    (artifacts / "hipify.diff").write_text(
        unified_diff(source, hipified), encoding="utf-8"
    )

    hipified_text = hipified.read_text(encoding="utf-8")
    adapted_text, changes = apply_post_hipify_rules(hipified_text)
    adapted.write_text(adapted_text, encoding="utf-8")
    (artifacts / "rules.json").write_text(
        json.dumps([change.as_dict() for change in changes], indent=2) + "\n",
        encoding="utf-8",
    )
    for change in changes:
        logger.info(
            "[RULE] %s line=%d %s -> %s",
            change.rule_id,
            change.line,
            change.before,
            change.after,
        )
    logger.info("[RULES] applied=%d", len(changes))
    if not changes and adapted_text != hipified_text:
        raise ProbeError(
            "rules",
            "adapted source changed without a recorded rule",
            artifacts=artifacts,
        )

    build_log = artifacts / "build.log"
    result = run_logged(
        [
            hipcc,
            "-x",
            "hip",
            "-std=c++17",
            "-O2",
            f"--offload-arch={detected_arch}",
            str(adapted),
            "-o",
            str(binary),
        ],
        build_log,
    )
    if result.returncode != 0 or not binary.is_file():
        raise ProbeError(
            "C1",
            "HIP build failed",
            tail=log_tail(build_log),
            artifacts=artifacts,
        )
    logger.info("[C1] PASS")

    run_log = artifacts / "run.log"
    result = run_logged([str(binary)], run_log)
    output = run_log.read_text(encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise ProbeError(
            "C2",
            f"probe exited with {result.returncode}",
            tail=output,
            artifacts=artifacts,
        )
    for line in output.splitlines():
        logger.info("%s", line)
    logger.info("[C2] PASS")

    if "RESULT=PASS" not in output:
        raise ProbeError(
            "C3",
            "RESULT=PASS marker missing",
            tail=output,
            artifacts=artifacts,
        )
    logger.info("[C3] PASS")
    logger.info("ARTIFACTS=%s", artifacts)

    return RunResult(
        probe=name,
        arch=detected_arch,
        out_dir=artifacts,
        changes=tuple(changes),
    )
