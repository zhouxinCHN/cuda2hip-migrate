"""Run one packaged CUDA probe through HIPIFY, rules, compilers and checks."""

from __future__ import annotations

import json
import logging
import re
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
BUILD_SPEC_SUFFIX = ".build.json"
ALLOWED_BUILD_KEYS = frozenset({"libraries"})
DEFAULT_SUCCESS_MARKER = "RESULT=PASS"


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
    success_marker: str


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


def build_flags_from_spec(probe: str, spec: object) -> tuple[str, ...]:
    """Validate one probe build-spec object and return hipcc link flags."""
    if not isinstance(spec, dict):
        raise ProbeError("build-spec", f"{probe}: build spec must be a JSON object")
    unknown = set(spec) - ALLOWED_BUILD_KEYS
    if unknown:
        names = ", ".join(sorted(str(key) for key in unknown))
        raise ProbeError("build-spec", f"{probe}: unsupported build spec key(s): {names}")
    libraries = spec.get("libraries", [])
    if not isinstance(libraries, list) or not all(
        isinstance(library, str) and library for library in libraries
    ):
        raise ProbeError("build-spec", f"{probe}: 'libraries' must be a list of names")
    return tuple(f"-l{library}" for library in libraries)


def probe_build_flags(name: str) -> tuple[str, ...]:
    """Return extra hipcc flags declared by the probe's optional build spec.

    A probe without ``<probe>.build.json`` gets no extra flags, so existing
    probes keep the plain build command.
    """
    spec_resource = files(PROBE_PACKAGE).joinpath(name + BUILD_SPEC_SUFFIX)
    if not spec_resource.is_file():
        return ()
    try:
        spec = json.loads(spec_resource.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ProbeError("build-spec", f"{name}: invalid build spec JSON: {error}") from error
    return build_flags_from_spec(name, spec)


def source_name(source_path: Path) -> str:
    """Derive a safe artifact name from an external source file name."""
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(source_path).stem)
    if not name or name.startswith("."):
        raise ProbeError("source", f"cannot derive an artifact name from {source_path}")
    return name


def run_probe(name: str, *, out_dir: Path | None = None, arch: str | None = None) -> RunResult:
    """Run one packaged probe with the default acceptance marker."""
    return _run_source(
        name,
        probe_source_text(name),
        out_dir=out_dir,
        arch=arch,
        include_dirs=(),
        libraries=(),
        success_marker=DEFAULT_SUCCESS_MARKER,
    )


def run_source_file(
    source_path: Path,
    *,
    out_dir: Path | None = None,
    arch: str | None = None,
    include_dirs: tuple[Path, ...] = (),
    libraries: tuple[str, ...] = (),
    success_marker: str = DEFAULT_SUCCESS_MARKER,
) -> RunResult:
    """Run an external CUDA source file, for example one cuda-samples file.

    The file is read and copied into the artifacts; it is never modified in
    place. ``include_dirs`` are forwarded to ``hipify-clang`` and ``hipcc`` so a
    sample can resolve its own helper headers.
    """
    path = Path(source_path)
    if not path.is_file():
        raise ProbeError("source", f"source file not found: {path}")
    if not success_marker:
        raise ProbeError("source", "success marker must not be empty")
    return _run_source(
        source_name(path),
        path.read_text(encoding="utf-8"),
        out_dir=out_dir,
        arch=arch,
        include_dirs=tuple(Path(directory) for directory in include_dirs),
        libraries=tuple(libraries),
        success_marker=success_marker,
    )


def _run_source(
    name: str,
    source_text: str,
    *,
    out_dir: Path | None,
    arch: str | None,
    include_dirs: tuple[Path, ...],
    libraries: tuple[str, ...],
    success_marker: str,
) -> RunResult:
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

    include_flags = [f"-I{directory}" for directory in include_dirs]
    if include_flags:
        logger.info("[INCLUDE] %s", " ".join(include_flags))

    hipify_log = artifacts / "hipify.log"
    result = run_logged(
        [hipify, str(source), *include_flags, "-o", str(hipified)], hipify_log
    )
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

    build_flags = (
        probe_build_flags(name) + tuple(f"-l{library}" for library in libraries)
    )
    if build_flags:
        logger.info("[BUILD] flags=%s", " ".join(build_flags))

    build_log = artifacts / "build.log"
    result = run_logged(
        [
            hipcc,
            "-x",
            "hip",
            "-std=c++17",
            "-O2",
            f"--offload-arch={detected_arch}",
            *include_flags,
            *build_flags,
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

    if success_marker not in output:
        raise ProbeError(
            "C3",
            f"{success_marker} marker missing",
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
        success_marker=success_marker,
    )
