"""Run one packaged CUDA probe through HIPIFY, rules, compilers and checks."""

from __future__ import annotations

import json
import logging
import re
import tempfile
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from cuda2hip_migrate.preprocess import (
    apply_include_redirects,
    unmatched_redirects,
)
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


SPEC_KEYS = frozenset(
    {
        "sample",
        "source",
        "include_dirs",
        "libraries",
        "success_marker",
        "expected",
        "include_redirects",
        "optimization",
    }
)
EXPECTATIONS = ("hipify-direct", "rule-required", "unknown")
OPTIMIZATION_LEVELS = ("-O0", "-O1", "-O2", "-O3")
DEFAULT_OPTIMIZATION = "-O2"


@dataclass(frozen=True)
class SampleSpec:
    """Committed description of how one external sample is adapted."""

    sample: str
    source: Path
    include_dirs: tuple[Path, ...]
    libraries: tuple[str, ...]
    success_marker: str
    expected: str
    include_redirects: tuple[tuple[str, str], ...]
    optimization: str


def _spec_string_list(path: Path, raw: dict[str, object], key: str) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ProbeError("spec", f"{path}: '{key}' must be a list of non-empty strings")
    return tuple(str(item) for item in value)


def load_sample_spec(path: Path) -> SampleSpec:
    """Load and validate one sample spec file."""
    if not path.is_file():
        raise ProbeError("spec", f"sample spec not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ProbeError("spec", f"{path}: invalid JSON: {error}") from error
    if not isinstance(raw, dict):
        raise ProbeError("spec", f"{path}: spec must be a JSON object")

    unknown = set(raw) - SPEC_KEYS
    if unknown:
        names = ", ".join(sorted(str(key) for key in unknown))
        raise ProbeError("spec", f"{path}: unsupported key(s): {names}")

    for key in ("sample", "source"):
        if not isinstance(raw.get(key), str) or not raw[key]:
            raise ProbeError("spec", f"{path}: '{key}' is required and must be a string")
    sample = str(raw["sample"])
    source = Path(str(raw["source"]))

    expected = raw.get("expected", "unknown")
    if expected not in EXPECTATIONS:
        raise ProbeError(
            "spec", f"{path}: 'expected' must be one of: {', '.join(EXPECTATIONS)}"
        )

    marker = raw.get("success_marker", DEFAULT_SUCCESS_MARKER)
    if not isinstance(marker, str) or not marker:
        raise ProbeError("spec", f"{path}: 'success_marker' must be a non-empty string")

    redirects_raw = raw.get("include_redirects", {})
    if not isinstance(redirects_raw, dict) or not all(
        isinstance(header, str) and header and isinstance(target, str) and target
        for header, target in redirects_raw.items()
    ):
        raise ProbeError(
            "spec",
            f"{path}: 'include_redirects' must map non-empty header names to targets",
        )

    optimization = raw.get("optimization", DEFAULT_OPTIMIZATION)
    if optimization not in OPTIMIZATION_LEVELS:
        raise ProbeError(
            "spec",
            f"{path}: 'optimization' must be one of: {', '.join(OPTIMIZATION_LEVELS)}",
        )

    return SampleSpec(
        sample=sample,
        source=source,
        include_dirs=tuple(
            Path(item) for item in _spec_string_list(path, raw, "include_dirs")
        ),
        libraries=_spec_string_list(path, raw, "libraries"),
        success_marker=marker,
        expected=str(expected),
        include_redirects=tuple(
            (str(header), str(target)) for header, target in redirects_raw.items()
        ),
        optimization=str(optimization),
    )


def check_expectation(expected: str | None, applied: int) -> None:
    """Fail the run when the declared adaptation class does not match reality."""
    if expected is None or expected == "unknown":
        return
    if expected == "hipify-direct" and applied:
        raise ProbeError(
            "expectation",
            f"declared hipify-direct but {applied} rule hit(s) were applied",
        )
    if expected == "rule-required" and not applied:
        raise ProbeError("expectation", "declared rule-required but no rule matched")


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
        include_redirects=(),
        expected=None,
        optimization=DEFAULT_OPTIMIZATION,
    )


def run_source_file(
    source_path: Path,
    *,
    out_dir: Path | None = None,
    arch: str | None = None,
    include_dirs: tuple[Path, ...] = (),
    libraries: tuple[str, ...] = (),
    success_marker: str = DEFAULT_SUCCESS_MARKER,
    include_redirects: tuple[tuple[str, str], ...] = (),
    optimization: str = DEFAULT_OPTIMIZATION,
) -> RunResult:
    """Run an external CUDA source file, for example one cuda-samples file.

    The file is read and copied into the artifacts; it is never modified in
    place. ``include_dirs`` are forwarded to ``hipify-clang`` and ``hipcc`` so a
    sample can resolve its own helper headers. ``include_redirects`` are
    pre-HIPIFY include rewrites, used to route a sample at a compatibility
    header without editing it.
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
        include_redirects=tuple(include_redirects),
        expected=None,
        optimization=optimization,
    )


def run_sample_spec(path: Path, *, out_dir: Path | None = None, arch: str | None = None) -> RunResult:
    """Run an external source described by a committed sample spec file."""
    spec = load_sample_spec(Path(path))
    if not spec.source.is_file():
        raise ProbeError("source", f"spec source not found: {spec.source}")
    return _run_source(
        source_name(spec.source),
        spec.source.read_text(encoding="utf-8"),
        out_dir=out_dir,
        arch=arch,
        include_dirs=spec.include_dirs,
        libraries=spec.libraries,
        success_marker=spec.success_marker,
        include_redirects=spec.include_redirects,
        expected=spec.expected,
        optimization=spec.optimization,
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
    include_redirects: tuple[tuple[str, str], ...],
    expected: str | None,
    optimization: str,
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

    for header in unmatched_redirects(source_text, include_redirects):
        logger.warning(
            "[REDIRECT-MISS] %s is declared in the spec but no matching include "
            "was found; the overlay header is not in use",
            header,
        )

    preprocessed_text, redirect_changes = apply_include_redirects(
        source_text, include_redirects
    )
    if redirect_changes:
        preprocessed_dir = artifacts / "preprocessed"
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        hipify_input = preprocessed_dir / f"{name}.cu"
        hipify_input.write_text(preprocessed_text, encoding="utf-8")
        (artifacts / "redirects.json").write_text(
            json.dumps([change.as_dict() for change in redirect_changes], indent=2)
            + "\n",
            encoding="utf-8",
        )
        for change in redirect_changes:
            logger.info(
                "[REDIRECT] %s line=%d %s -> %s",
                change.rule_id,
                change.line,
                change.before,
                change.after,
            )
        logger.info("[REDIRECTS] applied=%d", len(redirect_changes))
    else:
        hipify_input = source

    hipified = hipified_dir / f"{name}.hip.cu"
    adapted = adapted_dir / f"{name}.hip.cu"
    binary = artifacts / name

    logger.info("probe=%s arch=%s out=%s", name, detected_arch, artifacts)

    include_flags = [f"-I{directory}" for directory in include_dirs]
    if include_flags:
        logger.info("[INCLUDE] %s", " ".join(include_flags))

    hipify_log = artifacts / "hipify.log"
    result = run_logged(
        [hipify, str(hipify_input), *include_flags, "-o", str(hipified)], hipify_log
    )
    if result.returncode != 0 or not hipified.is_file():
        raise ProbeError(
            "hipify",
            "HIPIFY failed",
            tail=log_tail(hipify_log),
            artifacts=artifacts,
        )

    (artifacts / "hipify.diff").write_text(
        unified_diff(hipify_input, hipified), encoding="utf-8"
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
    if expected is not None:
        logger.info("[EXPECTED] %s rules=%d", expected, len(changes))
    check_expectation(expected, len(changes))
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

    if optimization != DEFAULT_OPTIMIZATION:
        logger.info("[BUILD] optimization=%s", optimization)

    build_log = artifacts / "build.log"
    result = run_logged(
        [
            hipcc,
            "-x",
            "hip",
            "-std=c++17",
            optimization,
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
