"""Run one packaged CUDA probe through HIPIFY, rules, compilers and checks."""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from cuda2hip_migrate.cuda_root import CudaRoot, CudaRootError, resolve_cuda_root
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

#: hipify-clang can exit 0 while reporting a failure; these markers are checked
#: in addition to the exit status and the existence of the output file.
HIPIFY_ERROR_MARKERS = ("[HIPIFY] error", "Error while processing")
HIP_TAIL = ".hip"


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
        "schema_version",
        "cuda_samples_commit",
        "sample",
        "source",
        "include_dirs",
        "libraries",
        "success_marker",
        "expected",
        "expected_redirects",
        "include_redirects",
        "optional_redirects",
        "hipify_forced_includes",
        "optimization",
        "cudart_version",
    }
)
SPEC_SCHEMA_VERSION = 1
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
    expected_redirects: int | None
    include_redirects: tuple[tuple[str, str], ...]
    optional_redirects: tuple[str, ...]
    hipify_forced_includes: tuple[Path, ...]
    optimization: str
    schema_version: int
    cuda_samples_commit: str | None
    cudart_version: int | None = None


def _spec_string_list(path: Path, raw: dict[str, object], key: str) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ProbeError("spec", f"{path}: '{key}' must be a list of non-empty strings")
    return tuple(str(item) for item in value)


def resolve_forced_includes(
    spec_path: Path,
    include_dirs: tuple[Path, ...],
    names: tuple[str, ...],
) -> tuple[Path, ...]:
    """Resolve parse-only forced includes against the declared include dirs."""
    resolved: list[Path] = []
    for name in names:
        for directory in include_dirs:
            candidate = directory / name
            if candidate.is_file():
                resolved.append(candidate)
                break
        else:
            raise ProbeError(
                "spec",
                f"{spec_path}: hipify_forced_includes entry {name!r} was not found "
                "under any declared include_dir",
            )
    return tuple(resolved)


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

    schema_version = raw.get("schema_version", SPEC_SCHEMA_VERSION)
    if schema_version != SPEC_SCHEMA_VERSION:
        raise ProbeError(
            "spec",
            f"{path}: unsupported schema_version {schema_version!r}; "
            f"expected {SPEC_SCHEMA_VERSION}",
        )

    commit = raw.get("cuda_samples_commit")
    if commit is not None and (not isinstance(commit, str) or not commit):
        raise ProbeError("spec", f"{path}: 'cuda_samples_commit' must be a string")

    for key in ("sample", "source"):
        if not isinstance(raw.get(key), str) or not raw[key]:
            raise ProbeError("spec", f"{path}: '{key}' is required and must be a string")
    sample = str(raw["sample"])
    # Paths are resolved relative to the spec file, never the process CWD, so a
    # committed spec behaves the same from any working directory.
    base_dir = path.resolve().parent
    source = base_dir / str(raw["source"])

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

    expected_redirects = raw.get("expected_redirects")
    if expected_redirects is not None and (
        isinstance(expected_redirects, bool)
        or not isinstance(expected_redirects, int)
        or expected_redirects < 0
    ):
        raise ProbeError(
            "spec", f"{path}: 'expected_redirects' must be a non-negative integer"
        )

    cudart_version = raw.get("cudart_version")
    if cudart_version is not None and (
        isinstance(cudart_version, bool)
        or not isinstance(cudart_version, int)
        or cudart_version < 1000
    ):
        raise ProbeError(
            "spec",
            f"{path}: 'cudart_version' must be a CUDA_VERSION integer such as 12000",
        )

    include_dirs = tuple(
        base_dir / item for item in _spec_string_list(path, raw, "include_dirs")
    )
    forced_includes = resolve_forced_includes(
        path,
        include_dirs,
        _spec_string_list(path, raw, "hipify_forced_includes"),
    )

    return SampleSpec(
        sample=sample,
        source=source,
        include_dirs=include_dirs,
        libraries=_spec_string_list(path, raw, "libraries"),
        success_marker=marker,
        expected=str(expected),
        expected_redirects=expected_redirects,
        include_redirects=tuple(
            (str(header), str(target)) for header, target in redirects_raw.items()
        ),
        optional_redirects=_spec_string_list(path, raw, "optional_redirects"),
        hipify_forced_includes=forced_includes,
        optimization=str(optimization),
        schema_version=int(schema_version),
        cuda_samples_commit=None if commit is None else str(commit),
        cudart_version=None if cudart_version is None else int(cudart_version),
    )


def check_expectation(expected: str | None, applied: int) -> None:
    """Fail the run when the declared adaptation class does not match reality.

    ``expected`` describes post-HIPIFY rule hits only. Redirect counts are
    covered separately by :func:`check_redirect_expectation`.
    """
    if expected is None or expected == "unknown":
        return
    if expected == "hipify-direct" and applied:
        raise ProbeError(
            "expectation",
            f"declared hipify-direct but {applied} rule hit(s) were applied",
        )
    if expected == "rule-required" and not applied:
        raise ProbeError("expectation", "declared rule-required but no rule matched")


def check_redirect_expectation(
    expected: int | None, applied: int, *, spec_path: Path | None = None
) -> None:
    """Fail the run when the declared redirect count does not match reality."""
    if expected is None or applied == expected:
        return
    where = f" in {spec_path}" if spec_path else ""
    raise ProbeError(
        "expectation",
        f"declared expected_redirects={expected} but {applied} redirect(s) "
        f"were applied{where}",
    )


def check_redirects(
    name: str,
    unmatched: tuple[str, ...],
    optional: tuple[str, ...],
    *,
    spec_path: Path | None = None,
) -> tuple[str, ...]:
    """Fail when a declared redirect matched nothing, unless it is optional.

    Returns the unmatched headers that were declared optional.
    """
    allowed = set(optional)
    missing = tuple(header for header in unmatched if header not in allowed)
    if missing:
        where = f" in {spec_path}" if spec_path else ""
        raise ProbeError(
            "redirect",
            f"{name}{where}: declared include redirect(s) not found in source: "
            f"{', '.join(missing)}; the overlay header would be unused. "
            "Declare the header in 'optional_redirects' if that is intended.",
        )
    return tuple(header for header in unmatched if header in allowed)


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
        optional_redirects=(),
        expected_redirects=None,
        spec_path=None,
        forced_includes=(),
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
    optional_redirects: tuple[str, ...] = (),
    forced_includes: tuple[Path, ...] = (),
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
        optional_redirects=tuple(optional_redirects),
        expected_redirects=None,
        spec_path=None,
        forced_includes=tuple(Path(item) for item in forced_includes),
        expected=None,
        optimization=optimization,
    )


def run_sample_spec(path: Path, *, out_dir: Path | None = None, arch: str | None = None) -> RunResult:
    """Run an external source described by a committed sample spec file."""
    spec_file = Path(path).resolve()
    spec = load_sample_spec(spec_file)
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
        optional_redirects=spec.optional_redirects,
        expected_redirects=spec.expected_redirects,
        spec_path=spec_file,
        forced_includes=spec.hipify_forced_includes,
        expected=spec.expected,
        optimization=spec.optimization,
        cudart_version=spec.cudart_version,
    )


def hipify_report_errors(output: str) -> tuple[str, ...]:
    """Return the failure markers hipify-clang printed despite its exit status."""
    return tuple(marker for marker in HIPIFY_ERROR_MARKERS if marker in output)


def build_hipify_command(
    *,
    hipify: str,
    source: Path,
    output: Path,
    cuda: CudaRoot,
    include_dirs: tuple[Path, ...],
    forced_includes: tuple[Path, ...] = (),
    cudart_version: int | None = None,
) -> list[str]:
    """Build the hipify-clang command line.

    Kept separate from the run so the flags that are easy to get wrong are
    covered by unit tests instead of by a GPU:

    - ``--cuda-path`` must be absolute or hipify-clang cannot find CUDA;
    - ``--skip-excluded-preprocessor-conditional-blocks`` keeps Windows-only
      code out of the parse;
    - ``--local-headers`` is what makes a quoted local header (``*.cuh``)
      translate at all;
    - ``CUDART_VERSION`` must be defined when a sample branches on it, because
      HIP does not define it and an undefined identifier evaluates to 0, which
      turns a skipped CUDA block into live code.
    """
    command = [
        hipify,
        str(source),
        f"--cuda-path={cuda.path}",
        "--skip-excluded-preprocessor-conditional-blocks",
        "--local-headers",
        "--no-backup",
        "-o",
        str(output),
    ]
    for directory in include_dirs:
        command += ["-I", str(directory)]

    clang_args: list[str] = []
    if cudart_version is not None:
        clang_args.append(f"-DCUDART_VERSION={cudart_version}")
    for forced in forced_includes:
        # hipify-clang forwards everything after "--" to clang.
        clang_args += ["-include", str(forced)]
    if clang_args:
        command += ["--", *clang_args]
    return command


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
    optional_redirects: tuple[str, ...] = (),
    expected_redirects: int | None = None,
    spec_path: Path | None = None,
    forced_includes: tuple[Path, ...] = (),
    expected: str | None,
    optimization: str,
    cudart_version: int | None = None,
) -> RunResult:
    hipify = require_tool("hipify-clang")
    hipcc = require_tool("hipcc")
    try:
        detected_arch = detect_arch(arch)
    except ToolError as error:
        raise ProbeError("environment", str(error)) from error

    artifacts = (out_dir or Path(tempfile.mkdtemp(prefix=f"{name}-"))).resolve()
    try:
        cuda = resolve_cuda_root(artifacts)
    except CudaRootError as error:
        raise ProbeError("environment", str(error)) from error
    (artifacts / "cuda-root.json").write_text(
        json.dumps(cuda.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    logger.info("[CUDA-ROOT] %s (%s)", cuda.path, cuda.detection)
    logger.info("[CUDA-ROOT] version=%s synthetic=%s", cuda.version_string, cuda.synthetic)

    source_dir = artifacts / "source"
    hipified_dir = artifacts / "hipified"
    adapted_dir = artifacts / "adapted"
    for directory in (source_dir, hipified_dir, adapted_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source = source_dir / f"{name}.cu"
    source.write_text(source_text, encoding="utf-8")

    skipped = check_redirects(
        name,
        unmatched_redirects(source_text, include_redirects),
        optional_redirects,
        spec_path=spec_path,
    )
    for header in skipped:
        logger.info(
            "[REDIRECT-SKIP] %s is declared optional and no matching include "
            "was found",
            header,
        )

    preprocessed_text, redirect_changes = apply_include_redirects(
        source_text, include_redirects
    )
    check_redirect_expectation(
        expected_redirects, len(redirect_changes), spec_path=spec_path
    )
    if redirect_changes:
        preprocessed_dir = artifacts / "preprocessed"
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        hipify_input = preprocessed_dir / f"{name}.cu"
        hipify_input.write_text(preprocessed_text, encoding="utf-8")
        for change in redirect_changes:
            logger.info(
                "[REDIRECT] %s line=%d %s -> %s",
                change.rule_id,
                change.line,
                change.before,
                change.after,
            )
    else:
        hipify_input = source
    (artifacts / "redirects.json").write_text(
        json.dumps([change.as_dict() for change in redirect_changes], indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("[REDIRECTS] applied=%d", len(redirect_changes))

    hipified = hipified_dir / f"{name}.hip.cu"
    adapted = adapted_dir / f"{name}.hip.cu"
    binary = artifacts / name

    logger.info("probe=%s arch=%s out=%s", name, detected_arch, artifacts)

    include_flags = [f"-I{directory}" for directory in include_dirs]
    if include_flags:
        logger.info("[INCLUDE] %s", " ".join(include_flags))

    hipify_command = build_hipify_command(
        hipify=hipify,
        source=hipify_input,
        output=hipified,
        cuda=cuda,
        include_dirs=include_dirs,
        forced_includes=forced_includes,
        cudart_version=cudart_version,
    )
    if forced_includes:
        logger.info("[HIPIFY-PARSE-INCLUDE] forced includes=%d", len(forced_includes))

    hipify_log = artifacts / "hipify.log"
    result = run_logged(hipify_command, hipify_log)
    hipify_output = log_tail(hipify_log, lines=200)
    if result.returncode != 0:
        raise ProbeError(
            "hipify",
            f"HIPIFY failed with exit code {result.returncode}",
            tail=hipify_output,
            artifacts=artifacts,
        )
    # hipify-clang can exit 0 while reporting a failure in its output.
    reported = hipify_report_errors(hipify_output)
    if reported:
        raise ProbeError(
            "hipify",
            f"HIPIFY reported an error despite exit code 0 ({', '.join(reported)})",
            tail=hipify_output,
            artifacts=artifacts,
        )
    if not hipified.is_file() or hipified.stat().st_size == 0:
        raise ProbeError(
            "hipify",
            "HIPIFY produced no output",
            tail=hipify_output,
            artifacts=artifacts,
        )

    # hipify-clang writes translated local headers next to their input with a
    # ".hip" suffix; move them into the hipified tree so quoted includes resolve.
    for produced in sorted(hipify_input.parent.glob(f"*{HIP_TAIL}")):
        target = hipified_dir / produced.name[: -len(HIP_TAIL)]
        shutil.move(str(produced), str(target))
        logger.info("[HIPIFY-LOCAL] %s", target.name)

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

    define_flags = (
        [f"-DCUDART_VERSION={cudart_version}"] if cudart_version is not None else []
    )
    build_log = artifacts / "build.log"
    result = run_logged(
        [
            hipcc,
            "-x",
            "hip",
            "-std=c++17",
            optimization,
            f"--offload-arch={detected_arch}",
            *define_flags,
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
