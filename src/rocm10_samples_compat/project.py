"""Project-level CUDA to HIP source migration.

This module is deliberately independent from the CUDA Samples manifest. A
project is described by an input tree and, optionally, a
``compile_commands.json`` file. The converter never builds or runs the result:
building is a separate concern and C1/C2/C3 evidence belongs to the sample
acceptance layer.

The output separates the source a user compiles from the audit trail::

    <output>/src/          converted source tree, relative layout preserved
    <output>/reports/      JSON report, change log, HIPIFY log
    <output>/.migration/   source/preprocessed/hipified stages
    <output>/rocm-migration.cmake

No input file is written to. A failed translation is recorded as blocking and
is not presented as a completed conversion.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__, environment
from .cuda_root import CudaRoot, resolve_cuda_root
from .hipify import HipifyError, hipify_source
from .redirects import (
    DEFAULT_REDIRECTS,
    Redirect,
    apply_redirects,
    matched_header,
)
from .rules import (
    RULE_IDS,
    RULE_REQUIREMENTS,
    RuleChange,
    apply_post_hipify_rules,
    rule_hits,
)
from .rules.cpp import find_directives

REPORT_SCHEMA = 1
RULES_PROFILE = "rocm10-cuda12.9"

TEXT_SUFFIXES = frozenset(
    {".cu", ".c", ".cpp", ".cxx", ".cc", ".cuh", ".h", ".hpp", ".hh", ".inl", ".ipp"}
)
TRANSLATABLE_SUFFIXES = frozenset({".cu", ".c", ".cpp", ".cxx", ".cc"})
HEADER_SUFFIXES = TEXT_SUFFIXES - TRANSLATABLE_SUFFIXES
CUDA_SOURCE_SUFFIXES = frozenset({".cu", ".cuh"})

#: Identifiers that make a header worth a standalone HIPIFY pass even when its
#: includes do not name a CUDA header directly. This is intentionally narrow:
#: translating every project header would change unrelated host code.
CUDA_HEADER_IDENTIFIERS = frozenset(
    {
        "__global__",
        "__device__",
        "__host__",
        "__constant__",
        "__managed__",
        "__shared__",
        "__syncthreads",
        "__ldg",
        "__shfl",
        "__ballot",
        "__activemask",
        "atomicAdd",
        "threadIdx",
        "blockIdx",
        "blockDim",
        "gridDim",
        "warpSize",
    }
)

DEFAULT_EXCLUDED_NAMES = frozenset({".git", ".hg", ".svn", "__pycache__"})
DEFAULT_EXCLUDED_PATTERNS = (
    "*.pyc",
    "*.o",
    "*.obj",
    "*.a",
    "*.so",
    "*.dylib",
    "*.dll",
    "*.exe",
)

#: NVIDIA compiler options that consume the next argument. They are recorded as
#: ignored rather than forwarded blindly to hipify-clang.
_FLAGS_WITH_VALUE = frozenset(
    {
        "-o",
        "-MF",
        "-MT",
        "-MQ",
        "-Xcompiler",
        "-Xptxas",
        "-Xlinker",
        "-Xcudafe",
        "-Xnvlink",
        "-ccbin",
        "--compiler-bindir",
        "-arch",
        "--gpu-architecture",
        "-gencode",
        "--generate-code",
        "--default-stream",
        "--threads",
        "-maxrregcount",
        "--maxrregcount",
        "-isysroot",
        "--sysroot",
        "-target",
        "--target",
    }
)

_INCLUDE_FLAGS = frozenset({"-I", "-isystem", "-iquote", "-idirafter"})
_INCLUDE_PREFIXES = ("-I", "-isystem", "-iquote", "-idirafter")

#: Prefixes used by residual-CUDA inspection. These produce review findings,
#: not automatic rewrites.
CUDA_API_PREFIXES = (
    "cuda",
    "cublas",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "cudnn",
    "nvrtc",
    "nvml",
)

_INCLUDE_RE = re.compile(
    r"^\s*#\s*include\s*(?:<(?P<angle>[^>]+)>|\"(?P<quoted>[^\"]+)\")"
)
_IDENTIFIER_START_RE = re.compile(r"[A-Za-z_]")


class ProjectError(RuntimeError):
    """An input or configuration error prevented migration from starting."""


class UnitConflict(ProjectError):
    """One source file is compiled in more than one incompatible way."""


@dataclass(frozen=True)
class IncludeArgument:
    flag: str
    path: Path

    def as_dict(self) -> dict[str, str]:
        return {"flag": self.flag, "path": str(self.path)}


@dataclass(frozen=True)
class TranslationUnit:
    """The parse-relevant view of one compilation database entry."""

    path: Path  # relative to the input root
    includes: tuple[IncludeArgument, ...] = ()
    defines: tuple[str, ...] = ()
    undefines: tuple[str, ...] = ()
    standard: str | None = None
    forced_includes: tuple[Path, ...] = ()
    ignored_arguments: tuple[str, ...] = ()
    targets: tuple[str, ...] = ()
    compiler: str | None = None
    translate: bool = True

    def signature(self) -> tuple[object, ...]:
        return (
            self.includes,
            self.defines,
            self.undefines,
            self.standard,
            self.forced_includes,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path.as_posix(),
            "includes": [item.as_dict() for item in self.includes],
            "defines": list(self.defines),
            "undefines": list(self.undefines),
            "standard": self.standard,
            "forced_includes": [str(item) for item in self.forced_includes],
            "ignored_arguments": list(self.ignored_arguments),
            "targets": list(self.targets),
            "compiler": self.compiler,
            "translate": self.translate,
        }


@dataclass(frozen=True)
class MigrationRequest:
    input_root: Path
    output_root: Path
    cuda_root: CudaRoot | None = None
    compile_commands: Path | None = None
    include_dirs: tuple[Path, ...] = ()
    defines: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()
    source_patterns: tuple[str, ...] = ()
    force: bool = False
    rules_profile: str = RULES_PROFILE


@dataclass(frozen=True)
class ProjectIssue:
    kind: str
    message: str
    path: str | None = None
    line: int | None = None
    symbol: str | None = None
    occurrences: int = 1
    severity: str = "warning"
    blocking: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "message": self.message,
            "path": self.path,
            "line": self.line,
            "symbol": self.symbol,
            "occurrences": self.occurrences,
            "severity": self.severity,
            "blocking": self.blocking,
        }


@dataclass(frozen=True)
class ConversionOutcome:
    status: str
    report: dict[str, object]
    report_path: Path
    output_root: Path

    @property
    def exit_code(self) -> int:
        return 2 if self.status == "blocked" else 0


@dataclass
class _RunState:
    issues: list[ProjectIssue] = field(default_factory=list)
    changes: list[tuple[str, RuleChange]] = field(default_factory=list)
    unit_records: list[dict[str, object]] = field(default_factory=list)
    header_records: list[dict[str, object]] = field(default_factory=list)
    redirects: tuple[Redirect, ...] = ()
    redirect_hits: list[dict[str, str]] = field(default_factory=list)
    features: set[str] = field(default_factory=set)
    logs: list[str] = field(default_factory=list)
    copied_files: list[str] = field(default_factory=list)
    translated_units: int = 0
    failed_units: int = 0
    translated_headers: int = 0
    failed_headers: int = 0
    adapter_defines: set[str] = field(default_factory=set)

    def add(self, issue: ProjectIssue) -> None:
        self.issues.append(issue)


def _within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _discover_files(
    root: Path, patterns: tuple[str, ...], *, exclude_names: frozenset[str]
) -> list[Path]:
    """Return regular files under ``root`` in stable relative-path order."""
    found: list[Path] = []
    for current, dirs, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        relative_dir = current_path.relative_to(root)
        kept_dirs: list[str] = []
        for name in sorted(dirs):
            candidate = current_path / name
            relative = relative_dir / name
            if name in exclude_names or _matches_any(relative, patterns, is_dir=True):
                continue
            if candidate.is_symlink():
                # Following directory symlinks invites cycles and escapes the
                # input tree. The copy stage records this as a finding.
                continue
            kept_dirs.append(name)
        dirs[:] = kept_dirs
        for name in sorted(files):
            candidate = current_path / name
            relative = relative_dir / name
            if _matches_any(relative, patterns, is_dir=False):
                continue
            if candidate.is_symlink() and not candidate.is_file():
                continue
            if candidate.is_file():
                found.append(relative)
    return found


def _matches_any(relative: Path, patterns: tuple[str, ...], *, is_dir: bool) -> bool:
    text = relative.as_posix()
    for pattern in patterns:
        if not pattern:
            continue
        normalized = pattern.rstrip("/")
        if "/" not in normalized:
            if relative.name == normalized or fnmatch.fnmatchcase(relative.name, normalized):
                return True
        if fnmatch.fnmatchcase(text, normalized):
            return True
        if is_dir and fnmatch.fnmatchcase(text + "/", pattern):
            return True
    return False


def _copy_project_tree(
    source: Path,
    destination: Path,
    *,
    excludes: tuple[str, ...],
    state: _RunState,
) -> None:
    patterns = tuple(excludes) + DEFAULT_EXCLUDED_PATTERNS
    files = _discover_files(source, patterns, exclude_names=DEFAULT_EXCLUDED_NAMES)

    for relative in files:
        src = source / relative
        dst = destination / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_symlink():
            resolved = src.resolve()
            if not _within(resolved, source):
                state.add(
                    ProjectIssue(
                        kind="symlink-escape",
                        path=relative.as_posix(),
                        message="symlink resolves outside the input tree; not copied",
                        severity="error",
                        blocking=True,
                    )
                )
                continue
            if not resolved.is_file():
                state.add(
                    ProjectIssue(
                        kind="unsupported-symlink",
                        path=relative.as_posix(),
                        message="file symlink does not resolve to a regular file",
                        severity="error",
                        blocking=True,
                    )
                )
                continue
        shutil.copy2(src, dst, follow_symlinks=True)
        state.copied_files.append(relative.as_posix())

    # Directory symlinks are skipped by _discover_files. Report each one so a
    # partial tree is never mistaken for a complete copy.
    for current, dirs, _files in os.walk(source, followlinks=False):
        current_path = Path(current)
        for name in dirs:
            candidate = current_path / name
            if not candidate.is_symlink():
                continue
            relative = candidate.relative_to(source).as_posix()
            state.add(
                ProjectIssue(
                    kind="directory-symlink",
                    path=relative,
                    message="directory symlinks are not followed or copied",
                    severity="error",
                    blocking=True,
                )
            )


def _prepare_output(request: MigrationRequest) -> tuple[Path, Path]:
    input_root = request.input_root.expanduser().resolve()
    output_root = request.output_root.expanduser().resolve()

    if not input_root.is_dir():
        raise ProjectError(f"input directory does not exist: {input_root}")
    if output_root in (Path("/"), Path.home().resolve()):
        raise ProjectError(f"refusing to use {output_root} as the output directory")
    if _within(output_root, input_root):
        raise ProjectError("output directory must not be inside the input directory")
    if _within(input_root, output_root):
        raise ProjectError("input directory must not be inside the output directory")

    if output_root.exists():
        if not output_root.is_dir():
            raise ProjectError(f"output path exists and is not a directory: {output_root}")
        if any(output_root.iterdir()):
            if not request.force:
                raise ProjectError(
                    f"output directory is not empty: {output_root}; use --force to replace it"
                )
            shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    return input_root, output_root


def _resolve_compile_path(value: str, directory: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = directory / path
    return path.resolve()


def _arg_value(arguments: list[str], index: int, flag: str) -> tuple[str, int]:
    if index + 1 >= len(arguments):
        raise ProjectError(f"compile argument {flag!r} has no value")
    return arguments[index + 1], index + 2


def _parse_compile_arguments(
    arguments: list[str],
    *,
    directory: Path,
    source_path: Path,
) -> tuple[
    tuple[IncludeArgument, ...],
    tuple[str, ...],
    tuple[str, ...],
    str | None,
    tuple[Path, ...],
    tuple[str, ...],
]:
    """Extract parse-relevant options from an NVCC command line.

    Optimisation, warning, code-generation and linker options do not affect the
    source-to-source parse. They are recorded in ``ignored`` instead of being
    forwarded, because hipify-clang rejects or misinterprets several NVCC-only
    options.
    """
    includes: list[IncludeArgument] = []
    defines: list[str] = []
    undefines: list[str] = []
    forced: list[Path] = []
    ignored: list[str] = []
    standard: str | None = None

    index = 0
    while index < len(arguments):
        arg = arguments[index]
        if not arg:
            index += 1
            continue

        matched_include = next(
            (flag for flag in _INCLUDE_PREFIXES if arg == flag or arg.startswith(flag + "=")),
            None,
        )
        if matched_include is not None:
            flag = matched_include
            if arg == flag:
                value, index = _arg_value(arguments, index, flag)
            elif arg.startswith(flag + "="):
                value = arg[len(flag) + 1 :]
                index += 1
            else:
                value = arg[len(flag) :]
                index += 1
            includes.append(
                IncludeArgument(flag=flag, path=_resolve_compile_path(value, directory))
            )
            continue

        # -I/path (no '='), the form emitted by CMake.
        include_prefix = next(
            (flag for flag in ("-I", "-isystem", "-iquote", "-idirafter") if arg.startswith(flag)),
            None,
        )
        if include_prefix is not None and len(arg) > len(include_prefix):
            value = arg[len(include_prefix) :]
            includes.append(
                IncludeArgument(
                    flag=include_prefix,
                    path=_resolve_compile_path(value, directory),
                )
            )
            index += 1
            continue

        if arg == "-D":
            value, index = _arg_value(arguments, index, arg)
            defines.append(value)
            continue
        if arg.startswith("-D") and len(arg) > 2:
            defines.append(arg[2:])
            index += 1
            continue
        if arg == "-U":
            value, index = _arg_value(arguments, index, arg)
            undefines.append(value)
            continue
        if arg.startswith("-U") and len(arg) > 2:
            undefines.append(arg[2:])
            index += 1
            continue

        if arg == "-std":
            standard, index = _arg_value(arguments, index, arg)
            continue
        if arg.startswith("-std="):
            standard = arg.split("=", 1)[1]
            index += 1
            continue

        if arg == "-include":
            value, index = _arg_value(arguments, index, arg)
            forced.append(_resolve_compile_path(value, directory))
            continue

        if arg in _FLAGS_WITH_VALUE:
            _value, index = _arg_value(arguments, index, arg)
            ignored.append(arg)
            continue

        if arg == source_path.name or arg == str(source_path):
            index += 1
            continue

        # Compiler, -c and -o are structural rather than semantic. Everything
        # else is recorded for review; forwarding unknown options is unsafe.
        if arg not in {"-c"}:
            ignored.append(arg)
        index += 1

    return (
        tuple(includes),
        tuple(defines),
        tuple(undefines),
        standard,
        tuple(forced),
        tuple(ignored),
    )


def _entry_arguments(entry: dict[str, object]) -> list[str]:
    arguments = entry.get("arguments")
    if isinstance(arguments, list) and all(isinstance(item, str) for item in arguments):
        return list(arguments)
    command = entry.get("command")
    if isinstance(command, str) and command.strip():
        return shlex.split(command)
    raise ProjectError(f"compile database entry has neither arguments nor command: {entry!r}")


def _normalise_units(
    candidates: list[TranslationUnit],
    *,
    input_root: Path,
) -> tuple[TranslationUnit, ...]:
    """Merge identical entries and reject one file with incompatible flags."""
    by_path: dict[Path, TranslationUnit] = {}
    targets: dict[Path, set[str]] = {}
    for unit in candidates:
        if unit.path in by_path:
            existing = by_path[unit.path]
            if existing.signature() != unit.signature():
                raise UnitConflict(
                    f"{unit.path.as_posix()} is compiled with incompatible options; "
                    "one converted file cannot represent both configurations"
                )
            targets[unit.path].update(unit.targets)
        else:
            by_path[unit.path] = unit
            targets[unit.path] = set(unit.targets)

    merged: list[TranslationUnit] = []
    for relative in sorted(by_path, key=lambda item: item.as_posix()):
        unit = by_path[relative]
        merged.append(
            TranslationUnit(
                path=relative,
                includes=unit.includes,
                defines=unit.defines,
                undefines=unit.undefines,
                standard=unit.standard,
                forced_includes=unit.forced_includes,
                ignored_arguments=unit.ignored_arguments,
                targets=tuple(sorted(targets[relative])),
                compiler=unit.compiler,
                translate=unit.translate,
            )
        )
    return tuple(merged)


def _units_from_compile_commands(
    database: Path,
    *,
    input_root: Path,
    excludes: tuple[str, ...],
    source_patterns: tuple[str, ...],
) -> tuple[TranslationUnit, ...]:
    if not database.is_file():
        raise ProjectError(f"compile database does not exist: {database}")
    try:
        data = json.loads(database.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProjectError(f"cannot read compile database {database}: {error}") from error
    if not isinstance(data, list):
        raise ProjectError("compile database must be a JSON array")

    patterns = tuple(excludes) + DEFAULT_EXCLUDED_PATTERNS
    candidates: list[TranslationUnit] = []
    for raw in data:
        if not isinstance(raw, dict):
            raise ProjectError("compile database contains a non-object entry")
        raw_file = raw.get("file")
        raw_directory = raw.get("directory")
        if not isinstance(raw_file, str) or not isinstance(raw_directory, str):
            raise ProjectError("compile database entry needs string 'file' and 'directory'")
        directory = Path(raw_directory).expanduser().resolve()
        source_path = _resolve_compile_path(raw_file, directory)
        if not _within(source_path, input_root):
            continue
        relative = source_path.relative_to(input_root)
        if _matches_any(relative, patterns, is_dir=False):
            continue
        if source_patterns and not any(
            fnmatch.fnmatchcase(relative.as_posix(), pattern) for pattern in source_patterns
        ):
            continue
        if source_path.suffix.lower() not in TRANSLATABLE_SUFFIXES:
            continue

        arguments = _entry_arguments(raw)
        compiler = Path(arguments[0]).name if arguments else None
        compiler_key = (compiler or "").lower()
        translate = (
            source_path.suffix.lower() in CUDA_SOURCE_SUFFIXES or "nvcc" in compiler_key
        )
        includes, defines, undefines, standard, forced, ignored = _parse_compile_arguments(
            arguments[1:] if arguments else [],
            directory=directory,
            source_path=source_path,
        )
        output = raw.get("output")
        targets = (str(output),) if isinstance(output, str) and output else ()
        candidates.append(
            TranslationUnit(
                path=relative,
                includes=includes,
                defines=defines,
                undefines=undefines,
                standard=standard,
                forced_includes=forced,
                ignored_arguments=ignored,
                targets=targets,
                compiler=compiler,
                translate=translate,
            )
        )

    units = _normalise_units(candidates, input_root=input_root)
    if not units:
        raise ProjectError(
            "compile database contains no source file inside the input tree"
        )
    return units


def _units_from_tree(
    input_root: Path,
    *,
    excludes: tuple[str, ...],
    source_patterns: tuple[str, ...],
    include_dirs: tuple[Path, ...],
    defines: tuple[str, ...],
) -> tuple[TranslationUnit, ...]:
    patterns = tuple(excludes) + DEFAULT_EXCLUDED_PATTERNS
    files = _discover_files(input_root, patterns, exclude_names=DEFAULT_EXCLUDED_NAMES)
    includes = tuple(
        IncludeArgument(flag="-I", path=path.expanduser().resolve())
        for path in include_dirs
    )
    units: list[TranslationUnit] = []
    for relative in files:
        if relative.suffix.lower() not in TRANSLATABLE_SUFFIXES:
            continue
        if source_patterns and not any(
            fnmatch.fnmatchcase(relative.as_posix(), pattern) for pattern in source_patterns
        ):
            continue
        units.append(
            TranslationUnit(
                path=relative,
                includes=includes,
                defines=defines,
            )
        )
    if not units:
        raise ProjectError("no translatable source files found in the input tree")
    return tuple(units)


def _contains_include(texts: list[str], header: str) -> bool:
    for text in texts:
        for _start, _end, directive in find_directives(text):
            if matched_header(directive) == header:
                return True
    return False


def _select_redirects(
    texts: list[str], *, overlay: Path, state: _RunState
) -> tuple[Redirect, ...]:
    active: list[Redirect] = []
    for redirect in DEFAULT_REDIRECTS:
        if not _contains_include(texts, redirect.header):
            continue
        replacement = overlay / redirect.replacement
        if not replacement.is_file():
            state.add(
                ProjectIssue(
                    kind="unsupported-helper",
                    path=redirect.header,
                    message=(
                        f"{redirect.header} is present but the compatibility overlay "
                        f"does not provide {redirect.replacement}"
                    ),
                    severity="error",
                    blocking=True,
                )
            )
            continue
        active.append(redirect)
    return tuple(active)


def _mapped_path(path: Path, *, input_root: Path, stage_root: Path) -> Path:
    if _within(path, input_root):
        return stage_root / path.relative_to(input_root)
    return path


def _unit_includes(unit: TranslationUnit, *, input_root: Path, stage_root: Path) -> list[str]:
    args: list[str] = []
    for include in unit.includes:
        mapped = _mapped_path(include.path, input_root=input_root, stage_root=stage_root)
        if include.flag == "-I":
            args.extend(["-I", str(mapped)])
        else:
            # hipify-clang exposes only -I as a first-class option. Other
            # Clang include-search flags must be forwarded to its Clang tooling
            # invocation one argument at a time.
            args.extend([f"--extra-arg={include.flag}", f"--extra-arg={mapped}"])
    return args


def _defined_names(defines: tuple[str, ...]) -> set[str]:
    return {define.split("=", 1)[0] for define in defines}


def _effective_defines(
    unit: TranslationUnit,
    *,
    request_defines: tuple[str, ...],
    cuda_root: CudaRoot,
) -> tuple[str, ...]:
    defines = list(unit.defines)
    for extra in request_defines:
        name = extra.split("=", 1)[0]
        defines = [item for item in defines if item.split("=", 1)[0] != name]
        defines.append(extra)
    if cuda_root.version is not None and "CUDART_VERSION" not in _defined_names(tuple(defines)):
        defines.append(f"CUDART_VERSION={cuda_root.version}")
    return tuple(defines)


def _hipify_arguments(
    unit: TranslationUnit,
    *,
    preprocessed: Path,
    overlay: Path,
    input_root: Path,
    cuda_root: CudaRoot,
    request_defines: tuple[str, ...],
    redirects: tuple[Redirect, ...],
    state: _RunState,
) -> tuple[list[str], tuple[str, ...]]:
    defines = _effective_defines(
        unit, request_defines=request_defines, cuda_root=cuda_root
    )
    if cuda_root.version is not None and "CUDART_VERSION" not in _defined_names(
        unit.defines + request_defines
    ):
        state.adapter_defines.add(f"CUDART_VERSION={cuda_root.version}")

    extra_args = ["--skip-excluded-preprocessor-conditional-blocks"]
    extra_args.extend(f"-D{define}" for define in defines)
    extra_args.extend(f"--extra-arg=-U{define}" for define in unit.undefines)
    if unit.standard:
        extra_args.append(f"--extra-arg=-std={unit.standard}")
    for forced in unit.forced_includes:
        mapped = _mapped_path(forced, input_root=input_root, stage_root=preprocessed)
        extra_args.extend(["--extra-arg=-include", f"--extra-arg={mapped}"])

    extra_args.extend(
        _unit_includes(unit, input_root=input_root, stage_root=preprocessed)
    )
    if redirects:
        extra_args.extend(["-I", str(overlay)])
    return extra_args, defines


def _restore_after_failure(source: Path, output: Path) -> None:
    """Restore the routed source when hipify-clang removed its output."""
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)


def _move_generated_headers(
    *,
    preprocessed: Path,
    hipified: Path,
    preexisting_hips: set[Path],
    translated_headers: dict[Path, bytes],
    state: _RunState,
    context: str,
) -> None:
    generated_files = set(preprocessed.rglob("*.hip")) - preexisting_hips
    for generated in sorted(generated_files):
        relative = generated.relative_to(preprocessed)
        target_relative = Path(relative.as_posix()[: -len(".hip")])
        content = generated.read_bytes()
        if relative in translated_headers:
            if translated_headers[relative] != content:
                state.add(
                    ProjectIssue(
                        kind="header-configuration-conflict",
                        path=target_relative.as_posix(),
                        message=(
                            "the same local header translated to different content "
                            f"for {context}"
                        ),
                        severity="error",
                        blocking=True,
                    )
                )
            generated.unlink(missing_ok=True)
            continue
        target = hipified / target_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        translated_headers[relative] = content
        generated.unlink(missing_ok=True)


def _header_needs_translation(text: str) -> bool:
    for _start, _end, directive in find_directives(text):
        match = _INCLUDE_RE.match(directive)
        if match is None:
            continue
        header = (match.group("angle") or match.group("quoted")).lower()
        if any(header.startswith(prefix) for prefix in CUDA_API_PREFIXES):
            return True
    for name, _offset in _iter_identifiers(text):
        if name in CUDA_HEADER_IDENTIFIERS or _is_cuda_identifier(name):
            return True
    return False


def _header_configs(
    header: Path,
    *,
    units: tuple[TranslationUnit, ...],
    input_root: Path,
) -> tuple[TranslationUnit, ...]:
    """Pick the translation-unit option sets that can see a header.

    A quoted header next to its source is normally translated by
    ``--local-headers``. That mode does not translate a header reached through
    ``-I``, so headers left unchanged get standalone passes. If several
    compatible option sets reach the header, each is tried and differing output
    is a blocking conflict.
    """
    candidates: list[TranslationUnit] = []
    for unit in units:
        if not unit.translate:
            continue
        source_parent = (input_root / unit.path).parent
        visible = source_parent == header.parent or any(
            _within(header, include.path)
            for include in unit.includes
            if include.path.is_dir()
        )
        if visible:
            candidates.append(unit)
    if not candidates:
        candidates = [unit for unit in units if unit.translate][:1]

    unique: list[TranslationUnit] = []
    seen: set[tuple[object, ...]] = set()
    for unit in candidates:
        signature = unit.signature()
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(unit)
    return tuple(unique)


def _hipify_headers(
    *,
    units: tuple[TranslationUnit, ...],
    preprocessed: Path,
    hipified: Path,
    overlay: Path,
    input_root: Path,
    cuda_root: CudaRoot,
    request_defines: tuple[str, ...],
    redirects: tuple[Redirect, ...],
    preexisting_hips: set[Path],
    translated_headers: dict[Path, bytes],
    state: _RunState,
) -> None:
    for source in sorted(preprocessed.rglob("*")):
        if not source.is_file() or source.suffix.lower() not in HEADER_SUFFIXES:
            continue
        relative = source.relative_to(preprocessed)
        output = hipified / relative
        if output.is_file() and output.read_bytes() != source.read_bytes():
            # --local-headers already produced a translated version for a
            # translation unit. Do not overwrite it with a less-informed pass.
            continue
        try:
            text = source.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if not _header_needs_translation(text):
            continue

        configs = _header_configs(
            input_root / relative, units=units, input_root=input_root
        )
        if not configs:
            continue
        first_content: bytes | None = None
        failures = 0
        successes = 0
        for unit in configs:
            extra_args, defines = _hipify_arguments(
                unit,
                preprocessed=preprocessed,
                overlay=overlay,
                input_root=input_root,
                cuda_root=cuda_root,
                request_defines=request_defines,
                redirects=redirects,
                state=state,
            )
            try:
                result = hipify_source(
                    source,
                    output,
                    cuda_root=cuda_root,
                    include_dirs=(),
                    local_headers=True,
                    extra_args=tuple(extra_args),
                )
            except HipifyError as error:
                failures += 1
                state.failed_headers += 1
                state.add(
                    ProjectIssue(
                        kind="hipify-header-failure",
                        path=relative.as_posix(),
                        message=str(error),
                        severity="error",
                        blocking=True,
                    )
                )
                if first_content is not None:
                    output.write_bytes(first_content)
                else:
                    _restore_after_failure(source, output)
                for generated in set(preprocessed.rglob("*.hip")) - preexisting_hips:
                    generated.unlink(missing_ok=True)
                state.logs.append(
                    f"=== header {relative.as_posix()} FAILED ===\n{error}\n"
                )
                continue

            successes += 1
            content = output.read_bytes()
            if first_content is None:
                first_content = content
            elif content != first_content:
                state.add(
                    ProjectIssue(
                        kind="header-configuration-conflict",
                        path=relative.as_posix(),
                        message=(
                            "the header translated to different content for "
                            "different compile options"
                        ),
                        severity="error",
                        blocking=True,
                    )
                )
                output.write_bytes(first_content)
            else:
                output.write_bytes(first_content)

            state.logs.append(
                f"=== header {relative.as_posix()} ===\n"
                f"$ {' '.join(result.result.command)}\n"
                f"{result.result.output}\n"
            )
            _move_generated_headers(
                preprocessed=preprocessed,
                hipified=hipified,
                preexisting_hips=preexisting_hips,
                translated_headers=translated_headers,
                state=state,
                context=f"header {relative.as_posix()}",
            )

        if successes:
            state.translated_headers += 1
        state.header_records.append(
            {
                "path": relative.as_posix(),
                "configurations": len(configs),
                "effective_defines": list(defines),
                "successes": successes,
                "failures": failures,
                "status": "translated" if successes and not failures else "hipify-failed",
            }
        )


def _hipify_units(
    *,
    units: tuple[TranslationUnit, ...],
    preprocessed: Path,
    hipified: Path,
    overlay: Path,
    input_root: Path,
    cuda_root: CudaRoot,
    request_defines: tuple[str, ...],
    redirects: tuple[Redirect, ...],
    state: _RunState,
) -> None:
    shutil.copytree(preprocessed, hipified)
    preexisting_hips = {
        path for path in preprocessed.rglob("*.hip") if path.is_file()
    }
    translated_headers: dict[Path, bytes] = {}

    for unit in units:
        if not unit.translate:
            continue
        source = preprocessed / unit.path
        output = hipified / unit.path
        if not source.is_file():
            state.failed_units += 1
            state.add(
                ProjectIssue(
                    kind="missing-source",
                    path=unit.path.as_posix(),
                    message="compile database names a source that is absent from the copied tree",
                    severity="error",
                    blocking=True,
                )
            )
            continue

        extra_args, defines = _hipify_arguments(
            unit,
            preprocessed=preprocessed,
            overlay=overlay,
            input_root=input_root,
            cuda_root=cuda_root,
            request_defines=request_defines,
            redirects=redirects,
            state=state,
        )

        try:
            result = hipify_source(
                source,
                output,
                cuda_root=cuda_root,
                include_dirs=(),
                local_headers=True,
                extra_args=tuple(extra_args),
            )
        except HipifyError as error:
            state.failed_units += 1
            state.add(
                ProjectIssue(
                    kind="hipify-failure",
                    path=unit.path.as_posix(),
                    message=str(error),
                    severity="error",
                    blocking=True,
                )
            )
            state.unit_records.append(
                {**unit.as_dict(), "status": "hipify-failed", "output": None}
            )
            _restore_after_failure(source, output)
            for generated in set(preprocessed.rglob("*.hip")) - preexisting_hips:
                generated.unlink(missing_ok=True)
            state.logs.append(
                f"=== {unit.path.as_posix()} FAILED ===\n{error}\n"
            )
            continue

        state.translated_units += 1
        state.logs.append(
            f"=== {unit.path.as_posix()} ===\n"
            f"$ {' '.join(result.result.command)}\n"
            f"{result.result.output}\n"
        )
        state.unit_records.append(
            {
                **unit.as_dict(),
                "effective_defines": list(defines),
                "status": "translated",
                "output": unit.path.as_posix(),
            }
        )
        _move_generated_headers(
            preprocessed=preprocessed,
            hipified=hipified,
            preexisting_hips=preexisting_hips,
            translated_headers=translated_headers,
            state=state,
            context=f"translation unit {unit.path.as_posix()}",
        )

    for unit in units:
        if unit.translate:
            continue
        state.unit_records.append({**unit.as_dict(), "status": "not-a-cuda-unit"})
    if not any(unit.translate for unit in units):
        raise ProjectError("compile database contains no CUDA translation unit")

    _hipify_headers(
        units=units,
        preprocessed=preprocessed,
        hipified=hipified,
        overlay=overlay,
        input_root=input_root,
        cuda_root=cuda_root,
        request_defines=request_defines,
        redirects=redirects,
        preexisting_hips=preexisting_hips,
        translated_headers=translated_headers,
        state=state,
    )


def _apply_rules(hipified: Path, adapted: Path, state: _RunState) -> None:
    shutil.copytree(hipified, adapted)
    for path in sorted(adapted.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            state.add(
                ProjectIssue(
                    kind="non-utf8-source",
                    path=path.relative_to(adapted).as_posix(),
                    message="file was copied but not scanned for deterministic rules",
                )
            )
            continue
        updated, changes = apply_post_hipify_rules(text)
        if changes:
            path.write_text(updated, encoding="utf-8")
            relative = path.relative_to(adapted).as_posix()
            state.changes.extend((relative, change) for change in changes)


def _iter_identifiers(text: str):
    """Yield identifiers outside comments and literals with their offsets."""
    index = 0
    length = len(text)
    while index < length:
        ch = text[index]
        following = text[index + 1] if index + 1 < length else ""
        if ch == "/" and following == "/":
            newline = text.find("\n", index + 2)
            index = length if newline < 0 else newline + 1
            continue
        if ch == "/" and following == "*":
            end = text.find("*/", index + 2)
            index = length if end < 0 else end + 2
            continue
        if ch in ('"', "'"):
            quote = ch
            index += 1
            while index < length:
                if text[index] == "\\":
                    index += 2
                    continue
                if text[index] == quote:
                    index += 1
                    break
                index += 1
            continue
        if not _IDENTIFIER_START_RE.match(ch):
            index += 1
            continue
        end = index + 1
        while end < length and (text[end].isalnum() or text[end] == "_"):
            end += 1
        yield text[index:end], index
        index = end


def _is_cuda_identifier(name: str) -> bool:
    for prefix in CUDA_API_PREFIXES:
        if not name.startswith(prefix) or name == prefix:
            continue
        remainder = name[len(prefix) :]
        if remainder[0].isupper() or remainder[0].isdigit() or remainder[0] == "_":
            return True
    return False


def _inspect_residuals(adapted: Path, state: _RunState) -> None:
    findings: dict[tuple[str, str, str], dict[str, object]] = {}
    for path in sorted(adapted.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(adapted).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for start, _end, directive in find_directives(text):
            match = _INCLUDE_RE.match(directive)
            if not match:
                continue
            header = match.group("angle") or match.group("quoted")
            lower = header.lower()
            if not any(lower.startswith(prefix) for prefix in CUDA_API_PREFIXES):
                continue
            line = text.count("\n", 0, start) + 1
            key = (relative, "residual-cuda-include", header)
            findings.setdefault(
                key,
                {
                    "line": line,
                    "message": f"CUDA-related include remains after HIPIFY: {header}",
                    "occurrences": 1,
                },
            )

        for name, offset in _iter_identifiers(text):
            kind = None
            if name in {"asm", "__asm__", "__asm"}:
                kind = "inline-assembly"
                message = "inline assembly remains and requires semantic review"
            elif _is_cuda_identifier(name):
                kind = "residual-cuda-identifier"
                message = f"CUDA identifier remains after HIPIFY: {name}"
            if kind is None:
                continue
            key = (relative, kind, name)
            if key in findings:
                findings[key]["occurrences"] = int(findings[key]["occurrences"]) + 1
                continue
            findings[key] = {
                "line": text.count("\n", 0, offset) + 1,
                "message": message,
                "occurrences": 1,
            }

    for (relative, kind, symbol), finding in sorted(findings.items()):
        state.add(
            ProjectIssue(
                kind=kind,
                path=relative,
                line=int(finding["line"]),
                symbol=symbol,
                occurrences=int(finding["occurrences"]),
                message=str(finding["message"]),
                severity="warning",
                blocking=False,
            )
        )


def _requirements(state: _RunState, overlay: Path) -> dict[str, object]:
    features = set(state.features)
    if state.redirects:
        features.add("common-helpers")
    for _relative, change in state.changes:
        requirement = RULE_REQUIREMENTS.get(change.rule_id)
        if requirement is not None:
            features.add(requirement.feature)

    includes: set[str] = set()
    libraries: set[str] = set()
    cmake_targets: set[str] = set()
    if features & {"common-helpers", "profiler-roctx"}:
        includes.add(str(overlay))
    for feature in sorted(features):
        for requirement in RULE_REQUIREMENTS.values():
            if requirement.feature != feature:
                continue
            libraries.update(requirement.libraries)
            cmake_targets.update(requirement.cmake_targets)

    return {
        "features": sorted(features),
        "include_dirs": sorted(includes),
        "compile_definitions": sorted(state.adapter_defines),
        "libraries": sorted(libraries),
        "cmake_targets": sorted(cmake_targets),
    }


def _cmake_quote(value: str) -> str:
    return '"' + value.replace("\\", "/").replace('"', '\\"') + '"'


def _write_cmake(path: Path, requirements: dict[str, object]) -> None:
    include_dirs = [str(item) for item in requirements["include_dirs"]]  # type: ignore[index]
    compile_definitions = [  # type: ignore[index]
        str(item) for item in requirements["compile_definitions"]
    ]
    libraries = [str(item) for item in requirements["libraries"]]  # type: ignore[index]
    features = [str(item) for item in requirements["features"]]  # type: ignore[index]

    lines = [
        "# Generated by rocm10-samples-compat. Do not edit.",
        "# Link rocm_cuda_migration_dependencies into targets built from src/.",
        "add_library(rocm_cuda_migration_dependencies INTERFACE)",
    ]
    if include_dirs:
        lines.append("target_include_directories(rocm_cuda_migration_dependencies INTERFACE")
        lines.extend(f"  {_cmake_quote(directory)}" for directory in include_dirs)
        lines.append(")")
    if compile_definitions:
        lines.append("target_compile_definitions(rocm_cuda_migration_dependencies INTERFACE")
        lines.extend(f"  {_cmake_quote(define)}" for define in compile_definitions)
        lines.append(")")
    if libraries:
        lines.append("target_link_libraries(rocm_cuda_migration_dependencies INTERFACE")
        lines.extend(f"  {_cmake_quote(library)}" for library in libraries)
        lines.append(")")

    if "profiler-roctx" in features:
        lines.extend(
            [
                "if(NOT TARGET ROCmCudaMigrate::profiler)",
                "  add_library(ROCmCudaMigrate::profiler INTERFACE IMPORTED)",
                "  set_target_properties(ROCmCudaMigrate::profiler PROPERTIES",
                '    INTERFACE_LINK_LIBRARIES "rocprofiler-sdk-roctx")',
                "endif()",
                "target_link_libraries(rocm_cuda_migration_dependencies",
                "  INTERFACE ROCmCudaMigrate::profiler)",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_hashes(input_root: Path, units: tuple[TranslationUnit, ...]) -> dict[str, str]:
    return {
        unit.path.as_posix(): _file_sha256(input_root / unit.path)
        for unit in units
        if (input_root / unit.path).is_file()
    }


def _write_reports(
    output_root: Path,
    report: dict[str, object],
    state: _RunState,
) -> Path:
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / "migration.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    changes_path = reports / "changes.jsonl"
    with changes_path.open("w", encoding="utf-8") as handle:
        for relative, change in state.changes:
            payload = {"file": relative, **change.as_dict()}
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    (reports / "hipify.log").write_text("\n".join(state.logs), encoding="utf-8")
    return report_path


def convert_project(request: MigrationRequest) -> ConversionOutcome:
    """Convert one project tree without compiling or running it."""
    if request.rules_profile != RULES_PROFILE:
        raise ProjectError(
            f"unsupported rules profile {request.rules_profile!r}; expected {RULES_PROFILE!r}"
        )

    input_root, output_root = _prepare_output(request)
    state = _RunState()
    work_root = output_root / ".migration"
    work_root.mkdir()
    source_stage = work_root / "source"
    preprocessed_stage = work_root / "preprocessed"
    hipified_stage = work_root / "hipified"
    adapted_stage = output_root / "src"

    cuda_root = request.cuda_root or resolve_cuda_root(work_root)
    compile_commands = (
        request.compile_commands.expanduser().resolve()
        if request.compile_commands is not None
        else None
    )

    if compile_commands is not None:
        units = _units_from_compile_commands(
            compile_commands,
            input_root=input_root,
            excludes=request.excludes,
            source_patterns=request.source_patterns,
        )
    else:
        units = _units_from_tree(
            input_root,
            excludes=request.excludes,
            source_patterns=request.source_patterns,
            include_dirs=request.include_dirs,
            defines=request.defines,
        )

    _copy_project_tree(
        input_root,
        source_stage,
        excludes=request.excludes,
        state=state,
    )
    shutil.copytree(source_stage, preprocessed_stage)

    texts: list[str] = []
    text_paths: list[Path] = []
    for relative in state.copied_files:
        path = preprocessed_stage / relative
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            texts.append(path.read_text(encoding="utf-8"))
            text_paths.append(path)
        except UnicodeDecodeError:
            state.add(
                ProjectIssue(
                    kind="non-utf8-source",
                    path=relative,
                    message="file was copied but excluded from pre-HIPIFY routing",
                )
            )

    overlay = Path(__file__).resolve().parent / "overlay"
    redirects = _select_redirects(texts, overlay=overlay, state=state)
    state.redirects = redirects
    for path, text in zip(text_paths, texts):
        updated, changes = apply_redirects(text, redirects)
        if changes:
            path.write_text(updated, encoding="utf-8")
            relative = path.relative_to(preprocessed_stage).as_posix()
            state.changes.extend((relative, change) for change in changes)
            for change in changes:
                state.redirect_hits.append(
                    {
                        "file": relative,
                        "header": change.before,
                        "replacement": change.after,
                    }
                )

    # A source outside the compile database is still copied but cannot safely be
    # translated with guessed options. Surface it instead of silently leaving
    # CUDA code in the output tree.
    if compile_commands is not None:
        selected = {unit.path for unit in units}
        for relative in state.copied_files:
            candidate = input_root / relative
            if (
                candidate.suffix.lower() in CUDA_SOURCE_SUFFIXES
                and Path(relative) not in selected
            ):
                state.add(
                    ProjectIssue(
                        kind="source-not-in-compile-database",
                        path=relative,
                        message=(
                            "CUDA source is not in the compile database and was not "
                            "translated"
                        ),
                        severity="warning",
                        blocking=False,
                    )
                )

    try:
        _hipify_units(
            units=units,
            preprocessed=preprocessed_stage,
            hipified=hipified_stage,
            overlay=overlay,
            input_root=input_root,
            cuda_root=cuda_root,
            request_defines=request.defines,
            redirects=redirects,
            state=state,
        )
    except HipifyError as error:
        state.add(
            ProjectIssue(
                kind="hipify-failure",
                message=str(error),
                severity="error",
                blocking=True,
            )
        )

    filter_stage = hipified_stage if hipified_stage.is_dir() else preprocessed_stage
    _apply_rules(filter_stage, adapted_stage, state)
    _inspect_residuals(adapted_stage, state)

    requirements = _requirements(state, overlay)
    _write_cmake(output_root / "rocm-migration.cmake", requirements)

    blocking = any(issue.blocking for issue in state.issues)
    post_rule_changes = [
        change for _relative, change in state.changes if change.rule_id in RULE_IDS
    ]
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "status": "blocked" if blocking else "completed",
        "tool": {"name": "rocm10-samples-compat", "version": __version__},
        "rules_profile": request.rules_profile,
        "rule_ids": list(RULE_IDS),
        "input": {
            "root": str(input_root),
            "files_copied": len(state.copied_files),
            "translation_unit_hashes": _input_hashes(input_root, units),
        },
        "output": {
            "root": str(output_root),
            "source": str(adapted_stage),
            "reports": str(output_root / "reports"),
        },
        "environment": environment.evidence_environment(),
        "cuda_root": cuda_root.as_dict(),
        "compile_commands": str(compile_commands) if compile_commands else None,
        "units": state.unit_records,
        "unit_count": len(units),
        "translated_units": state.translated_units,
        "failed_units": state.failed_units,
        "headers": state.header_records,
        "translated_headers": state.translated_headers,
        "failed_headers": state.failed_headers,
        "redirects": {
            "enabled": [
                {
                    "header": redirect.header,
                    "replacement": redirect.replacement,
                }
                for redirect in redirects
            ],
            "hits": state.redirect_hits,
        },
        "rule_hits": rule_hits(post_rule_changes),
        "changes_count": len(state.changes),
        "requirements": requirements,
        "issues": [issue.as_dict() for issue in state.issues],
        "blocking_issues": sum(1 for issue in state.issues if issue.blocking),
        "review_issues": sum(1 for issue in state.issues if not issue.blocking),
    }
    report_path = _write_reports(output_root, report, state)
    return ConversionOutcome(
        status=str(report["status"]),
        report=report,
        report_path=report_path,
        output_root=output_root,
    )
