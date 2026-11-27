"""End-to-end migration pipeline for one CUDA Samples v12.9 sample.

The pipeline is a fixed sequence of stages, each of which writes its own
directory so that the input, the HIPIFY output and the adapted source remain
separately auditable:

    source/        verbatim copy of the pinned sample directory
    preprocessed/  after pre-HIPIFY include routing
    hipified/      after hipify-clang
    adapted/       after deterministic post-HIPIFY rules

A stage that cannot complete stops the pipeline and records which acceptance
level was never reached. Nothing is retried, guessed or patched by hand.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path

from . import environment
from .checkout import Checkout, sample_directory
from .cuda_root import CudaRoot, resolve_cuda_root
from .hipify import HipifyError, hipify_source
from .process import run_logged, unified_diff
from .redirects import (
    DEFAULT_REDIRECTS,
    Redirect,
    apply_redirects,
    matched_header,
    unmatched_redirects,
)
from .rules import RULE_IDS, RuleChange, apply_post_hipify_rules, rule_hits
from .spec import ADAPTATION_DEFINES, SampleSpec

STAGES = ("source", "preprocessed", "hipified", "adapted")
TEXT_SUFFIXES = frozenset({".cu", ".cpp", ".c", ".cuh", ".h", ".hpp"})
HIPIFY_SUFFIXES = frozenset({".cu", ".cpp", ".c"})
BUILD_SUFFIXES = frozenset({".cu", ".cpp", ".c"})
HIP_TAIL = ".hip"


class StageError(RuntimeError):
    """A pipeline stage failed; the acceptance level it blocked is recorded."""

    def __init__(
        self,
        stage: str,
        level: str,
        message: str,
        *,
        tail: str = "",
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.level = level
        self.tail = tail


@dataclass(frozen=True)
class SampleRun:
    sample: str
    status: str
    evidence: dict[str, object]
    directory: Path

    @property
    def c1(self) -> bool:
        return bool(self.evidence["build"]["c1"])  # type: ignore[index]

    @property
    def c2(self) -> bool:
        return bool(self.evidence["run"]["c2"])  # type: ignore[index]

    @property
    def c3(self) -> bool:
        return bool(self.evidence["c3"]["passed"])  # type: ignore[index]


def overlay_directory() -> Path:
    """Return the packaged Common helper overlay directory."""
    return Path(str(files("rocm10_samples_compat").joinpath("overlay")))


def _stage_dir(root: Path, stage: str) -> Path:
    directory = root / stage
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    return directory


def _copy_sources(sample_dir: Path, destination: Path) -> list[str]:
    copied: list[str] = []
    for path in sorted(sample_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        shutil.copy2(path, destination / path.name)
        copied.append(path.name)
    if not copied:
        raise StageError("source", "C1", f"no compilable sources in {sample_dir}")
    return copied


def _preprocess(stage: Path, redirects: tuple[Redirect, ...]) -> list[RuleChange]:
    changes: list[RuleChange] = []
    texts: list[str] = []
    for path in sorted(stage.iterdir()):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        texts.append(text)
        updated, file_changes = apply_redirects(text, redirects)
        if file_changes:
            path.write_text(updated, encoding="utf-8")
            changes.extend(file_changes)

    missing = unmatched_redirects(texts, redirects)
    if missing:
        names = ", ".join(redirect.header for redirect in missing)
        raise StageError(
            "redirect",
            "C1",
            f"declared redirect(s) matched no include: {names}",
        )
    return changes


def _hipify(
    preprocessed: Path,
    hipified: Path,
    *,
    cuda_root: CudaRoot,
    include_dirs: tuple[Path, ...],
    defines: tuple[str, ...],
) -> None:
    # Start from the preprocessed tree so that files HIPIFY does not translate
    # (local headers it was not asked to process, data files) still reach the
    # build directory unchanged.
    for path in sorted(preprocessed.iterdir()):
        if path.is_file():
            shutil.copy2(path, hipified / path.name)

    for name in sorted(path.name for path in preprocessed.iterdir()):
        source = preprocessed / name
        if not source.is_file() or source.suffix.lower() not in HIPIFY_SUFFIXES:
            continue
        hipify_source(
            source,
            hipified / name,
            cuda_root=cuda_root,
            include_dirs=include_dirs,
            local_headers=True,
            extra_args=tuple(
                ["--skip-excluded-preprocessor-conditional-blocks"]
                + [f"-D{define}" for define in defines]
            ),
        )

    # hipify-clang writes translated local headers next to their input with a
    # ".hip" suffix; move them into place so quoted includes resolve.
    for produced in sorted(preprocessed.glob(f"*{HIP_TAIL}")):
        target = hipified / produced.name[: -len(HIP_TAIL)]
        shutil.move(str(produced), str(target))


def _apply_rules(hipified: Path, adapted: Path) -> list[RuleChange]:
    for path in sorted(hipified.iterdir()):
        if path.is_file():
            shutil.copy2(path, adapted / path.name)

    changes: list[RuleChange] = []
    for path in sorted(adapted.iterdir()):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        updated, file_changes = apply_post_hipify_rules(text)
        if file_changes:
            path.write_text(updated, encoding="utf-8")
            changes.extend(file_changes)
    return changes


def _build(
    adapted: Path,
    *,
    spec: SampleSpec,
    checkout: Checkout,
    arch: str,
    work: Path,
) -> tuple[list[str], Path, str]:
    sources = sorted(
        path.name
        for path in adapted.iterdir()
        if path.is_file() and path.suffix.lower() in BUILD_SUFFIXES
    )
    missing = [name for name in spec.sources if name not in sources]
    if missing:
        raise StageError("build", "C1", f"adapted tree is missing source(s): {missing}")

    binary = work / spec.name
    command = [
        "hipcc",
        "-O3",
        "-std=c++17",
        f"--offload-arch={arch}",
        *[f"-D{define}" for define in ADAPTATION_DEFINES],
        "-I",
        str(checkout.common),
        "-I",
        str(overlay_directory()),
        "-I",
        str(adapted),
        *sources,
        "-o",
        str(binary),
        *[f"-l{library}" for library in spec.libraries],
    ]
    result = run_logged(command, cwd=adapted)
    log = work / "build.log"
    log.write_text(
        "$ " + " ".join(command) + "\n\n" + result.stdout + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise StageError(
            "build", "C1", f"hipcc exited {result.returncode}", tail=result.tail
        )
    if not binary.is_file():
        raise StageError("build", "C1", "hipcc reported success but produced no binary")
    return command, binary, str(log)


def _run(
    binary: Path,
    *,
    spec: SampleSpec,
    work: Path,
) -> tuple[list[str], int, str, Path]:
    run_dir = work / "run"
    run_dir.mkdir(exist_ok=True)
    command = [str(binary), *spec.run_args]
    result = run_logged(command, cwd=run_dir, timeout=spec.timeout_seconds)
    log = work / "run.log"
    log.write_text(result.output + "\n", encoding="utf-8")
    return command, result.returncode, result.output, log


def _classify(changes: list[RuleChange], redirect_changes: list[RuleChange]) -> str:
    if changes:
        return "rule-fixed"
    if redirect_changes:
        return "rule-fixed"
    return "hipify-direct"


def _merge_changes(changes: list[RuleChange]) -> list[dict[str, object]]:
    return [change.as_dict() for change in changes]


def run_sample(
    spec: SampleSpec,
    checkout: Checkout,
    *,
    artifacts_root: Path,
    redirects: tuple[Redirect, ...] = DEFAULT_REDIRECTS,
    arch: str | None = None,
) -> SampleRun:
    """Run one sample through every stage and return its evidence."""
    work = (artifacts_root / spec.name).resolve()
    work.mkdir(parents=True, exist_ok=True)
    sample_dir = sample_directory(checkout, spec.path)

    cuda_root = resolve_cuda_root(work)
    reported_arch = arch or environment.primary_gfx_architecture()
    if not reported_arch:
        raise StageError("build", "C1", "rocminfo reported no gfx architecture")
    evidence: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample": spec.name,
        "sample_path": spec.path,
        "spec": spec.as_dict(),
        "cuda_samples": {"tag": checkout.tag, "commit": checkout.commit},
        "environment": environment.evidence_environment(),
        "cuda_root": cuda_root.as_dict(),
        "arch": reported_arch,
        "redirects": {
            "declared": [
                {
                    "header": redirect.header,
                    "replacement": redirect.replacement,
                    "optional": redirect.optional,
                }
                for redirect in redirects
            ],
            "hits": [],
        },
        "rules": {"registry": [], "hits": {}},
        "build": {"c1": False},
        "run": {"c2": False},
        "c3": {"mode": spec.c3_mode, "marker": spec.c3_marker, "passed": False},
        "status": "blocked",
        "blocked_at": None,
        "limitations": [],
    }
    evidence["limitations"] = _limitations(cuda_root)  # type: ignore[assignment]

    evidence["rules"]["registry"] = list(RULE_IDS)  # type: ignore[index]

    try:
        source_stage = _stage_dir(work, "source")
        _copy_sources(sample_dir, source_stage)

        preprocessed = _stage_dir(work, "preprocessed")
        for path in sorted(source_stage.iterdir()):
            shutil.copy2(path, preprocessed / path.name)
        redirect_changes = _preprocess(preprocessed, redirects)
        if redirect_changes:
            (work / "redirects.json").write_text(
                json.dumps(_merge_changes(redirect_changes), indent=2), encoding="utf-8"
            )
        matched = sorted(
            {
                header
                for header in (matched_header(change.before) for change in redirect_changes)
                if header is not None
            }
        )
        evidence["redirects"]["hits"] = [{"header": header} for header in matched]  # type: ignore[index]

        hipified = _stage_dir(work, "hipified")
        _hipify(
            preprocessed,
            hipified,
            cuda_root=cuda_root,
            # The ROCm include directory is deliberately absent: hipify-clang
            # defines __HIP_PLATFORM_NVIDIA__, so making HIP headers reachable
            # during translation selects HIP's NVIDIA shim and fails. The
            # overlay header is parse-safe without them (see its __HIPCC__
            # branch), and hipcc finds HIP headers on its own.
            #
            # The source directory is absent too: a quoted local header such as
            # scalarProd_kernel.cuh resolves relative to its source file, and
            # putting the source directory on the include path makes hipify-clang
            # parse the translation unit as a local header and fail.
            include_dirs=(checkout.common, overlay_directory()),
            defines=ADAPTATION_DEFINES,
        )

        for name in spec.sources:
            before = (preprocessed / name).read_text(encoding="utf-8")
            after = (hipified / name).read_text(encoding="utf-8")
            diff = unified_diff(before, after, f"preprocessed/{name}", f"hipified/{name}")
            if diff:
                with (work / "hipify.diff").open("a", encoding="utf-8") as handle:
                    handle.write(diff)

        adapted = _stage_dir(work, "adapted")
        rule_changes = _apply_rules(hipified, adapted)
        if rule_changes:
            (work / "rules.json").write_text(
                json.dumps(_merge_changes(rule_changes), indent=2), encoding="utf-8"
            )
        evidence["rules"]["hits"] = rule_hits(rule_changes)  # type: ignore[index]

        command, binary, build_log = _build(
            adapted, spec=spec, checkout=checkout, arch=reported_arch, work=work
        )
        evidence["build"] = {
            "c1": True,
            "command": command,
            "log": build_log,
        }

    except HipifyError as error:
        evidence["blocked_at"] = "hipify"
        evidence["hipify_error"] = str(error)
        evidence["status"] = "blocked"
        _write_evidence(work, evidence)
        return SampleRun(spec.name, "blocked", evidence, work)
    except StageError as error:
        evidence["blocked_at"] = error.level if error.stage == "build" else error.stage
        evidence["stage_error"] = {"stage": error.stage, "message": str(error), "tail": error.tail}
        if error.stage == "build":
            evidence["build"] = {
                "c1": False,
                "log": str(work / "build.log"),
                "tail": error.tail,
            }
        _write_evidence(work, evidence)
        return SampleRun(spec.name, "blocked", evidence, work)

    try:
        run_command, exit_code, output, run_log = _run(binary, spec=spec, work=work)
    except Exception as error:  # noqa: BLE001 - a crash must still produce evidence
        evidence["run"] = {"c2": False, "error": str(error), "log": str(work / "run.log")}
        evidence["blocked_at"] = "C2"
        _write_evidence(work, evidence)
        return SampleRun(spec.name, "blocked", evidence, work)

    c2 = exit_code == 0
    marker_present = (
        spec.c3_mode == "marker"
        and spec.c3_marker is not None
        and spec.c3_marker in output
    )
    c3 = c2 and (marker_present if spec.c3_mode == "marker" else True)
    evidence["run"] = {
        "c2": c2,
        "exit_code": exit_code,
        "command": run_command,
        "log": str(run_log),
    }
    evidence["c3"] = {
        "mode": spec.c3_mode,
        "marker": spec.c3_marker,
        "marker_present": marker_present,
        "passed": c3,
    }
    evidence["status"] = _classify(rule_changes, redirect_changes)
    if not c2:
        evidence["blocked_at"] = "C2"
        evidence["status"] = "blocked"
    elif not c3:
        evidence["blocked_at"] = "C3"
        evidence["status"] = "blocked"

    _write_evidence(work, evidence)
    return SampleRun(spec.name, str(evidence["status"]), evidence, work)


def _limitations(cuda_root: CudaRoot) -> list[str]:
    limitations: list[str] = []
    if cuda_root.version is not None and cuda_root.version != 12090:
        limitations.append(
            "hipify-clang parsed against CUDA "
            f"{cuda_root.version_string} headers, not the CUDA 12.9 baseline"
        )
    detected = environment.detected_rocm_version()
    if detected and not detected.startswith(environment.EXPECTED_ROCM):
        limitations.append(
            f"executed on ROCm {detected}, not the ROCm {environment.EXPECTED_ROCM} target"
        )
    return limitations


def _write_evidence(work: Path, evidence: dict[str, object]) -> None:
    (work / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8"
    )
