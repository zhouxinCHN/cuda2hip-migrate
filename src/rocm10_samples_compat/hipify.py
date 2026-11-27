"""Run ``hipify-clang`` on a preprocessed source file.

``hipify-clang`` reports some failures without a non-zero exit status, so a run
counts as successful only when the process succeeded, no HIPIFY error line was
emitted, and a non-empty output file exists. Anything else stops the pipeline at
the HIPIFY stage instead of producing an adapted file that hides the failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cuda_root import CudaRoot
from .process import ToolResult, require_tool, run_logged
from .workspace import temp_environment

#: Failure markers hipify-clang prints in place of a non-zero exit status.
#:
#: Deliberately narrow: a bare ``error:`` also appears in the benign
#: ``ptxas``/``nvlink`` architecture diagnostics that a successful translation
#: emits, so treating every ``error:`` line as fatal would fail good runs.
ERROR_MARKERS = ("[HIPIFY] error", "Error while processing")


class HipifyError(RuntimeError):
    """HIPIFY failed or produced no output."""

    def __init__(self, message: str, *, result: ToolResult | None = None):
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class HipifyResult:
    source: Path
    output: Path
    result: ToolResult


def hipify_source(
    source: Path,
    output: Path,
    *,
    cuda_root: CudaRoot,
    include_dirs: tuple[Path, ...],
    local_headers: bool = False,
    extra_args: tuple[str, ...] = (),
    temp_dir: Path | None = None,
) -> HipifyResult:
    """Translate one CUDA source file into its HIP equivalent.

    ``source`` must already be the redirected pre-HIPIFY copy; the pinned
    checkout is never passed to this function.  When ``temp_dir`` is supplied,
    both hipify-clang's ``--temp-dir`` and the child's ``TMPDIR`` (plus the
    portable ``TMP``/``TEMP`` spellings) point inside the caller's output root.
    """
    executable = require_tool("hipify-clang")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    environment = None
    command = [
        executable,
        str(source),
        f"--cuda-path={cuda_root.path}",
        "--no-backup",
        "-o",
        str(output),
    ]
    if temp_dir is not None:
        resolved_temp = Path(temp_dir).expanduser().resolve()
        command.append(f"--temp-dir={resolved_temp}")
        environment = temp_environment(resolved_temp)
    if local_headers:
        command.append("--local-headers")
    for directory in include_dirs:
        command.extend(["-I", str(directory)])
    command.extend(extra_args)

    result = run_logged(command, env=environment)
    text = result.output

    if result.returncode != 0:
        raise HipifyError(
            f"hipify-clang exited {result.returncode} for {source.name}\n{result.tail}",
            result=result,
        )
    if "[HIPIFY] error" in text or "Error while processing" in text:
        raise HipifyError(
            f"hipify-clang reported an error for {source.name}\n{result.tail}",
            result=result,
        )
    if not output.is_file() or output.stat().st_size == 0:
        raise HipifyError(
            f"hipify-clang produced no output for {source.name}\n{result.tail}",
            result=result,
        )
    return HipifyResult(source=source, output=output, result=result)
