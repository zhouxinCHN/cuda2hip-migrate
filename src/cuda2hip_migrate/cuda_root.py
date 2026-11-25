"""Resolve a CUDA installation root that ``hipify-clang`` can parse against.

``hipify-clang`` translates CUDA sources with clang, which needs a CUDA
installation in NVIDIA's layout::

    <root>/include/cuda_runtime.h
    <root>/nvvm/libdevice/libdevice.10.bc
    <root>/bin/                         (toolchain binaries)

The toolkit packaged by Debian/Ubuntu (``nvidia-cuda-toolkit``) installs headers
into ``/usr/include``, libdevice into ``/usr/lib/nvidia-cuda-toolkit/libdevice``
and binaries into ``/usr/lib/nvidia-cuda-toolkit/bin``, which does not match.

Three details are load-bearing and were measured on ROCm 7.2.4:

- a missing ``bin/`` makes hipify-clang log ``CUDA SDK version detected: 0`` and
  then fail on every CUDA identifier, while still exiting 0;
- a relative ``--cuda-path`` reproduces that same failure;
- the CUDA version that was actually parsed must be reported, because a v12.0
  sample parsed against newer headers is a real limitation, not a detail.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

LIBDEVICE = Path("nvvm/libdevice/libdevice.10.bc")
RUNTIME_HEADER = Path("include/cuda_runtime.h")
TOOLCHAIN_BIN = Path("bin")

#: Debian/Ubuntu toolkit locations.
SYSTEM_INCLUDE = Path("/usr/include")
SYSTEM_LIBDEVICE = Path("/usr/lib/nvidia-cuda-toolkit/libdevice")


class CudaRootError(RuntimeError):
    """No CUDA installation usable by hipify-clang was found."""


@dataclass(frozen=True)
class CudaRoot:
    path: Path
    version: int | None
    version_string: str
    synthetic: bool
    detection: str

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "cuda_version": self.version,
            "cuda_version_string": self.version_string,
            "synthetic": self.synthetic,
            "detection": self.detection,
        }


def _cuda_version(include_dir: Path) -> int | None:
    header = include_dir / "cuda.h"
    if not header.is_file():
        return None
    match = re.search(
        r"#define\s+CUDA_VERSION\s+(\d+)",
        header.read_text(encoding="utf-8", errors="replace"),
    )
    return int(match.group(1)) if match else None


def _format_version(version: int | None) -> str:
    """Format CUDA_VERSION (major * 1000 + minor * 10) as ``major.minor``."""
    if version is None:
        return "unknown"
    major, remainder = divmod(version, 1000)
    return f"{major}.{remainder // 10}"


def _is_complete(root: Path) -> bool:
    return (root / RUNTIME_HEADER).is_file() and (root / LIBDEVICE).is_file()


def _system_bin_dir() -> Path | None:
    """Return the directory holding the CUDA toolchain binaries."""
    toolkit_bin = SYSTEM_LIBDEVICE.parent / TOOLCHAIN_BIN
    if toolkit_bin.is_dir():
        return toolkit_bin
    nvcc = shutil.which("nvcc")
    return Path(nvcc).parent if nvcc else None


def _candidate_roots() -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    for name in ("CUDA_PATH", "CUDA_HOME"):
        value = os.environ.get(name)
        if value:
            candidates.append((f"${name}", Path(value)))
    candidates.append(("/usr/local/cuda", Path("/usr/local/cuda")))
    for entry in sorted(Path("/usr/local").glob("cuda-*")):
        candidates.append((str(entry), entry))
    return candidates


def resolve_cuda_root(work_dir: Path) -> CudaRoot:
    """Return a CUDA root suitable for ``hipify-clang --cuda-path``.

    ``work_dir`` receives the synthetic root when the installed toolkit does not
    use NVIDIA's layout. The synthetic root holds only symlinks and is rebuilt on
    every call, so it is reproducible and never touches the system.
    """
    # hipify-clang only recognises an absolute --cuda-path.
    work_dir = Path(work_dir).resolve()

    for label, candidate in _candidate_roots():
        if _is_complete(candidate) and (candidate / TOOLCHAIN_BIN).is_dir():
            version = _cuda_version(candidate / "include")
            return CudaRoot(
                path=candidate.resolve(),
                version=version,
                version_string=_format_version(version),
                synthetic=False,
                detection=f"standard layout at {label}",
            )

    if (SYSTEM_INCLUDE / "cuda_runtime.h").is_file() and (
        SYSTEM_LIBDEVICE / "libdevice.10.bc"
    ).is_file():
        version = _cuda_version(SYSTEM_INCLUDE)
        synthetic = work_dir / "cuda-root"
        links = [
            (synthetic / "include", SYSTEM_INCLUDE),
            (synthetic / LIBDEVICE, SYSTEM_LIBDEVICE / "libdevice.10.bc"),
        ]
        bin_dir = _system_bin_dir()
        if bin_dir is not None:
            links.append((synthetic / TOOLCHAIN_BIN, bin_dir))
        for link, target in links:
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target)
        return CudaRoot(
            path=synthetic,
            version=version,
            version_string=_format_version(version),
            synthetic=True,
            detection=(
                "Debian/Ubuntu layout: headers "
                f"{SYSTEM_INCLUDE}, libdevice {SYSTEM_LIBDEVICE}, "
                f"bin {bin_dir} (symlinked root)"
            ),
        )

    raise CudaRootError(
        "no CUDA installation usable by hipify-clang; set CUDA_PATH to a toolkit "
        f"containing {RUNTIME_HEADER}, {LIBDEVICE} and {TOOLCHAIN_BIN}"
    )
