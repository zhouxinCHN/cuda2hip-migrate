"""Per-sample build and acceptance specs.

A spec is the only place that knows how one sample is compiled, run and
accepted. Everything else in the pipeline is generic. Specs live in package data
so that a spec cannot drift away from the manifest it belongs to.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files

SPEC_PACKAGE = "rocm10_samples_compat.data.specs"
SPEC_SUFFIX = ".json"

C3_MODES = ("marker", "exit-status")

#: CUDA 12.9 preprocessor semantics.
#:
#: HIPIFY leaves skipped conditional blocks in its output verbatim. A block
#: guarded by ``#if CUDART_VERSION < 5000`` would therefore become *active*
#: under hipcc, because HIP does not define CUDART_VERSION and an undefined
#: identifier evaluates to 0. Defining the CUDA baseline value keeps every
#: conditional branch in the same state it has in the CUDA build.
ADAPTATION_DEFINES: tuple[str, ...] = ("CUDART_VERSION=12090",)


class SpecError(RuntimeError):
    """A sample spec is missing or invalid."""


@dataclass(frozen=True)
class SampleSpec:
    name: str
    path: str
    sources: tuple[str, ...]
    libraries: tuple[str, ...]
    run_args: tuple[str, ...]
    timeout_seconds: int
    c3_mode: str
    c3_marker: str | None
    notes: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "path": self.path,
            "sources": list(self.sources),
            "libraries": list(self.libraries),
            "run_args": list(self.run_args),
            "timeout_seconds": self.timeout_seconds,
            "c3_mode": self.c3_mode,
            "c3_marker": self.c3_marker,
            "notes": self.notes,
        }


def spec_names() -> tuple[str, ...]:
    root = files(SPEC_PACKAGE)
    return tuple(
        sorted(
            entry.name[: -len(SPEC_SUFFIX)]
            for entry in root.iterdir()
            if entry.is_file() and entry.name.endswith(SPEC_SUFFIX)
        )
    )


def load_spec(name: str) -> SampleSpec:
    available = spec_names()
    if name not in available:
        raise SpecError(
            f"no spec for sample {name!r}; available: {', '.join(available) or '<none>'}"
        )
    resource = files(SPEC_PACKAGE).joinpath(name + SPEC_SUFFIX)
    data = json.loads(resource.read_text(encoding="utf-8"))
    return _validate(name, data)


def _validate(name: str, data: dict[str, object]) -> SampleSpec:
    if data.get("schema_version") != 1:
        raise SpecError(f"{name}: unsupported spec schema")
    if data.get("name") != name:
        raise SpecError(f"{name}: spec name field does not match its file name")

    path = data.get("path")
    if not isinstance(path, str) or not path:
        raise SpecError(f"{name}: path must be a non-empty string")

    sources = data.get("sources")
    if not isinstance(sources, list) or not sources or not all(
        isinstance(item, str) and item for item in sources
    ):
        raise SpecError(f"{name}: sources must be a non-empty list of file names")

    libraries = data.get("libraries", [])
    if not isinstance(libraries, list) or not all(isinstance(item, str) for item in libraries):
        raise SpecError(f"{name}: libraries must be a list of strings")

    run_args = data.get("run", {}).get("args", [])
    if not isinstance(run_args, list) or not all(isinstance(item, str) for item in run_args):
        raise SpecError(f"{name}: run.args must be a list of strings")
    timeout = data.get("run", {}).get("timeout_seconds", 300)
    if not isinstance(timeout, int) or timeout <= 0:
        raise SpecError(f"{name}: run.timeout_seconds must be a positive integer")

    c3 = data.get("c3", {})
    if not isinstance(c3, dict):
        raise SpecError(f"{name}: c3 must be an object")
    mode = c3.get("mode")
    if mode not in C3_MODES:
        raise SpecError(f"{name}: c3.mode must be one of {C3_MODES}")
    marker = c3.get("marker")
    if mode == "marker" and (not isinstance(marker, str) or not marker):
        raise SpecError(f"{name}: c3.marker is required when mode is 'marker'")
    if mode == "exit-status" and marker is not None:
        raise SpecError(f"{name}: c3.marker must be absent when mode is 'exit-status'")

    notes = data.get("notes", "")
    if not isinstance(notes, str):
        raise SpecError(f"{name}: notes must be a string")

    return SampleSpec(
        name=name,
        path=path,
        sources=tuple(sources),
        libraries=tuple(libraries),
        run_args=tuple(run_args),
        timeout_seconds=timeout,
        c3_mode=mode,
        c3_marker=marker,
        notes=notes,
    )
