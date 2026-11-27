"""Load and validate the pinned CUDA Samples v12.9 manifest."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

MANIFEST_RESOURCE = "cuda-samples-v12.9.json"
BATCH_ORDER = (
    "P0_baseline",
    "P1_semantics_and_business",
    "P1_probes",
    "P2_remaining_basic",
    "P3_semantic_and_library",
    "P4_special",
    "P5_high_risk_rewrite",
    "P6_deferred_platform",
)


class ManifestError(RuntimeError):
    """The manifest is missing or internally inconsistent."""


def load_manifest() -> dict[str, Any]:
    resource = files("rocm10_samples_compat.data").joinpath(MANIFEST_RESOURCE)
    data = json.loads(resource.read_text(encoding="utf-8"))
    validate_manifest(data)
    return data


def validate_manifest(data: dict[str, Any]) -> None:
    if data.get("schema_version") != 1:
        raise ManifestError("unsupported manifest schema; expected 1")

    samples = data.get("samples")
    batches = data.get("batches")
    if not isinstance(samples, list) or not samples:
        raise ManifestError("manifest must contain a non-empty samples list")
    if not isinstance(batches, dict):
        raise ManifestError("manifest must contain a batches object")

    names = [sample.get("name") for sample in samples]
    if any(not isinstance(name, str) or not name for name in names):
        raise ManifestError("every sample must have a non-empty name")
    if len(names) != len(set(names)):
        raise ManifestError("sample names must be unique")

    known = set(names)
    for batch, members in batches.items():
        if batch not in BATCH_ORDER:
            raise ManifestError(f"unknown batch: {batch}")
        if not isinstance(members, list) or not all(isinstance(item, str) for item in members):
            raise ManifestError(f"batch {batch} must be a list of names")
        missing = sorted(set(members) - known)
        if missing:
            raise ManifestError(f"batch {batch} references unknown samples: {missing}")

    fixed = data.get("fixed_decisions", {})
    if fixed.get("cuda_samples_tag") != "v12.9":
        raise ManifestError("manifest must pin cuda_samples_tag to v12.9")
    if fixed.get("target_rocm") != "10.0.0":
        raise ManifestError("manifest must pin target_rocm to 10.0.0")
    if fixed.get("cuda_toolkit") != "12.9":
        raise ManifestError("manifest must pin cuda_toolkit to 12.9")


def batch_members(data: dict[str, Any], batch: str) -> tuple[str, ...]:
    if batch not in data["batches"]:
        raise ManifestError(f"unknown batch: {batch}")
    return tuple(data["batches"][batch])


def summary(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "fixed_decisions": data["fixed_decisions"],
        "counts": data["counts"],
        "batches": {batch: len(members) for batch, members in data["batches"].items()},
    }
