#!/usr/bin/env python3
"""Scan CUDA API usage and sample-set changes across cuda-samples tags.

The scanner reads source directly from Git tags, strips comments and literals,
and extracts function-call names. It distinguishes API-set changes in samples
that exist in multiple versions from API changes caused only by sample removal.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

TAGS = ("v12.0", "v12.9", "v13.5")
CPP_ROOTS = ("Samples", "cpp")
SOURCE_SUFFIXES = {
    ".cu",
    ".cuh",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".hxx",
    ".inl",
}
BUILD_FILES = {"Makefile", "CMakeLists.txt"}
# Sample-local helpers sometimes use cuda/cu prefixes. These are not CUDA APIs.
SAMPLE_LOCAL_PREFIXES = (
    "cuda_",
    "cudaConsumer",
    "cudaProducer",
    "cudaDeviceCreate",
    "cudaGraphsManual",
    "cudaGraphsUsingStreamCapture",
    "cudaVk",
    "cudaNvSci",
    "cuHook",
    "cublasX",
)
SAMPLE_LOCAL_NAMES = {
    "cudaAcquireSync",
    "cudaBlockSize",
    "cudaCalculateSlopeKernel",
    "cudaCheckError",
    "cudaDeviceInit",
    "cudaGenerateSpectrumKernel",
    "cudaGetValueMismatch",
    "cudaGridSize",
    "cudaInit",
    "cudaProcess",
    "cudaReleaseSync",
    "cudaUpdateHeightmapKernel",
    "cudaUpdateVkImage",
    "cuEqual",
    "cuGet",
    "cuRand",
}
CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


def is_api_call(name: str) -> bool:
    if name in SAMPLE_LOCAL_NAMES or name.startswith(SAMPLE_LOCAL_PREFIXES):
        return False
    if name.startswith("cuda") and len(name) > 4:
        return True
    if name.startswith("cuvid") and len(name) > 5:
        return True
    if name.startswith("cu") and len(name) > 2 and name[2].isupper():
        return True
    return name.startswith(
        (
            "npp",
            "cublas",
            "cufft",
            "cusparse",
            "cusolver",
            "curand",
            "nvjpeg",
            "nvrtc",
            "nvml",
            "nvscibuf",
        )
    )


def api_family(name: str) -> str:
    if name.startswith("cuda"):
        return "CUDA Runtime"
    if name.startswith("cuvid"):
        return "CUVID"
    if name.startswith("cu") and len(name) > 2 and name[2].isupper():
        return "CUDA Driver"
    if name.startswith("npp"):
        return "NPP"
    if name.startswith("cublas"):
        return "cuBLAS"
    if name.startswith("cufft"):
        return "cuFFT"
    if name.startswith("cusparse"):
        return "cuSPARSE"
    if name.startswith("cusolver"):
        return "cuSOLVER"
    if name.startswith("curand"):
        return "cuRAND"
    if name.startswith("nvjpeg"):
        return "nvJPEG"
    if name.startswith("nvrtc"):
        return "NVRTC"
    if name.startswith("nvml"):
        return "NVML"
    if name.startswith("nvscibuf"):
        return "NvSciBuf"
    return "Other"


def strip_cpp(text: str) -> str:
    """Remove comments and string/char literals while preserving structure."""
    result: list[str] = []
    index = 0
    state = "code"
    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if state == "code":
            if char == "/" and following == "/":
                state = "line_comment"
                result.append("  ")
                index += 2
                continue
            if char == "/" and following == "*":
                state = "block_comment"
                result.append("  ")
                index += 2
                continue
            if char == '"':
                state = "string"
                result.append(" ")
                index += 1
                continue
            if char == "'":
                state = "char"
                result.append(" ")
                index += 1
                continue
            result.append(char)
            index += 1
            continue
        if state == "line_comment":
            if char == "\n":
                state = "code"
                result.append("\n")
            else:
                result.append(" ")
            index += 1
            continue
        if state == "block_comment":
            if char == "*" and following == "/":
                state = "code"
                result.append("  ")
                index += 2
            else:
                result.append("\n" if char == "\n" else " ")
                index += 1
            continue
        if state in ("string", "char"):
            if char == "\\":
                result.append("  ")
                index += 2
                continue
            if (state == "string" and char == '"') or (state == "char" and char == "'"):
                state = "code"
            result.append("\n" if char == "\n" else " ")
            index += 1
    return "".join(result)


def git_archive_files(repo: Path, tag: str) -> list[tuple[str, bytes]]:
    archive = subprocess.check_output(
        ["git", "-C", str(repo), "archive", "--format=tar", tag]
    )
    files: list[tuple[str, bytes]] = []
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            extracted = tar.extractfile(member)
            if extracted is not None:
                files.append((member.name, extracted.read()))
    return files


def collect_version(repo: Path, tag: str) -> dict:
    files = git_archive_files(repo, tag)
    sample_dirs: set[str] = set()
    sample_roots: dict[str, str] = {}
    python_dirs: set[str] = set()
    for path, _data in files:
        parts = path.split("/")
        if len(parts) >= 4 and parts[0] in CPP_ROOTS and parts[-1] in BUILD_FILES:
            root = "/".join(parts[:3])
            sample_dirs.add(root)
            sample_roots[root] = "/".join(parts[1:3])
        if len(parts) >= 4 and parts[0] == "python" and parts[-1].endswith(".py"):
            python_dirs.add("/".join(parts[1:3]))

    api_by_sample: dict[str, set[str]] = defaultdict(set)
    occurrences = Counter()
    source_file_count = 0
    for path, data in files:
        parts = path.split("/")
        if len(parts) < 4 or parts[0] not in CPP_ROOTS:
            continue
        sample_root = "/".join(parts[:3])
        if sample_root not in sample_dirs or Path(path).suffix.lower() not in SOURCE_SUFFIXES:
            continue
        sample = sample_roots[sample_root]
        source_file_count += 1
        text = strip_cpp(data.decode("utf-8", errors="ignore"))
        for match in CALL_RE.finditer(text):
            name = match.group(1)
            if not is_api_call(name):
                continue
            api_by_sample[sample].add(name)
            occurrences[name] += 1

    all_apis = set(occurrences)
    family_counts: Counter[str] = Counter()
    family_samples: dict[str, set[str]] = defaultdict(set)
    for sample, apis in api_by_sample.items():
        for api in apis:
            family_counts[api_family(api)] += 1
            family_samples[api_family(api)].add(sample)

    sample_usage = Counter()
    for apis in api_by_sample.values():
        sample_usage.update(apis)

    return {
        "tag": tag,
        "sample_count": len(sample_dirs),
        "python_sample_count": len(python_dirs),
        "source_file_count": source_file_count,
        "api_unique": len(all_apis),
        "api_occurrences": sum(occurrences.values()),
        "samples_with_api": sum(bool(apis) for apis in api_by_sample.values()),
        "samples_without_api": len(sample_dirs)
        - sum(bool(apis) for apis in api_by_sample.values()),
        "families": {
            family: {
                "unique_apis": len({api for api in all_apis if api_family(api) == family}),
                "occurrences": sum(
                    count for api, count in occurrences.items() if api_family(api) == family
                ),
                "samples": len(samples),
            }
            for family, samples in sorted(family_samples.items())
        },
        "top_apis": [
            {
                "api": api,
                "family": api_family(api),
                "samples": sample_usage[api],
                "occurrences": occurrences[api],
            }
            for api in sorted(
                all_apis, key=lambda name: (-sample_usage[name], -occurrences[name], name)
            )[:40]
        ],
        "api_set": sorted(all_apis),
        "api_by_sample": {
            sample: sorted(apis) for sample, apis in sorted(api_by_sample.items())
        },
        "sample_names": sorted(sample_roots.values()),
        "python_dirs": sorted(python_dirs),
    }


def compare(a: dict, b: dict) -> dict:
    a_apis = set(a["api_set"])
    b_apis = set(b["api_set"])
    common_samples = sorted(set(a["sample_names"]) & set(b["sample_names"]))
    api_by_sample_a = {k: set(v) for k, v in a["api_by_sample"].items()}
    api_by_sample_b = {k: set(v) for k, v in b["api_by_sample"].items()}
    changed = []
    aggregate_added: set[str] = set()
    aggregate_removed: set[str] = set()
    for sample in common_samples:
        before = api_by_sample_a.get(sample, set())
        after = api_by_sample_b.get(sample, set())
        if before != after:
            changed.append(
                {
                    "sample": sample,
                    "added": sorted(after - before),
                    "removed": sorted(before - after),
                }
            )
            aggregate_added.update(after - before)
            aggregate_removed.update(before - after)
    return {
        "from": a["tag"],
        "to": b["tag"],
        "api_common": len(a_apis & b_apis),
        "api_added": sorted(b_apis - a_apis),
        "api_removed": sorted(a_apis - b_apis),
        "api_jaccard": round(len(a_apis & b_apis) / len(a_apis | b_apis), 3)
        if a_apis | b_apis
        else 1.0,
        "sample_added": sorted(set(b["sample_names"]) - set(a["sample_names"])),
        "sample_removed": sorted(set(a["sample_names"]) - set(b["sample_names"])),
        "common_samples": len(common_samples),
        "same_api_set_samples": len(common_samples) - len(changed),
        "changed_api_set_samples": len(changed),
        "api_added_in_common_samples": sorted(aggregate_added),
        "api_removed_in_common_samples": sorted(aggregate_removed),
        "changed_samples": changed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "third-party" / "cuda-samples",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    versions = {tag: collect_version(args.repo, tag) for tag in TAGS}
    result = {
        "tags": list(TAGS),
        "versions": versions,
        "comparisons": [
            compare(versions["v12.0"], versions["v12.9"]),
            compare(versions["v12.9"], versions["v13.5"]),
            compare(versions["v12.0"], versions["v13.5"]),
        ],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for tag in TAGS:
        version = versions[tag]
        print(
            f"{tag}: samples={version['sample_count']} "
            f"python={version['python_sample_count']} "
            f"api_unique={version['api_unique']} "
            f"api_occurrences={version['api_occurrences']}"
        )
    for comparison in result["comparisons"]:
        print(
            f"{comparison['from']}->{comparison['to']}: "
            f"jaccard={comparison['api_jaccard']} "
            f"common_samples={comparison['common_samples']} "
            f"same_api_set={comparison['same_api_set_samples']} "
            f"changed_api_set={comparison['changed_api_set_samples']} "
            f"sample_added={len(comparison['sample_added'])} "
            f"sample_removed={len(comparison['sample_removed'])}"
        )
    print(f"JSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
