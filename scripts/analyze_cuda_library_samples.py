#!/usr/bin/env python3
"""Measure API usage in CUDALibrarySamples and combine it with cuda-samples v13.5.

CUDALibrarySamples is an independent repository. Its sample names do not map 1:1
to the 26 library samples removed from cuda-samples v13.5, so this script
reports API-family coverage rather than pretending the names are equivalent.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from analyze_cuda_samples_versions import (
    SOURCE_SUFFIXES,
    api_family as base_api_family,
    is_api_call,
    strip_cpp,
)

CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
SKIP_PARTS = {".git", "build", "dist", "third_party", "__pycache__"}

# The C APIs use camelCase after the library prefix. Requiring an uppercase
# first character excludes sample helpers such as cufft_check and
# cusolver_finalize. Keep the few real snake_case APIs as explicit exceptions.
C_API_PREFIXES = (
    "nvMatmulHeuristics",
    "cusolverdx",
    "cusolverMp",
    "cusolverMg",
    "cusolver",
    "cutensorMp",
    "cutensorMg",
    "cutensor",
    "cusparseLt",
    "cusparse",
    "cublasdx",
    "cublasLt",
    "cublasMp",
    "cublas",
    "cufftdx",
    "cufftMp",
    "cufft",
    "nvcompdx",
    "nvcomp",
    "nvjpeg2k",
    "nvjpeg",
    "nvtiff",
    "nvrtc",
    "cudss",
    "cuest",
)
SAMPLE_LOCAL_NAMES = {"cudss_finalize", "cuestCheck"}
SNAKE_CASE_APIS = {"nvtiff_status_string"}


def is_library_api_call(name: str) -> bool:
    if name in SAMPLE_LOCAL_NAMES or name.endswith("_t"):
        return False
    if name in SNAKE_CASE_APIS:
        return True
    for prefix in C_API_PREFIXES:
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            return bool(suffix) and suffix[0].isupper()
    return is_api_call(name)


def family_for(name: str) -> str:
    lower = name.lower()
    special_prefixes = (
        ("cublasmp", "cuBLASMp"),
        ("cublaslt", "cuBLASLt"),
        ("cublasdx", "cuBLASDx"),
        ("cufftmp", "cuFFTMp"),
        ("cufftdx", "cuFFTDx"),
        ("cusolvermp", "cuSOLVERMp"),
        ("cusolvermg", "cuSOLVERMg"),
        ("cusolverdx", "cuSOLVERDx"),
        ("cusparselt", "cuSPARSELt"),
        ("cutensormp", "cuTENSORMp"),
        ("cutensormg", "cuTENSORMg"),
        ("cutensor", "cuTENSOR"),
        ("cudss", "cuDSS"),
        ("cuest", "cuEST"),
        ("nvcompdx", "nvCOMPDx"),
        ("nvcomp", "nvCOMP"),
        ("nvtiff", "nvTIFF"),
        ("nvmatmulheuristics", "nvMatmulHeuristics"),
    )
    for prefix, family in special_prefixes:
        if prefix in lower:
            return family
    if "cublas" in lower:
        return "cuBLAS"
    if "cufft" in lower:
        return "cuFFT"
    if "cusolver" in lower:
        return "cuSOLVER"
    if "cusparse" in lower:
        return "cuSPARSE"
    if "curand" in lower or "mrg32k3a" in lower or "philox" in lower:
        return "cuRAND"
    if "nvjpeg2000" in lower or "nvjpeg2k" in lower:
        return "nvJPEG2000"
    if "nvjpeg" in lower:
        return "nvJPEG"
    if "npp" in lower:
        return "NPP"
    return base_api_family(name)


def source_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        files.append(path)
    return files


def nearest_cmake_dir(path: Path, root: Path) -> Path | None:
    current = path.parent
    while current != root.parent:
        if (current / "CMakeLists.txt").is_file():
            return current
        if current == root:
            break
        current = current.parent
    return None


def sample_label(path: Path, root: Path) -> str:
    result = nearest_cmake_dir(path, root)
    if result is None:
        return str(path.parent.relative_to(root))
    return str(result.relative_to(root))


def collect_library(root: Path) -> dict[str, Any]:
    api_by_sample: dict[str, set[str]] = defaultdict(set)
    occurrences = Counter()
    files = source_files(root)
    sample_dirs: set[str] = set()
    source_count = 0
    for path in files:
        sample = sample_label(path, root)
        sample_dirs.add(sample)
        source_count += 1
        text = strip_cpp(path.read_text(encoding="utf-8", errors="ignore"))
        for match in CALL_RE.finditer(text):
            name = match.group(1)
            if not is_library_api_call(name):
                continue
            api_by_sample[sample].add(name)
            occurrences[name] += 1

    python_dirs = {
        str(path.parent.relative_to(root))
        for path in root.rglob("*.py")
        if not any(part in SKIP_PARTS for part in path.parts)
    }
    all_apis = set(occurrences)
    family_occurrences = Counter()
    family_unique: dict[str, set[str]] = defaultdict(set)
    family_samples: dict[str, set[str]] = defaultdict(set)
    sample_usage = Counter()
    for api, count in occurrences.items():
        family_occurrences[family_for(api)] += count
    for sample, apis in api_by_sample.items():
        sample_usage.update(apis)
        for api in apis:
            family = family_for(api)
            family_unique[family].add(api)
            family_samples[family].add(sample)

    return {
        "repository": "https://github.com/NVIDIA/CUDALibrarySamples",
        "commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "sample_count": len(sample_dirs),
        "python_sample_count": len(python_dirs),
        "source_file_count": source_count,
        "api_unique": len(all_apis),
        "api_occurrences": sum(occurrences.values()),
        "api_set": sorted(all_apis),
        "api_by_sample": {sample: sorted(apis) for sample, apis in sorted(api_by_sample.items())},
        "sample_names": sorted(sample_dirs),
        "python_dirs": sorted(python_dirs),
        "families": {
            family: {
                "unique_apis": len(apis),
                "occurrences": family_occurrences[family],
                "samples": len(family_samples[family]),
            }
            for family, apis in sorted(family_unique.items())
        },
        "top_apis": [
            {
                "api": api,
                "family": family_for(api),
                "samples": sample_usage[api],
                "occurrences": occurrences[api],
            }
            for api in sorted(all_apis, key=lambda name: (-sample_usage[name], -occurrences[name], name))[:40]
        ],
    }


def removed_samples(v129: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    names = comparison["sample_removed"]
    api_by_sample = v129["api_by_sample"]
    removed_apis = set()
    groups: dict[str, list[str]] = defaultdict(list)
    for name in names:
        if "CUBLAS" in name or "matrixMulCUBLAS" in name:
            group = "cuBLAS"
        elif "CUFFT" in name:
            group = "cuFFT"
        elif "Solver" in name:
            group = "cuSOLVER"
        elif "conjugateGradient" in name:
            group = "cuSPARSE"
        elif "NPP" in name:
            group = "NPP"
        elif "nvJPEG" in name:
            group = "nvJPEG"
        elif "Mersenne" in name:
            group = "cuRAND"
        else:
            group = "other"
        groups[group].append(name)
    library_names = [
        name for group, group_names in groups.items() if group != "other" for name in group_names
    ]
    for name in library_names:
        removed_apis.update(api_by_sample.get(name, []))
    return {
        "count": len(library_names),
        "all_removed_count": len(names),
        "samples": library_names,
        "other_samples": groups.get("other", []),
        "groups": {key: sorted(value) for key, value in sorted(groups.items())},
        "api_set": sorted(removed_apis),
    }


def combined(
    v135: dict[str, Any], library: dict[str, Any], removed: dict[str, Any]
) -> dict[str, Any]:
    v135_apis = set(v135["api_set"])
    lib_apis = set(library["api_set"])
    removed_apis = set(removed["api_set"])
    union = v135_apis | lib_apis
    family_occurrences = Counter()
    family_unique: dict[str, set[str]] = defaultdict(set)
    family_samples = Counter()
    for api in union:
        family = family_for(api)
        family_unique[family].add(api)
    for source in (v135, library):
        for family, data in source["families"].items():
            family_occurrences[family] += data["occurrences"]
            family_samples[family] += data["samples"]
    return {
        "cuda_samples_v13_5": {
            "sample_count": v135["sample_count"],
            "python_sample_count": v135["python_sample_count"],
            "api_unique": v135["api_unique"],
            "api_occurrences": v135["api_occurrences"],
        },
        "cuda_library_samples": {
            "sample_count": library["sample_count"],
            "python_sample_count": library["python_sample_count"],
            "api_unique": library["api_unique"],
            "api_occurrences": library["api_occurrences"],
        },
        "combined": {
            "sample_count": v135["sample_count"] + library["sample_count"],
            "python_sample_count": v135["python_sample_count"] + library["python_sample_count"],
            "api_unique": len(union),
            "api_occurrences": v135["api_occurrences"] + library["api_occurrences"],
            "families": {
                family: {
                    "unique_apis": len(apis),
                    "occurrences": family_occurrences[family],
                    "samples": family_samples[family],
                }
                for family, apis in sorted(family_unique.items())
            },
        },
        "removed_v129_to_v135": {
            "sample_count": removed["count"],
            "groups": removed["groups"],
            "api_unique": len(removed_apis),
            "api_retained_in_library": sorted(removed_apis & lib_apis),
            "api_not_seen_in_library": sorted(removed_apis - lib_apis),
            "library_only_apis": sorted(lib_apis - v135_apis),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--v129-json", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    v129 = json.loads(args.v129_json.read_text(encoding="utf-8"))
    comparison = next(
        item
        for item in v129["comparisons"]
        if item["from"] == "v12.9" and item["to"] == "v13.5"
    )
    library = collect_library(args.repo)
    removed = removed_samples(v129["versions"]["v12.9"], comparison)
    result = {
        "cuda_library_samples": library,
        "removed_v129_to_v135": removed,
        "combined": combined(v129["versions"]["v13.5"], library, removed),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    combined_data = result["combined"]
    print(
        "CUDALibrarySamples:",
        f"samples={library['sample_count']}",
        f"python={library['python_sample_count']}",
        f"api_unique={library['api_unique']}",
        f"api_occurrences={library['api_occurrences']}",
    )
    print(
        "Combined:",
        f"samples={combined_data['combined']['sample_count']}",
        f"python={combined_data['combined']['python_sample_count']}",
        f"api_unique={combined_data['combined']['api_unique']}",
        f"api_occurrences={combined_data['combined']['api_occurrences']}",
    )
    removed_result = result["combined"]["removed_v129_to_v135"]
    print(
        "Library samples:",
        removed["count"],
        "all removed:",
        removed["all_removed_count"],
        "removed APIs:",
        len(removed["api_set"]),
        "retained:",
        len(removed_result["api_retained_in_library"]),
        "not seen:",
        len(removed_result["api_not_seen_in_library"]),
    )
    print(f"JSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
