"""Command line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from cuda2hip_migrate import __version__
from cuda2hip_migrate.process import ToolError
from cuda2hip_migrate.runner import (
    DEFAULT_SUCCESS_MARKER,
    ProbeError,
    probe_names,
    run_probe,
    run_source_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cuda2hip-migrate",
        description="Deterministic HIPIFY post-processing and CUDA probe validation.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list packaged probes")
    list_parser.add_argument("--json", action="store_true", help="emit JSON")

    run_parser = subparsers.add_parser("run", help="translate, build and run one probe")
    source_group = run_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--probe", help="packaged probe name without .cu"
    )
    source_group.add_argument(
        "--source",
        type=Path,
        help="external CUDA source file, e.g. one cuda-samples file",
    )
    run_parser.add_argument("--out", type=Path, help="artifact directory")
    run_parser.add_argument("--arch", help="HIP offload architecture, e.g. gfx1100")
    run_parser.add_argument(
        "--include-dir",
        type=Path,
        action="append",
        default=[],
        dest="include_dirs",
        help="include directory passed to hipify-clang and hipcc (repeatable)",
    )
    run_parser.add_argument(
        "--library",
        action="append",
        default=[],
        dest="libraries",
        help="library to link with hipcc as -l<name> (repeatable)",
    )
    run_parser.add_argument(
        "--success-marker",
        default=DEFAULT_SUCCESS_MARKER,
        help=f"output marker required for C3 (default: {DEFAULT_SUCCESS_MARKER})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        names = probe_names()
        if args.json:
            print(json.dumps(names))
        else:
            for name in names:
                print(name)
        return 0

    if args.command == "run":
        try:
            if args.source is not None:
                result = run_source_file(
                    args.source,
                    out_dir=args.out,
                    arch=args.arch,
                    include_dirs=tuple(args.include_dirs),
                    libraries=tuple(args.libraries),
                    success_marker=args.success_marker,
                )
            else:
                result = run_probe(args.probe, out_dir=args.out, arch=args.arch)
        except (ProbeError, ToolError) as error:
            if isinstance(error, ProbeError):
                print(f"ERROR [{error.stage}]: {error}", file=sys.stderr)
                if error.tail:
                    print(error.tail, file=sys.stderr)
                if error.artifacts:
                    print(f"ARTIFACTS={error.artifacts}", file=sys.stderr)
            else:
                print(f"ERROR [environment]: {error}", file=sys.stderr)
            return 1
        print(f"OK probe={result.probe} rules={len(result.changes)} artifacts={result.out_dir}")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2
