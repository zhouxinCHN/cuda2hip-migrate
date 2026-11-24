"""Command line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from cuda2hip_migrate import __version__
from cuda2hip_migrate.process import ToolError
from cuda2hip_migrate.runner import ProbeError, probe_names, run_probe


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
    run_parser.add_argument("--probe", required=True, help="probe name without .cu")
    run_parser.add_argument("--out", type=Path, help="artifact directory")
    run_parser.add_argument("--arch", help="HIP offload architecture, e.g. gfx1100")
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
