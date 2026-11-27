"""Command line interface for the compatibility project."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rocm10_samples_compat import __version__
from rocm10_samples_compat.checkout import CheckoutError, verify_checkout
from rocm10_samples_compat.cuda_root import CudaRootError, load_cuda_root
from rocm10_samples_compat.environment import (
    collect_environment,
    strict_environment_errors,
)
from rocm10_samples_compat.manifest import (
    BATCH_ORDER,
    ManifestError,
    load_manifest,
    summary,
)
from rocm10_samples_compat.pipeline import run_sample
from rocm10_samples_compat.process import ToolError
from rocm10_samples_compat.project import (
    RULES_PROFILE,
    MigrationRequest,
    ProjectError,
    convert_project,
)
from rocm10_samples_compat.spec import SpecError, load_spec, spec_names


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rocm10-samples-compat",
        description="ROCm 10.0.0 / CUDA Samples v12.9 compatibility project.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    env_parser = subparsers.add_parser("check-env", help="report target environment")
    env_parser.add_argument("--json", action="store_true")
    env_parser.add_argument("--strict", action="store_true")

    subparsers.add_parser("show-manifest", help="show pinned manifest summary")
    list_parser = subparsers.add_parser("list-batches", help="list priority batches")
    list_parser.add_argument("--json", action="store_true")
    sample_parser = subparsers.add_parser("list-samples", help="list samples in one batch")
    sample_parser.add_argument("--batch", required=True, choices=BATCH_ORDER)
    sample_parser.add_argument("--json", action="store_true")

    checkout_parser = subparsers.add_parser(
        "verify-checkout", help="verify the pinned cuda-samples checkout"
    )
    checkout_parser.add_argument("--checkout", default=None)

    specs_parser = subparsers.add_parser("list-specs", help="list samples with a spec")
    specs_parser.add_argument("--json", action="store_true")

    convert_parser = subparsers.add_parser(
        "convert",
        help="convert an arbitrary CUDA project tree to HIP sources",
    )
    convert_parser.add_argument("--input", required=True, type=Path)
    convert_parser.add_argument("--output", required=True, type=Path)
    convert_parser.add_argument("--cuda-root", default=None, type=Path)
    convert_parser.add_argument("--compile-commands", default=None, type=Path)
    convert_parser.add_argument(
        "--no-configure",
        dest="auto_configure",
        action="store_false",
        default=True,
        help=(
            "do not configure a CMake project automatically; use tree mode "
            "unless --compile-commands is supplied"
        ),
    )
    convert_parser.add_argument(
        "--configure-arg",
        action="append",
        default=[],
        help="extra CMake configure argument (repeatable)",
    )
    convert_parser.add_argument(
        "--include",
        action="append",
        default=[],
        type=Path,
        help="include directory for tree mode (repeatable)",
    )
    convert_parser.add_argument(
        "--define",
        action="append",
        default=[],
        help="preprocessor define NAME or NAME=VALUE (repeatable)",
    )
    convert_parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="fnmatch pattern selecting sources (repeatable)",
    )
    convert_parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="fnmatch pattern excluding input paths (repeatable)",
    )
    convert_parser.add_argument(
        "--rules-profile", default=RULES_PROFILE, help="deterministic rule profile"
    )
    convert_parser.add_argument(
        "--force", action="store_true", help="replace a non-empty output directory"
    )
    convert_parser.add_argument("--json", action="store_true")

    run_parser = subparsers.add_parser("run-sample", help="migrate, build and run one sample")
    run_parser.add_argument("--sample", required=True)
    run_parser.add_argument("--artifacts", default="artifacts")
    run_parser.add_argument("--arch", default=None, help="gfx target for --offload-arch")
    run_parser.add_argument("--checkout", default=None)
    run_parser.add_argument("--json", action="store_true")

    batch_parser = subparsers.add_parser("run-batch", help="run every sample in one batch")
    batch_parser.add_argument("--batch", required=True, choices=BATCH_ORDER)
    batch_parser.add_argument("--artifacts", default="artifacts")
    batch_parser.add_argument("--arch", default=None)
    batch_parser.add_argument("--checkout", default=None)
    batch_parser.add_argument("--json", action="store_true")
    return parser


def _load_manifest_or_error() -> tuple[dict[str, Any] | None, int | None]:
    try:
        return load_manifest(), None
    except ManifestError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return None, 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "check-env":
        environment = collect_environment()
        if args.json:
            _print_json(environment)
        else:
            print(f"expected ROCm: {environment['expected']['rocm']}")
            print(f"expected CUDA Toolkit: {environment['expected']['cuda_toolkit']}")
            print(f"expected cuda-samples: {environment['expected']['cuda_samples_tag']}")
            print(f"detected ROCm: {environment['detected']['rocm_version']}")
            print("gfx: " + ", ".join(environment["detected"]["gfx_architectures"]))
            for name, version in environment["detected"]["tools"].items():
                print(f"{name}: {version or 'MISSING'}")
        errors = strict_environment_errors(environment) if args.strict else []
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1 if errors else 0

    if args.command == "convert":
        try:
            cuda_root = load_cuda_root(args.cuda_root) if args.cuda_root else None
            outcome = convert_project(
                MigrationRequest(
                    input_root=args.input,
                    output_root=args.output,
                    cuda_root=cuda_root,
                    compile_commands=args.compile_commands,
                    include_dirs=tuple(args.include),
                    defines=tuple(args.define),
                    excludes=tuple(args.exclude),
                    source_patterns=tuple(args.source),
                    force=args.force,
                    auto_configure=args.auto_configure,
                    configure_args=tuple(args.configure_arg),
                    rules_profile=args.rules_profile,
                )
            )
        except (ProjectError, CudaRootError, ToolError) as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 1

        if args.json:
            _print_json(outcome.report)
        else:
            report = outcome.report
            print(f"status: {outcome.status}")
            print(f"source: {report['output']['source']}")  # type: ignore[index]
            print(f"compile commands: {report['compile_commands_source']}")
            print(f"report: {outcome.report_path}")
            print(
                f"units: {report['translated_units']}/{report['unit_count']} translated"
            )
            print(
                f"headers: {report['translated_headers']} translated, "
                f"{report['failed_headers']} failed"
            )
            print(f"changes: {report['changes_count']}")
            print(f"blocking issues: {report['blocking_issues']}")
            print(f"review issues: {report['review_issues']}")
        return outcome.exit_code

    if args.command in ("show-manifest", "list-batches", "list-samples", "run-batch"):
        manifest, failure = _load_manifest_or_error()
        if failure is not None:
            return failure
        assert manifest is not None
        if args.command == "show-manifest":
            _print_json(summary(manifest))
            return 0
        if args.command == "list-batches":
            batches = {
                batch: len(members) for batch, members in manifest["batches"].items()
            }
            if args.json:
                _print_json(batches)
            else:
                for batch in BATCH_ORDER:
                    print(f"{batch}: {batches[batch]}")
            return 0
        if args.command == "list-samples":
            members = manifest["batches"][args.batch]
            if args.json:
                _print_json(members)
            else:
                for sample in members:
                    print(sample)
            return 0

    if args.command == "verify-checkout":
        try:
            checkout = verify_checkout(Path(args.checkout) if args.checkout else None)
        except CheckoutError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 1
        print(f"checkout: {checkout.root}")
        print(f"tag: {checkout.tag}")
        print(f"commit: {checkout.commit}")
        return 0

    if args.command == "list-specs":
        names = spec_names()
        if args.json:
            _print_json(list(names))
        else:
            for name in names:
                print(name)
        return 0

    if args.command in ("run-sample", "run-batch"):
        try:
            checkout = verify_checkout(Path(args.checkout) if args.checkout else None)
        except CheckoutError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 1
        if args.command == "run-sample":
            names = (args.sample,)
        else:
            manifest, failure = _load_manifest_or_error()
            if failure is not None:
                return failure
            assert manifest is not None
            names = tuple(manifest["batches"][args.batch])
        return _run_samples(names, checkout, Path(args.artifacts), args.arch, args.json)

    parser.error(f"unknown command: {args.command}")
    return 2


def _run_samples(
    names: tuple[str, ...],
    checkout: object,
    artifacts: Path,
    arch: str | None,
    as_json: bool,
) -> int:
    results: list[dict[str, object]] = []
    failures = 0
    for name in names:
        try:
            spec = load_spec(name)
        except SpecError as error:
            print(f"SKIP {name}: {error}", file=sys.stderr)
            continue
        try:
            run = run_sample(  # type: ignore[arg-type]
                spec, checkout, artifacts_root=artifacts, arch=arch
            )
        except Exception as error:  # noqa: BLE001 - a broken sample must not stop the batch
            print(f"ERROR {name}: {error}", file=sys.stderr)
            failures += 1
            continue
        summary_row = {
            "sample": run.sample,
            "status": run.status,
            "c1": run.c1,
            "c2": run.c2,
            "c3": run.c3,
            "blocked_at": run.evidence.get("blocked_at"),
            "evidence": str(run.directory / "evidence.json"),
        }
        results.append(summary_row)
        if not run.c3:
            failures += 1
        if not as_json:
            print(
                f"{run.sample:<16} status={run.status:<12} "
                f"C1={_tick(run.c1)} C2={_tick(run.c2)} C3={_tick(run.c3)}"
                + (
                    f" blocked_at={summary_row['blocked_at']}"
                    if summary_row["blocked_at"]
                    else ""
                )
            )
    if as_json:
        _print_json(results)
    return 1 if failures else 0


def _tick(value: bool) -> str:
    return "PASS" if value else "FAIL"
