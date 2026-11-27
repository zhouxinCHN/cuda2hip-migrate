"""Tests for project-level conversion and compile database handling."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rocm10_samples_compat.cuda_root import CudaRoot
from rocm10_samples_compat.hipify import HipifyError, HipifyResult
from rocm10_samples_compat.process import ToolResult
from rocm10_samples_compat.project import (
    MigrationRequest,
    ProjectError,
    UnitConflict,
    _apply_rules,
    _inspect_residuals,
    _normalise_units,
    _parse_compile_arguments,
    _prepare_output,
    _RunState,
    _select_redirects,
    _units_from_compile_commands,
    _units_from_tree,
    convert_project,
)
from rocm10_samples_compat.rules import profiler


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class CompileArgumentTests(unittest.TestCase):
    def test_extracts_parse_relevant_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            includes, defines, undefines, standard, forced, ignored = (
                _parse_compile_arguments(
                    [
                        "-Iinclude",
                        "-I",
                        "generated",
                        "-isysteminclude/system",
                        "-DFEATURE=1",
                        "-D",
                        "OTHER",
                        "-UOLD",
                        "-std=c++20",
                        "-include",
                        "prefix.h",
                        "-gencode",
                        "arch=compute_90,code=sm_90",
                        "-Xcompiler",
                        "-fPIC",
                        "-O3",
                        "-c",
                        "kernel.cu",
                        "-o",
                        "kernel.o",
                    ],
                    directory=root,
                    source_path=root / "kernel.cu",
                )
            )
            self.assertEqual(
                [(item.flag, item.path) for item in includes],
                [
                    ("-I", root / "include"),
                    ("-I", root / "generated"),
                    ("-isystem", root / "include/system"),
                ],
            )
            self.assertEqual(defines, ("FEATURE=1", "OTHER"))
            self.assertEqual(undefines, ("OLD",))
            self.assertEqual(standard, "c++20")
            self.assertEqual(forced, (root / "prefix.h",))
            self.assertIn("-gencode", ignored)
            self.assertIn("-Xcompiler", ignored)
            self.assertIn("-O3", ignored)
            self.assertNotIn("-c", ignored)

    def test_arguments_with_equals_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            includes, _defines, _undefines, standard, _forced, _ignored = (
                _parse_compile_arguments(
                    ["-I=include", "-isystem=/opt/include", "-std=c++17"],
                    directory=root,
                    source_path=root / "x.cu",
                )
            )
            self.assertEqual(includes[0].path, root / "include")
            self.assertEqual(includes[1].path, Path("/opt/include"))
            self.assertEqual(standard, "c++17")


class CompileDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        _write(self.root / "src" / "main.cu", "__global__ void k() {}\n")
        _write(self.root / "src" / "host.cpp", "int main() { return 0; }\n")
        _write(self.root / "include" / "value.cuh", "#define VALUE 1\n")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _database(self, entries: list[dict[str, object]]) -> Path:
        return _write(
            self.root / "build" / "compile_commands.json",
            json.dumps(entries),
        )

    def test_uses_each_entry_and_skips_host_compiler_units(self) -> None:
        database = self._database(
            [
                {
                    "directory": str(self.root),
                    "file": "src/main.cu",
                    "command": (
                        "nvcc -I include -DMAIN=1 -std=c++17 -gencode "
                        "arch=compute_90,code=sm_90 -c src/main.cu"
                    ),
                },
                {
                    "directory": str(self.root),
                    "file": str(self.root / "src" / "host.cpp"),
                    "command": "g++ -I include -c src/host.cpp",
                },
                {
                    "directory": "/elsewhere",
                    "file": "/elsewhere/not-ours.cu",
                    "command": "nvcc -c not-ours.cu",
                },
            ]
        )
        units = _units_from_compile_commands(
            database,
            input_root=self.root,
            excludes=(),
            source_patterns=(),
        )
        self.assertEqual([unit.path.as_posix() for unit in units], ["src/host.cpp", "src/main.cu"])
        by_path = {unit.path.as_posix(): unit for unit in units}
        self.assertTrue(by_path["src/main.cu"].translate)
        self.assertFalse(by_path["src/host.cpp"].translate)
        unit = by_path["src/main.cu"]
        self.assertEqual(unit.defines, ("MAIN=1",))
        self.assertEqual(unit.standard, "c++17")
        self.assertEqual(unit.includes[0].path, self.root / "include")
        self.assertIn("-gencode", unit.ignored_arguments)

    def test_incompatible_duplicate_is_rejected(self) -> None:
        database = self._database(
            [
                {
                    "directory": str(self.root),
                    "file": "src/main.cu",
                    "arguments": ["nvcc", "-DVALUE=1", "-c", "src/main.cu"],
                },
                {
                    "directory": str(self.root),
                    "file": "src/main.cu",
                    "arguments": ["nvcc", "-DVALUE=2", "-c", "src/main.cu"],
                },
            ]
        )
        with self.assertRaises(UnitConflict):
            _units_from_compile_commands(
                database,
                input_root=self.root,
                excludes=(),
                source_patterns=(),
            )

    def test_source_pattern_and_exclude_are_applied(self) -> None:
        database = self._database(
            [
                {
                    "directory": str(self.root),
                    "file": "src/main.cu",
                    "arguments": ["nvcc", "-c", "src/main.cu"],
                }
            ]
        )
        units = _units_from_compile_commands(
            database,
            input_root=self.root,
            source_patterns=("src/*.cu",),
            excludes=(),
        )
        self.assertEqual(len(units), 1)
        with self.assertRaises(ProjectError):
            _units_from_compile_commands(
                database,
                input_root=self.root,
                source_patterns=("other/*.cu",),
                excludes=(),
            )

    def test_identical_duplicate_entries_are_merged(self) -> None:
        candidates = []
        for output in ("a.o", "b.o"):
            candidates.append(
                _units_from_compile_commands(
                    self._database(
                        [
                            {
                                "directory": str(self.root),
                                "file": "src/main.cu",
                                "arguments": ["nvcc", "-c", "src/main.cu"],
                                "output": output,
                            }
                        ]
                    ),
                    input_root=self.root,
                    excludes=(),
                    source_patterns=(),
                )[0]
            )
        merged = _normalise_units(candidates, input_root=self.root)
        self.assertEqual(len(merged), 1)


class TreeModeTests(unittest.TestCase):
    def test_recurses_and_preserves_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root / "a" / "one.cu", "__global__ void a() {}\n")
            _write(root / "b" / "two.cu", "__global__ void b() {}\n")
            _write(root / "b" / "notes.txt", "not source\n")
            _write(root / ".git" / "config", "ignored\n")
            units = _units_from_tree(
                root,
                excludes=(),
                source_patterns=(),
                include_dirs=(root / "include",),
                defines=("FEATURE=1",),
            )
            self.assertEqual([unit.path.as_posix() for unit in units], ["a/one.cu", "b/two.cu"])
            self.assertEqual(units[0].defines, ("FEATURE=1",))


class OutputSafetyTests(unittest.TestCase):
    def test_output_inside_input_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ProjectError):
                _prepare_output(
                    MigrationRequest(input_root=root, output_root=root / "out")
                )

    def test_non_empty_output_needs_force(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            output = root / "output"
            _write(output / "keep.txt", "old\n")
            with self.assertRaises(ProjectError):
                _prepare_output(MigrationRequest(input_root=source, output_root=output))
            resolved_input, resolved_output = _prepare_output(
                MigrationRequest(
                    input_root=source, output_root=output, force=True
                )
            )
            self.assertEqual(resolved_input, source.resolve())
            self.assertFalse((resolved_output / "keep.txt").exists())


class RedirectSelectionTests(unittest.TestCase):
    def test_helper_is_enabled_only_when_present(self) -> None:
        state = _RunState()
        overlay = Path(__file__).resolve().parents[1] / "src/rocm10_samples_compat/overlay"
        self.assertEqual(
            _select_redirects(["#include <stdio.h>\n"], overlay=overlay, state=state),
            (),
        )
        self.assertEqual(state.issues, [])
        redirects = _select_redirects(
            ["#include <helper_cuda.h>\n"], overlay=overlay, state=state
        )
        self.assertEqual([redirect.header for redirect in redirects], ["helper_cuda.h"])


class RuleStageTests(unittest.TestCase):
    def test_profiler_rule_change_is_recorded_with_its_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(
                root / "kernel.cu",
                "#include <hip/hip_runtime.h>\nhipProfilerStart();\n",
            )
            state = _RunState()
            _apply_rules(root, root / "out", state)
            self.assertEqual(len(state.changes), 2)
            self.assertEqual({path for path, _change in state.changes}, {"kernel.cu"})
            self.assertEqual(
                [change.rule_id for _path, change in state.changes],
                [profiler.RULE_ID, profiler.RULE_ID],
            )

    def test_residual_scan_skips_comments_and_reports_cuda_calls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(
                root / "kernel.cu",
                "// cudaMalloc appears in a comment\n"
                'const char *s = "cudaFree";\n'
                "#include <cuda_runtime.h>\n"
                "cudaMalloc(&p, 10);\n"
                "asm volatile(\"nop;\");\n",
            )
            state = _RunState()
            _inspect_residuals(root, state)
            kinds = {issue.kind for issue in state.issues}
            self.assertIn("residual-cuda-include", kinds)
            self.assertIn("residual-cuda-identifier", kinds)
            self.assertIn("inline-assembly", kinds)
            symbols = {issue.symbol for issue in state.issues}
            self.assertIn("cudaMalloc", symbols)
            self.assertNotIn("cudaFree", symbols)


class HeaderConfigurationTests(unittest.TestCase):
    def test_header_with_incompatible_configs_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "input"
            output_root = root / "output"
            _write(
                source_root / "include" / "kernel.cuh",
                "#include <cuda_runtime.h>\n__device__ void k();\n",
            )
            _write(source_root / "one.cu", "__global__ void one() {}\n")
            _write(source_root / "two.cu", "__global__ void two() {}\n")
            database = _write(
                source_root / "compile_commands.json",
                json.dumps(
                    [
                        {
                            "directory": str(source_root),
                            "file": "one.cu",
                            "arguments": [
                                "nvcc",
                                "-Iinclude",
                                "-DFEATURE=1",
                                "-c",
                                "one.cu",
                            ],
                        },
                        {
                            "directory": str(source_root),
                            "file": "two.cu",
                            "arguments": [
                                "nvcc",
                                "-Iinclude",
                                "-DFEATURE=2",
                                "-c",
                                "two.cu",
                            ],
                        },
                    ]
                ),
            )

            def fake_hipify(
                source: Path, output: Path, **kwargs: object
            ) -> HipifyResult:
                output.parent.mkdir(parents=True, exist_ok=True)
                text = source.read_text(encoding="utf-8")
                if source.suffix == ".cuh":
                    extra = " ".join(str(item) for item in kwargs.get("extra_args", ()))
                    variant = "one" if "-DFEATURE=1" in extra else "two"
                    text += f"// {variant}\n"
                output.write_text(text, encoding="utf-8")
                return HipifyResult(
                    source=source,
                    output=output,
                    result=ToolResult(
                        command=("hipify-clang", str(source)),
                        returncode=0,
                        stdout="",
                        stderr="",
                    ),
                )

            cuda_root = CudaRoot(
                path=root / "cuda",
                version=12090,
                version_string="12.9",
                synthetic=False,
                detection="test",
            )
            with mock.patch(
                "rocm10_samples_compat.project.hipify_source",
                side_effect=fake_hipify,
            ):
                outcome = convert_project(
                    MigrationRequest(
                        input_root=source_root,
                        output_root=output_root,
                        cuda_root=cuda_root,
                        compile_commands=database,
                    )
                )
            self.assertEqual(outcome.status, "blocked")
            self.assertTrue(
                any(
                    issue["kind"] == "header-configuration-conflict"
                    for issue in outcome.report["issues"]
                )
            )


class ConvertProjectTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input = self.root / "input"
        self.output = self.root / "output"
        _write(
            self.input / "src" / "main.cu",
            "#include <helper_cuda.h>\n"
            "#include <hip/hip_runtime.h>\n"
            "int main() { hipProfilerStart(); return 0; }\n",
        )
        _write(self.input / "src" / "include" / "value.cuh", "#define VALUE 1\n")
        _write(
            self.input / "src" / "include" / "device_math.cuh",
            "#include <cuda_runtime.h>\n"
            "__device__ inline float twice(float x) { return x * 2.0f; }\n",
        )
        _write(self.input / "assets" / "data.bin", "payload\n")
        _write(self.input / ".git" / "config", "not copied\n")
        self.database = _write(
            self.input / "compile_commands.json",
            json.dumps(
                [
                    {
                        "directory": str(self.input),
                        "file": "src/main.cu",
                        "command": (
                            "nvcc -I src/include -DFEATURE=1 -std=c++17 "
                            "-c src/main.cu -o main.o"
                        ),
                    }
                ]
            ),
        )
        self.cuda_root = CudaRoot(
            path=self.root / "cuda",
            version=12090,
            version_string="12.9",
            synthetic=False,
            detection="test",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _fake_hipify(self, source: Path, output: Path, **kwargs: object) -> HipifyResult:
        output.parent.mkdir(parents=True, exist_ok=True)
        text = source.read_text(encoding="utf-8").replace(
            "#include <cuda_runtime.h>", "#include <hip/hip_runtime.h>"
        )
        output.write_text(text, encoding="utf-8")
        return HipifyResult(
            source=source,
            output=output,
            result=ToolResult(
                command=("hipify-clang", str(source)),
                returncode=0,
                stdout="",
                stderr="",
            ),
        )

    def test_conversion_writes_source_report_requirements_and_keeps_input(self) -> None:
        before = (self.input / "src" / "main.cu").read_bytes()
        with mock.patch(
            "rocm10_samples_compat.project.hipify_source", side_effect=self._fake_hipify
        ):
            outcome = convert_project(
                MigrationRequest(
                    input_root=self.input,
                    output_root=self.output,
                    cuda_root=self.cuda_root,
                    compile_commands=self.database,
                )
            )

        self.assertEqual(outcome.status, "completed")
        self.assertEqual(outcome.exit_code, 0)
        self.assertTrue((self.output / "src" / "src" / "main.cu").is_file())
        self.assertTrue((self.output / "src" / "src" / "include" / "value.cuh").is_file())
        translated_header = (
            self.output / "src" / "src" / "include" / "device_math.cuh"
        ).read_text(encoding="utf-8")
        self.assertIn("#include <hip/hip_runtime.h>", translated_header)
        self.assertTrue((self.output / "src" / "assets" / "data.bin").is_file())
        self.assertFalse((self.output / "src" / ".git").exists())
        self.assertTrue((self.output / "reports" / "migration.json").is_file())
        self.assertTrue((self.output / "rocm-migration.cmake").is_file())
        self.assertEqual((self.input / "src" / "main.cu").read_bytes(), before)

        report = json.loads((self.output / "reports" / "migration.json").read_text())
        self.assertEqual(report["translated_units"], 1)
        self.assertEqual(report["translated_headers"], 1)
        self.assertEqual(report["headers"][0]["path"], "src/include/device_math.cuh")
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["redirects"]["enabled"][0]["header"], "helper_cuda.h")
        self.assertIn("profiler-roctx", report["requirements"]["features"])
        self.assertIn(
            "rocprofiler-sdk-roctx", report["requirements"]["libraries"]
        )
        self.assertIn(
            "CUDART_VERSION=12090", report["requirements"]["compile_definitions"]
        )
        cmake = (self.output / "rocm-migration.cmake").read_text()
        self.assertIn("ROCmCudaMigrate::profiler", cmake)
        self.assertIn("rocprofiler-sdk-roctx", cmake)
        self.assertIn("CUDART_VERSION=12090", cmake)

    def test_blocked_conversion_restores_source_and_exits_two(self) -> None:
        with mock.patch(
            "rocm10_samples_compat.project.hipify_source",
            side_effect=HipifyError("translation failed"),
        ):
            outcome = convert_project(
                MigrationRequest(
                    input_root=self.input,
                    output_root=self.output,
                    cuda_root=self.cuda_root,
                    compile_commands=self.database,
                )
            )
        self.assertEqual(outcome.status, "blocked")
        self.assertEqual(outcome.exit_code, 2)
        self.assertTrue((self.output / "src" / "src" / "main.cu").is_file())
        report = json.loads((self.output / "reports" / "migration.json").read_text())
        self.assertEqual(report["failed_units"], 1)
        self.assertTrue(any(issue["kind"] == "hipify-failure" for issue in report["issues"]))

    def test_force_replaces_output(self) -> None:
        _write(self.output / "stale.txt", "old\n")
        with mock.patch(
            "rocm10_samples_compat.project.hipify_source", side_effect=self._fake_hipify
        ):
            outcome = convert_project(
                MigrationRequest(
                    input_root=self.input,
                    output_root=self.output,
                    cuda_root=self.cuda_root,
                    compile_commands=self.database,
                    force=True,
                )
            )
        self.assertEqual(outcome.status, "completed")
        self.assertFalse((self.output / "stale.txt").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
