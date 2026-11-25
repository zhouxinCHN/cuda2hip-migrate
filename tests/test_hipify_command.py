"""Tests for the hipify-clang command line and its failure detection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cuda2hip_migrate.cuda_root import CudaRoot
from cuda2hip_migrate.runner import (
    HIPIFY_ERROR_MARKERS,
    ProbeError,
    build_hipify_command,
    hipify_report_errors,
    load_sample_spec,
)


def _cuda_root(path: Path) -> CudaRoot:
    return CudaRoot(
        path=path,
        version=12000,
        version_string="12.0",
        synthetic=False,
        detection="test",
    )


class BuildHipifyCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("/tmp/cuda-root")

    def test_cuda_path_is_absolute_and_present(self) -> None:
        command = build_hipify_command(
            hipify="hipify-clang",
            source=Path("/tmp/a.cu"),
            output=Path("/tmp/a.hip.cu"),
            cuda=_cuda_root(self.root),
            include_dirs=(),
        )
        self.assertIn(f"--cuda-path={self.root}", command)

    def test_parses_only_the_selected_preprocessor_blocks(self) -> None:
        command = build_hipify_command(
            hipify="hipify-clang",
            source=Path("/tmp/a.cu"),
            output=Path("/tmp/a.hip.cu"),
            cuda=_cuda_root(self.root),
            include_dirs=(),
        )
        self.assertIn("--skip-excluded-preprocessor-conditional-blocks", command)

    def test_translates_quoted_local_headers(self) -> None:
        command = build_hipify_command(
            hipify="hipify-clang",
            source=Path("/tmp/a.cu"),
            output=Path("/tmp/a.hip.cu"),
            cuda=_cuda_root(self.root),
            include_dirs=(),
        )
        self.assertIn("--local-headers", command)

    def test_include_dirs_are_passed_as_separate_arguments(self) -> None:
        command = build_hipify_command(
            hipify="hipify-clang",
            source=Path("/tmp/a.cu"),
            output=Path("/tmp/a.hip.cu"),
            cuda=_cuda_root(self.root),
            include_dirs=(Path("/one"), Path("/two")),
        )
        self.assertEqual(command.count("-I"), 2)
        self.assertIn("/one", command)
        self.assertIn("/two", command)

    def test_cudart_version_is_a_clang_argument(self) -> None:
        command = build_hipify_command(
            hipify="hipify-clang",
            source=Path("/tmp/a.cu"),
            output=Path("/tmp/a.hip.cu"),
            cuda=_cuda_root(self.root),
            include_dirs=(),
            cudart_version=12000,
        )
        self.assertIn("--", command)
        self.assertGreater(command.index("--"), command.index("-o"))
        self.assertEqual(command[-1], "-DCUDART_VERSION=12000")

    def test_cudart_version_absent_by_default(self) -> None:
        command = build_hipify_command(
            hipify="hipify-clang",
            source=Path("/tmp/a.cu"),
            output=Path("/tmp/a.hip.cu"),
            cuda=_cuda_root(self.root),
            include_dirs=(),
        )
        self.assertNotIn("--", command)
        self.assertFalse(any(item.startswith("-DCUDART") for item in command))

    def test_forced_includes_follow_the_separator(self) -> None:
        command = build_hipify_command(
            hipify="hipify-clang",
            source=Path("/tmp/a.cu"),
            output=Path("/tmp/a.hip.cu"),
            cuda=_cuda_root(self.root),
            include_dirs=(),
            forced_includes=(Path("/inc/shim.h"),),
            cudart_version=12000,
        )
        separator = command.index("--")
        self.assertEqual(
            command[separator + 1 :],
            ["-DCUDART_VERSION=12000", "-include", "/inc/shim.h"],
        )


class HipifyErrorDetectionTests(unittest.TestCase):
    def test_clean_output_reports_nothing(self) -> None:
        self.assertEqual(hipify_report_errors("[HIPIFY] CUDA_VERSION: 12000\n"), ())

    def test_error_marker_is_reported_despite_exit_zero(self) -> None:
        output = "1 error generated.\nError while processing /tmp/a.cu\n"
        self.assertEqual(hipify_report_errors(output), ("Error while processing",))

    def test_hipify_prefixed_error_is_reported(self) -> None:
        output = "[HIPIFY] error: Hipifying failed: /tmp/a.cu\n"
        self.assertEqual(hipify_report_errors(output), ("[HIPIFY] error",))

    def test_markers_are_documented_constants(self) -> None:
        self.assertIn("[HIPIFY] error", HIPIFY_ERROR_MARKERS)
        self.assertIn("Error while processing", HIPIFY_ERROR_MARKERS)


class CudartVersionSpecTests(unittest.TestCase):
    def _write_spec(self, directory: str, extra: str) -> Path:
        path = Path(directory) / "spec.json"
        path.write_text(
            "{\n"
            '  "schema_version": 1,\n'
            '  "sample": "test",\n'
            '  "source": "source.cu",\n'
            '  "success_marker": "RESULT=PASS",\n'
            f"  {extra}\n"
            "}\n",
            encoding="utf-8",
        )
        return path

    def test_cudart_version_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_spec(directory, '"cudart_version": 12000')
            spec = load_sample_spec(path)
        self.assertEqual(spec.cudart_version, 12000)

    def test_cudart_version_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_spec(directory, '"libraries": []')
            spec = load_sample_spec(path)
        self.assertIsNone(spec.cudart_version)

    def test_non_integer_cudart_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_spec(directory, '"cudart_version": "12.0"')
            with self.assertRaises(ProbeError):
                load_sample_spec(path)

    def test_boolean_cudart_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_spec(directory, '"cudart_version": true')
            with self.assertRaises(ProbeError):
                load_sample_spec(path)


if __name__ == "__main__":
    unittest.main()
