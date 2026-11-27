"""Tests for HIPIFY failure detection.

hipify-clang does not always signal failure through its exit status alone, but a
bare ``error:`` line is not a failure signal either: a successful translation of
a CUDA source prints ``ptxas``/``nvlink`` architecture diagnostics that start
with ``error:``. These tests pin the distinction.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rocm10_samples_compat.cuda_root import CudaRoot
from rocm10_samples_compat.hipify import ERROR_MARKERS, hipify_source
from rocm10_samples_compat.process import ToolResult


class ErrorMarkerTests(unittest.TestCase):
    def test_hipify_error_line_is_a_marker(self) -> None:
        self.assertIn("[HIPIFY] error", ERROR_MARKERS)

    def test_processing_failure_is_a_marker(self) -> None:
        self.assertIn("Error while processing", ERROR_MARKERS)

    def test_bare_error_prefix_is_not_a_marker(self) -> None:
        """A bare ``error:`` also appears in benign, successful runs."""
        self.assertNotIn("error: ", ERROR_MARKERS)
        self.assertNotIn("error:", ERROR_MARKERS)

    def test_benign_ptxas_diagnostics_do_not_match(self) -> None:
        output = (
            "error: must pass in an explicit nvptx64 gpu architecture to 'ptxas'\n"
            "error: must pass in an explicit nvptx64 gpu architecture to 'nvlink'\n"
            " !!!!!!! CUDA SDK version detected: 19\n"
        )
        self.assertFalse(any(marker in output for marker in ERROR_MARKERS))

    def test_real_failure_matches(self) -> None:
        output = "1 error generated when compiling for host.\nError while processing /tmp/a.cu\n"
        self.assertTrue(any(marker in output for marker in ERROR_MARKERS))


class HipifyTempDirectoryTests(unittest.TestCase):
    def test_temp_dir_reaches_command_and_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.cu"
            source.write_text("__global__ void k() {}\n", encoding="utf-8")
            output = root / "output" / "input.hip"
            temp_dir = root / "output" / ".tmp" / "project" / "hipify"

            def fake_run(command: list[str], **kwargs: object) -> ToolResult:
                self.assertIn(f"--temp-dir={temp_dir.resolve()}", command)
                self.assertEqual(
                    kwargs["env"]["TMPDIR"], str(temp_dir.resolve())  # type: ignore[index]
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("// hip\n", encoding="utf-8")
                return ToolResult(
                    command=tuple(command),
                    returncode=0,
                    stdout="",
                    stderr="",
                )

            cuda_root = CudaRoot(
                path=root / "cuda",
                version=12090,
                version_string="12.9",
                synthetic=False,
                detection="test",
            )
            with (
                mock.patch(
                    "rocm10_samples_compat.hipify.require_tool",
                    return_value="/usr/bin/hipify-clang",
                ),
                mock.patch(
                    "rocm10_samples_compat.hipify.run_logged",
                    side_effect=fake_run,
                ),
            ):
                result = hipify_source(
                    source,
                    output,
                    cuda_root=cuda_root,
                    include_dirs=(),
                    temp_dir=temp_dir,
                )
            self.assertEqual(result.output, output)
            self.assertTrue(temp_dir.is_dir())


if __name__ == "__main__":
    unittest.main()
