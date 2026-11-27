"""Tests for HIPIFY failure detection.

hipify-clang does not always signal failure through its exit status alone, but a
bare ``error:`` line is not a failure signal either: a successful translation of
a CUDA source prints ``ptxas``/``nvlink`` architecture diagnostics that start
with ``error:``. These tests pin the distinction.
"""

from __future__ import annotations

import unittest

from rocm10_samples_compat.hipify import ERROR_MARKERS


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


if __name__ == "__main__":
    unittest.main()
