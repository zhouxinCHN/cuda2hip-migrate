"""Tests for the output-root temporary-directory policy."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from rocm10_samples_compat.workspace import (
    WorkspaceError,
    temp_environment,
    temp_root,
    unit_slug,
)


class WorkspaceTests(unittest.TestCase):
    def test_temp_root_is_under_output_and_sanitized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            root = temp_root(output, "../project name")
            self.assertEqual(root.parent.parent, output.resolve())
            self.assertEqual(root.name, "project-name")
            self.assertTrue(root.is_dir())

    def test_temp_environment_points_all_spellings_at_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "tmp"
            environment = temp_environment(root)
            self.assertEqual(
                environment,
                {
                    "TMPDIR": str(root.resolve()),
                    "TMP": str(root.resolve()),
                    "TEMP": str(root.resolve()),
                },
            )
            self.assertTrue(root.is_dir())

    def test_symlinked_tmp_cannot_escape_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            outside = root / "outside"
            output.mkdir()
            outside.mkdir()
            try:
                os.symlink(outside, output / ".tmp")
            except OSError as error:
                self.skipTest(f"symlinks are unavailable: {error}")
            with self.assertRaises(WorkspaceError):
                temp_root(output, "project")

    def test_empty_name_has_a_fallback(self) -> None:
        self.assertEqual(unit_slug("..."), "project")
        self.assertEqual(unit_slug("a/b c"), "a-b-c")


if __name__ == "__main__":
    unittest.main(verbosity=2)
