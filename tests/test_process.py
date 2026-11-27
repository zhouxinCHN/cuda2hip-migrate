"""Tests for the shared subprocess runner."""

from __future__ import annotations

import os
import subprocess
import unittest
from unittest import mock

from rocm10_samples_compat.process import run_logged


class RunLoggedTests(unittest.TestCase):
    def test_environment_overrides_are_merged_with_the_parent(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["tool"], returncode=0, stdout="ok\n", stderr=""
        )
        with mock.patch(
            "rocm10_samples_compat.process.subprocess.run",
            return_value=completed,
        ) as run:
            result = run_logged(
                ["tool", "--flag"], env={"TMPDIR": "/tool/temp"}
            )
        self.assertEqual(result.returncode, 0)
        environment = run.call_args.kwargs["env"]
        self.assertEqual(environment["TMPDIR"], "/tool/temp")
        self.assertEqual(environment["PATH"], os.environ.get("PATH"))
        self.assertIsNone(run.call_args.kwargs["cwd"])

    def test_no_environment_keeps_inherited_environment(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["tool"], returncode=0, stdout="", stderr=""
        )
        with mock.patch(
            "rocm10_samples_compat.process.subprocess.run",
            return_value=completed,
        ) as run:
            run_logged(["tool"])
        self.assertIsNone(run.call_args.kwargs["env"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
