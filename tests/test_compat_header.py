"""Host-only tests for the cuda-samples overlay's argument parsing.

The overlay header delegates flag parsing to
``compat/cuda-samples-v12.0/include/cuda_samples_compat/argparse.h``, which has
no CUDA or HIP dependency. That lets the parsing rules be compiled and run with
a plain host compiler, with no GPU and no ROCm installation.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

INCLUDE_ROOT = (
    Path(__file__).resolve().parent.parent
    / "compat"
    / "cuda-samples-v12.0"
    / "include"
)

PROGRAM = r"""
#include "cuda_samples_compat/argparse.h"

#include <cstdio>

int main() {
  const char *cases[][3] = {
      {"-device=0", "device", "0"},
      {"--device=0", "device", "0"},
      {"---device=3", "device", "3"},
      {"device=5", "device", "5"},
      {"-device=-1", "device", "-1"},
      {"-device=abc", "device", "abc"},
      {"-wA=320", "device", ""},
  };

  for (const auto &entry : cases) {
    const char *argv[] = {"prog", entry[0]};
    const char *value = nullptr;
    const bool found = cuda_samples_compat::find_flag_value(2, argv, entry[1], &value);
    std::printf("%s:%s:%s\n", entry[0], found ? "found" : "missing",
                found ? value : "-");
  }

  return 0;
}
"""


@unittest.skipUnless(shutil.which("g++"), "g++ is required for host header tests")
class OverlayArgparseTests(unittest.TestCase):
    def run_program(self) -> dict[str, str | None]:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "argparse_test.cpp"
            binary = Path(directory) / "argparse_test"
            source.write_text(PROGRAM, encoding="utf-8")
            build = subprocess.run(
                [
                    "g++",
                    "-std=c++17",
                    f"-I{INCLUDE_ROOT}",
                    str(source),
                    "-o",
                    str(binary),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            results: dict[str, str | None] = {}
            for line in run.stdout.splitlines():
                flag, state, value = line.split(":", 2)
                results[flag] = value if state == "found" else None
            return results

    def test_single_and_double_dash_are_equivalent(self) -> None:
        results = self.run_program()
        self.assertEqual(results["-device=0"], "0")
        self.assertEqual(results["--device=0"], "0")
        self.assertEqual(results["---device=3"], "3")

    def test_flag_without_dashes_is_accepted(self) -> None:
        self.assertEqual(self.run_program()["device=5"], "5")

    def test_negative_and_non_numeric_values_are_passed_through(self) -> None:
        results = self.run_program()
        self.assertEqual(results["-device=-1"], "-1")
        self.assertEqual(results["-device=abc"], "abc")

    def test_other_flags_are_not_matched(self) -> None:
        self.assertIsNone(self.run_program()["-wA=320"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
