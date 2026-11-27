from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from rocm10_samples_compat.cli import main


class CliTests(unittest.TestCase):
    def test_list_batches(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["list-batches", "--json"]), 0)
        batches = json.loads(output.getvalue())
        self.assertEqual(batches["P0_baseline"], 6)
        self.assertEqual(batches["P1_probes"], 3)

    def test_list_p0_samples(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["list-samples", "--batch", "P0_baseline"]), 0)
        names = output.getvalue().splitlines()
        self.assertEqual(names[0], "deviceQuery")
        self.assertIn("scalarProd", names)

    def test_show_manifest(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["show-manifest"]), 0)
        data = json.loads(output.getvalue())
        self.assertEqual(data["fixed_decisions"]["target_rocm"], "10.0.0")

    def test_verify_checkout_accepts_checkout_option(self) -> None:
        """The parser must define --checkout, not fail with AttributeError."""
        with tempfile.TemporaryDirectory() as directory:
            errors = io.StringIO()
            with redirect_stderr(errors):
                code = main(["verify-checkout", "--checkout", directory])
            self.assertEqual(code, 1)
            self.assertIn("not a git checkout", errors.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
