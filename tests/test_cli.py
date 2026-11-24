from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout

from cuda2hip_migrate.cli import main


class CliTests(unittest.TestCase):
    def test_list_returns_success(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["list"]), 0)
        self.assertIn("probe_malloc_memcpy", output.getvalue())

    def test_list_json_is_valid(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["list", "--json"]), 0)
        names = json.loads(output.getvalue())
        self.assertIn("probe_pinned_host_mem", names)

    def test_unknown_probe_returns_failure(self) -> None:
        self.assertEqual(main(["run", "--probe", "does_not_exist"]), 1)

    def test_missing_source_returns_failure(self) -> None:
        self.assertEqual(main(["run", "--source", "does_not_exist.cu"]), 1)

    def test_probe_and_source_are_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            main(["run", "--probe", "probe_malloc_memcpy", "--source", "x.cu"])

    def test_source_or_probe_is_required(self) -> None:
        with self.assertRaises(SystemExit):
            main(["run"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
