from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cuda2hip_migrate.runner import (
    ProbeError,
    check_redirect_expectation,
    check_redirects,
)


class RequiredRedirectTests(unittest.TestCase):
    def test_matching_redirect_passes(self) -> None:
        self.assertEqual(check_redirects("demo", (), ()), ())

    def test_unmatched_redirect_fails(self) -> None:
        with self.assertRaisesRegex(ProbeError, "declared include redirect"):
            check_redirects("demo", ("helper_cuda.h",), ())

    def test_failure_names_header_and_spec(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = Path(directory) / "demo.json"
            with self.assertRaises(ProbeError) as caught:
                check_redirects(
                    "demo", ("helper_cuda.h",), (), spec_path=spec
                )
        message = str(caught.exception)
        self.assertIn("helper_cuda.h", message)
        self.assertIn(str(spec), message)

    def test_optional_redirect_may_be_unmatched(self) -> None:
        self.assertEqual(
            check_redirects("demo", ("helper_cuda.h",), ("helper_cuda.h",)),
            ("helper_cuda.h",),
        )

    def test_optional_does_not_cover_other_headers(self) -> None:
        with self.assertRaisesRegex(ProbeError, "helper_drvapi.h"):
            check_redirects(
                "demo",
                ("helper_cuda.h", "helper_drvapi.h"),
                ("helper_cuda.h",),
            )


class RedirectExpectationTests(unittest.TestCase):
    def test_undeclared_expectation_is_ignored(self) -> None:
        check_redirect_expectation(None, 0)
        check_redirect_expectation(None, 3)

    def test_matching_count_passes(self) -> None:
        check_redirect_expectation(1, 1)
        check_redirect_expectation(0, 0)

    def test_mismatched_count_fails(self) -> None:
        with self.assertRaisesRegex(ProbeError, "expected_redirects=1"):
            check_redirect_expectation(1, 0)
        with self.assertRaisesRegex(ProbeError, "expected_redirects=0"):
            check_redirect_expectation(0, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
