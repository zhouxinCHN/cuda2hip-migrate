"""Tests for pinned-checkout verification."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rocm10_samples_compat.checkout import (
    EXPECTED_COMMIT,
    CheckoutError,
    sample_directory,
    verify_checkout,
)


class VerifyCheckoutTests(unittest.TestCase):
    def test_missing_checkout_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(CheckoutError) as context:
                verify_checkout(Path(directory))
        self.assertIn("not a git checkout", str(context.exception))

    def test_missing_samples_directory_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / ".git").mkdir()
            with self.assertRaises(CheckoutError) as context:
                verify_checkout(Path(directory))
        self.assertIn("Samples", str(context.exception))

    def test_sample_directory_outside_checkout_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = type("Fake", (), {"samples": Path(directory)})()
            with self.assertRaises(CheckoutError):
                sample_directory(checkout, "0_Introduction/doesNotExist")  # type: ignore[arg-type]

    def test_pinned_commit_is_v129(self) -> None:
        self.assertEqual(EXPECTED_COMMIT, "cab7c66b4fcf35b09ea8fbc4c3c1a9272e92d997")


if __name__ == "__main__":
    unittest.main()
