from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cuda2hip_migrate.runner import (
    ProbeError,
    check_expectation,
    load_sample_spec,
)


def write_spec(directory: str, payload: object) -> Path:
    path = Path(directory) / "spec.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class SampleSpecTests(unittest.TestCase):
    def test_loads_full_spec(self) -> None:
        payload = {
            "sample": "cuda-samples-v12.0/demo",
            "source": "third-party/cuda-samples/Samples/demo/demo.cu",
            "include_dirs": ["compat/include", "third-party/cuda-samples/Common"],
            "libraries": ["hipblas"],
            "success_marker": "Test PASSED",
            "expected": "hipify-direct",
            "include_redirects": {"helper_cuda.h": "cuda_samples_compat/helper_cuda.h"},
        }
        with tempfile.TemporaryDirectory() as directory:
            spec = load_sample_spec(write_spec(directory, payload))
        self.assertEqual(spec.sample, "cuda-samples-v12.0/demo")
        self.assertEqual(spec.source, Path(payload["source"]))
        self.assertEqual(len(spec.include_dirs), 2)
        self.assertEqual(spec.libraries, ("hipblas",))
        self.assertEqual(spec.success_marker, "Test PASSED")
        self.assertEqual(spec.expected, "hipify-direct")
        self.assertEqual(
            spec.include_redirects,
            (("helper_cuda.h", "cuda_samples_compat/helper_cuda.h"),),
        )

    def test_defaults(self) -> None:
        payload = {"sample": "s", "source": "x.cu"}
        with tempfile.TemporaryDirectory() as directory:
            spec = load_sample_spec(write_spec(directory, payload))
        self.assertEqual(spec.success_marker, "RESULT=PASS")
        self.assertEqual(spec.expected, "unknown")
        self.assertEqual(spec.include_dirs, ())
        self.assertEqual(spec.libraries, ())
        self.assertEqual(spec.include_redirects, ())

    def test_unknown_key_is_rejected(self) -> None:
        payload = {"sample": "s", "source": "x.cu", "cmake": True}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ProbeError, "unsupported key"):
                load_sample_spec(write_spec(directory, payload))

    def test_missing_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ProbeError, "'source' is required"):
                load_sample_spec(write_spec(directory, {"sample": "s"}))

    def test_bad_expected_is_rejected(self) -> None:
        payload = {"sample": "s", "source": "x.cu", "expected": "hopefully"}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ProbeError, "'expected' must be one of"):
                load_sample_spec(write_spec(directory, payload))

    def test_optimization_defaults_to_o2(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = load_sample_spec(
                write_spec(directory, {"sample": "s", "source": "x.cu"})
            )
        self.assertEqual(spec.optimization, "-O2")

    def test_optimization_is_honoured(self) -> None:
        payload = {"sample": "s", "source": "x.cu", "optimization": "-O0"}
        with tempfile.TemporaryDirectory() as directory:
            spec = load_sample_spec(write_spec(directory, payload))
        self.assertEqual(spec.optimization, "-O0")

    def test_bad_optimization_is_rejected(self) -> None:
        payload = {"sample": "s", "source": "x.cu", "optimization": "-Os"}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ProbeError, "'optimization' must be one of"):
                load_sample_spec(write_spec(directory, payload))

    def test_bad_redirects_are_rejected(self) -> None:
        payload = {"sample": "s", "source": "x.cu", "include_redirects": ["a"]}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ProbeError, "include_redirects"):
                load_sample_spec(write_spec(directory, payload))

    def test_missing_spec_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ProbeError, "sample spec not found"):
                load_sample_spec(Path(directory) / "absent.json")


class ExpectationTests(unittest.TestCase):
    def test_unknown_expected_never_fails(self) -> None:
        check_expectation(None, 3)
        check_expectation("unknown", 0)

    def test_hipify_direct_without_rules_passes(self) -> None:
        check_expectation("hipify-direct", 0)

    def test_hipify_direct_with_rules_fails(self) -> None:
        with self.assertRaisesRegex(ProbeError, "declared hipify-direct"):
            check_expectation("hipify-direct", 1)

    def test_rule_required_without_rules_fails(self) -> None:
        with self.assertRaisesRegex(ProbeError, "declared rule-required"):
            check_expectation("rule-required", 0)

    def test_rule_required_with_rules_passes(self) -> None:
        check_expectation("rule-required", 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
