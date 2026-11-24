from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from cuda2hip_migrate.runner import (
    ProbeError,
    check_expectation,
    load_sample_spec,
    resolve_forced_includes,
)


def write_spec(directory: str, payload: object, name: str = "spec.json") -> Path:
    path = Path(directory) / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class SampleSpecTests(unittest.TestCase):
    def test_loads_full_spec(self) -> None:
        payload = {
            "schema_version": 1,
            "sample": "cuda-samples-v12.0/demo",
            "cuda_samples_commit": "abc123",
            "source": "third-party/cuda-samples/Samples/demo/demo.cu",
            "include_dirs": ["compat/include", "third-party/cuda-samples/Common"],
            "libraries": ["hipblas"],
            "success_marker": "Test PASSED",
            "expected": "hipify-direct",
            "expected_redirects": 1,
            "include_redirects": {"helper_cuda.h": "cuda_samples_compat/helper_cuda.h"},
            "optional_redirects": ["helper_cuda_drvapi.h"],
            "optimization": "-O0",
        }
        with tempfile.TemporaryDirectory() as directory:
            spec = load_sample_spec(write_spec(directory, payload))
            base = Path(directory).resolve()
        self.assertEqual(spec.sample, "cuda-samples-v12.0/demo")
        self.assertEqual(spec.cuda_samples_commit, "abc123")
        self.assertEqual(spec.source, base / payload["source"])
        self.assertEqual(spec.include_dirs, (base / "compat/include", base / "third-party/cuda-samples/Common"))
        self.assertEqual(spec.libraries, ("hipblas",))
        self.assertEqual(spec.success_marker, "Test PASSED")
        self.assertEqual(spec.expected, "hipify-direct")
        self.assertEqual(spec.expected_redirects, 1)
        self.assertEqual(
            spec.include_redirects,
            (("helper_cuda.h", "cuda_samples_compat/helper_cuda.h"),),
        )
        self.assertEqual(spec.optional_redirects, ("helper_cuda_drvapi.h",))
        self.assertEqual(spec.optimization, "-O0")
        self.assertEqual(spec.schema_version, 1)

    def test_defaults(self) -> None:
        payload = {"sample": "s", "source": "x.cu"}
        with tempfile.TemporaryDirectory() as directory:
            spec = load_sample_spec(write_spec(directory, payload))
        self.assertEqual(spec.success_marker, "RESULT=PASS")
        self.assertEqual(spec.expected, "unknown")
        self.assertEqual(spec.expected_redirects, None)
        self.assertEqual(spec.include_dirs, ())
        self.assertEqual(spec.libraries, ())
        self.assertEqual(spec.include_redirects, ())
        self.assertEqual(spec.optional_redirects, ())
        self.assertEqual(spec.optimization, "-O2")
        self.assertEqual(spec.schema_version, 1)
        self.assertIsNone(spec.cuda_samples_commit)

    def test_paths_are_relative_to_the_spec_not_the_cwd(self) -> None:
        payload = {"sample": "s", "source": "samples/demo.cu"}
        with tempfile.TemporaryDirectory() as directory:
            spec_dir = Path(directory) / "samples" / "v12.0"
            spec_dir.mkdir(parents=True)
            nested = spec_dir / "samples" / "demo.cu"
            nested.parent.mkdir(parents=True)
            nested.write_text("int main() { return 0; }\n", encoding="utf-8")
            spec_path = write_spec(str(spec_dir), payload)

            original_cwd = Path.cwd()
            os.chdir("/")
            try:
                spec = load_sample_spec(spec_path)
                self.assertTrue(spec.source.is_absolute())
                self.assertTrue(spec.source.is_file())
                self.assertEqual(spec.source.resolve(), nested.resolve())
            finally:
                os.chdir(original_cwd)

    def test_unknown_key_is_rejected(self) -> None:
        payload = {"sample": "s", "source": "x.cu", "cmake": True}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ProbeError, "unsupported key"):
                load_sample_spec(write_spec(directory, payload))

    def test_missing_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ProbeError, "'source' is required"):
                load_sample_spec(write_spec(directory, {"sample": "s"}))

    def test_bad_schema_version_is_rejected(self) -> None:
        payload = {"schema_version": 2, "sample": "s", "source": "x.cu"}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ProbeError, "unsupported schema_version"):
                load_sample_spec(write_spec(directory, payload))

    def test_bad_expected_is_rejected(self) -> None:
        payload = {"sample": "s", "source": "x.cu", "expected": "hopefully"}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ProbeError, "'expected' must be one of"):
                load_sample_spec(write_spec(directory, payload))

    def test_bad_expected_redirects_is_rejected(self) -> None:
        payload = {"sample": "s", "source": "x.cu", "expected_redirects": -1}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ProbeError, "expected_redirects"):
                load_sample_spec(write_spec(directory, payload))

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


class ForcedIncludeTests(unittest.TestCase):
    def test_resolves_against_include_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            include_dir = Path(directory) / "include"
            (include_dir / "shim").mkdir(parents=True)
            target = include_dir / "shim" / "host_math.h"
            target.write_text("// shim\n", encoding="utf-8")
            resolved = resolve_forced_includes(
                Path(directory) / "spec.json", (include_dir,), ("shim/host_math.h",)
            )
        self.assertEqual(resolved, (target,))

    def test_missing_forced_include_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ProbeError, "hipify_forced_includes"):
                resolve_forced_includes(
                    Path(directory) / "spec.json", (Path(directory),), ("absent.h",)
                )

    def test_spec_records_resolved_forced_includes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            include_dir = Path(directory) / "include"
            (include_dir / "cuda_samples_compat").mkdir(parents=True)
            shim = include_dir / "cuda_samples_compat" / "host_math_parse_shim.h"
            shim.write_text("// shim\n", encoding="utf-8")
            payload = {
                "sample": "s",
                "source": "x.cu",
                "include_dirs": ["include"],
                "hipify_forced_includes": ["cuda_samples_compat/host_math_parse_shim.h"],
            }
            spec = load_sample_spec(write_spec(directory, payload))
        self.assertEqual(spec.hipify_forced_includes, (shim,))


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
