from __future__ import annotations

import unittest

from cuda2hip_migrate.runner import (
    ProbeError,
    build_flags_from_spec,
    probe_build_flags,
)


class BuildSpecTests(unittest.TestCase):
    def test_spec_with_libraries_becomes_link_flags(self) -> None:
        self.assertEqual(
            build_flags_from_spec("probe", {"libraries": ["hipblas"]}),
            ("-lhipblas",),
        )

    def test_spec_without_libraries_adds_no_flags(self) -> None:
        self.assertEqual(build_flags_from_spec("probe", {}), ())

    def test_unknown_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(ProbeError, "unsupported build spec key"):
            build_flags_from_spec("probe", {"include_dirs": ["/opt/rocm"]})

    def test_non_object_spec_is_rejected(self) -> None:
        with self.assertRaisesRegex(ProbeError, "must be a JSON object"):
            build_flags_from_spec("probe", ["hipblas"])

    def test_non_string_library_is_rejected(self) -> None:
        with self.assertRaisesRegex(ProbeError, "list of names"):
            build_flags_from_spec("probe", {"libraries": [7]})

    def test_probe_without_spec_has_no_flags(self) -> None:
        self.assertEqual(probe_build_flags("probe_malloc_memcpy"), ())

    def test_cublas_probe_declares_hipblas(self) -> None:
        self.assertEqual(probe_build_flags("probe_cublas_gemm"), ("-lhipblas",))


if __name__ == "__main__":
    unittest.main(verbosity=2)
