"""Tests for CUDA-root resolution used by hipify-clang."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rocm10_samples_compat import cuda_root


class VersionFormattingTests(unittest.TestCase):
    def test_reformats_cuda_version(self) -> None:
        self.assertEqual(cuda_root._format_version(12000), "12.0")
        self.assertEqual(cuda_root._format_version(12090), "12.9")
        self.assertEqual(cuda_root._format_version(None), "unknown")

    def test_reads_version_from_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            include = Path(directory)
            (include / "cuda.h").write_text(
                "#define CUDA_VERSION 12090  /* 12.9 */\n", encoding="utf-8"
            )
            self.assertEqual(cuda_root._cuda_version(include), 12090)

    def test_missing_header_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(cuda_root._cuda_version(Path(directory)))


class ExplicitRootTests(unittest.TestCase):
    def test_loads_a_complete_explicit_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "include").mkdir()
            (root / "include" / "cuda_runtime.h").write_text("", encoding="utf-8")
            (root / "include" / "cuda.h").write_text(
                "#define CUDA_VERSION 12090\n", encoding="utf-8"
            )
            (root / "nvvm" / "libdevice").mkdir(parents=True)
            (root / "nvvm" / "libdevice" / "libdevice.10.bc").write_text(
                "", encoding="utf-8"
            )
            (root / "bin").mkdir()
            loaded = cuda_root.load_cuda_root(root)
            self.assertEqual(loaded.version, 12090)
            self.assertFalse(loaded.synthetic)

    def test_rejects_incomplete_explicit_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(cuda_root.CudaRootError):
                cuda_root.load_cuda_root(Path(directory))


class SyntheticRootTests(unittest.TestCase):
    def test_error_when_no_installation_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(cuda_root, "_candidate_roots", return_value=[]), \
                 mock.patch.object(cuda_root, "SYSTEM_INCLUDE", Path(directory) / "nope"), \
                 mock.patch.object(cuda_root, "SYSTEM_LIBDEVICE", Path(directory) / "nope2"):
                with self.assertRaises(cuda_root.CudaRootError):
                    cuda_root.resolve_cuda_root(Path(directory))

    def test_synthetic_root_links_include_libdevice_and_bin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            system_include = root / "sys-include"
            system_include.mkdir()
            (system_include / "cuda_runtime.h").write_text("", encoding="utf-8")
            (system_include / "cuda.h").write_text("#define CUDA_VERSION 12000\n", encoding="utf-8")
            libdevice = root / "libdevice"
            libdevice.mkdir()
            (libdevice / "libdevice.10.bc").write_text("", encoding="utf-8")
            toolkit_bin = root / "toolkit" / "bin"
            toolkit_bin.mkdir(parents=True)

            with mock.patch.object(cuda_root, "_candidate_roots", return_value=[]), \
                 mock.patch.object(cuda_root, "SYSTEM_INCLUDE", system_include), \
                 mock.patch.object(cuda_root, "SYSTEM_LIBDEVICE", libdevice), \
                 mock.patch.object(cuda_root, "_system_bin_dir", return_value=toolkit_bin):
                resolved = cuda_root.resolve_cuda_root(root / "work")

            self.assertTrue(resolved.synthetic)
            self.assertEqual(resolved.version, 12000)
            self.assertEqual(resolved.version_string, "12.0")
            self.assertTrue((resolved.path / "include").is_symlink())
            self.assertTrue((resolved.path / "nvvm" / "libdevice" / "libdevice.10.bc").is_symlink())
            self.assertTrue((resolved.path / "bin").is_symlink())

    def test_synthetic_root_without_bin_is_rejected(self) -> None:
        """A bin/-less root makes hipify report CUDA SDK version 0 and fail."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            system_include = root / "sys-include"
            system_include.mkdir()
            (system_include / "cuda_runtime.h").write_text("", encoding="utf-8")
            libdevice = root / "libdevice"
            libdevice.mkdir()
            (libdevice / "libdevice.10.bc").write_text("", encoding="utf-8")

            with mock.patch.object(cuda_root, "_candidate_roots", return_value=[]), \
                 mock.patch.object(cuda_root, "SYSTEM_INCLUDE", system_include), \
                 mock.patch.object(cuda_root, "SYSTEM_LIBDEVICE", libdevice), \
                 mock.patch.object(cuda_root, "_system_bin_dir", return_value=None):
                with self.assertRaises(cuda_root.CudaRootError) as context:
                    cuda_root.resolve_cuda_root(root / "work")
            self.assertIn("bin/", str(context.exception))
            self.assertFalse((root / "work" / "cuda-root").exists())

    def test_standard_layout_without_bin_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "include").mkdir()
            (root / "include" / "cuda_runtime.h").write_text("", encoding="utf-8")
            (root / "nvvm" / "libdevice").mkdir(parents=True)
            (root / "nvvm" / "libdevice" / "libdevice.10.bc").write_text(
                "", encoding="utf-8"
            )

            with mock.patch.object(
                cuda_root, "_candidate_roots", return_value=[("test", root)]
            ), mock.patch.object(cuda_root, "SYSTEM_INCLUDE", root / "nope"), \
                 mock.patch.object(cuda_root, "SYSTEM_LIBDEVICE", root / "nope2"):
                with self.assertRaises(cuda_root.CudaRootError):
                    cuda_root.resolve_cuda_root(root / "work")

    def test_standard_layout_is_preferred(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "include").mkdir()
            (root / "include" / "cuda_runtime.h").write_text("", encoding="utf-8")
            (root / "include" / "cuda.h").write_text(
                "#define CUDA_VERSION 12090\n", encoding="utf-8"
            )
            (root / "nvvm" / "libdevice").mkdir(parents=True)
            (root / "nvvm" / "libdevice" / "libdevice.10.bc").write_text("", encoding="utf-8")
            (root / "bin").mkdir()

            with mock.patch.object(cuda_root, "_candidate_roots", return_value=[("test", root)]):
                resolved = cuda_root.resolve_cuda_root(root / "work")

            self.assertFalse(resolved.synthetic)
            self.assertEqual(resolved.version, 12090)


if __name__ == "__main__":
    unittest.main()
