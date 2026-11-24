from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cuda2hip_migrate.runner import ProbeError, run_source_file, source_name


class SourceNameTests(unittest.TestCase):
    def test_uses_stem(self) -> None:
        self.assertEqual(source_name(Path("/tmp/samples/vectorAdd.cu")), "vectorAdd")

    def test_sanitizes_unsafe_characters(self) -> None:
        self.assertEqual(source_name(Path("my sample 1.cu")), "my_sample_1")

    def test_rejects_empty_name(self) -> None:
        with self.assertRaisesRegex(ProbeError, "cannot derive an artifact name"):
            source_name(Path(".cu"))


class SourceRunValidationTests(unittest.TestCase):
    def test_missing_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "absent.cu"
            with self.assertRaisesRegex(ProbeError, "source file not found"):
                run_source_file(missing)

    def test_empty_success_marker_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.cu"
            source.write_text("int main() { return 0; }\n", encoding="utf-8")
            with self.assertRaisesRegex(ProbeError, "must not be empty"):
                run_source_file(source, success_marker="")


if __name__ == "__main__":
    unittest.main(verbosity=2)
