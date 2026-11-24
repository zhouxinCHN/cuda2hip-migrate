"""Checks that the committed sample specs are internally consistent.

These run wherever the repository is checked out. Source-existence checks are
skipped when the external cuda-samples checkout is absent, which is the normal
case in CI.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from cuda2hip_migrate.runner import load_sample_spec

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_DIR = REPO_ROOT / "samples" / "cuda-samples-v12.0"
CHECKOUT = REPO_ROOT / "third-party" / "cuda-samples"


def committed_specs() -> list[Path]:
    return sorted(SPEC_DIR.glob("*.json"))


class CommittedSpecTests(unittest.TestCase):
    def test_specs_are_present(self) -> None:
        self.assertTrue(committed_specs(), "no committed sample specs found")

    def test_specs_load_and_use_current_schema(self) -> None:
        for path in committed_specs():
            with self.subTest(spec=path.name):
                spec = load_sample_spec(path)
                self.assertEqual(spec.schema_version, 1)

    def test_sample_name_matches_file_name(self) -> None:
        for path in committed_specs():
            with self.subTest(spec=path.name):
                spec = load_sample_spec(path)
                self.assertEqual(spec.sample.rsplit("/", 1)[-1], path.stem)

    def test_every_spec_pins_the_checkout_commit(self) -> None:
        for path in committed_specs():
            with self.subTest(spec=path.name):
                spec = load_sample_spec(path)
                self.assertIsNotNone(spec.cuda_samples_commit)
                self.assertRegex(str(spec.cuda_samples_commit), r"^[0-9a-f]{40}$")

    def test_redirect_targets_resolve_inside_declared_include_dirs(self) -> None:
        for path in committed_specs():
            spec = load_sample_spec(path)
            for _header, target in spec.include_redirects:
                with self.subTest(spec=path.name, target=target):
                    resolved = [directory / target for directory in spec.include_dirs]
                    self.assertTrue(
                        any(candidate.is_file() for candidate in resolved),
                        f"{target} not found under any include_dir of {path.name}",
                    )

    def test_redirect_targets_are_declared_by_expected_count(self) -> None:
        for path in committed_specs():
            spec = load_sample_spec(path)
            with self.subTest(spec=path.name):
                if spec.include_redirects:
                    self.assertEqual(spec.expected_redirects, 1)
                else:
                    self.assertEqual(spec.expected_redirects, 0)


@unittest.skipUnless(
    (CHECKOUT / "Samples").is_dir(), "cuda-samples checkout is not present"
)
class CommittedSpecCheckoutTests(unittest.TestCase):
    def test_sources_exist(self) -> None:
        for path in committed_specs():
            with self.subTest(spec=path.name):
                spec = load_sample_spec(path)
                self.assertTrue(spec.source.is_file(), spec.source)

    def test_source_paths_point_into_the_checkout(self) -> None:
        for path in committed_specs():
            with self.subTest(spec=path.name):
                spec = load_sample_spec(path)
                self.assertTrue(
                    spec.source.resolve().is_relative_to(CHECKOUT.resolve()),
                    spec.source,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
