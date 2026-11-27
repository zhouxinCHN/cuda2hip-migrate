from __future__ import annotations

import unittest

from rocm10_samples_compat.manifest import BATCH_ORDER, load_manifest, summary


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = load_manifest()

    def test_pins_fixed_versions(self) -> None:
        fixed = self.manifest["fixed_decisions"]
        self.assertEqual(fixed["target_rocm"], "10.0.0")
        self.assertEqual(fixed["cuda_toolkit"], "12.9")
        self.assertEqual(fixed["cuda_samples_tag"], "v12.9")
        self.assertEqual(fixed["deliverable_form"], "skills")

    def test_contains_all_v129_samples(self) -> None:
        self.assertEqual(self.manifest["counts"]["samples"], 188)
        self.assertEqual(len(self.manifest["samples"]), 188)
        names = {sample["name"] for sample in self.manifest["samples"]}
        self.assertIn("vectorAdd", names)
        self.assertIn("matrixMul", names)
        self.assertIn("simpleTexture", names)
        self.assertNotIn("cudaTensorCoreGemm_v13", names)

    def test_batches_cover_known_samples(self) -> None:
        names = {sample["name"] for sample in self.manifest["samples"]}
        self.assertEqual(tuple(self.manifest["batches"]), BATCH_ORDER)
        for batch, members in self.manifest["batches"].items():
            self.assertTrue(set(members) <= names, batch)

    def test_expected_baseline_order(self) -> None:
        self.assertEqual(
            self.manifest["batches"]["P0_baseline"],
            [
                "deviceQuery",
                "vectorAdd",
                "matrixMul",
                "asyncAPI",
                "simpleStreams",
                "scalarProd",
            ],
        )

    def test_summary_has_counts(self) -> None:
        result = summary(self.manifest)
        self.assertEqual(result["counts"]["samples"], 188)
        self.assertEqual(result["batches"]["P0_baseline"], 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
