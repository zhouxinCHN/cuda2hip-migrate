from __future__ import annotations

import unittest

from cuda2hip_migrate.rules import apply_post_hipify_rules


class HostMallocArityRuleTests(unittest.TestCase):
    RULE = "HIP-HOSTMALLOC-ARITY-001"

    def test_removes_redundant_third_argument(self) -> None:
        source = "hipHostMalloc(&p, 4096, 0, hipHostMallocDefault);\n"
        output, changes = apply_post_hipify_rules(source)
        self.assertEqual(output, "hipHostMalloc(&p, 4096, hipHostMallocDefault);\n")
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].rule_id, self.RULE)

    def test_handles_multiline_call(self) -> None:
        source = (
            "hipHostMalloc(\n"
            "    reinterpret_cast<void **>(&p),\n"
            "    bytes,\n"
            "    0,\n"
            "    hipHostMallocDefault);\n"
        )
        output, changes = apply_post_hipify_rules(source)
        self.assertIn(
            "hipHostMalloc(reinterpret_cast<void **>(&p), bytes, hipHostMallocDefault);",
            output,
        )
        self.assertEqual(len(changes), 1)

    def test_three_argument_call_is_unchanged(self) -> None:
        source = "hipHostMalloc(&p, 4096, hipHostMallocDefault);\n"
        output, changes = apply_post_hipify_rules(source)
        self.assertEqual(output, source)
        self.assertEqual(changes, [])

    def test_non_zero_third_argument_is_unchanged(self) -> None:
        source = "hipHostMalloc(&p, 4096, 1, hipHostMallocDefault);\n"
        output, changes = apply_post_hipify_rules(source)
        self.assertEqual(output, source)
        self.assertEqual(changes, [])

    def test_other_flags_are_unchanged(self) -> None:
        source = "hipHostMalloc(&p, 4096, 0, hipHostMallocMapped);\n"
        output, changes = apply_post_hipify_rules(source)
        self.assertEqual(output, source)
        self.assertEqual(changes, [])

    def test_comment_is_unchanged(self) -> None:
        source = "// hipHostMalloc(&p, 4096, 0, hipHostMallocDefault);\n"
        output, changes = apply_post_hipify_rules(source)
        self.assertEqual(output, source)
        self.assertEqual(changes, [])

    def test_string_literal_is_unchanged(self) -> None:
        source = 'const char *s = "hipHostMalloc(&p, 4096, 0, hipHostMallocDefault);";\n'
        output, changes = apply_post_hipify_rules(source)
        self.assertEqual(output, source)
        self.assertEqual(changes, [])

    def test_different_identifier_is_unchanged(self) -> None:
        source = "my_hipHostMalloc(&p, 4096, 0, hipHostMallocDefault);\n"
        output, changes = apply_post_hipify_rules(source)
        self.assertEqual(output, source)
        self.assertEqual(changes, [])

    def test_member_call_is_unchanged(self) -> None:
        """A qualified name is not the HIP API and must not be rewritten."""
        for source in (
            "wrapper.hipHostMalloc(&p, 4096, 0, hipHostMallocDefault);\n",
            "wrapper->hipHostMalloc(&p, 4096, 0, hipHostMallocDefault);\n",
            "hip::hipHostMalloc(&p, 4096, 0, hipHostMallocDefault);\n",
        ):
            output, changes = apply_post_hipify_rules(source)
            self.assertEqual(output, source)
            self.assertEqual(changes, [])

    def test_rule_is_idempotent(self) -> None:
        source = "hipHostMalloc(&p, 4096, 0, hipHostMallocDefault);\n"
        once, first_changes = apply_post_hipify_rules(source)
        twice, second_changes = apply_post_hipify_rules(once)
        self.assertEqual(once, twice)
        self.assertEqual(len(first_changes), 1)
        self.assertEqual(second_changes, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
