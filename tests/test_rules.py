"""Positive, negative and idempotence tests for every deterministic rule."""

from __future__ import annotations

import unittest

from rocm10_samples_compat.rules import RULE_IDS, apply_post_hipify_rules, rule_hits
from rocm10_samples_compat.rules import cublas, hostmalloc, profiler


class CublasIncludeRuleTests(unittest.TestCase):
    def test_rewrites_hipify_include(self) -> None:
        text = "#include <hipblas.h>\nint main() { return 0; }\n"
        updated, changes = cublas.apply(text)
        self.assertIn("#include <hipblas/hipblas.h>", updated)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].rule_id, cublas.RULE_ID)
        self.assertEqual(changes[0].line, 1)

    def test_leaves_correct_include_alone(self) -> None:
        text = "#include <hipblas/hipblas.h>\n"
        updated, changes = cublas.apply(text)
        self.assertEqual(updated, text)
        self.assertEqual(changes, [])

    def test_ignores_comment_and_string(self) -> None:
        text = (
            "// #include <hipblas.h>\n"
            'const char *doc = "#include <hipblas.h>";\n'
            "/* #include <hipblas.h> */\n"
        )
        updated, changes = cublas.apply(text)
        self.assertEqual(updated, text)
        self.assertEqual(changes, [])

    def test_is_idempotent(self) -> None:
        text = "#include <hipblas.h>\n"
        once, _ = cublas.apply(text)
        twice, changes = cublas.apply(once)
        self.assertEqual(once, twice)
        self.assertEqual(changes, [])


class HostMallocArityRuleTests(unittest.TestCase):
    def test_removes_redundant_zero_argument(self) -> None:
        text = "hipHostMalloc(&ptr, size, 0, hipHostMallocDefault);\n"
        updated, changes = hostmalloc.apply(text)
        self.assertEqual(updated, "hipHostMalloc(&ptr, size, hipHostMallocDefault);\n")
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].rule_id, hostmalloc.RULE_ID)

    def test_keeps_unexpected_four_argument_form(self) -> None:
        text = "hipHostMalloc(&ptr, size, 4, hipHostMallocDefault);\n"
        updated, changes = hostmalloc.apply(text)
        self.assertEqual(updated, text)
        self.assertEqual(changes, [])

    def test_keeps_three_argument_form(self) -> None:
        text = "hipHostMalloc(&ptr, size, hipHostMallocDefault);\n"
        updated, changes = hostmalloc.apply(text)
        self.assertEqual(updated, text)
        self.assertEqual(changes, [])

    def test_ignores_comment_and_string(self) -> None:
        text = (
            "// hipHostMalloc(&p, 1, 0, hipHostMallocDefault);\n"
            'const char *doc = "hipHostMalloc(&p, 1, 0, hipHostMallocDefault)";\n'
        )
        updated, changes = hostmalloc.apply(text)
        self.assertEqual(updated, text)
        self.assertEqual(changes, [])

    def test_ignores_member_call(self) -> None:
        text = "wrapper.hipHostMalloc(&ptr, size, 0, hipHostMallocDefault);\n"
        updated, changes = hostmalloc.apply(text)
        self.assertEqual(updated, text)
        self.assertEqual(changes, [])

    def test_handles_nested_arguments(self) -> None:
        text = "hipHostMalloc(&ptr, sizeof(float) * n, 0, hipHostMallocDefault);\n"
        updated, _changes = hostmalloc.apply(text)
        self.assertEqual(updated, "hipHostMalloc(&ptr, sizeof(float) * n, hipHostMallocDefault);\n")

    def test_is_idempotent(self) -> None:
        text = "hipHostMalloc(&ptr, size, 0, hipHostMallocDefault);\n"
        once, _ = hostmalloc.apply(text)
        twice, changes = hostmalloc.apply(once)
        self.assertEqual(once, twice)
        self.assertEqual(changes, [])


class RegistryTests(unittest.TestCase):
    def test_registry_order_is_stable_and_unique(self) -> None:
        self.assertEqual(len(RULE_IDS), len(set(RULE_IDS)))
        self.assertEqual(RULE_IDS, (cublas.RULE_ID, hostmalloc.RULE_ID, profiler.RULE_ID))

    def test_registry_is_idempotent(self) -> None:
        text = (
            "#include <hipblas.h>\n"
            "hipHostMalloc(&ptr, size, 0, hipHostMallocDefault);\n"
        )
        once, first = apply_post_hipify_rules(text)
        twice, second = apply_post_hipify_rules(once)
        self.assertEqual(once, twice)
        self.assertEqual(second, [])
        self.assertEqual(rule_hits(first), {cublas.RULE_ID: 1, hostmalloc.RULE_ID: 1})


if __name__ == "__main__":
    unittest.main()
