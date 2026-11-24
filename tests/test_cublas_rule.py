from __future__ import annotations

import unittest

from cuda2hip_migrate.rules import apply_post_hipify_rules


class CublasIncludeRuleTests(unittest.TestCase):
    RULE = "HIP-CUBLAS-INCLUDE-001"

    def test_rewrites_hipify_include(self) -> None:
        source = "#include <cuda_runtime.h>\n#include <hipblas.h>\n"
        output, changes = apply_post_hipify_rules(source)
        self.assertEqual(
            output, "#include <cuda_runtime.h>\n#include <hipblas/hipblas.h>\n"
        )
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].rule_id, self.RULE)
        self.assertEqual(changes[0].line, 2)

    def test_preserves_trailing_whitespace(self) -> None:
        source = "#include <hipblas.h>   \nint main() {}\n"
        output, _changes = apply_post_hipify_rules(source)
        self.assertEqual(output, "#include <hipblas/hipblas.h>   \nint main() {}\n")

    def test_already_correct_include_is_unchanged(self) -> None:
        source = "#include <hipblas/hipblas.h>\n"
        output, changes = apply_post_hipify_rules(source)
        self.assertEqual(output, source)
        self.assertEqual(changes, [])

    def test_original_cuda_include_is_unchanged(self) -> None:
        source = "#include <cublas_v2.h>\n"
        output, changes = apply_post_hipify_rules(source)
        self.assertEqual(output, source)
        self.assertEqual(changes, [])

    def test_line_comment_is_unchanged(self) -> None:
        source = "// #include <hipblas.h>\n"
        output, changes = apply_post_hipify_rules(source)
        self.assertEqual(output, source)
        self.assertEqual(changes, [])

    def test_block_comment_is_unchanged(self) -> None:
        source = "/*\n#include <hipblas.h>\n*/\nint main() {}\n"
        output, changes = apply_post_hipify_rules(source)
        self.assertEqual(output, source)
        self.assertEqual(changes, [])

    def test_string_literal_is_unchanged(self) -> None:
        source = 'const char *line = "#include <hipblas.h>";\n'
        output, changes = apply_post_hipify_rules(source)
        self.assertEqual(output, source)
        self.assertEqual(changes, [])

    def test_non_directive_occurrence_is_unchanged(self) -> None:
        source = "int include_hipblas = 0;  // hipblas.h\n"
        output, changes = apply_post_hipify_rules(source)
        self.assertEqual(output, source)
        self.assertEqual(changes, [])

    def test_rule_is_idempotent(self) -> None:
        source = "#include <hipblas.h>\n"
        once, first_changes = apply_post_hipify_rules(source)
        twice, second_changes = apply_post_hipify_rules(once)
        self.assertEqual(once, twice)
        self.assertEqual(len(first_changes), 1)
        self.assertEqual(second_changes, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
