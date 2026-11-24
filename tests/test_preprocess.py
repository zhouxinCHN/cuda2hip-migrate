from __future__ import annotations

import unittest

from cuda2hip_migrate.preprocess import (
    RULE_ID,
    apply_include_redirects,
    unmatched_redirects,
)

REDIRECTS = (("helper_cuda.h", "cuda_samples_compat/helper_cuda.h"),)


class IncludeRedirectTests(unittest.TestCase):
    def test_redirects_exact_include(self) -> None:
        source = "#include <helper_cuda.h>\nint main() {}\n"
        output, changes = apply_include_redirects(source, REDIRECTS)
        self.assertEqual(
            output, '#include "cuda_samples_compat/helper_cuda.h"\nint main() {}\n'
        )
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].rule_id, RULE_ID)
        self.assertEqual(changes[0].line, 1)

    def test_preserves_trailing_line_comment(self) -> None:
        source = "#include <helper_cuda.h>  // error check\n"
        output, changes = apply_include_redirects(source, REDIRECTS)
        self.assertEqual(
            output,
            '#include "cuda_samples_compat/helper_cuda.h"  // error check\n',
        )
        self.assertEqual(len(changes), 1)

    def test_other_header_is_unchanged(self) -> None:
        source = "#include <helper_string.h>\n"
        output, changes = apply_include_redirects(source, REDIRECTS)
        self.assertEqual(output, source)
        self.assertEqual(changes, [])

    def test_comment_and_string_are_unchanged(self) -> None:
        source = '// #include <helper_cuda.h>\nconst char *s = "#include <helper_cuda.h>";\n'
        output, changes = apply_include_redirects(source, REDIRECTS)
        self.assertEqual(output, source)
        self.assertEqual(changes, [])

    def test_already_quoted_include_is_unchanged(self) -> None:
        source = '#include "cuda_samples_compat/helper_cuda.h"\n'
        output, changes = apply_include_redirects(source, REDIRECTS)
        self.assertEqual(output, source)
        self.assertEqual(changes, [])

    def test_already_redirected_output_is_idempotent(self) -> None:
        source = "#include <helper_cuda.h>\n"
        once, first = apply_include_redirects(source, REDIRECTS)
        twice, second = apply_include_redirects(once, REDIRECTS)
        self.assertEqual(once, twice)
        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])

    def test_no_redirects_returns_input(self) -> None:
        source = "#include <helper_cuda.h>\n"
        output, changes = apply_include_redirects(source, ())
        self.assertEqual(output, source)
        self.assertEqual(changes, [])


class UnmatchedRedirectTests(unittest.TestCase):
    def test_declared_header_present(self) -> None:
        source = "#include <helper_cuda.h>  // comment\n"
        self.assertEqual(unmatched_redirects(source, REDIRECTS), ())

    def test_declared_header_absent(self) -> None:
        source = "#include <stdio.h>\n"
        self.assertEqual(
            unmatched_redirects(source, REDIRECTS), ("helper_cuda.h",)
        )

    def test_commented_include_does_not_count(self) -> None:
        source = "// #include <helper_cuda.h>\n"
        self.assertEqual(
            unmatched_redirects(source, REDIRECTS), ("helper_cuda.h",)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
