"""Tests for pre-HIPIFY include routing."""

from __future__ import annotations

import unittest

from rocm10_samples_compat.redirects import (
    DEFAULT_REDIRECTS,
    Redirect,
    apply_redirects,
    matched_header,
    unmatched_redirects,
)

HELPER = Redirect(
    header="helper_cuda.h",
    replacement="rocm10_compat/helper_cuda.h",
    reason="test",
)


class ApplyRedirectTests(unittest.TestCase):
    def test_rewrites_angle_include(self) -> None:
        text = "#include <helper_cuda.h>\n"
        updated, changes = apply_redirects(text, (HELPER,))
        self.assertEqual(updated, '#include "rocm10_compat/helper_cuda.h"\n')
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].line, 1)

    def test_rewrites_quoted_include(self) -> None:
        text = '#include "helper_cuda.h"\n'
        updated, _changes = apply_redirects(text, (HELPER,))
        self.assertEqual(updated, '#include "rocm10_compat/helper_cuda.h"\n')

    def test_preserves_trailing_comment(self) -> None:
        text = "#include <helper_cuda.h>  // error checking\n"
        updated, _changes = apply_redirects(text, (HELPER,))
        self.assertEqual(
            updated, '#include "rocm10_compat/helper_cuda.h"  // error checking\n'
        )

    def test_leaves_other_includes_alone(self) -> None:
        text = "#include <helper_functions.h>\n#include <cuda_runtime.h>\n"
        updated, changes = apply_redirects(text, (HELPER,))
        self.assertEqual(updated, text)
        self.assertEqual(changes, [])

    def test_ignores_comment(self) -> None:
        text = "// #include <helper_cuda.h>\n/*#include <helper_cuda.h>*/\n"
        updated, changes = apply_redirects(text, (HELPER,))
        self.assertEqual(updated, text)
        self.assertEqual(changes, [])

    def test_ignores_string_literal(self) -> None:
        text = 'const char *s = "#include <helper_cuda.h>";\n'
        updated, changes = apply_redirects(text, (HELPER,))
        self.assertEqual(updated, text)
        self.assertEqual(changes, [])

    def test_is_idempotent(self) -> None:
        text = "#include <helper_cuda.h>\n"
        once, _ = apply_redirects(text, (HELPER,))
        twice, changes = apply_redirects(once, (HELPER,))
        self.assertEqual(once, twice)
        self.assertEqual(changes, [])

    def test_no_redirects_is_a_no_op(self) -> None:
        text = "#include <helper_cuda.h>\n"
        updated, changes = apply_redirects(text, ())
        self.assertEqual(updated, text)
        self.assertEqual(changes, [])


class UnmatchedRedirectTests(unittest.TestCase):
    def test_reports_declared_but_unused_redirect(self) -> None:
        missing = unmatched_redirects(["#include <stdio.h>\n"], DEFAULT_REDIRECTS)
        self.assertEqual([redirect.header for redirect in missing], ["helper_cuda.h"])

    def test_optional_redirect_may_stay_unmatched(self) -> None:
        optional = Redirect(
            header="helper_cuda_drvapi.h",
            replacement="rocm10_compat/helper_cuda_drvapi.h",
            reason="test",
            optional=True,
        )
        self.assertEqual(unmatched_redirects(["#include <stdio.h>\n"], (optional,)), ())

    def test_matched_across_files(self) -> None:
        texts = ["#include <stdio.h>\n", "#include <helper_cuda.h>\n"]
        self.assertEqual(unmatched_redirects(texts, (HELPER,)), ())

    def test_header_extraction(self) -> None:
        self.assertEqual(matched_header("#include <helper_cuda.h>"), "helper_cuda.h")
        self.assertEqual(matched_header('#include "helper_cuda.h"'), "helper_cuda.h")
        self.assertIsNone(matched_header("#pragma once"))


if __name__ == "__main__":
    unittest.main()
