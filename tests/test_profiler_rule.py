"""Positive, negative and idempotence tests for HIP-PROFILER-ROCTX-001."""

from __future__ import annotations

import unittest

from rocm10_samples_compat.rules import profiler

SAMPLE = (
    "#include <hip/hip_runtime.h>\n"
    "\n"
    "int main() {\n"
    "  checkCudaErrors(hipProfilerStart());\n"
    "  run();\n"
    "  checkCudaErrors(hipProfilerStop());\n"
    "  return 0;\n"
    "}\n"
)


class ProfilerRuleTests(unittest.TestCase):
    def test_rewrites_both_calls(self) -> None:
        updated, changes = profiler.apply(SAMPLE)
        self.assertIn("checkCudaErrors(rocm10_compat::hipProfilerStart());", updated)
        self.assertIn("checkCudaErrors(rocm10_compat::hipProfilerStop());", updated)
        self.assertNotIn("(hipProfilerStart())", updated)
        self.assertNotIn("(hipProfilerStop())", updated)
        self.assertEqual([c.rule_id for c in changes], [profiler.RULE_ID] * 3)

    def test_adds_compat_include_after_runtime_include(self) -> None:
        updated, _changes = profiler.apply(SAMPLE)
        self.assertTrue(updated.startswith("#include <hip/hip_runtime.h>\n"))
        self.assertIn(profiler.COMPAT_INCLUDE, updated)
        self.assertLess(
            updated.index("#include <hip/hip_runtime.h>"),
            updated.index(profiler.COMPAT_INCLUDE),
        )
        self.assertEqual(updated.count(profiler.COMPAT_INCLUDE), 1)

    def test_anchors_on_runtime_api_include_too(self) -> None:
        text = "#include <hip/hip_runtime_api.h>\nhipProfilerStart();\n"
        updated, _changes = profiler.apply(text)
        self.assertLess(
            updated.index("#include <hip/hip_runtime_api.h>"),
            updated.index(profiler.COMPAT_INCLUDE),
        )

    def test_keeps_include_when_already_present(self) -> None:
        text = SAMPLE.replace(
            "#include <hip/hip_runtime.h>\n",
            "#include <hip/hip_runtime.h>\n" + profiler.COMPAT_INCLUDE + "\n",
        )
        updated, changes = profiler.apply(text)
        self.assertEqual(updated.count(profiler.COMPAT_INCLUDE), 1)
        self.assertEqual(len(changes), 2)  # the two calls only

    def test_tolerates_whitespace_inside_the_call(self) -> None:
        updated, changes = profiler.apply("hipProfilerStart( );\n")
        self.assertIn("rocm10_compat::hipProfilerStart();", updated)
        self.assertEqual(len(changes), 2)  # call + include

    def test_leaves_non_zero_argument_call_alone(self) -> None:
        text = "hipProfilerStart(0);\n"
        updated, changes = profiler.apply(text)
        self.assertEqual(updated, text)
        self.assertEqual(changes, [])

    def test_leaves_the_ext_api_alone(self) -> None:
        text = "hipProfilerStartExt(cb);\n"
        updated, changes = profiler.apply(text)
        self.assertEqual(updated, text)
        self.assertEqual(changes, [])

    def test_ignores_comment_string_and_literal(self) -> None:
        text = (
            "// hipProfilerStart();\n"
            'const char *doc = "hipProfilerStart();";\n'
            "/* hipProfilerStop(); */\n"
        )
        updated, changes = profiler.apply(text)
        self.assertEqual(updated, text)
        self.assertEqual(changes, [])

    def test_ignores_member_and_qualified_call(self) -> None:
        text = "obj.hipProfilerStart();\nns::hipProfilerStart();\nptr->hipProfilerStart();\n"
        updated, changes = profiler.apply(text)
        self.assertEqual(updated, text)
        self.assertEqual(changes, [])

    def test_records_line_numbers(self) -> None:
        _updated, changes = profiler.apply(SAMPLE)
        calls = [change for change in changes if change.before]
        self.assertEqual([change.line for change in calls], [4, 6])

    def test_is_idempotent(self) -> None:
        once, _first = profiler.apply(SAMPLE)
        twice, changes = profiler.apply(once)
        self.assertEqual(once, twice)
        self.assertEqual(changes, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
