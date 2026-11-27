from __future__ import annotations

import unittest
from unittest import mock

from rocm10_samples_compat import environment as environment_module
from rocm10_samples_compat.environment import strict_environment_errors


def environment(rocm_version: str | None, gfx: list[str], missing: list[str]) -> dict:
    return {
        "expected": {
            "rocm": "10.0.0",
            "cuda_toolkit": "12.9",
            "cuda_samples_tag": "v12.9",
            "cuda_samples_commit": "unused",
        },
        "detected": {
            "rocm_version": rocm_version,
            "gfx_architectures": gfx,
            "tools": {},
            "hipify_versions": None,
        },
        "missing_tools": missing,
    }


class EnvironmentTests(unittest.TestCase):
    def test_valid_environment(self) -> None:
        self.assertEqual(
            strict_environment_errors(environment("10.0.0", ["gfx1100"], [])), []
        )

    def test_wrong_rocm_version(self) -> None:
        errors = strict_environment_errors(environment("7.2.4", ["gfx1100"], []))
        self.assertTrue(any("ROCm 10.0.0 required" in error for error in errors))

    def test_missing_tools_and_gfx(self) -> None:
        errors = strict_environment_errors(environment("10.0.0", [], ["hipcc"]))
        self.assertTrue(any("hipcc" in error for error in errors))
        self.assertTrue(any("no gfx architecture" in error for error in errors))


class PrimaryArchitectureTests(unittest.TestCase):
    def test_preserves_rocminfo_agent_order(self) -> None:
        output = (
            "  Name:                    gfx1201\n"
            "  Name:                    gfx1036\n"
        )
        with mock.patch.object(environment_module.shutil, "which", return_value="/bin/rocminfo"), \
             mock.patch.object(environment_module, "_run", return_value=(0, output)):
            self.assertEqual(
                environment_module._gfx_architectures(), ("gfx1201", "gfx1036")
            )
            self.assertEqual(environment_module.primary_gfx_architecture(), "gfx1201")


if __name__ == "__main__":
    unittest.main(verbosity=2)
