"""Tests for sample specs and the packaged overlay contract."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from rocm10_samples_compat.pipeline import overlay_directory
from rocm10_samples_compat.spec import ADAPTATION_DEFINES, SpecError, load_spec, spec_names

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "rocm10_samples_compat"
OVERLAY_HEADER = PACKAGE_ROOT / "overlay" / "rocm10_compat" / "helper_cuda.h"


class SpecTests(unittest.TestCase):
    def test_every_packaged_spec_loads(self) -> None:
        self.assertTrue(spec_names())
        for name in spec_names():
            spec = load_spec(name)
            self.assertEqual(spec.name, name)
            self.assertIn(spec.c3_mode, ("marker", "exit-status"))

    def test_p0_samples_have_specs(self) -> None:
        expected = {
            "deviceQuery",
            "vectorAdd",
            "matrixMul",
            "asyncAPI",
            "simpleStreams",
            "scalarProd",
        }
        self.assertTrue(expected.issubset(set(spec_names())))

    def test_unknown_spec_is_reported(self) -> None:
        with self.assertRaises(SpecError):
            load_spec("doesNotExist")

    def test_adaptation_define_pins_cuda_baseline(self) -> None:
        self.assertIn("CUDART_VERSION=12090", ADAPTATION_DEFINES)


class OverlayContractTests(unittest.TestCase):
    """The parse branch and the build branch must declare the same surface.

    This pins names, arity and the parameters that are intentionally different.
    It does not prove the two branches behave identically: the parse branch has
    no bodies at all, and the collector types differ by design (the CUDA-side
    call sites pass ``cudaError_t``, an unscoped enum, where the build branch
    takes ``hipError_t``).
    """

    #: Function name -> parameter count, for both branches.
    EXPECTED_ARITY = {
        "check": 4,
        "getLastError": 0,
        "getLastCudaError": 1,
        "_ConvertSMVer2Cores": 2,
        "_ConvertSMVer2ArchName": 2,
        "gpuDeviceInit": 1,
        "gpuGetMaxGflopsDeviceId": 0,
        "findCudaDevice": 2,
        "checkCudaCapabilities": 2,
    }

    SIGNATURE = re.compile(r"inline\s+(?P<head>[^;{()]*?)\((?P<params>[^;{]*?)\)", re.S)

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = OVERLAY_HEADER.read_text(encoding="utf-8")
        cls.parse_branch = cls.text.split("#else  // hipify-clang parse")[1].split(
            "#endif  // __HIPCC__"
        )[0]
        cls.build_branch = cls.text.split("#if defined(__HIPCC__)", 2)[2]

    @classmethod
    def _signatures(cls, fragment: str) -> dict[str, tuple[str, list[str]]]:
        signatures: dict[str, tuple[str, list[str]]] = {}
        for match in cls.SIGNATURE.finditer(fragment):
            head = " ".join(match.group("head").split())
            identifiers = re.findall(r"[A-Za-z_]\w*", head)
            if not identifiers:
                continue
            name = identifiers[-1]
            if name not in cls.EXPECTED_ARITY:
                continue
            return_type = head[: head.rfind(name)].strip()
            params = " ".join(match.group("params").split())
            parameters = [] if not params else [item.strip() for item in params.split(",")]
            signatures[name] = (return_type, parameters)
        return signatures

    def test_both_branches_declare_the_same_names(self) -> None:
        self.assertEqual(
            set(self._signatures(self.parse_branch)),
            set(self._signatures(self.build_branch)),
        )

    def test_arity_is_pinned_for_every_helper(self) -> None:
        for branch in (self.parse_branch, self.build_branch):
            arity = {
                name: len(params)
                for name, (_ret, params) in self._signatures(branch).items()
            }
            self.assertEqual(arity, self.EXPECTED_ARITY)

    def test_collector_type_difference_is_explicit(self) -> None:
        """The documented difference must stay deliberate, not drift silently."""
        parse_params = self._signatures(self.parse_branch)["check"][1]
        build_params = self._signatures(self.build_branch)["check"][1]
        self.assertTrue(parse_params[0].startswith("int result"))
        self.assertTrue(build_params[0].startswith("hipError_t result"))

    def test_parse_branch_has_no_bodies(self) -> None:
        """A body in the parse branch would be compiled only by hipify-clang."""
        self.assertNotIn("{", self.parse_branch.split("inline")[-1])

    def test_hipcc_branch_is_guarded(self) -> None:
        self.assertTrue(self.text.count("#if defined(__HIPCC__)") >= 1)
        self.assertIn("checkCudaErrors", self.text)

    def test_overlay_directory_contains_the_header(self) -> None:
        header = overlay_directory() / "rocm10_compat" / "helper_cuda.h"
        self.assertTrue(header.is_file())


if __name__ == "__main__":
    unittest.main()
