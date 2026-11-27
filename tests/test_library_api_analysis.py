"""Tests for CUDA library sample API classification."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_cuda_library_samples import family_for, is_library_api_call


class LibraryApiAnalysisTests(unittest.TestCase):
    def test_recognizes_c_apis_and_families(self) -> None:
        cases = {
            "cublasSgemm": "cuBLAS",
            "cublasLtMatmul": "cuBLASLt",
            "cublasMpMatmul": "cuBLASMp",
            "cublasdxCreateDescriptor": "cuBLASDx",
            "cufftExecC2C": "cuFFT",
            "cufftMpAttachComm": "cuFFTMp",
            "cusolverDnSgesvd": "cuSOLVER",
            "cusolverMgGetrf": "cuSOLVERMg",
            "cusolverMpPotrf": "cuSOLVERMp",
            "cusparseSpMV": "cuSPARSE",
            "cusparseLtMatmul": "cuSPARSELt",
            "cutensorContract": "cuTENSOR",
            "cutensorMgCreate": "cuTENSORMg",
            "cutensorMpCreate": "cuTENSORMp",
            "cudssExecute": "cuDSS",
            "cuestCreate": "cuEST",
            "nvcompBatchedLZ4CompressAsync": "nvCOMP",
            "nvcompdxCreateDescriptor": "nvCOMPDx",
            "nvtiffDecode": "nvTIFF",
            "nvMatmulHeuristicsCreate": "nvMatmulHeuristics",
        }
        for name, family in cases.items():
            with self.subTest(name=name):
                self.assertTrue(is_library_api_call(name))
                self.assertEqual(family_for(name), family)

    def test_accepts_real_snake_case_and_device_apis(self) -> None:
        self.assertTrue(is_library_api_call("nvtiff_status_string"))
        self.assertTrue(is_library_api_call("curand_uniform"))
        self.assertTrue(is_library_api_call("cublasMpGemm_bufferSize"))

    def test_rejects_helpers_and_type_casts(self) -> None:
        names = (
            "cufft_check",
            "cusolver_finalize",
            "cudss_finalize",
            "cuestCheck",
            "cublasdx_dgemm_emulation",
            "cufftdx_execution",
            "nvjpegBackend_t",
            "cublasMpEmulationStrategy_t",
        )
        for name in names:
            with self.subTest(name=name):
                self.assertFalse(is_library_api_call(name))


if __name__ == "__main__":
    unittest.main()
