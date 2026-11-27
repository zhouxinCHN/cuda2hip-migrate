"""Optional end-to-end test for generic conversion on a ROCm host.

Skipped on machines without hipify-clang/hipcc or the pinned sample checkout,
so the default CI stays a CPU-only unit test suite. On the Docker target it
performs a real HIPIFY conversion, hipcc build and execution.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from rocm10_samples_compat.environment import primary_gfx_architecture
from rocm10_samples_compat.project import MigrationRequest, convert_project

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = (
    REPO_ROOT
    / "third-party"
    / "cuda-samples"
    / "Samples"
    / "0_Introduction"
    / "vectorAdd"
    / "vectorAdd.cu"
)
COMMON = REPO_ROOT / "third-party" / "cuda-samples" / "Common"


class GenericConversionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("hipify-clang") is None or shutil.which("hipcc") is None:
            raise unittest.SkipTest("hipify-clang or hipcc is not installed")
        if shutil.which("cmake") is None:
            raise unittest.SkipTest("cmake is not installed")
        if not SAMPLE.is_file() or not COMMON.is_dir():
            raise unittest.SkipTest("pinned cuda-samples checkout is not available")

    def test_generic_convert_builds_and_runs_vector_add(self) -> None:
        arch = primary_gfx_architecture()
        if arch is None:
            raise unittest.SkipTest("rocminfo reported no GPU architecture")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "input"
            output_root = root / "output"
            source_root.mkdir()
            shutil.copy2(SAMPLE, source_root / "vectorAdd.cu")
            (source_root / "nested").mkdir()
            (source_root / "nested" / "asset.txt").write_text(
                "preserved\n", encoding="utf-8"
            )
            database = source_root / "compile_commands.json"
            database.write_text(
                json.dumps(
                    [
                        {
                            "directory": str(source_root),
                            "file": "vectorAdd.cu",
                            "arguments": [
                                "nvcc",
                                f"-I{COMMON}",
                                "-std=c++17",
                                "-c",
                                "vectorAdd.cu",
                                "-o",
                                "vectorAdd.o",
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            outcome = convert_project(
                MigrationRequest(
                    input_root=source_root,
                    output_root=output_root,
                    compile_commands=database,
                )
            )
            self.assertEqual(outcome.status, "completed", outcome.report["issues"])
            self.assertEqual(outcome.report["blocking_issues"], 0)
            self.assertTrue(
                (output_root / "src" / "nested" / "asset.txt").is_file()
            )
            converted = (output_root / "src" / "vectorAdd.cu").read_text(
                encoding="utf-8"
            )
            self.assertIn("#include <hip/hip_runtime.h>", converted)
            self.assertIn("rocm10_compat/helper_cuda.h", converted)

            requirements = outcome.report["requirements"]
            self.assertIn("common-helpers", requirements["features"])
            overlay_include = Path(requirements["include_dirs"][0])
            binary = root / "vectorAdd"
            command = [
                "hipcc",
                "-O3",
                "-std=c++17",
                f"--offload-arch={arch}",
                *[
                    f"-D{item}"
                    for item in requirements["compile_definitions"]
                ],
                "-I",
                str(COMMON),
                "-I",
                str(overlay_include),
                str(output_root / "src" / "vectorAdd.cu"),
                "-o",
                str(binary),
            ]
            build = subprocess.run(
                command, capture_output=True, text=True, check=False, timeout=300
            )
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            run = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False, timeout=120
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("Test PASSED", run.stdout)

    def test_header_reached_through_include_dir_is_translated_and_builds(self) -> None:
        arch = primary_gfx_architecture()
        if arch is None:
            raise unittest.SkipTest("rocminfo reported no GPU architecture")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "input"
            output_root = root / "output"
            include_root = source_root / "include"
            include_root.mkdir(parents=True)
            (include_root / "device_math.cuh").write_text(
                "#pragma once\n"
                "#include <cuda_runtime.h>\n"
                "__device__ inline float twice(float x) { return x * 2.0f; }\n",
                encoding="utf-8",
            )
            (source_root / "main.cu").write_text(
                "#include <cstdio>\n"
                "#include <cuda_runtime.h>\n"
                '#include "device_math.cuh"\n'
                "__global__ void scale(float *data) {\n"
                "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
                "  if (i < 2) data[i] = twice(data[i]);\n"
                "}\n"
                "int main() {\n"
                "  float host[2] = {1.0f, 2.0f};\n"
                "  float *device = nullptr;\n"
                "  cudaMalloc(&device, sizeof(host));\n"
                "  cudaMemcpy(device, host, sizeof(host), cudaMemcpyHostToDevice);\n"
                "  scale<<<1, 2>>>(device);\n"
                "  cudaMemcpy(host, device, sizeof(host), cudaMemcpyDeviceToHost);\n"
                "  cudaFree(device);\n"
                '  printf("HEADER %s\\n", '
                '(host[0] == 2.0f && host[1] == 4.0f) ? "PASS" : "FAIL");\n'
                "  return (host[0] == 2.0f && host[1] == 4.0f) ? 0 : 1;\n"
                "}\n",
                encoding="utf-8",
            )
            database = source_root / "compile_commands.json"
            database.write_text(
                json.dumps(
                    [
                        {
                            "directory": str(source_root),
                            "file": "main.cu",
                            "arguments": [
                                "nvcc",
                                "-Iinclude",
                                "-std=c++17",
                                "-c",
                                "main.cu",
                                "-o",
                                "main.o",
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            outcome = convert_project(
                MigrationRequest(
                    input_root=source_root,
                    output_root=output_root,
                    compile_commands=database,
                )
            )
            self.assertEqual(outcome.status, "completed", outcome.report["issues"])
            self.assertEqual(outcome.report["translated_headers"], 1)
            translated_header = (
                output_root / "src" / "include" / "device_math.cuh"
            ).read_text(encoding="utf-8")
            self.assertIn("#include <hip/hip_runtime.h>", translated_header)
            self.assertNotIn("#include <cuda_runtime.h>", translated_header)

            requirements = outcome.report["requirements"]
            binary = root / "header-test"
            include_args: list[str] = []
            for include_dir in [
                output_root / "src" / "include",
                *[Path(item) for item in requirements["include_dirs"]],
            ]:
                include_args.extend(["-I", str(include_dir)])
            command = [
                "hipcc",
                "-O3",
                "-std=c++17",
                f"--offload-arch={arch}",
                *[
                    f"-D{item}"
                    for item in requirements["compile_definitions"]
                ],
                *include_args,
                str(output_root / "src" / "main.cu"),
                "-o",
                str(binary),
            ]
            build = subprocess.run(
                command, capture_output=True, text=True, check=False, timeout=300
            )
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            run = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False, timeout=120
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("HEADER PASS", run.stdout)

    def test_auto_configure_generates_compile_database_and_builds(self) -> None:
        arch = primary_gfx_architecture()
        if arch is None:
            raise unittest.SkipTest("rocminfo reported no GPU architecture")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "input"
            output_root = root / "output"
            (source_root / "include").mkdir(parents=True)
            (source_root / "include" / "value.h").write_text(
                "#pragma once\n#define AUTO_VALUE 2.0f\n",
                encoding="utf-8",
            )
            (source_root / "main.cu").write_text(
                "#include <cstdio>\n"
                "#include <cuda_runtime.h>\n"
                "#include <value.h>\n"
                "__global__ void scale(float *data) {\n"
                "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
                "  if (i < 2) data[i] *= AUTO_VALUE;\n"
                "}\n"
                "int main() {\n"
                "  float host[2] = {1.0f, 2.0f};\n"
                "  float *device = nullptr;\n"
                "  cudaMalloc(&device, sizeof(host));\n"
                "  cudaMemcpy(device, host, sizeof(host), cudaMemcpyHostToDevice);\n"
                "  scale<<<1, 2>>>(device);\n"
                "  cudaMemcpy(host, device, sizeof(host), cudaMemcpyDeviceToHost);\n"
                "  cudaFree(device);\n"
                '  printf("AUTO %s\\n", '
                '(host[0] == 2.0f && host[1] == 4.0f) ? "PASS" : "FAIL");\n'
                "  return (host[0] == 2.0f && host[1] == 4.0f) ? 0 : 1;\n"
                "}\n",
                encoding="utf-8",
            )
            (source_root / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.20)\n"
                "project(auto_configure_probe LANGUAGES CXX CUDA)\n"
                "set(CMAKE_CUDA_ARCHITECTURES 52)\n"
                "add_executable(auto_configure_probe main.cu)\n"
                "target_compile_features(auto_configure_probe PRIVATE cxx_std_17 cuda_std_17)\n"
                "target_include_directories(auto_configure_probe PRIVATE include)\n",
                encoding="utf-8",
            )

            outcome = convert_project(
                MigrationRequest(
                    input_root=source_root,
                    output_root=output_root,
                )
            )
            self.assertEqual(outcome.status, "completed", outcome.report["issues"])
            self.assertEqual(outcome.report["compile_commands_source"], "configured")
            configure = outcome.report["configure"]  # type: ignore[index]
            self.assertEqual(configure["mode"], "configured")
            database = (
                output_root
                / ".tmp"
                / "input"
                / "cmake-build"
                / "compile_commands.json"
            )
            self.assertTrue(database.is_file())
            includes = outcome.report["units"][0]["includes"]  # type: ignore[index]
            self.assertIn(
                str(source_root / "include"),
                [item["path"] for item in includes],  # type: ignore[index]
            )

            requirements = outcome.report["requirements"]
            binary = root / "auto-configure-test"
            include_args: list[str] = []
            for include_dir in [
                output_root / "src" / "include",
                *[Path(item) for item in requirements["include_dirs"]],
            ]:
                include_args.extend(["-I", str(include_dir)])
            command = [
                "hipcc",
                "-O3",
                "-std=c++17",
                f"--offload-arch={arch}",
                *[
                    f"-D{item}"
                    for item in requirements["compile_definitions"]
                ],
                *include_args,
                str(output_root / "src" / "main.cu"),
                "-o",
                str(binary),
            ]
            build = subprocess.run(
                command, capture_output=True, text=True, check=False, timeout=300
            )
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            run = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False, timeout=120
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("AUTO PASS", run.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
