# Generic conversion verification — ROCm 10.0.0 / gfx1201

Status: **developer-preview verification**. It proves the commands and tested
paths below. It does not claim support for arbitrary CUDA applications.

## Environment

| Item | Value |
|---|---|
| ROCm | `10.0.0` |
| HIP | `7.15.26333-0000000`, AMD clang `23.0.0git` |
| hipify-clang | AMD LLVM `23.0.0git` |
| GPU | AMD Radeon RX 9070 XT, `gfx1201` |
| CUDA headers | 12.9 (`CUDA_VERSION 12090`) |
| Container | `mark/ubuntu:24.04-rocm10`, repo bind-mounted read/write |
| Date | 2026-11-27 |

## Automated verification

```bash
docker exec ubuntu2404 bash -lc \
  'cd /home/mark/rocm10-cuda-samples-v12.9 && make test'
```

Result: **99 tests, OK**.

The two integration tests run only when `hipify-clang`, `hipcc`, the pinned
checkout and a GPU are present. They cover:

1. compile-database conversion of `vectorAdd` → HIPIFY → `hipcc` → run
   `Test PASSED`;
2. a project whose CUDA header is reached through `-I` → standalone header
   translation → `hipcc` → run `HEADER PASS`.

The other tests cover recursive layout, compile-database parsing, NVCC argument
filtering, incompatible duplicate rejection, redirect activation, header
configuration conflicts, blocked-conversion recovery, report/CMake output and
input immutability.

## Manual checks

### `matrixMul` with profiler dependency

Input: the pinned `0_Introduction/matrixMul/matrixMul.cu`, with a compile
database containing the Common include directory.

```bash
PYTHONPATH=src python3 -m rocm10_samples_compat convert \
  --input /tmp/general-matrixmul \
  --output /tmp/general-matrixmul-hip \
  --cuda-root /usr/local/cuda-12.9 \
  --compile-commands /tmp/general-matrixmul/compile_commands.json
```

Result: `status=completed`, 1/1 unit translated, 4 recorded changes, no review
findings. The report listed:

```text
features:            common-helpers, profiler-roctx
libraries:           rocprofiler-sdk-roctx
compile_definitions: CUDART_VERSION=12090
```

Building the converted tree with `hipcc --offload-arch=gfx1201` and the reported
dependencies produced:

```text
Checking computed result for correctness: Result = PASS
```

This exercises the profiler rule, the namespace wrapper, the overlay include and
the rocTX link dependency through the generic path.

### Generated CMake file

A minimal CMake project included the generated `rocm-migration.cmake` and
configured successfully. The check verifies that
`rocm_cuda_migration_dependencies` and the imported
`ROCmCudaMigrate::profiler` target are accepted by CMake. It does not prove that
an arbitrary project's own CMake targets can be rebuilt without changes.

## P0 regression

After the generic-converter changes:

```bash
make verify
make run-p0 ARTIFACTS=artifacts-rocm10
```

All six P0 samples still pass C1/C2/C3; evidence records ROCm `10.0.0`,
CUDA headers `12090`, `gfx1201`, and `"limitations": []`.

## Boundaries

- No claim beyond the cases above.
- No golden comparison against an NVIDIA result.
- No claim that `convert` output builds without adapting the project's own build
  configuration.
- Residual CUDA findings are review warnings, not proof of compatibility.
- Generic conversion success is not sample acceptance and not ROCm support.
