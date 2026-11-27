# P0 Baseline Evidence — ROCm 10.0.0 / gfx1201

Status: **target environment**. This replaces
[`p0-baseline-rocm7.2.4-gfx1100.md`](p0-baseline-rocm7.2.4-gfx1100.md), which is
interim-only and must not be quoted as a current result.

## Environment

| Item | Value |
|---|---|
| CUDA Samples | `v12.9` @ `cab7c66b4fcf35b09ea8fbc4c3c1a9272e92d997` (verified, clean) |
| ROCm detected | `10.0.0` (target, matches) |
| HIP | `7.15.26333-0000000`, AMD clang `23.0.0git` |
| hipify-clang | AMD LLVM `23.0.0git`, from `/opt/rocm/bin` |
| GPU | AMD Radeon RX 9070 XT, `gfx1201`, 16 GB (iGPU `gfx1036` also enumerated) |
| CUDA headers used by HIPIFY | **12.9** (`CUDA_VERSION 12090`, `CUDART_VERSION 12090`) |
| hipcc flags | `-O3 -std=c++17 --offload-arch=gfx1201 -DCUDART_VERSION=12090` |
| Host | Ubuntu 24.04.5, kernel `7.0.0-42-generic` |
| Container | `mark/ubuntu:24.04-rocm10` (Ubuntu 24.04 base, user `mark` uid 1001) |
| Date | 2026-11-26 |

Both limitations recorded by the interim run are gone: the headers really are
12.9 and the execution really is on ROCm 10.0.0. Every `evidence.json` in this
run carries `"limitations": []`.

## Result matrix

| Sample | C1 build | C2 exit 0 | C3 | Status | Blocked at |
|---|---|---|---|---|---|
| deviceQuery | PASS | PASS | PASS (`Result = PASS`) | rule-fixed | — |
| vectorAdd | PASS | PASS | PASS (`Test PASSED`) | rule-fixed | — |
| matrixMul | PASS | PASS | PASS (`Result = PASS`) | rule-fixed (profiler rule) | — |
| asyncAPI | PASS | PASS | PASS (exit status only) | rule-fixed (profiler rule) | — |
| simpleStreams | PASS | PASS | PASS (exit status only) | rule-fixed | — |
| scalarProd | PASS | PASS | PASS (`Test passed`) | rule-fixed | — |

`run-batch --batch P0_baseline` exits 0. This is the first P0 run in which every
sample reaches C3.

## The `hipProfilerStart()` decision (recorded 2026-11-26)

The interim decision was "keep blocked until the ROCm 10.0.0 run says otherwise".
The run said otherwise in one respect and confirmed it in another: on the target
environment the call **still** returns `801 (hipErrorNotSupported)`, exactly as
on ROCm 7.2.4, so waiting for the environment change bought nothing.

### What was measured

`hipProfilerStart`/`hipProfilerStop` are `HIP_DEPRECATED("use roctracer/rocTX
instead")` in `hip/hip_runtime_api.h`, which documents `@returns
#hipErrorNotSupported`. The exported symbols carry the `hip_4.2` ABI version.

| Condition | Result |
|---|---|
| before any device init | 801 |
| after `hipInit(0)` + `hipSetDevice(0)` | 801 |
| repeated `start, start, stop, stop` | 801 each |
| with a real kernel between start and stop | 801 |
| under an attached `rocprofv3` | **801** |

The last row matters: the return value does not depend on whether a profiler is
listening, so this is an unconditional stub, not a precondition that was missing.

A `LD_PRELOAD` shim that returned success for both calls (diagnostic only, since
removed) let `matrixMul` and `asyncAPI` run to completion and pass their own
self-checks, which establishes that 801 was the **only** blocker for both samples.

### Decision

`HIP-PROFILER-ROCTX-001` maps the calls to rocTX:

```text
hipProfilerStart()  ->  rocm10_compat::hipProfilerStart()  ->  roctxProfilerResume(0)
hipProfilerStop()   ->  rocm10_compat::hipProfilerStop()   ->  roctxProfilerPause(0)
```

Reasons this preserves the sample's meaning where a no-op would not:
`roctxProfilerResume/Pause` pause and resume profiling for the whole process,
which is the same contract as `cudaProfilerStart/Stop`. Measured on ROCm 10.0.0:
both return 0 when no tool is attached and when the requested state already
holds, so the sample behaves as it does on CUDA without a profiler.

A no-op stub was rejected even though it also reaches C3 — and that is exactly
the argument against it: the shim experiment shows the sample's self-check passes
with instrumentation disabled, so a no-op would report `Result = PASS` while
measuring nothing at all.

**Link requirement:** the wrapper needs `librocprofiler-sdk-roctx`.
`libROCTX64.so` does not export `roctxProfilerResume/Pause`; linking `-lroctx64`
fails with undefined references. `matrixMul` and `asyncAPI` declare
`"libraries": ["rocprofiler-sdk-roctx"]` in their spec for this reason.

## CUDA 12.9 header requirements discovered by this run

hipify-clang needs a CUDA root with `include/`, `nvvm/libdevice/libdevice.10.bc`
and `bin/`. Meeting that is not enough for ROCm 10's clang 23; three packages are
required in addition, each of which fails the parse if missing:

| Package | Why it is needed |
|---|---|
| `libcurand-dev-12-9` | `__clang_cuda_runtime_wrapper.h` force-includes `curand_mtgp32_kernel.h` |
| `cuda-cccl-12-9` | the same wrapper needs `<nv/target>` |
| `cuda-profiler-api-12-9` | `cuda_profiler_api.h` is a separate package in CUDA 12.x, not part of `cuda-cudart-dev` |

A missing profiler header fails at the HIPIFY stage with
`'cuda_profiler_api.h' file not found`, which looks like a translation failure but
is a toolkit-completeness problem.

## Reproducing the environment

ROCm 10.0.0 installs to `/opt/rocm` from
`https://stable.repo.amd.com/rocm/core/packages/ubuntu2404/`; the package set used
here is `amdrocm-core-sdk10.0-gfx1201`, `amdrocm-hipify10.0`,
`amdrocm-amdsmi10.0` (80 archives, ~1 GB).

Three things the install does not do for you in a container:

1. `/opt/rocm/.info/version` is absent, and `environment.py` reads exactly that
   file to determine the ROCm version. It must contain `10.0.0`.
2. `/etc/ld.so.conf.d/rocm.conf` must list `/opt/rocm/lib` **and**
   `/opt/rocm/lib/llvm/lib`; without the second entry `hipify-clang` dies with
   `libLLVM.so.23.0git: cannot open shared object file`, and without the first,
   any compiled HIP binary fails on `libamdhip64.so.7`.
3. GPU access is AMD device passthrough, not `--gpus all`:
   `--device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 110
   --security-opt seccomp=unconfined`. Numeric GIDs are required because the base
   image has no `render` group to name.

## Reproduction

```bash
make test                                     # 74 tests
python3 -m rocm10_samples_compat verify-checkout
python3 -m rocm10_samples_compat run-batch --batch P0_baseline --arch gfx1201
```

`--arch gfx1201` is currently mandatory on this machine:
`primary_gfx_architecture()` returns the first entry of a sorted list, which is
the iGPU `gfx1036`, not the discrete `gfx1201`.

Artifacts are under `artifacts-rocm10/<sample>/` for this run; the interim
`artifacts/` tree is left untouched. Both are git-ignored; regenerate rather than
commit.

## What this does not prove

- No `gfx1100` or other-architecture behaviour; results do not transfer.
- No golden numeric comparison against a CUDA reference; C3 is the sample's own
  self-check or its exit status.
- No performance statement. `matrixMul` printed 2217 GFlop/s, which the sample
  itself says is not a benchmark result.
- `simpleStreams` and `asyncAPI` still rest on exit status only.
- No statement about the remaining 182 samples.
