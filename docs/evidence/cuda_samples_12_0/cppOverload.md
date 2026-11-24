# cuda-samples v12.0 / cppOverload: overlay + untranslated device header

Date: 2026-11-24

Environment: ROCm 7.2.4, hipify-clang 22.0.0git, hipcc, gfx1100

Sample origin: `third-party/cuda-samples`, tag `v12.0`, commit
`2b689228b746248b7bb7d2778d5ea0701669f7d3`.

## Status: HIPIFY direct + Common overlay, C1/C2/C3 pass

Three kernel overloads selected through function pointers, plus
`cudaFuncSetCacheConfig` / `cudaFuncGetAttributes` — API surfaces no earlier
validated sample used.

## Command

```bash
cuda2hip-migrate run --spec samples/cuda-samples-v12.0/cppOverload.json
```

The spec adds the sample's own directory to `include_dirs`, because the sample
includes `"cppOverload_kernel.cuh"`.

## Translation surface

```text
#include <helper_cuda.h>   -> redirected to cuda_samples_compat/helper_cuda.h
#include <helper_math.h>   -> #include <hip/hip_vector_types.h>      (HIPIFY)
#include <helper_string.h> -> unchanged, CPU-only, from the original Common
cudaFuncAttributes        -> hipFuncAttributes
cudaFuncSetCacheConfig    -> hipFuncSetCacheConfig
cudaFuncGetAttributes     -> hipFuncGetAttributes
cudaFuncCachePreferShared -> hipFuncCachePreferShared
```

All `OUTPUT_ATTR` fields used by the sample exist in `hipFuncAttributes`
(`sharedSizeBytes`, `constSizeBytes`, `localSizeBytes`, `maxThreadsPerBlock`,
`numRegs`, `ptxVersion`, `binaryVersion`).

## Acceptance

```text
[REDIRECT] PRE-HIPIFY-INCLUDE-REDIRECT line=34
  #include <helper_cuda.h> -> #include "cuda_samples_compat/helper_cuda.h"
[REDIRECTS] applied=1
[RULES] applied=0
[EXPECTED] hipify-direct rules=0
[C1] PASS
Device 0: "AMD Radeon RX 7900 XTX"
...
simple_kernel(const int *pIn, int *pOut, int a) PASSED
simple_kernel(const int2 *pIn, int *pOut, int a) PASSED
simple_kernel(const int *pIn1, const int *pIn2, int *pOut, int a) PASSED
[C2] PASS
[C3] PASS
```

C3 uses the last kernel's `PASSED` line, which proves the run reached the final
check. Correctness across all three kernels is carried by C2: the sample exits
with `EXIT_SUCCESS` only when `testResult` is true, i.e. when every kernel
matched its host reference. Neither check alone is treated as sufficient.

Removing the overlay include directory from the same build fails with
`'cuda_samples_compat/helper_cuda.h' file not found`, so the overlay is in use.
`source/cppOverload.cu` is byte-identical to the checkout file.

## New boundary: an untranslated device header

`cppOverload_kernel.cuh` is a second file in the sample directory. The runner
translates only the main `.cu`; the `.cuh` is included as-is and compiled by
`hipcc` as HIP. That works here because it contains only `__global__` kernels
with CUDA/HIP-common syntax (no `cuda*` runtime calls), but it is **not** a
general multi-file translation story:

- the `.cuh` is not HIPIFY-ed, so `hipified/` does not describe it;
- a secondary file that calls CUDA runtime APIs would not compile and would need
  its own translation or an explicit unsupported classification;
- `include_dirs` must point at the sample directory, which the spec records.

## Recorded semantic mismatches

- `cudaFuncSetCacheConfig(..., cudaFuncCachePreferShared)` translates and
  returns success, but CUDA's L1/shared carve-out preference is an NVIDIA
  concept. The sample does not verify that the preference took effect, and this
  evidence makes no claim that it did.
- `cudaFuncGetAttributes` reports HIP values in CUDA-shaped fields (`PTX
  Version: 110`, `Binary Version: 110` here). Those are AMD code-object
  version numbers, not NVIDIA PTX versions. The sample only prints them.
- The `prop.major < 2` gate is a CUDA compute-capability check applied to AMD
  architecture numbers (gfx1100 reports 11.0); it passes for a different reason
  than it would on NVIDIA hardware.

## Boundaries

- Three int kernels with one block size; no half-precision, no libraries.
- The `.cuh` is not translated (see above).
- Attribute values, occupancy and cache behaviour are printed, not validated.
- ROCm 10.0 is not validated.
