# cuda-samples v12.0 / simpleOccupancy: Common overlay validated

Date: 2026-11-24

Environment: ROCm 7.2.4, hipify-clang 22.0.0git, hipcc, gfx1100

Sample origin: `third-party/cuda-samples`, tag `v12.0`, commit
`2b689228b746248b7bb7d2778d5ea0701669f7d3`.

## Why this sample

It is the first validated sample that actually calls a Common helper:
`checkCudaErrors` around `hipMalloc`, `hipMemcpy`, `hipEventCreate`,
`hipOccupancyMaxPotentialBlockSize`, `hipDeviceSynchronize` and friends. That is
what makes it a test of the overlay rather than a test of header lookup.

## Command

```bash
cuda2hip-migrate run --spec samples/cuda-samples-v12.0/simpleOccupancy.json
```

The spec declares:

```json
{
  "schema_version": 1,
  "cuda_samples_commit": "2b689228b746248b7bb7d2778d5ea0701669f7d3",
  "include_dirs": [
    "../../compat/cuda-samples-v12.0/include",
    "../../third-party/cuda-samples/Common"
  ],
  "success_marker": "Test PASSED",
  "expected": "hipify-direct",
  "expected_redirects": 1,
  "optimization": "-O0",
  "include_redirects": {"helper_cuda.h": "cuda_samples_compat/helper_cuda.h"}
}
```

`expected_redirects: 1` makes the redirect count part of acceptance: a run with
zero redirects fails instead of silently building without the overlay. Spec
paths are relative to the spec file, so this spec was also run from `/tmp` with
the same result, and the loader is covered by a CWD-independence test.

## Pre-HIPIFY redirect

The original source keeps `#include <helper_cuda.h>`; the copy handed to
`hipify-clang` gets one deterministic edit, recorded in `redirects.json`:

```text
[REDIRECT] PRE-HIPIFY-INCLUDE-REDIRECT line=29
  #include <helper_cuda.h>  // helper functions for CUDA error check
  -> #include "cuda_samples_compat/helper_cuda.h"  // helper functions for CUDA error check
```

`source/simpleOccupancy.cu` stays byte-identical to the checkout;
`preprocessed/simpleOccupancy.cu` holds the redirect; `hipify.diff` is
preprocessed -> hipified.

## The overlay is genuinely used

Removing the overlay include directory from the same build fails:

```text
adapted/simpleOccupancy.hip.cu:31:10: fatal error:
  'cuda_samples_compat/helper_cuda.h' file not found
```

So the sample compiles only because `checkCudaErrors` comes from the overlay,
not from NVIDIA's header and not from HIPIFY output.

## Acceptance

```text
[RULES] applied=0
[EXPECTED] hipify-direct rules=0
[BUILD] optimization=-O0
[C1] PASS
starting Simple Occupancy
[ Manual configuration with 32 threads per block ]
Potential occupancy: 100%
...
[ Automatic, occupancy-based configuration ]
Suggested block size: 1024
Minimum grid size for maximum occupancy: 96
Potential occupancy: 100%
Test PASSED
[C2] PASS
[C3] PASS
```

This validates `checkCudaErrors` at run time (it is called successfully many
times) and the HIP path of the overlay's device probing through
`hipGetDeviceProperties`-based occupancy queries.

## Why `-O0`

At the default `-O2` this sample fails its own verification:

```text
element 65536 expected 0 actual 0
Test failed
```

`65536 * 65536` overflows 32-bit signed `int` in the sample's verification loop
(`if (array[i] != i * i)`), which is undefined behaviour; the same expression is
optimized differently by nvcc and by clang. Rebuilding the identical translated
source at `-O0` passes, confirming the cause is the sample's overflow, not the
translation or the overlay.

The optimization level is an explicit, validated spec field (`-O0`..`-O3`,
default `-O2`) and is logged as `[BUILD] optimization=...`. C1/C2/C3 are still
enforced; no marker was relaxed.

## Boundaries

- Only `checkCudaErrors` and `findCudaDevice` existence are exercised across
  samples; `findCudaDevice` is not called by this sample (it uses
  `cudaGetDevice` directly). matrixMul is the sample that calls it, and it is
  blocked at run time for an unrelated reason (see `matrixMul.md`).
- `gpuGetMaxGflopsDeviceId`, compute-capability checks, driver-API helpers,
  `helper_cuda_drvapi.h`, image/GL/NVIDIA-library helpers are not implemented.
- Occupancy numbers and timings are not performance claims.
- ROCm 10.0 is not validated.
