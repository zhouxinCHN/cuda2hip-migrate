# cuda-samples v12.0 / Common helper overlay: scope

Date: 2026-11-24

Environment: ROCm 7.2.4, hipify-clang 22.0.0git, hipcc, gfx1100

## Why an overlay

`-I <checkout>/Common` makes `hipify-clang` and `hipcc` find sample helper
headers. It does not migrate anything: the header contents stay NVIDIA CUDA
code, and HIPIFY maps `helper_cuda.h` to `<hip/hip_runtime_api.h>` anyway, so a
sample that calls `checkCudaErrors` fails to compile.

`-I` is a header-lookup mechanism, not a Common migration mechanism. The overlay
below is the migration mechanism; `-I` remains part of the pipeline.

## Layout

```text
compat/cuda-samples-v12.0/
`-- include/
    `-- cuda_samples_compat/
        `-- helper_cuda.h        # minimal HIP implementation

samples/cuda-samples-v12.0/
|-- vectorAdd.json               # smoke test, no redirect
|-- simpleOccupancy.json         # overlay, C1/C2/C3
`-- matrixMul.json               # overlay, C1 only, run blocked
```

Include order for builds: overlay directory first, then the original `Common`.
CPU-only Common headers keep coming from the original checkout.

## Pre-HIPIFY include redirect

The original sample is never edited. The runner copies it, applies one
deterministic rewrite before HIPIFY, and records it:

```text
#include <helper_cuda.h>  ->  #include "cuda_samples_compat/helper_cuda.h"
```

This must happen before HIPIFY. After HIPIFY the directive is
`<hip/hip_runtime_api.h>`, which is indistinguishable from a legitimate direct
include of that header, so a post-HIPIFY rule cannot reliably reverse it.

Redirects only fire on a line-start `#include` outside comments, strings and
character literals, preserve trailing line comments, and are recorded in
`redirects.json` with line and before/after text. A declared redirect that never
matches produces `[REDIRECT-MISS]`, so an unused overlay cannot look like a
successful migration.

## Header classification and intended strategy

| Header type | Examples | Strategy |
|---|---|---|
| CPU-only utilities | `helper_string.h`, `helper_timer.h`, `helper_image.h` | keep using the original `Common` copies; compile-verified |
| CUDA runtime wrappers | `helper_cuda.h` | minimal HIP overlay |
| CUDA Driver API wrappers | `helper_cuda_drvapi.h` | separate overlay, only when a driver sample is validated |
| Device math helpers | `helper_math.h` | verify the original first; adapt only if needed |
| NVIDIA library helpers | `helper_cusolver.h`, `helper_nvJPEG.hxx` | separate decision or mark unsupported; no generic HIP mapping |
| Graphics/platform helpers | `helper_gl.h` | out of scope for this pipeline |

## Implemented in the overlay so far

```text
checkCudaErrors(expr)        hipError_t check: prints file/line/code/string, exits on failure
findCudaDevice(argc, argv)   honours -device=<n>, validates range, sets the device, prints it
```

Not implemented on purpose: `gpuGetMaxGflopsDeviceId`-style selection,
compute-capability gates, driver API helpers, `cudaDeviceProp` tuning. The
971-line NVIDIA header was not rewritten; only what validated samples call exists
here.

The overlay is a header with two modes. `hipcc` defines `__HIPCC__`/`__HIP__`,
so the HIP branch is what builds. `hipify-clang` parses it in CUDA mode and only
needs the surface to type-check, so a mirror branch against the CUDA runtime
keeps parsing working; that branch is never linked into a binary.

## What this does not establish

- Support for all of Common, for all cuda-samples, or for any customer project.
- That samples calling unimplemented helpers (`sdkGetTimerValue`,
  `gpuGetMaxGflopsDeviceId`, image/GL helpers) work.
- ROCm 10.0 behaviour.
