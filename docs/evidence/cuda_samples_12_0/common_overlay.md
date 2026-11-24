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
character literals, support both `<header>` and `"header"` forms, preserve
trailing line comments, and are recorded in `redirects.json` with line and
before/after text. `redirects.json` is written on every run, as `[]` when no
redirect applied.

A declared redirect that never matches **fails the run**:

```text
ERROR [redirect]: plain in <spec>: declared include redirect(s) not found in
source: helper_cuda.h; the overlay header would be unused. Declare the header in
'optional_redirects' if that is intended.
```

That is the fix for a real false pass: an earlier revision only logged
`[REDIRECT-MISS]` and still reported C1/C2/C3 success, which was directly
contradicted by the wording of this document. `optional_redirects` now carries
the intentional exception, and `expected_redirects` pins the exact redirect
count per sample.

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
_ConvertSMVer2Cores(a, b)    AMD analogue: 64 lanes per compute unit
gpuGetMaxGflopsDeviceId()    ranks devices by lanes x multiprocessors x clockRate
gpuDeviceInit(dev_id)        validates the id, selects it, prints it, returns -1 when invalid

EXIT_WAIVED                  defined when not already defined, as NVIDIA's header does
helper_string.h              included verbatim from the original Common checkout
```

Not implemented on purpose: compute-capability gates, driver API helpers,
`cudaDeviceProp` tuning, image/GL/NVIDIA-library helpers. The 971-line NVIDIA
header was not rewritten; only what validated samples call exists here.

AMD semantic caveats: the three device-selection helpers are NVIDIA-shaped
functions implemented with AMD proxies (lanes-per-CU instead of SM cores,
lanes x multiprocessors x clock instead of a GFLOPs estimate). They influence
banners and workload clamps in the samples that use them, never correctness.

## Parse-only shim

`cuda_samples_compat/host_math_parse_shim.h` exists because `hipify-clang`'s CUDA
math shim declares `max`/`min` as `__device__`-only while nvcc exposes them to
host code. Samples that call them on the host fail to parse, so no HIP output is
produced at all. The shim pulls `std::max`/`std::min` into the global namespace,
is passed to `hipify-clang` with `-- -include <shim>`, and is never compiled into
a binary (`hipcc` provides host `max`/`min` itself).

It lives in the same overlay tree but is a different mechanism: parse-time only,
selected per spec by `hipify_forced_includes`, resolved against `include_dirs`,
and logged as `[HIPIFY-PARSE-INCLUDE]`.

## Device selection argument parsing

`findCudaDevice(argc, argv)` honours `-device=<n>`. Parsing lives in
`cuda_samples_compat/argparse.h`, which has no CUDA or HIP dependency, so
`tests/test_compat_header.py` compiles it with `g++` and runs it on the host:

```text
-device=0   -> found, "0"        --device=0 -> found, "0"
---device=3 -> found, "3"        device=5   -> found, "5"
-device=-1  -> found, "-1"       -wA=320    -> not the device flag
```

Any number of leading `-` is stripped, matching the NVIDIA helper's
`stringRemoveDelimiter`. An earlier revision stripped at most one `-` and
compared the remainder to `device`, so `-device=0` and `--device=0` were both
ignored and the default device was always used. That is fixed and covered by
host tests that need no GPU.

The overlay is a header with two modes. `hipcc` defines `__HIPCC__`/`__HIP__`,
so the HIP branch is what builds. `hipify-clang` parses it in CUDA mode and only
needs the surface to type-check, so a mirror branch against the CUDA runtime
keeps parsing working; that branch is never linked into a binary.

## What this does not establish

- Support for all of Common, for all cuda-samples, or for any customer project.
- That samples calling unimplemented helpers (`sdkGetTimerValue`,
  `gpuGetMaxGflopsDeviceId`, image/GL helpers) work.
- ROCm 10.0 behaviour.
