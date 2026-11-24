# cuda-samples v12.0 / fp16ScalarProduct: overlay runs findCudaDevice to C3

Date: 2026-11-24

Environment: ROCm 7.2.4, hipify-clang 22.0.0git, hipcc, gfx1100

Sample origin: `third-party/cuda-samples`, tag `v12.0`, commit
`2b689228b746248b7bb7d2778d5ea0701669f7d3`.

## Status: HIPIFY direct + Common overlay, C1/C2/C3 pass

This is the first validated sample where the overlay's `findCudaDevice` runs to
completion in a passing run, and the first that needs the quoted include form.

## Command

```bash
cuda2hip-migrate run --spec samples/cuda-samples-v12.0/fp16ScalarProduct.json
```

## What the sample adds to the evidence

| Surface | Result |
|---|---|
| `#include "helper_cuda.h"` (quoted form) | redirected before HIPIFY |
| `#include "cuda_fp16.h"` (quoted form) | HIPIFY rewrites to `"hip/hip_fp16.h"` |
| `findCudaDevice(argc, argv)` | overlay implementation runs, prints `Device 0: "AMD Radeon RX 7900 XTX"` |
| `checkCudaErrors` (12 call sites) | overlay macros compile and pass |
| `EXIT_WAIVED` | NVIDIA `helper_cuda.h` defines it; the overlay now provides the same surface |
| `half2`, `__hadd2`, `__hfma2`, `__float2half2_rn`, `__low2float`, `__high2float` | HIPIFY passes them through; they compile against `hip/hip_fp16.h` |
| `cudaMallocHost` / `cudaMalloc` / `cudaMemcpy` / `cudaFree` | translated one-to-one |

## Transcript

```text
[REDIRECT] PRE-HIPIFY-INCLUDE-REDIRECT line=29
  #include "helper_cuda.h" -> #include "cuda_samples_compat/helper_cuda.h"
[REDIRECTS] applied=1
[RULES] applied=0
[EXPECTED] hipify-direct rules=0
[C1] PASS
Device 0: "AMD Radeon RX 7900 XTX"
Result native operators	: 657090.000000
Result intrinsics	: 657090.000000
&&&& fp16ScalarProduct PASSED
[C2] PASS
[C3] PASS
```

`source/fp16ScalarProduct.cu` is byte-identical to the checkout file; the only
change before HIPIFY is the recorded include redirect. Removing the overlay
include directory from the same build fails with
`'cuda_samples_compat/helper_cuda.h' file not found`, so the overlay is genuinely
in use.

## Overlay surface added for this sample

`helper_cuda.h` now also includes the CPU-only `helper_string.h` from the
original Common checkout and defines `EXIT_WAIVED` (guarded by `#ifndef`), which
matches what NVIDIA's header provides transitively. Both are documented in
`common_overlay.md`.

## Known semantic mismatch (recorded, not fixed)

The sample gates itself with a CUDA compute-capability check:

```cpp
if (devProp.major < 5 || (devProp.major == 5 && devProp.minor < 3)) { ... EXIT_WAIVED; }
```

HIP reports AMD architecture numbers in the same fields (gfx1100 reports
major 11, minor 0), so the gate passes — but for a different reason than on
NVIDIA hardware. This is a sample-level assumption being reused on a different
architecture, not a translation defect. A sample whose AMD arch number were
below the gate would exit with `EXIT_WAIVED` (2) and fail C2, which is the
correct outcome for this pipeline.

## Boundaries

- The comparison is intrinsic-vs-native half math inside one run
  (`fabs(intrinsics - native) < 1e-5`); inputs come from `rand()` seeded with
  `time()`, so absolute values vary between runs while the comparison stays
  within-run. Absolute results are not reproducible and no numeric claim is made.
- Only the two `half2` reduction kernels are covered; no other half-precision
  API, no mixed-precision libraries.
- `findCudaDevice` still does not implement NVIDIA's
  `gpuGetMaxGflopsDeviceId`-style multi-GPU selection. It selects device 0 unless
  `-device=<n>` is passed, and that parsing is covered by host tests, not by a
  multi-GPU run.
- ROCm 10.0 is not validated.
