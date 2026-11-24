# cuda-samples v12.0 / matrixMul: overlay compiles, run blocked by HIP API

Date: 2026-11-24

Environment: ROCm 7.2.4, hipify-clang 22.0.0git, hipcc, gfx1100

Sample origin: `third-party/cuda-samples`, tag `v12.0`, commit
`2b689228b746248b7bb7d2778d5ea0701669f7d3`.

## Status: blocked, not supported

- C1 (build): **PASS** with the Common overlay.
- C2 (run): **blocked** by `hipProfilerStart()` returning `hipErrorNotSupported`
  (801) on this platform.
- C3: not reached.

No rule was added and no source edit was made to work around it. Disabling
profiling calls through a rule would change the program's behaviour and hide a
real platform limitation.

## What the sample exercises

`checkCudaErrors` on 28 runtime calls, `findCudaDevice(argc, argv)`, shared
memory tiling, `cudaMallocHost`, non-blocking stream, event timing and
`cudaProfilerStart` / `cudaProfilerStop`.

## Command

```bash
cuda2hip-migrate run --spec samples/cuda-samples-v12.0/matrixMul.json
```

```json
{
  "include_dirs": [
    "compat/cuda-samples-v12.0/include",
    "third-party/cuda-samples/Common"
  ],
  "success_marker": "Test PASSED",
  "expected": "hipify-direct",
  "include_redirects": {"helper_cuda.h": "cuda_samples_compat/helper_cuda.h"}
}
```

## Transcript

```text
[REDIRECT] PRE-HIPIFY-INCLUDE-REDIRECT line=52
  #include <helper_cuda.h> -> #include "cuda_samples_compat/helper_cuda.h"
[REDIRECTS] applied=1
[RULES] applied=0
[EXPECTED] hipify-direct rules=0
[C1] PASS
HIP error at adapted/matrixMul.hip.cu:348 code=801(operation not supported) "hipProfilerStart()"
[Matrix Multiply Using CUDA] - Starting...
Device 0: "AMD Radeon RX 7900 XTX"
MatrixA(320,320), MatrixB(640,320)
ERROR [C2]: probe exited with 1
```

Two things are established by this run even though it is blocked:

1. The overlay works at run time: `checkCudaErrors` compiles and executes, and
   the `Device 0: "AMD Radeon RX 7900 XTX"` line is printed by the overlay's
   `findCudaDevice` implementation.
2. The blocker is an unsupported HIP runtime API, not a translation defect: the
   error is raised by `hipProfilerStart`, which HIP marks deprecated, and it is
   reached after device selection succeeded.

## Possible next steps (not taken)

- A documented, explicitly-scoped rule that maps `cudaProfilerStart` /
  `cudaProfilerStop` to no-ops for profiling-free validation runs, recorded in
  `rules.json` and revalidated after a ROCm upgrade.
- An upstream/environment route, if profiling support becomes available.
- Move on to samples that do not call the profiler API.

Until one of these is decided, matrixMul is declared **blocked**, and no
cuda-samples support claim includes it.

## Boundaries

- One environment (`gfx1100`, ROCm 7.2.4). Another GPU driver or ROCm release
  may return success for `hipProfilerStart`.
- The C2 failure says nothing about the matrix multiplication result, because the
  sample exits before the kernel runs.
