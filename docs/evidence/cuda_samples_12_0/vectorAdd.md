# cuda-samples v12.0 / vectorAdd: HIPIFY-direct evidence

Date: 2026-11-24

Environment: ROCm 7.2.4, hipify-clang 22.0.0git, hipcc, gfx1100

Sample origin: `third-party/cuda-samples` (git tag `v12.0`, commit
`2b689228b746248b7bb7d2778d5ea0701669f7d3`, "Updating samples for 12.0"). The
checkout is a nested Git repository and is excluded from this repository via
`.gitignore`; only this evidence and the runner invocation are committed.

## Sample

`Samples/0_Introduction/vectorAdd/vectorAdd.cu`, unmodified: 50 000-element
float vector addition, one kernel launch, host-side verification and
`Test PASSED` on success.

## Command

```bash
cuda2hip-migrate run \
  --source third-party/cuda-samples/Samples/0_Introduction/vectorAdd/vectorAdd.cu \
  --include-dir third-party/cuda-samples/Common \
  --success-marker "Test PASSED" \
  --out /tmp/vecadd
```

`--include-dir` is required because the sample includes `<helper_cuda.h>` and
`hipify-clang` must be able to resolve it; without the include directory HIPIFY
fails with `'helper_cuda.h' file not found`.

## Translation result

`hipify-clang` exit 0, no rule hits. Two points are worth recording:

```text
#include <cuda_runtime.h>  ->  #include <hip/hip_runtime.h>
#include <helper_cuda.h>   ->  #include <hip/hip_runtime_api.h>
```

HIPIFY maps the NVIDIA helper header to a HIP runtime header, so the CUDA
Samples helper dependency disappears from the translated source. `cudaMalloc`,
`cudaMemcpy`, `cudaFree`, `cudaGetLastError`, `cudaGetErrorString` and the
default-stream kernel launch are translated one-to-one.

## Acceptance

```text
[RULES] applied=0
[C1] PASS
[Vector addition of 50000 elements]
Copy input data from the host memory to the CUDA device
CUDA kernel launch with 196 blocks of 256 threads
Copy output data from the CUDA device to the host memory
Test PASSED
Done
[C2] PASS
[C3] PASS
```

C3 uses the sample's own marker (`Test PASSED`) through `--success-marker`. The
marker is per-run configuration; the default stays `RESULT=PASS`, so packaged
probes are unaffected and the acceptance threshold is not lowered.

The archived `source/vectorAdd.cu` is byte-identical to the file in the
cuda-samples checkout, and `rules.json` is `[]`.

## Boundaries

- One sample out of the cuda-samples v12.0 set. No general cuda-samples support
  is claimed, and other samples are unclassified until they are run.
- The sample is not built through its own Makefile or CMake project; the runner
  invokes `hipcc` directly, so the sample's build flags are not exercised.
  `helper_cuda.h` maps to a HIP header here only because this sample does not
  call helper functions. Samples that call `checkCudaErrors`,
  `findCudaDevice`, `sdkGetTimerValue` or similar helpers are expected to
  behave differently and must be measured, not assumed.
- Only the default stream and one launch configuration are covered.
- The sample's own tolerance (`1e-5`) is reused; performance is out of scope.
- ROCm 10.0 is not validated.
