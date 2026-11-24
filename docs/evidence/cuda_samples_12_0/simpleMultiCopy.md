# cuda-samples v12.0 / simpleMultiCopy: overlay growth + HIPIFY parse shim

Date: 2026-11-24

Environment: ROCm 7.2.4, hipify-clang 22.0.0git, hipcc, gfx1100

Sample origin: `third-party/cuda-samples`, tag `v12.0`, commit
`2b689228b746248b7bb7d2778d5ea0701669f7d3`.

## Status: HIPIFY direct + Common overlay, C1/C2/C3 pass

First validated sample where HIPIFY itself failed to parse the source, and the
first to need `gpuDeviceInit`, `gpuGetMaxGflopsDeviceId` and
`_ConvertSMVer2Cores` from the overlay.

## Command

```bash
cuda2hip-migrate run --spec samples/cuda-samples-v12.0/simpleMultiCopy.json
```

## Blocker 1: HIPIFY could not parse the sample at all

```text
simpleMultiCopy.cu:144:7: error: no matching function for call to 'max'
  max((32.0f / (_ConvertSMVer2Cores(...) * (float)deviceProp.multiProcessorCount)), 1.0f);
note: candidate function not viable: call to __device__ function from __host__ function
```

`hipify-clang` parses CUDA sources with its own CUDA math shim, which declares
`max`/`min` as `__device__`-only. nvcc exposes them to host code as `__host__
__device__`, and this sample calls `max(float, float)` on the host for a
workload clamp. The failure happens **before** any HIP file exists, so no
post-HIPIFY rule could fix it, and the original source is not edited.

Resolution — a parse-only shim, applied with `-include` and not compiled into any
binary:

```cpp
#include <algorithm>
using std::max;
using std::min;
```

`hipify-clang` forwards clang options after `--`, so the runner builds:

```text
hipify-clang <source> -I<overlay> -I<Common> -o <hipified> -- -include <shim>
```

A macro (`#define max(a, b) ...`) was tried first and breaks libstdc++ headers
(`algorithmfwd.h: expected unqualified-id`), so overload injection is used
instead. `hipcc` needs no shim: it exposes host `max`/`min` itself.

Spec field: `hipify_forced_includes`, resolved against `include_dirs` and logged
as `[HIPIFY-PARSE-INCLUDE]`.

## Blocker 2: three overlay helpers were missing

| Helper | AMD implementation | Note |
|---|---|---|
| `_ConvertSMVer2Cores(major, minor)` | returns 64 lanes per compute unit | NVIDIA maps SM version to cores/SM; AMD has no SM version |
| `gpuGetMaxGflopsDeviceId()` | ranks devices by lanes x multiprocessors x clockRate | proxy for NVIDIA's compute-capability GFLOPs estimate |
| `gpuDeviceInit(dev_id)` | validates the id, selects it, prints it, returns -1 when invalid | matches the NVIDIA contract used by the sample |

These affect banners and workload scaling only. The sample computes
`scale_factor = max(32 / (cores * multiProcessors), 1)`, which is 1 here, so the
workload size is unchanged.

Recorded caveat: HIP reports 48 multiprocessors for this device because it
counts workgroup processors in WGP mode, so `48 x 64 = 3072` under-reports the
hardware's lane count (~96 CUs x 64). The figure is printed and used for a clamp
only; no performance claim is derived from it.

## Acceptance

```text
[REDIRECT] PRE-HIPIFY-INCLUDE-REDIRECT line=52
  #include <helper_cuda.h> -> #include "cuda_samples_compat/helper_cuda.h"
[REDIRECTS] applied=1
[HIPIFY-PARSE-INCLUDE] -include <overlay>/cuda_samples_compat/host_math_parse_shim.h
[RULES] applied=0
[EXPECTED] hipify-direct rules=0
[C1] PASS
[simpleMultiCopy] - Starting...
> Using CUDA device [0]: AMD Radeon RX 7900 XTX
...
Average measured timings over 10 repetitions:
 Avg. time when execution fully serialized: 5.702680 ms
 Avg. time when overlapped using 4 streams : 3.914765 ms
...
Measured throughput:
 Fully serialized execution   : 5.883976 GB/s
 Overlapped using 4 streams   : 8.571251 GB/s
[C2] PASS
[C3] PASS
```

`source/simpleMultiCopy.cu` is byte-identical to the checkout file; the redirect
and the parse-only shim are the only preprocessing, both recorded.

## C3 marker rationale

The sample has no success line. `test()` compares all four streams' host outputs
against 1 and the exit status is `EXIT_SUCCESS` only when every comparison
passes, so correctness is carried by C2. The marker is
`Overlapped using 4 streams`, the last throughput line printed before
verification, which proves the run reached the final measurement report. Neither
check alone is treated as sufficient.

## Boundaries

- Timings, GB/s figures and the "speedup" line are measurements of this machine
  under this driver; they are not claims about overlap quality or performance.
- `deviceOverlap` / `asyncEngineCount` are printed as HIP reports them; the
  printed "(X) Can overlap two CPU<>GPU data transfers" line reflects HIP field
  values, not a verified overlap capability.
- The `-device=<n>` branch of the sample (`gpuDeviceInit`) was not exercised:
  this run took the `gpuGetMaxGflopsDeviceId` branch. Its parsing and validation
  are covered by host tests and spec-level reasoning, not by a multi-GPU run.
- `STREAM_COUNT` is 4 and `nreps` is 10; other configurations are not covered.
- ROCm 10.0 is not validated.
