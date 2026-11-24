# probe_cublas_gemm: pre-rule failure and fix evidence

Date: 2026-11-24

Environment: ROCm 7.2.4, hipify-clang 22.0.0git, hipblas 3.2.70204, gfx1100

## Purpose

Record the first probe that fails for two independent reasons after HIPIFY: a
header path that HIPIFY rewrites into a location ROCm does not use, and a
library that must be linked. The header failure is fixed by a deterministic
rule; the link failure is build configuration and is expressed as a per-probe
build spec.

## Original CUDA source

```cuda
#include <cublas_v2.h>
...
cublasCreate(&handle);
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, kN, kN, kN, &alpha,
            device_a, kN, device_b, kN, &beta, device_c, kN);
cublasDestroy(handle);
```

## HIPIFY output

```cpp
#include <hipblas.h>
...
hipblasCreate(&handle);
hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, kN, kN, kN, &alpha,
             device_a, kN, device_b, kN, &beta, device_c, kN);
hipblasDestroy(handle);
```

The library calls, types and enums translate correctly. HIPIFY is not the
problem on the call surface.

## Failure 1: header path

```text
/tmp/cublas-hipify/probe_cublas_gemm.hip.cu:5:10: fatal error: 'hipblas.h' file not found
    5 | #include <hipblas.h>
      |          ^~~~~~~~~~~
1 error generated when compiling for gfx1100.
```

On this ROCm installation the header is `/opt/rocm/include/hipblas/hipblas.h`;
`/opt/rocm/include/hipblas.h` does not exist. HIPIFY's mapping to a flat
`hipblas.h` therefore does not compile.

## Failure 2: missing link library

With the include corrected but no library flag:

```text
ld.lld: error: undefined symbol: hipblasCreate
ld.lld: error: undefined symbol: hipblasSgemm
ld.lld: error: undefined symbol: hipblasDestroy
```

## Fixes

Header, as a deterministic rule:

```text
HIP-CUBLAS-INCLUDE-001
Condition: a line-start #include directive, outside comments and literals,
           exactly "#include <hipblas.h>"
Action: rewrite to "#include <hipblas/hipblas.h>"
```

Implemented in `src/cuda2hip_migrate/rules/cublas.py`. Positive, negative,
comment/string and idempotence tests are in `tests/test_cublas_rule.py`. The
`#include <cublas_v2.h>` form present in the original CUDA source is not
touched, because the rule only ever sees HIPIFY output.

Link library, as packaged build configuration:

```json
{
  "libraries": ["hipblas"]
}
```

Stored as `src/cuda2hip_migrate/probes/probe_cublas_gemm.build.json`; the runner
turns `libraries` into `-lhipblas` for that probe only. Probes without a build
spec keep the plain `hipcc` command. Validation tests are in
`tests/test_build_spec.py`.

## Result after both fixes

```text
[RULE] HIP-CUBLAS-INCLUDE-001 line=5 #include <hipblas.h> -> #include <hipblas/hipblas.h>
[RULES] applied=1
[BUILD] flags=-lhipblas
[C1] PASS
PROBE=cublas_gemm device=AMD Radeon RX 7900 XTX order=column-major n=64 elements=4096 RESULT=PASS
[C2] PASS
[C3] PASS
```

Recreated with:

```bash
cuda2hip-migrate run --probe probe_cublas_gemm --out /tmp/probe-cublas
```

## Boundaries

- Only `cublasSgemm` and the handle lifecycle are covered; batched, strided,
  half/bfloat16, `cublasLt` and handle workspaces are not.
- The include rule encodes a ROCm 7.2.4 header layout. A ROCm release that
  ships `hipblas.h` in the include root, or HIPIFY output that changes, requires
  revalidation; the rule is narrowed to the exact directive to keep that risk
  visible.
- Column-major ordering is identical in cuBLAS and hipBLAS, so no transposition
  logic is exercised.
- Correctness is checked with a 1e-2 relative tolerance against a host loop;
  this is not a performance or numerical-equivalence claim.
