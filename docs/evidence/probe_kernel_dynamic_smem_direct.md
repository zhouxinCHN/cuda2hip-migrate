# probe_kernel_dynamic_smem: HIPIFY-direct evidence

Date: 2026-11-24

Environment: ROCm 7.2.4, hipify-clang 22.0.0git, gfx1100

## Purpose

Record that dynamic shared memory, `__syncthreads` and a multi-block reduction
translate and run without any post-HIPIFY rule.

## Capability covered

- `extern __shared__ float scratch[]` inside the kernel;
- a third kernel-launch parameter for dynamically sized shared memory;
- tree reduction across `blockDim.x` lanes with `__syncthreads`;
- one partial sum per block, copied back and summed on the host;
- comparison against a host-side `double` sum with a fixed tolerance.

## HIPIFY translation

The kernel launch configuration, `extern __shared__`, `__syncthreads`,
`blockIdx`/`blockDim`/`threadIdx` and the runtime calls
(`cudaMalloc`, `cudaMemcpy`, `cudaDeviceSynchronize`, `cudaFree`,
`cudaGetLastError`, `cudaGetErrorString`) are passed through without edit. No
rule is required.

## Run result

```text
[RULES] applied=0
[C1] PASS
PROBE=kernel_dynamic_smem device=AMD Radeon RX 7900 XTX blocks=17 threads=256 shared_bytes=1024 elements=4129 RESULT=PASS
[C2] PASS
[C3] PASS
```

Recreated with:

```bash
cuda2hip-migrate run --probe probe_kernel_dynamic_smem --out /tmp/probe-dyn-smem
```

`rules.json` is `[]` and `adapted/` is byte-identical to `hipified/`.

## Boundaries

- The block size is 256 threads and the dynamic shared allocation is 1 KiB, so
  the probe does not exercise the >48 KiB opt-in path
  (`cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize)`) that HIP
  implements differently.
- It does not measure occupancy, bank conflicts or reduction performance.
- Floating-point comparison uses a 1e-3 tolerance on a 4129-element sum; this is
  a correctness check, not a numerical-equivalence proof.
