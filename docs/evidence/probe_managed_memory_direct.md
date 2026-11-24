# probe_managed_memory: HIPIFY-direct evidence

Date: 2026-11-24

Environment: ROCm 7.2.4, hipify-clang 22.0.0git, gfx1100

## Purpose

Record that managed (unified) memory is translated correctly by HIPIFY without
any post-HIPIFY rule.

## Capability covered

- `cudaMallocManaged` with the explicit `cudaMemAttachGlobal` flag form;
- a kernel writing into managed memory on the default stream;
- host dereference of the same allocation without an explicit copy;
- `cudaFree` of managed memory.

The probe deliberately avoids `cudaMemPrefetchAsync` and `cudaMemAdvise` so the
capability stays narrow and does not depend on page-migration tuning.

## HIPIFY translation

```text
cudaMallocManaged(..., cudaMemAttachGlobal) -> hipMallocManaged(..., hipMemAttachGlobal)
```

The three-argument managed allocation is accepted by HIP unchanged; no arity or
flag rule is required. `cudaGetErrorString`, `cudaGetLastError`,
`cudaDeviceSynchronize` and `cudaFree` translate directly as in the other
probes.

## Run result

```text
[RULES] applied=0
[C1] PASS
PROBE=managed_memory device=AMD Radeon RX 7900 XTX elements=8197 bytes=32788 RESULT=PASS
[C2] PASS
[C3] PASS
```

Recreated with:

```bash
cuda2hip-migrate run --probe probe_managed_memory --out /tmp/probe-managed-memory
```

`rules.json` is `[]` and `adapted/` is byte-identical to `hipified/`.

## Boundaries

- The probe proves that one managed allocation can be written by the GPU and
  read by the host on this platform.
- It does not prove concurrent access from host and device, oversubscription,
  multi-GPU migration or prefetch behaviour.
- ROCm managed-memory support can depend on the kernel driver and on
  `HSA_XNACK`/page-migration settings; revalidate on any environment change.
