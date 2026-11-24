# probe_texture_object: HIPIFY-direct evidence

Date: 2026-11-24

Environment: ROCm 7.2.4, hipify-clang 22.0.0git, gfx1100

## Purpose

Record that linear texture objects and device-side `tex1Dfetch` translate and
run without any post-HIPIFY rule.

## Capability covered

- `cudaCreateChannelDesc<float>()` and a `cudaResourceDesc` for a linear
  resource over device memory;
- a `cudaTextureDesc` with `cudaReadModeElementType`;
- `cudaCreateTextureObject`, a kernel fetching through `tex1Dfetch<float>`
  (`__device__` template from `hip/amd_detail/texture_indirect_functions.h`),
  then `cudaDestroyTextureObject`;
- exact float comparison against the host input, since element reads should be
  bit-preserving.

## HIPIFY translation

```text
cudaTextureObject_t        -> hipTextureObject_t
cudaCreateChannelDesc<T>() -> hipCreateChannelDesc<T>()
cudaResourceDesc           -> hipResourceDesc
cudaResourceTypeLinear     -> hipResourceTypeLinear
cudaTextureDesc            -> hipTextureDesc
cudaReadModeElementType    -> hipReadModeElementType
cudaCreateTextureObject    -> hipCreateTextureObject
cudaDestroyTextureObject   -> hipDestroyTextureObject
```

`tex1Dfetch` is unchanged; HIP provides it for texture objects. No rule is
required.

## Run result

```text
[RULES] applied=0
[C1] PASS
PROBE=texture_object device=AMD Radeon RX 7900 XTX elements=4113 resource=linear RESULT=PASS
[C2] PASS
[C3] PASS
```

Recreated with:

```bash
cuda2hip-migrate run --probe probe_texture_object --out /tmp/probe-texture
```

`rules.json` is `[]` and `adapted/` is byte-identical to `hipified/`.

## Boundaries

- Only a linear resource with `tex1Dfetch` is covered. Array, pitch2D, mipmapped
  and layered resources, texture references (as opposed to objects), normalized
  reads with linear filtering and 3D linear filtering are not.
- HIP documents that 3D linear filtering is unsupported on GFX90A boards; that
  path is not exercised here and this probe says nothing about it.
- Texture cache behaviour and performance are out of scope.
