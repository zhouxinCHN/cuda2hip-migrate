# Machine Resources

## Existing resources

| Machine | GPU | Architecture | Memory | Role |
|---|---|---|---:|---|
| AMD validation | Radeon RX 7900 XTX | `gfx1100` | 24 GB | Current AMD baseline |
| NVIDIA reference | GeForce RTX 4060 | Ada | 8 GB | Low-memory CUDA/NPP reference |

## Requested resources

| Priority | GPU | Architecture | Memory | Quantity | Purpose |
|---|---|---|---:|---:|---|
| P0 | Radeon RX 9070 XT | `gfx1201` | 16 GB | 1 | ROCm 10.0.0 RDNA4 validation |
| P0 | GeForce RTX 5070 Ti | Blackwell | 16 GB | 1 | Same-tier NVIDIA comparison |

## Why they are needed

### Radeon RX 9070 XT

- Covers RDNA4 / `gfx1201`;
- validates HIPIFY, hipcc and media libraries on the new architecture;
- provides cross-architecture comparison against `gfx1100` and `gfx1101`;
- exposes architecture-specific failures in texture, stream, event, UVM and
  dynamic shared-memory samples.

`gfx1201` is an optional build target in parts of ROCm 10.0.0 source. Official
ROCm support must be confirmed before purchase or acceptance.

### GeForce RTX 5070 Ti

- Provides a 16 GB same-tier NVIDIA reference;
- generates 16 GB-class CUDA/NPP golden results;
- replaces RTX 4060 for high-resolution and multi-stream comparisons;
- allows comparison against RX 9070 XT with identical inputs and flags.

RTX 4060 remains in service for low-memory smoke tests and existing regression.

## Minimum configuration

| Item | Recommendation |
|---|---|
| CPU architecture | x86_64 |
| System memory | 64 GB or more |
| Storage | 2 TB NVMe SSD or more |
| Power supply | 850 W or more, per full-system power |
| OS | Same Linux LTS version as the existing validation machine |
| Remote access | SSH or out-of-band management |
| Cooling | Continuous GPU load capable |

## Software

### AMD

```text
ROCm 10.0.0
amdgpu driver
hipcc
hipify-clang
rocminfo
hipBLAS / hipFFT / hipRAND / hipCUB
rocJPEG / RPP where required
libva / libdrm / Mesa where required
```

### NVIDIA

```text
Driver matching RTX 5070 Ti
Pinned CUDA Toolkit 12.9
nvcc
cuda-samples v12.9
cuBLAS / cuFFT / cuRAND / NPP
```

## Validation after arrival

1. Run `make env` on the target machine.
2. Run all project unit tests.
3. Run the P0 sample batch.
4. Record GPU, driver, ROCm/CUDA, compiler, optimization level and full logs.
5. Compare RX 9070 XT and RTX 5070 Ti with identical inputs and parameters.
6. Confirm official ROCm 10.0.0 support before making support claims.

## Boundaries

- New hardware does not itself constitute ROCm or CUDA compatibility validation.
- RX 9070 XT does not replace RX 7900 XTX for 24 GB workloads.
- RTX 5070 Ti does not replace existing RTX 4060 regression results.
- No performance or pass-rate claim is valid before measured evidence.
