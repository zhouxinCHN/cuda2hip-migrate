# cuda-samples v12.0 / vectorAdd: external-path smoke test

Date: 2026-11-24 (revised)

Environment: ROCm 7.2.4, hipify-clang 22.0.0git, hipcc, gfx1100

Sample origin: `third-party/cuda-samples`, git tag `v12.0`, commit
`2b689228b746248b7bb7d2778d5ea0701669f7d3`. The checkout is a nested Git
repository and is excluded from this repository via `.gitignore`.

## What this sample does and does not prove

It proves the external-source path works: an unmodified sample is copied,
translated, built and run, and its own success line is checked.

It does **not** prove cuda-samples `Common` helper support. HIPIFY rewrites

```text
#include <helper_cuda.h>  ->  #include <hip/hip_runtime_api.h>
```

so the NVIDIA helper header never reaches the compiler, and this sample calls no
helper function (`checkCudaErrors`, `findCudaDevice`, timing helpers) at all.
The include is resolvable only because `-I <checkout>/Common` is passed to
`hipify-clang`. The helper migration path is therefore untested here; see
`simpleOccupancy.md` for the sample that actually exercises the overlay.

The Makefile target is named `smoke-vectoradd` for this reason.

## Command

```bash
cuda2hip-migrate run --spec samples/cuda-samples-v12.0/vectorAdd.json
```

Spec content: source, `include_dirs` (`third-party/cuda-samples/Common`),
`success_marker` (`Test PASSED`), `expected: hipify-direct`, no redirects.

## Translation and acceptance

```text
[RULES] applied=0
[EXPECTED] hipify-direct rules=0
[C1] PASS
[Vector addition of 50000 elements]
Test PASSED
Done
[C2] PASS
[C3] PASS
```

`source/vectorAdd.cu` is byte-identical to the checkout file and `rules.json` is
`[]`.

## Boundaries

- One sample, default stream, single launch, no helper call.
- No sample build system is used; `hipcc` is invoked directly by the runner.
- Not evidence of Common helper support, of other samples, or of ROCm 10.0.
