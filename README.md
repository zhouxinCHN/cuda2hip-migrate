# cuda2hip-migrate

Deterministic post-HIPIFY processing and standalone CUDA probe validation for
ROCm.

This project is an early prototype. It proves the execution path and currently
contains two source rules; it is not a complete CUDA-to-HIP migration tool. The
project is self-contained and can be promoted to its own Git repository.

## Scope

```text
original CUDA probe
  -> HIPIFY
  -> deterministic post-HIPIFY rules
  -> hipcc build
  -> run
  -> C1/C2/C3 checks
```

- HIPIFY output and rule-processed output are kept separately.
- Rules are deterministic Python code with positive/negative tests.
- No AI model or patch file participates in the runtime path.
- Unmatched code is left unchanged.
- Probe sources are packaged resources so the installed CLI can find them.

## Requirements

- Linux with an AMD GPU supported by ROCm
- ROCm with `hipcc` and `rocminfo`
- `hipify-clang`
- Python 3.10+
- `diff` and `patch` are not required by this project

The current prototype was verified only with ROCm 7.2.4 and `gfx1100`.

## Development quick start

From the project root:

```bash
make test
make list
make probe-malloc
make probe-pinned
make probe-stream
make probe-managed
make probe-smem
make probe-cublas
make probe-texture
```

`make` sets `PYTHONPATH=src`, so no installation is needed for development.

Install as a local package:

```bash
python3 -m pip install -e .
cuda2hip-migrate list
cuda2hip-migrate run --probe probe_pinned_host_mem
```

## CLI

```text
cuda2hip-migrate list [--json]
cuda2hip-migrate run --probe NAME [--out PATH] [--arch GFX]
cuda2hip-migrate run --spec PATH [--out PATH] [--arch GFX]
cuda2hip-migrate run --source PATH [--include-dir DIR]... [--library NAME]...
                       [--success-marker TEXT] [--out PATH] [--arch GFX]
```

Examples:

```bash
cuda2hip-migrate run --probe probe_malloc_memcpy
cuda2hip-migrate run \
  --probe probe_pinned_host_mem \
  --out /tmp/probe-pinned \
  --arch gfx1100
cuda2hip-migrate run \
  --source third-party/cuda-samples/Samples/0_Introduction/vectorAdd/vectorAdd.cu \
  --include-dir third-party/cuda-samples/Common \
  --success-marker "Test PASSED" \
  --out /tmp/vecadd
```

The architecture is taken from `--arch`, then `HIP_ARCH`, then `rocminfo`.

`--source` runs an external CUDA file, such as one cuda-samples file, through
the same pipeline. The file is copied into `source/` and never modified.
`--include-dir` is forwarded to both `hipify-clang` and `hipcc`; `--library`
adds `-l<name>`; `--success-marker` replaces the C3 marker for that run only.
Packaged probes keep `RESULT=PASS` as the required marker.

## Probes

| Probe | Coverage | Status |
|---|---|---|
| `probe_malloc_memcpy` | device allocation, synchronous H2D/D2H copy, kernel read/write | HIPIFY direct |
| `probe_pinned_host_mem` | pinned host allocation and device round trip | fixed by rule |
| `probe_stream_event` | two non-default streams, async copies, cross-stream event ordering | HIPIFY direct |
| `probe_managed_memory` | managed (unified) memory written by the GPU and read by the host | HIPIFY direct |
| `probe_kernel_dynamic_smem` | dynamic shared memory, `__syncthreads`, multi-block reduction | HIPIFY direct |
| `probe_cublas_gemm` | cuBLAS SGEMM on device buffers (library layer) | fixed by rule + link flag |
| `probe_texture_object` | linear texture object with `tex1Dfetch` reads | HIPIFY direct |

Planned probes are listed in the parent migration plan, not added ahead of
review. Add one probe at a time and stop for review before continuing.

## cuda-samples v12.0

An external cuda-samples checkout is used for adaptation evidence and is
excluded from this repository (`third-party/` in `.gitignore`); the checkout is
a nested Git repository, so only specs, overlay headers and evidence are
committed.

| Sample | What it establishes | Status | Rules |
|---|---|---|---|
| `vectorAdd` | external-source path only; HIPIFY maps its helper include away and the sample calls no helper | HIPIFY direct, C1/C2/C3 (`smoke-vectoradd`) | 0 |
| `simpleOccupancy` | Common overlay is used and `checkCudaErrors` runs; built `-O0` because the sample's own `i * i` overflows `int` | HIPIFY direct + overlay, C1/C2/C3 | 0 |
| `matrixMul` | overlay compiles and `findCudaDevice` runs; `hipProfilerStart` returns 801 on this platform | **blocked** at C2 | 0 |

```bash
make smoke-vectoradd
make sample-simpleoccupancy
make sample-matrixmul
```

Each sample has a committed spec under `samples/cuda-samples-v12.0/`:

```json
{
  "sample": "cuda-samples-v12.0/simpleOccupancy",
  "source": "third-party/cuda-samples/Samples/0_Introduction/simpleOccupancy/simpleOccupancy.cu",
  "include_dirs": ["compat/cuda-samples-v12.0/include", "third-party/cuda-samples/Common"],
  "libraries": [],
  "success_marker": "Test PASSED",
  "expected": "hipify-direct",
  "optimization": "-O0",
  "include_redirects": {"helper_cuda.h": "cuda_samples_compat/helper_cuda.h"}
}
```

`expected` is enforced: a spec declaring `hipify-direct` fails if any rule fires.
`optimization` defaults to `-O2` and is logged when overridden.

### Common helper overlay

`-I <checkout>/Common` only makes headers findable; it does not migrate them, and
HIPIFY rewrites `helper_cuda.h` to a HIP runtime header. Samples that actually
call helpers therefore need an overlay plus a pre-HIPIFY redirect:

```text
compat/cuda-samples-v12.0/include/cuda_samples_compat/helper_cuda.h
  implemented: checkCudaErrors, findCudaDevice
  not implemented: gpuGetMaxGflopsDeviceId selection, compute-capability gates,
                   driver API helpers, image/GL/NVIDIA-library helpers
```

The redirect rewrites `#include <helper_cuda.h>` in the copied source before
HIPIFY, is recorded in `redirects.json`, and warns with `[REDIRECT-MISS]` when a
declared redirect matches nothing. The original checkout is never edited and the
CPU-only Common headers keep coming from it.

Evidence: [`common_overlay.md`](docs/evidence/cuda_samples_12_0/common_overlay.md),
[`vectorAdd.md`](docs/evidence/cuda_samples_12_0/vectorAdd.md),
[`simpleOccupancy.md`](docs/evidence/cuda_samples_12_0/simpleOccupancy.md),
[`matrixMul.md`](docs/evidence/cuda_samples_12_0/matrixMul.md).

Do not claim cuda-samples or `Common` support beyond the rows above. Sample
Makefiles/CMake projects, data files, multi-file samples, unimplemented helpers
and other samples are unclassified.

## Rules

Current registry: `src/cuda2hip_migrate/rules/__init__.py`.

```text
HIP-HOSTMALLOC-ARITY-001
condition: hipHostMalloc with 4 args; arg 3 == 0; arg 4 == hipHostMallocDefault
action: remove argument 3

HIP-CUBLAS-INCLUDE-001
condition: line-start #include directive, outside comments and literals, that is
           exactly "#include <hipblas.h>"
action: rewrite to "#include <hipblas/hipblas.h>"
```

Rules:

- run only after HIPIFY;
- skip comments, strings and character literals;
- require exact conditions;
- are idempotent;
- record rule ID, line and before/after text in `rules.json`.

Rule tests are in `tests/test_hostmalloc_rule.py` and `tests/test_cublas_rule.py`.

## Build specs

A probe may ship an optional build spec, `probes/<probe>.build.json`, for
library-level probes that need extra `hipcc` flags:

```json
{
  "libraries": ["hipblas"]
}
```

Only `libraries` is supported; each entry becomes `-l<name>` for that probe
alone. Probes without a build spec keep the plain build command. The resolved
flags are printed as `[BUILD] flags=...` and recorded in `build.log`. Currently
only `probe_cublas_gemm` uses a build spec. Validation tests are in
`tests/test_build_spec.py`.

## Acceptance

- **C1**: `hipcc` build succeeds.
- **C2**: probe exits with status 0.
- **C3**: probe emits `RESULT=PASS`.

A build alone is not treated as success. Performance is out of scope.

## Artifacts

Each run creates:

```text
<out>/
|-- source/<name>.cu           # original source, byte-identical copy
|-- preprocessed/<name>.cu     # only with include redirects; HIPIFY input
|-- redirects.json             # only with include redirects
|-- hipified/<name>.hip.cu     # untouched HIPIFY output
|-- adapted/<name>.hip.cu      # rules applied; built source
|-- hipify.diff                # HIPIFY input -> HIPIFY output, audit only
|-- rules.json                 # rule hits
|-- hipify.log
|-- build.log
|-- run.log
`-- <name>                     # executable
```

Generated artifacts are written to `/tmp` by default and are not committed.

## Project layout

```text
cuda2hip-migrate/
|-- .github/workflows/ci.yml   # Python tests only, no GPU runner
|-- pyproject.toml
|-- Makefile
|-- README.md
|-- .gitignore
|-- compat/cuda-samples-v12.0/ # Common helper overlay (HIP headers)
|-- samples/cuda-samples-v12.0/ # committed sample specs
|-- docs/evidence/             # curated evidence, committed
|-- skills/                    # Agent workflow documentation, committed
|-- src/cuda2hip_migrate/
|   |-- cli.py
|   |-- runner.py
|   |-- process.py
|   |-- rules/
|   `-- probes/                # .cu resources, included in package data
`-- tests/                     # standard-library unittest suite
```

## Tests

```bash
make test
```

The suite does not require a GPU or ROCm. GPU/ROCm execution is exercised by the
probe commands and must be recorded separately.

## Git hygiene

- Do not commit `__pycache__`, build outputs, virtual environments or run
  artifacts.
- Keep probe sources in `src/cuda2hip_migrate/probes/`; they are package data.
- Keep generated evidence curated under `docs/evidence/`.
- Update `pyproject.toml` version when behavior or package contents change.
- A rule change requires a test change in the same commit.
- No license has been selected for the standalone repository yet. Add a license
  file and matching `pyproject.toml` metadata before publishing.

## Known boundaries

- Only two deterministic rules are implemented.
- No cuda-samples or `Common` helper support is claimed beyond the sample table
  above; `matrixMul` is blocked by `hipProfilerStart` on the validated platform.
- ROCm 10.0 has not been validated.
- `hipify-clang` may print `ptxas`/`nvlink` architecture diagnostics while still
  exiting successfully and producing valid output; the runner trusts the exit
  status and output file, and this must be rechecked when tools are upgraded.
- These probes do not prove performance, long-run stability, concurrency safety
  or full CUDA semantic equivalence.
- See [`docs/evidence/probe_pinned_host_mem_pre_rule.md`](docs/evidence/probe_pinned_host_mem_pre_rule.md)
  and [`skills/cuda-sample-probe-migration/SKILL.md`](skills/cuda-sample-probe-migration/SKILL.md).
