---
name: cuda-sample-probe-migration
description: Use when adding, migrating, rerunning or reviewing a standalone CUDA probe in the cuda2hip-migrate repository with HIPIFY, deterministic Python rules and HIP on ROCm; covers rule tests, C1/C2/C3 acceptance, artifact capture and safe promotion of fixes.
---

# CUDA probe migration

## Scope

Use this workflow for one self-contained CUDA probe packaged in this
repository:

- The probe must not depend on cuda-samples `Common/helper_*`.
- The probe must not depend on a sample Makefile or CMake build.
- Process one probe at a time and stop for review before adding the next one.

The same pipeline also runs an external CUDA file with `--source`, which is how
cuda-samples v12.0 samples are adapted through committed specs under
`samples/cuda-samples-v12.0/` (`make sample-simpleoccupancy`). The external file
is copied into `source/` and never modified; the checkout stays outside the
repository (`third-party/` is ignored).

Do not extrapolate this workflow into a claim that all cuda-samples or customer
projects can be migrated automatically.

## External sample mode

```bash
cuda2hip-migrate run --spec samples/cuda-samples-v12.0/<sample>.json
```

A spec pins `schema_version`, `cuda_samples_commit`, source path,
`include_dirs`, `libraries`, `success_marker`, `expected` (`hipify-direct` /
`rule-required` / `unknown`), `expected_redirects`, `optimization`
(`-O0`..`-O3`, default `-O2`), `include_redirects` and `optional_redirects`.

- Paths are resolved relative to the spec file, never the CWD.
- `expected` covers **post-HIPIFY rules only** and is enforced; use
  `expected_redirects` for the exact redirect count.
- A declared redirect that matches nothing **fails the run**. Only list a header
  in `optional_redirects` when a sample may legitimately omit it; otherwise an
  unused overlay would look like a successful migration.

Ad-hoc runs still support `--source PATH --include-dir DIR --library NAME
--success-marker TEXT`.

Rules for sample adaptation:

- `-I`/`--include-dir` is header lookup, **not** helper migration. HIPIFY maps
  `helper_cuda.h` to `<hip/hip_runtime_api.h>`, so a sample that only includes it
  proves nothing about Common helpers.
- A sample that *calls* `checkCudaErrors`, `findCudaDevice`, `sdkGetTimerValue`
  or similar needs the overlay plus a pre-HIPIFY include redirect; measure it,
  never assume it.
- Redirect before HIPIFY, never reverse a post-HIPIFY `<hip/hip_runtime_api.h>`
  back to a helper header: the two cannot be distinguished. Both `<header>` and
  `"header"` include forms are redirected.
- The overlay has no CUDA/HIP dependency in its argument parsing
  (`cuda_samples_compat/argparse.h`), so it is host-testable with `g++`.
- When `hipify-clang` cannot parse a source at all (no HIP output is produced),
  a post-HIPIFY rule cannot help. Use `hipify_forced_includes` for a parse-only
  shim, record it in the spec and evidence, and keep it out of the build.
- Never edit the sample to make it pass. If a sample's own check fails for a
  sample-level reason (e.g. signed overflow), record it and use an explicit,
  logged build setting such as `optimization: "-O0"` rather than weakening C2/C3.
- Use the sample's own success line as `success_marker`; never lower C2.
- Record tag, commit, command, transcript and boundaries in `docs/evidence/`.
- A blocked sample is `blocked`, not `supported`. If a HIP API returns
  `hipErrorNotSupported`, report it; do not silence it with a no-op rule without
  an explicit decision.

## Pipeline

```text
Original CUDA source
  -> HIPIFY output (hipified/, preserved unchanged)
  -> deterministic Python rules (src/cuda2hip_migrate/rules/)
  -> adapted/ source
  -> hipcc build and run
```

No patch files are used. HIPIFY's own translation is not a fix record. Any
change that cannot be expressed as an explicit deterministic rule requires a
new rule or human review.

## Current probes and rules

| Probe | Status | Rules |
|---|---|---|
| `probe_malloc_memcpy` | HIPIFY direct | 0 |
| `probe_pinned_host_mem` | rule-fixed | `HIP-HOSTMALLOC-ARITY-001` |
| `probe_stream_event` | HIPIFY direct | 0 |
| `probe_managed_memory` | HIPIFY direct | 0 |
| `probe_kernel_dynamic_smem` | HIPIFY direct | 0 |
| `probe_cublas_gemm` | rule-fixed + link flag | `HIP-CUBLAS-INCLUDE-001` |
| `probe_texture_object` | HIPIFY direct | 0 |

External cuda-samples v12.0 status:

| Sample | Status | Rules |
|---|---|---|
| `vectorAdd` | smoke only: helper include mapped away, no helper called | 0 |
| `simpleOccupancy` | overlay used, C1/C2/C3 at `-O0` (sample `int` overflow at `-O2`) | 0 |
| `fp16ScalarProduct` | overlay's `findCudaDevice` runs to C3; quoted includes; `half2` math | 0 |
| `cppOverload` | `cudaFuncSetCacheConfig`/`GetAttributes`; secondary `.cuh` compiled, not translated | 0 |
| `simpleMultiCopy` | 4-stream pipeline; overlay device-selection helpers; HIPIFY parse shim for host `max` | 0 |
| `matrixMul` | blocked at C2: `hipProfilerStart` returns 801 | 0 |

Implemented rule:

```text
HIP-HOSTMALLOC-ARITY-001
Condition: hipHostMalloc has four arguments; argument 3 is 0 and argument 4 is
           hipHostMallocDefault
Action: remove argument 3

HIP-CUBLAS-INCLUDE-001
Condition: line-start #include directive, outside comments and literals, that is
           exactly "#include <hipblas.h>"
Action: rewrite to "#include <hipblas/hipblas.h>"
```

Pre-rule failure evidence:

```text
no matching function for call to 'hipHostMalloc'
requires at most 3 arguments, but 4 were provided
```

See `docs/evidence/probe_pinned_host_mem_pre_rule.md`.

`probe_cublas_gemm` adds a second pre-rule failure: HIPIFY emits
`#include <hipblas.h>` while ROCm 7.2.4 ships `hipblas/hipblas.h`, and the
library must be linked. See
`docs/evidence/probe_cublas_gemm_pre_rule.md`.

## hipify-clang environment rules

Measured on ROCm 7.2.4; each of these costs an hour to rediscover.

1. `hipify-clang` needs a CUDA root with `include/`, `nvvm/libdevice/` **and
   `bin/`**. The Debian/Ubuntu toolkit splits these, so the runner builds a
   symlinked root in the artifact directory and records it in `cuda-root.json`.
2. Without `bin/` hipify logs `CUDA SDK version detected: 0`, then fails every
   CUDA identifier **and still exits 0**.
3. `--cuda-path` must be absolute.
4. Never put the source directory on hipify's include path; it treats the
   translation unit as a local header and translation fails.
5. `--skip-excluded-preprocessor-conditional-blocks` keeps Windows-only code out
   of the parse; `--local-headers` is what makes a quoted `*.cuh` translate. The
   translated header appears next to its input as `<name>.hip` and the runner
   moves it into `hipified/`.
6. A sample that branches on `CUDART_VERSION` needs the macro defined (spec key
   `cudart_version`), because HIP does not define it and `#if CUDART_VERSION <
   N` then becomes live code.
7. Do not trust the exit status alone: the runner also fails on `[HIPIFY] error`
   / `Error while processing`, and on a missing or empty output file.

## Build specs

A probe that needs a library ships `probes/<probe>.build.json`:

```json
{"libraries": ["hipblas"]}
```

The runner turns each entry into `-l<name>` for that probe only. Do not add
global link flags for a single probe, and do not edit the original CUDA probe to
work around a link failure.

## Standard workflow

1. Read the original `.cu` and confirm that it covers one clear capability.
2. Run the probe and classify it as HIPIFY-direct or rule-required.
3. On failure, preserve the HIPIFY output and compiler error. Do not modify the
   original CUDA probe.
4. Prefer a deterministic rule in `rules/`, with positive, negative and
   idempotence tests.
5. Run the full unit test suite, then run the complete probe.
6. Confirm C1/C2/C3 and record the rule hits.
7. Update this skill and the probe status. Stop for review before starting the
   next probe.

Run:

```bash
cd <cuda2hip-migrate-repository>
make test
make probe-pinned
```

After installation:

```bash
cuda2hip-migrate run --probe probe_pinned_host_mem
```

Success criteria:

```text
[RULE] HIP-HOSTMALLOC-ARITY-001 line=35
[RULES] applied=1
PROBE=pinned_host_mem ... RESULT=PASS
[C1] PASS
[C2] PASS
[C3] PASS
```

## Rule requirements

- Rules operate only on HIPIFY output.
- Match conditions must be explicit; do not use broad text replacement.
- Skip comments, strings and character literals.
- Leave unmatched code unchanged and report the rule hit count.
- Rules must be idempotent.
- Every rule must have positive, negative and idempotence tests in `tests/`.
- Rules must not depend on a probe's incidental file path or line number.
- The runtime path must not call an AI model or modify rules automatically.

## Failure classification

- **HIPIFY fails or produces no file**: translation failure; C1 was not reached.
- **Unit tests fail**: the rule set is not usable; stop.
- **Zero rules matched but a compiler/runtime error remains**: a rule or human
  review is missing. Do not hide the error.
- **C1 fails**: inspect the `adapted/` source, hipcc arguments and HIP headers.
- **C2 fails**: inspect the runtime error, exit status and `run.log`.
- **C3 fails**: inspect the probe's internal comparison and output. Do not lower
  the validation threshold.

## Promotion criteria

Add a public rule only when:

- the trigger condition can be stated precisely;
- the change is repeatable and idempotent;
- positive and negative tests exist;
- the change does not depend on incidental code layout;
- regression does not break existing probes.

A one-off change that requires business judgment must not become a rule. Mark it
as `manual-fix` or `unsupported` and send it to human review.

## Boundaries

- The current probes are validated only on ROCm 7.2.4 and gfx1100. ROCm 10.0
  must be rerun separately.
- `hipify-clang` may print `ptxas`/`nvlink` architecture diagnostics while still
  exiting successfully and producing valid output. Treat the exit status and
  output file as authoritative, and recheck this after a tool upgrade.
- Do not modify the original CUDA probe to fit HIP. Post-HIPIFY changes must be
  recorded by deterministic rules.
- Probes do not prove performance, long-run stability, concurrency safety or
  full CUDA semantic equivalence.
