---
name: cuda-samples-v12.9-rocm10
description: Use when adapting, running or reviewing CUDA Samples v12.9 on ROCm 10.0.0 with HIPIFY, deterministic rules and C1/C2/C3 evidence; covers pinned versions, pre-HIPIFY include routing, the Common helper overlay, hipify-clang environment traps, rule promotion, failure classification and support boundaries.
---

# CUDA Samples v12.9 on ROCm 10.0.0

## Fixed versions

```text
Target ROCm:          10.0.0
CUDA Toolkit:         12.9
CUDA Samples:         v12.9
Samples commit:       cab7c66b4fcf35b09ea8fbc4c3c1a9272e92d997
Primary deliverable:  ROCm compatibility stack and development skills
```

Do not use v13.5 samples or v12.0 patches in this baseline.

## When to use

Use this skill for any C++ cuda-samples v12.9 adaptation or review. The project
manifest contains all 188 samples and their priority batches.

## Project commands

```bash
make test                                                     # unit tests, no GPU needed
make list                                                     # priority batches
make verify                                                   # pinned checkout check
make run-p0                                                   # P0 batch and evidence
python3 -m rocm10_samples_compat verify-checkout               # pin check, rejects dirty/detached-wrong
python3 -m rocm10_samples_compat list-specs
python3 -m rocm10_samples_compat run-sample --sample vectorAdd
python3 -m rocm10_samples_compat run-batch --batch P0_baseline
python3 -m rocm10_samples_compat check-env --strict            # fails on non-target ROCm
python3 -m rocm10_samples_compat convert --input DIR --output DIR \
  [--cuda-root DIR] [--compile-commands FILE | --no-configure] \
  [--configure-arg ARG]
```

`run-sample` and `run-batch` exit non-zero when any sample fails C3. `convert`
converts source only and exits 2 on a blocking conversion issue; its success is
not a build or support claim. With a root `CMakeLists.txt`, `convert`
automatically runs CMake configure into
`<output>/.tmp/<project>/cmake-build` to obtain compile commands. Use
`--no-configure` to force explicit tree mode; an existing
`--compile-commands` database wins over configuration.

## Pipelines

Sample acceptance:

```text
pinned cuda-samples v12.9 (read-only)
  -> source/        verbatim copy of the sample directory
  -> preprocessed/  pre-HIPIFY include routing
  -> hipified/      hipify-clang output
  -> adapted/       deterministic post-HIPIFY rules
  -> hipcc build    C1
  -> run            C2
  -> C3 marker / exit status
  -> evidence.json
```

Project conversion (`convert`, no build/run):

```text
arbitrary CUDA input tree (read-only)
  -> optional .tmp/<project>/cmake-build/compile_commands.json
  -> .migration/source/
  -> .migration/preprocessed/
  -> .migration/hipified/
  -> src/                  converted tree, relative layout preserved
  -> reports/migration.json, changes.jsonl, hipify.log, configure.log
  -> rocm-migration.cmake  compatibility dependencies for the real build
```

Every stage keeps its own directory. All tool-owned temporary data is under the
user-selected output: CMake build files, HIPIFY `--temp-dir`, and child
`TMPDIR/TMP/TEMP`. CMake configure executes project CMake code and is therefore
not a sandbox. The original input/checkout is never written to, and no patch
file exists anywhere in either pipeline. See `docs/generic-conversion.md` for
compile-database behavior and limitations.

## Agent failure-analysis workflow

Use this loop when preprocessing, HIPIFY conversion, extended conversion,
compilation or execution fails.

```text
preprocess -> HIPIFY -> extended conversion -> compile and run
     ^                                             |
     |                                             +-- success: finish and record evidence
     |                                             |
     +---------- modify and retest <---------------+-- failure: agent analysis
```

1. **Locate the failed stage.** Read the preprocessing report, HIPIFY output,
   `build.log` and `run.log`. Keep the original source, converted source, commands
   and complete logs.
2. **Build a minimal reproduction.** Reduce the failure to the CUDA construct or
   API involved and write down its required semantics.
3. **Choose the implementation site:**

   | Finding | Implementation site |
   |---|---|
   | Missing or incorrect systematic HIPIFY mapping | HIPIFY mapping or conversion logic |
   | Source structure left after HIPIFY, or a post-HIPIFY pattern | Extended conversion tool |
   | Missing declaration, signature, constant, library symbol or runtime behavior | ROCm API implementation |

   One failure may require coordinated changes in more than one component.
4. **Implement a general fix.** Cover the CUDA semantic class rather than a single
   sample. Add positive and negative tests, idempotence tests for conversion
   rules, and semantic tests for API or runtime changes.
5. **Retest from preprocessing.** Run the complete conversion, build and execution
   flow again. Finish only after C1/C2/C3 passes, then save the evidence.
6. **Promote the result into this skill.** Record the failure signature, minimal
   reproduction, analysis path, modified component, test and code revisions, and
   the final verification command. Later upgrades start from this record.

The agent performs failure analysis during development. Accepted fixes become
versioned HIPIFY, extended-tool or ROCm code; normal conversion, compilation and
execution then run without an agent.

## Non-negotiable rules

1. Preserve the original cuda-samples checkout unchanged.
2. `-I` is header lookup only; it is not Common helper migration.
3. HIPIFY maps `helper_cuda.h` to `hip/hip_runtime_api.h` and drops every helper
   definition the sample calls.
4. A sample that calls Common helpers needs an overlay plus a pre-HIPIFY include
   redirect.
5. A declared redirect that never matches must fail the run.
6. Pre-HIPIFY redirects and post-HIPIFY rules must both be recorded in evidence.
7. No patch files.
8. No AI model in the normal execution path. Agents analyze development failures;
   accepted fixes are committed as deterministic tool or ROCm code.
9. Rules must skip comments, strings and literals, and be idempotent.
10. Every rule requires positive, negative and idempotence tests.
11. Do not lower C1/C2/C3 to make a sample pass.
12. Do not claim support beyond recorded evidence.

## hipify-clang environment facts

Each claim below is followed by what was actually observed. Unless a claim says
otherwise, the observation is from ROCm 7.2.4 / hipify-clang 22.0.0git; the
temporary-directory and CMake response-file facts were measured on ROCm 10.0.0
/ hipify-clang 23. Distinguish *tool requirements* from *project conventions*;
only the former may be stated as a fact about hipify-clang.

### Tool requirements

1. **A CUDA root needs `bin/`.** With `include/`, `nvvm/libdevice/` and `bin/`
   the run reports `CUDA SDK version detected: 19`, exits 0 and produces output.
   Without `bin/` it reports `CUDA SDK version detected: 0`, prints
   `cannot find libdevice` / `cannot find CUDA installation`, **exits 1 and
   produces no output.** The runner refuses a root without `bin/` rather than
   discovering this later.
2. **Skipped conditional blocks stay in the output.** hipify skips a block that
   the preprocessor excludes but emits it verbatim. `deviceQuery.cpp` guards a
   driver-API path with `#if CUDART_VERSION < 5000`; HIP does not define
   `CUDART_VERSION`, an undefined identifier evaluates to `0`, and the dead block
   becomes live. Defining `CUDART_VERSION=12090` for both hipify and hipcc keeps
   the CUDA 12.9 branch selection.
3. **`--skip-excluded-preprocessor-conditional-blocks`** is required for v12.9
   sources: without it, Windows-only code (`sprintf_s`) fails the parse on Linux.
4. **`--local-headers`** is required for quoted local headers. The translated
   header (`scalarProd_kernel.cuh.hip`) is written next to its input and must be
   moved into place. It does **not** translate a header reached only through
   `-I`: hipify reports `No local headers detected` and leaves the header
   unchanged, so project conversion must translate such headers as standalone
   inputs.
5. **Check the output, not only the exit status.** Two failure markers are
   checked (`[HIPIFY] error`, `Error while processing`) plus a missing or empty
   output file. Translation failures do exit non-zero, but input/config failures
   do not: `hipify-clang /tmp/missing.cu` prints
   `[HIPIFY] error: source file: ... doesn't exist` and **exits 0**. A bare
   `error:` line is **not** a failure signal: a successful translation still
   prints `error: must pass in an explicit nvptx64 gpu architecture to 'ptxas'`.
6. **`--temp-dir` does not contain every temporary file by itself.** Measured
   on ROCm 10.0.0: `--temp-dir` held the final `.hip` temp file while Clang's
   preprocessing temp file still appeared under `TMPDIR`. Set both, with
   `TMPDIR` pointing into the caller's output root.
7. **CMake CUDA compile commands can hide include paths.** Measured with CMake
   3.28.3 and CUDA 12.9: CMake emits
   `--options-file CMakeFiles/<target>.dir/includes_CUDA.rsp`; the converter
   must expand `--options-file` and `@file` arguments before extracting `-I`.

### Project conventions (not tool requirements)

8. **`--cuda-path` is always absolute.** This version also accepted a relative
   path in a minimal run, so this is a reproducibility decision, not a
   requirement.
9. **The source directory is not added to hipify's `-I` path.** Adding it did
   break the v12.9 `vectorAdd` run (`EXIT_FAILURE`/`blockDim` undeclared), but
   the same flag on a minimal two-file source succeeded, so the trigger is
   sample-specific and is not a general rule.
10. **ROCm HIP headers are not made reachable during translation.** This does
    fail, but only when the translated source actually includes
    `<hip/hip_runtime.h>`: hipify defines `__HIP_PLATFORM_NVIDIA__`, HIP selects
    `nvidia_detail/nvidia_hip_runtime_api.h`, and it references CUDA 12.x-only
    identifiers (`cudaMemPrefetchAsync_v2`) that CUDA 12.0 headers do not have —
    exit 1, no output. Keeping HIP headers off the include path and branching the
    overlay header on `__HIPCC__` avoids the situation entirely.

### CUDA 12.9 header requirements

The target environment uses real CUDA 12.9 headers (`CUDA_VERSION=12090`). The
CUDA root needs `include/`, `nvvm/libdevice/libdevice.10.bc` and `bin/`. ROCm 10's
clang 23 additionally requires `libcurand-dev-12-9` and `cuda-cccl-12-9` during
parsing. Samples that include `cuda_profiler_api.h` also need
`cuda-profiler-api-12-9`. A missing package stops HIPIFY before compilation.

## Common helper overlay

`src/rocm10_samples_compat/overlay/rocm10_compat/helper_cuda.h` implements the
CUDA-coupled helpers on HIP with unchanged names and signatures.

**Design constraint:** the same file is parsed by hipify-clang (still CUDA, with
`__HIP_PLATFORM_NVIDIA__` defined) and compiled by hipcc (with `__HIPCC__`
defined). The header therefore has two branches:

- `#if defined(__HIPCC__)` — includes HIP and provides the real bodies.
- otherwise — declaration-only stubs so translation can proceed.

A unit test pins the function names, their arity, and the one intentional
difference (`check` takes `int` in the parse branch because the CUDA call sites
pass `cudaError_t`, and `hipError_t` in the build branch). It does **not** prove
the branches behave identically: the parse branch has no bodies at all.

Supported surface today: `check`, `checkCudaErrors`, `getLastError`,
`getLastCudaError`, `_ConvertSMVer2Cores`, `_ConvertSMVer2ArchName`,
`gpuDeviceInit`, `gpuGetMaxGflopsDeviceId`, `findCudaDevice`,
`checkCudaCapabilities`.

Anything else must fail at C1 with a clear missing-symbol error. Extend the
overlay only together with a test and recorded evidence.

Host-only Common headers (`helper_string.h`, `helper_timer.h`, `helper_image.h`,
`exception.h`) contain no CUDA calls and are used directly from the checkout via
`-I`; they must not be duplicated into the overlay.

`src/rocm10_samples_compat/overlay/rocm10_compat/hip_profiler.h` is a second,
separate overlay header. It provides `rocm10_compat::hipProfilerStart` and
`hipProfilerStop` on top of rocTX and is pulled in by `HIP-PROFILER-ROCTX-001`
after HIPIFY, so hipify-clang never parses it; its declaration-only branch exists
for symmetry with `helper_cuda.h`. It is not part of `helper_cuda.h` on purpose —
a sample may use a profiler session without using any Common helper.

`findCudaDevice` accepts both `-device=N` and `--device=N`.

## Deterministic rules

| Rule id | Stage | Purpose |
|---|---|---|
| `PRE-HIPIFY-INCLUDE-REDIRECT` | pre-HIPIFY | route `helper_cuda.h` to the overlay |
| `HIP-CUBLAS-INCLUDE-001` | post-HIPIFY | `hipblas.h` → `hipblas/hipblas.h` (ROCm installs it in a subdirectory) |
| `HIP-HOSTMALLOC-ARITY-001` | post-HIPIFY | drop HIPIFY's redundant `0` in the four-argument `hipHostMalloc` |
| `HIP-PROFILER-ROCTX-001` | post-HIPIFY | `hipProfilerStart/Stop` → `rocm10_compat::hipProfilerStart/Stop`, which wrap `roctxProfilerResume/Pause(0)` |

Rules are pure `str -> (str, changes)` functions, skip comments/strings/literals,
and must be idempotent. The pre-HIPIFY redirect fires for every P0 sample;
`HIP-PROFILER-ROCTX-001` fires for the two samples that bracket their compute
with a CUDA profiler session.

### `HIP-PROFILER-ROCTX-001`

`hipProfilerStart` / `hipProfilerStop` are `HIP_DEPRECATED` and documented to
return `hipErrorNotSupported`; the symbols left in `libamdhip64.so` are `hip_4.2`
ABI stubs. Measured on ROCm 10.0.0 they return 801 in every situation, including
under an attached `rocprofv3`, so the result does not depend on whether a
profiler is listening. A sample that brackets its compute with these calls dies
before doing any work.

The rule rewrites a **zero-argument** call to the qualified
`rocm10_compat::hipProfilerStart/Stop` and adds
`#include "rocm10_compat/hip_profiler.h"` after the HIP runtime include. Anything
else — arguments, member or qualified calls, `hipProfilerStartExt`, comments and
strings — is left untouched, which also makes the rule idempotent.

The wrapper preserves the instrumentation contract rather than erasing it:
`roctxProfilerResume(0)` / `roctxProfilerPause(0)` pause and resume profiling for
the whole process, the same contract as `cudaProfilerStart/Stop`. Measured on
ROCm 10.0.0: both return 0 when no tool is attached and when the requested state
already holds.

**Link requirement.** The wrapper needs `librocprofiler-sdk-roctx`; samples that
use it declare `rocprofiler-sdk-roctx` in `spec.libraries`.
`libROCTX64.so` does **not** export `roctxProfilerResume/Pause` — only
`librocprofiler-sdk-roctx.so` does. Linking `-lroctx64` fails with undefined
references.

## Acceptance

- **C1**: `hipcc` build succeeds.
- **C2**: sample exits 0.
- **C3**: sample success marker **or** exit status, recorded as `c3.mode`.

Compilation alone is not support. A sample can be `hipify-direct`, `rule-fixed`,
`manual-review`, `blocked` or `unsupported`; record the actual state.

`c3.mode = "exit-status"` is a weaker claim than a verified marker and must be
reported as such. `simpleStreams` has no verification at all.

## Failure classification

- HIPIFY failure: translation stage; C1 not reached.
- Redirect miss: configuration or Common helper coverage failure; stop the run.
- Rule test failure: the rule set is unusable; stop.
- C1 failure: inspect `adapted/`, include order and HIP headers.
- C2 failure: read `run.log`; a HIP `801`/`hipErrorNotSupported` on a profiling
  call is a platform gap, not a migration bug. Route it with
  `HIP-PROFILER-ROCTX-001`; do not stub the call to a no-op, because the sample's
  own self-check still passes with instrumentation disabled, so a no-op looks
  green while measuring nothing.
- C3 failure: inspect the sample's own verification; never lower the threshold.

## Evidence requirements

`artifacts/<sample>/evidence.json` records: sample path, tag and commit, ROCm and
HIP versions, gfx target, hipify-clang and CUDA-header versions, include
directories, compile defines, redirects declared and matched, rule hits, C1/C2/C3,
the C3 mode and marker, artifact paths, and explicit limitations.

## Machine resources

Existing: RX 7900 XTX `gfx1100` 24 GB; RTX 4060 8 GB (no driver on the current
validation host, so no CUDA golden can be generated there).

Requested: RX 9070 XT `gfx1201` 16 GB; RTX 5070 Ti 16 GB. Confirm official ROCm
10.0.0 support before procurement or acceptance.

## Current status (measured)

P0 batch on the **target** environment — ROCm 10.0.0 (HIP 7.15.26333, AMD clang
23.0.0git), `gfx1201` (RX 9070 XT), real CUDA 12.9 headers (`CUDA_VERSION 12090`):

| Sample | C1 | C2 | C3 | Status |
|---|---|---|---|---|
| deviceQuery | PASS | PASS | PASS | rule-fixed |
| vectorAdd | PASS | PASS | PASS | rule-fixed |
| matrixMul | PASS | PASS | PASS (`Result = PASS`) | rule-fixed (profiler rule) |
| asyncAPI | PASS | PASS | PASS (exit status) | rule-fixed (profiler rule) |
| simpleStreams | PASS | PASS | PASS (exit status) | rule-fixed |
| scalarProd | PASS | PASS | PASS | rule-fixed |

`run-batch --batch P0_baseline` exits 0. Evidence carries no limitations: neither
the ROCm-version nor the CUDA-header deviation applies any more. See
`docs/evidence/p0-baseline-rocm10-gfx1201.md`.

The interim ROCm 7.2.4 / `gfx1100` run is archived in
`docs/evidence/p0-baseline-rocm7.2.4-gfx1100.md` and must not be quoted as a
current result.

## Next action

P0 is complete on the target and the generic converter is a tested developer
preview. Automatic CMake configuration, response-file expansion and output-root
temporary containment are implemented and measured; the converter still stops
at source conversion. Before expanding either:

1. Add P1 sample specs and overlay coverage; do not infer support for samples
   without C1/C2/C3 evidence.
2. Exercise `convert` on real multi-target projects with generated or provided
   `compile_commands.json`; generic, stable rules should be promoted to HIPIFY.
3. The target machine is a container; see
   `docs/evidence/p0-baseline-rocm10-gfx1201.md` for the image, device
   passthrough flags and CUDA-root packages.
4. `primary_gfx_architecture()` now preserves rocminfo agent order, so
   `--arch gfx1201` is no longer required on this host, though it remains
   supported for explicit selection.
