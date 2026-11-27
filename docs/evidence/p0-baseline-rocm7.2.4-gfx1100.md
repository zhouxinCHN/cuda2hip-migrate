# P0 Baseline Evidence — ROCm 7.2.4 / gfx1100

Status: **interim environment**. This is not ROCm 10.0.0 evidence. It records what
the pipeline does on the machine that was available, so that the same commands
can be re-run unchanged when a ROCm 10.0.0 target exists.

## Environment

| Item | Value |
|---|---|
| CUDA Samples | `v12.9` @ `cab7c66b4fcf35b09ea8fbc4c3c1a9272e92d997` (verified, clean) |
| ROCm detected | `7.2.4` (target is `10.0.0`) |
| HIP | `7.2.53211-97f5574fe2`, AMD clang `22.0.0git` |
| hipify-clang | from `/opt/rocm-7.2.4/bin/hipify-clang` |
| GPU | AMD Radeon RX 7900 XTX, `gfx1100`, 24 GB |
| CUDA headers used by HIPIFY | `12.0` (Debian `nvidia-cuda-toolkit`), **not** 12.9 |
| hipcc flags | `-O3 -std=c++17 --offload-arch=gfx1100 -DCUDART_VERSION=12090` |
| Date | 2026-11-25 |

Two limitations follow directly from the table and are written into every
`evidence.json`:

1. hipify-clang parsed against CUDA 12.0 headers, not the 12.9 baseline.
2. Execution happened on ROCm 7.2.4, not the ROCm 10.0.0 target.

## Result matrix

| Sample | C1 build | C2 exit 0 | C3 | Status | Blocked at |
|---|---|---|---|---|---|
| deviceQuery | PASS | PASS | PASS (`Result = PASS`) | rule-fixed | — |
| vectorAdd | PASS | PASS | PASS (`Test PASSED`) | rule-fixed | — |
| matrixMul | PASS | **FAIL** | FAIL | blocked | C2 |
| asyncAPI | PASS | **FAIL** | FAIL | blocked | C2 |
| simpleStreams | PASS | PASS | PASS (exit status only) | rule-fixed | — |
| scalarProd | PASS | PASS | PASS (`Test passed`) | rule-fixed | — |

`rule-fixed` here means the pre-HIPIFY include redirect fired; no post-HIPIFY
rule matched in any P0 sample. The rule registry was still applied and recorded.

## Blocked samples

`matrixMul` and `asyncAPI` both stop at the same call:

```text
HIP error at matrixMul.cu:337 code=801(hipErrorNotSupported) "hipProfilerStart()"
HIP error at asyncAPI.cu:110  code=801(hipErrorNotSupported) "hipProfilerStart()"
```

This reproduces the v12.0 prototype finding exactly. It is recorded as `blocked
at C2` and is **not** worked around.

### Decision (recorded 2026-11-25): keep blocked

HIP 7.2.4 marks `hipProfilerStart` deprecated ("use roctracer/rocTX instead"),
and this machine does provide rocTX (`rocprofiler-sdk-roctx/roctx.h`,
`libroctx64.so`). Both a rocTX mapping and a no-op stub are technically
possible.

The decision is to change nothing yet:

- `801` is a ROCm 7.2.4 behaviour and the target is ROCm 10.0.0; fixing an
  interim environment would fix a problem that may not exist on the target;
- a rocTX mapping would add a library dependency and would not be semantically
  identical to the CUDA profiler session API;
- a no-op would let the samples reach C3 while profiling does nothing, which is
  the weakest possible outcome for a sample whose purpose is instrumentation.

Both samples stay `blocked at C2` until the P0 batch is re-run on ROCm 10.0.0.
If the failure persists there, revisit this decision with a rule id and a test,
not with an ad-hoc edit.

## C3 modes actually used

| Mode | Samples | Meaning |
|---|---|---|
| `marker` | deviceQuery, vectorAdd, matrixMul, scalarProd | the sample's own success string was found in `run.log` |
| `exit-status` | asyncAPI, simpleStreams | no success string exists; asyncAPI self-verifies through its exit code, simpleStreams performs no verification at all |

`simpleStreams` therefore has the weakest C3 in the batch: exit status only, no
value check. Do not describe it as verified.

## Pipeline facts established by this run

Stated as observations. Items 1-4 are tool behaviour; items 5-7 are project
conventions and are labelled as such.

1. `hipify-clang` does not accept the Debian/Ubuntu CUDA layout. A synthetic root
   with `include/`, `nvvm/libdevice/libdevice.10.bc` **and `bin/`** is required.
   Measured: with `bin/` the run reports `CUDA SDK version detected: 19` and
   writes output; without `bin/` it reports detection `0`, prints
   `cannot find libdevice` / `cannot find CUDA installation`, **exits 1 and
   writes nothing**. (An earlier draft of this document said the run still exits
   0; that was a measurement error in the shell pipeline, not tool behaviour.)
2. HIPIFY leaves skipped conditional blocks verbatim. `#if CUDART_VERSION < 5000`
   in `deviceQuery.cpp` would become *active* under hipcc because HIP does not
   define `CUDART_VERSION`; `-DCUDART_VERSION=12090` keeps CUDA 12.9 semantics.
   **Deviation to record:** the headers being parsed are 12.0, so the macro value
   reproduces 12.9 *conditional* semantics without 12.9 headers. The two can
   disagree; this is why the parsed header version is recorded separately.
3. `--skip-excluded-preprocessor-conditional-blocks` is required, otherwise
   Windows-only code (`sprintf_s`) fails the parse.
4. `--local-headers` is required for quoted local headers
   (`scalarProd_kernel.cuh`); the translated header is written next to its input
   with a `.hip` suffix and must be moved into place.
5. *(convention)* `--cuda-path` is always absolute here. This hipify-clang also
   accepted a relative path in a minimal run, so this is for reproducibility.
6. *(convention)* The source directory is kept off hipify's include path. It did
   break `vectorAdd`, but the same flag on a minimal source succeeded, so the
   trigger is sample-specific rather than a general rule.
7. *(convention)* ROCm HIP headers are kept off hipify's include path. Making
   them reachable fails only when the translated source includes
   `<hip/hip_runtime.h>`: `__HIP_PLATFORM_NVIDIA__` selects HIP's NVIDIA shim,
   which references CUDA 12.x-only identifiers missing from the installed 12.0
   headers.
8. Two output markers are treated as failure (`[HIPIFY] error`,
   `Error while processing`) plus a missing or empty output file. Translation
   failures exit non-zero, but input/config failures do not: a missing source
   file prints `[HIPIFY] error: source file: ... doesn't exist` and exits 0. A
   bare `error:` is not a failure signal: successful runs print `ptxas`/`nvlink`
   architecture diagnostics in that form.

## Reproduction

```bash
make test
python3 -m rocm10_samples_compat verify-checkout
python3 -m rocm10_samples_compat run-batch --batch P0_baseline
```

Artifacts for every sample are under `artifacts/<sample>/`:
`source/`, `preprocessed/`, `hipified/`, `adapted/`, `hipify.diff`,
`redirects.json`, `rules.json`, `build.log`, `run.log`, `evidence.json`.

`artifacts/` is git-ignored; regenerate it rather than commit it.

## What this does not prove

- No ROCm 10.0.0 behaviour.
- No `gfx1201` behaviour.
- No golden numeric comparison against a CUDA reference.
- No performance statement.
- No statement about the remaining 182 samples.
