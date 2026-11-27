# Agent Handoff Context

This document is the takeover context for the next agent. It records decisions,
current state, useful prior findings and known gaps. It intentionally omits long
history that does not change implementation decisions.

## 1. Fixed decisions

```text
Target ROCm:          10.0.0
CUDA Toolkit:         12.9
CUDA Samples:         v12.9
Samples commit:       cab7c66b4fcf35b09ea8fbc4c3c1a9272e92d997
Primary deliverable:  skills
```

- Use C++ cuda-samples v12.9 only.
- Deliver reusable skills, deterministic rules, manifests and evidence.
- Do not use patch files as the migration mechanism.
- Do not run AI models in the normal execution path.
- Do not change the original cuda-samples checkout.
- Do not lower C1/C2/C3 to make a sample pass.

## 2. Why v12.9

- CUDA Samples v12.9 uses CMake and the `Samples/` layout.
- It contains 188 C++ samples and no Python sample tree.
- It remains within the CUDA 12.9 source baseline.
- v13.5 introduces CUDA 13.5, `cpp/` + `python/`, CUDA Tile and newer CCCL,
  and is outside the current fixed baseline.

## 3. Current project state

Implemented:

- Python package scaffold;
- pinned v12.9 manifest with 188 samples;
- priority batches;
- environment check;
- manifest CLI;
- unit tests (53, no GPU required);
- development plan;
- machine resource plan;
- first skill document;
- P0 pipeline: pinned-checkout verification, CUDA-root resolution for
  hipify-clang, pre-HIPIFY include routing, HIPIFY runner, deterministic rule
  engine, Common helper overlay, C1/C2/C3 execution and evidence capture;
- per-sample specs for the P0 batch;
- P0 baseline evidence on ROCm 7.2.4 / `gfx1100` (interim, superseded);
- project-level `convert` developer preview for arbitrary source trees:
  recursive layout, optional per-unit compile database, report/change log,
  requirements aggregation and generated CMake dependency target; no build or
  run step (`docs/generic-conversion.md`);
- Docker-target verification: 99 tests, including real HIPIFY + hipcc
  integration tests for `vectorAdd` and an include-directory CUDA header, plus
  the P0 batch regression.

Not implemented:

- golden numeric comparison (no NVIDIA driver on the validation host);
- P1 and later sample specs;
- overlay coverage beyond the P0 helper surface;
- automatic project builds, CMakeLists rewriting or support claims in the
  generic `convert` command.

## 4. Reference prototype

A previous prototype exists outside this project:

```text
/home/mark/cuda2hip-migrate
```

It targets CUDA Samples v12.0. Treat it as a reference implementation only:

- useful architecture: source/preprocessed/hipified/adapted separation;
- useful rules: `HIP-HOSTMALLOC-ARITY-001`, `HIP-CUBLAS-INCLUDE-001`;
- useful preprocessing: pre-HIPIFY include redirect;
- useful tests and evidence structure;
- **do not reuse v12.0 patches or results as v12.9 evidence**.

If the prototype is unavailable on the new machine, continue from this project
and the development plan.

## 5. Technical lessons that must be preserved

1. `-I` only resolves headers; it does not migrate Common helpers.
2. HIPIFY can rewrite `#include <helper_cuda.h>` to
   `#include <hip/hip_runtime_api.h>`, removing helper definitions.
3. A sample that calls `checkCudaErrors` or `findCudaDevice` needs an overlay
   plus a pre-HIPIFY include redirect.
4. A declared redirect that never matches must fail, not only warn.
5. `expected` must account for both redirects and post-HIPIFY rules.
6. Sample spec relative paths must resolve consistently when the CLI is run from
   another working directory.
7. `findCudaDevice` must accept the flag forms used by cuda-samples, including
   `-device=N` and `--device=N`.
8. Rules must skip comments, strings and literals and be idempotent.
9. `matrixMul` on the previous environment was blocked at C2 by
   `hipProfilerStart()` returning 801. Do not hide that with a no-op rule
   without an explicit decision.
10. No support claim may exceed recorded evidence.

## 5a. Measured pipeline facts (ROCm 7.2.4, added after P0)

Each item states what was observed. Do not upgrade an observation into a general
claim about hipify-clang without a minimal reproduction.

**Tool requirements**

1. A CUDA root needs `bin/`. Complete root: `CUDA SDK version detected: 19`,
   exit 0, output written. Root without `bin/`: detection 0,
   `cannot find libdevice` / `cannot find CUDA installation`, **exit 1, no
   output**. The resolver rejects a `bin/`-less root up front.
2. HIPIFY keeps skipped conditional blocks verbatim and HIP does not define
   `CUDART_VERSION`, so `#if CUDART_VERSION < 5000` becomes live. Both the parse
   and the build define `CUDART_VERSION=12090`.
3. `--skip-excluded-preprocessor-conditional-blocks` is required for the
   Windows-only code in v12.9 samples.
4. `--local-headers` is required for quoted headers; the translated header is
   written next to its input with a `.hip` suffix and must be moved. It does
   not detect a header reached only through `-I`; the generic converter gives
   such CUDA headers a standalone pass and reports conflicts.
5. Two output markers are treated as failure (`[HIPIFY] error`,
   `Error while processing`), plus a missing or empty output file. A bare
   `error:` line is benign: successful runs print `ptxas`/`nvlink`
   architecture diagnostics that start that way.

**Project conventions, not tool requirements**

6. `--cuda-path` is always absolute here for reproducibility. This hipify-clang
   also accepted a relative path in a minimal run.
7. The source directory is kept off hipify's `-I` path. It did break the v12.9
   `vectorAdd` run, but the same flag on a minimal source succeeded, so the
   trigger is sample-specific.
8. ROCm HIP headers are kept off hipify's include path. Making them reachable
   fails only when the translated source includes `<hip/hip_runtime.h>`:
   `__HIP_PLATFORM_NVIDIA__` selects HIP's NVIDIA shim, which references CUDA
   12.x-only identifiers absent from the installed 12.0 headers.
9. The overlay header branches on `__HIPCC__` (real bodies for the build,
   declaration-only stubs for the parse). A unit test pins names, arity and the
   intentional `check(int)` / `check(hipError_t)` difference; it does not prove
   behavioural equivalence.

**Open deviation**

10. `CUDART_VERSION=12090` is defined while the parsed headers are 12.0. That
    reproduces 12.9 conditional semantics with 12.0 headers and can disagree.
    Strict validation needs real CUDA 12.9 headers.
11. `matrixMul` and `asyncAPI` are blocked at C2 by `hipProfilerStart()`
    returning 801 on ROCm 7.2.4, reproducing the v12.0 prototype finding.
    **Decided 2026-11-25: keep blocked** until the P0 batch is re-run on ROCm
    10.0.0. rocTX mapping and no-op stubs are both rejected for now; reopening
    requires a rule id plus a test, not an ad-hoc edit.

## 6. Baseline batches

P0:

```text
deviceQuery
vectorAdd
matrixMul
asyncAPI
simpleStreams
scalarProd
```

P1 semantics/business:

```text
simpleAtomicIntrinsics
simpleVoteIntrinsics
simpleTexture
convolutionFFT2D
NV12toBGRandResize
simpleCudaGraphs
```

P1 probes:

```text
inlinePTX
cdpSimplePrint
vectorAddDrv
```

## 7. Machine resources

Existing:

- AMD Radeon RX 7900 XTX, `gfx1100`, 24 GB;
- NVIDIA GeForce RTX 4060, 8 GB.

Requested:

- AMD Radeon RX 9070 XT, `gfx1201`, 16 GB;
- NVIDIA GeForce RTX 5070 Ti, 16 GB.

Reasons:

- RX 7900 XTX cannot cover RDNA4 / `gfx1201`;
- RTX 4060 is an 8 GB low-memory reference and cannot replace a 16 GB
  same-tier comparison;
- some ROCm 10.0.0 cases have reported partial support on current AMD
  platforms;
- RX 9070 XT and RTX 5070 Ti provide a 16 GB comparison pair.

Official ROCm 10.0.0 support must still be confirmed before procurement or
acceptance.

## 8. Next actions

Done on the target machine:

1. `make test` with the project-conversion suite and real integration test.
2. `make verify` (pinned checkout).
3. `make run-p0` on ROCm 10.0.0 / `gfx1201`.

Remaining:

1. Exercise `convert` on real multi-target projects and promote stable generic
   rules into HIPIFY.
2. Implement P1 specs and overlay coverage only after the P0 review.
3. Keep `convert` source-only: do not add C1/C2/C3 claims or automatic build
   rewriting without separate evidence.

## 9. Context retained

Retained because it affects implementation:

- version pinning;
- sample counts and priority batches;
- ROCm 10.0.0 target;
- v12.9 directory/build layout;
- Common helper overlay and redirect lessons;
- C1/C2/C3 acceptance;
- machine resource constraints;
- no-patch/no-runtime-AI rules;
- known blocked and partial-support boundaries.

## 10. Context intentionally dropped

Dropped because it does not change implementation:

- historical discussion of ZLUDA;
- long comparisons of unrelated compatibility-layer designs;
- v12.0-specific sample patches;
- v13.5 sample details beyond the decision not to use them;
- repeated API explanations already captured as engineering rules.
