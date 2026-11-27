# ROCm 10.0.0 CUDA Samples v12.9 Compatibility Project

Compatibility validation project for **CUDA Samples v12.9** on **ROCm 10.0.0**.

## Fixed decisions

| Item | Value |
|---|---|
| Target ROCm | `10.0.0` |
| CUDA Toolkit baseline | `12.9` |
| CUDA Samples | `v12.9` |
| CUDA Samples commit | `cab7c66b4fcf35b09ea8fbc4c3c1a9272e92d997` |
| Primary deliverable | Skills |
| Migration mechanism | HIPIFY + deterministic rules + evidence |
| Patch files as migration mechanism | Not allowed |

The project currently provides the repository scaffold, pinned sample manifest,
environment check, development plan, handoff context and the first skill, plus a
working P0 migration pipeline: pre-HIPIFY include routing, hipify-clang runner,
deterministic post-HIPIFY rules, a Common helper overlay and C1/C2/C3 evidence
capture.

It also has a project-level `convert` command for arbitrary CUDA source trees,
with optional per-translation-unit options from `compile_commands.json`. The
converter produces HIP source, a migration report, change log and a generated
CMake dependency target; it does not build, run or claim sample acceptance.
See [docs/generic-conversion.md](docs/generic-conversion.md).

## Quick start

```bash
cd rocm10-cuda-samples-v12.9

make test
make verify
make list
./scripts/fetch_cuda_samples.sh
./scripts/fetch_cuda_library_samples.sh
```

On the ROCm 10.0.0 target machine:

```bash
make env
make run-p0
```

`make env` is strict: it fails unless ROCm 10.0.0 and all required tools are
present. `make run-p0` writes `artifacts/<sample>/evidence.json`; override
`ARTIFACTS` or `ARCH` when needed.

### Generic project conversion

```bash
PYTHONPATH=src python3 -m rocm10_samples_compat convert \
  --input ./cuda-app \
  --output ../cuda-app-hip \
  --cuda-root /usr/local/cuda-12.9 \
  --compile-commands ./build/compile_commands.json
```

The installed package also provides `rocm-cuda-migrate` as an alias. Conversion
success is not a build, runtime, numerical or support claim. The measured Docker
verification is in
[docs/evidence/generic-conversion-rocm10.md](docs/evidence/generic-conversion-rocm10.md).

## Cross-version API analysis

The v12.0/v12.9/v13.5 API and sample-change report is
[docs/api-and-sample-changes.md](docs/api-and-sample-changes.md). Raw scan data is
in `docs/evidence/cuda_samples_api_versions.json`.

## CLI

```text
rocm10-samples-compat check-env [--json] [--strict]
rocm10-samples-compat convert --input DIR --output DIR
    [--compile-commands FILE] [--cuda-root DIR]
    [--include DIR] [--define NAME[=VALUE]]
    [--source PATTERN] [--exclude PATTERN] [--force] [--json]
rocm10-samples-compat show-manifest
rocm10-samples-compat list-batches [--json]
rocm10-samples-compat list-samples --batch <batch> [--json]
rocm10-samples-compat verify-checkout [--checkout DIR]
rocm10-samples-compat list-specs [--json]
rocm10-samples-compat run-sample --sample <name> [--artifacts DIR] [--arch gfxNNNN]
rocm10-samples-compat run-batch --batch <batch> [--artifacts DIR] [--arch gfxNNNN]
```

Installed entry points: `rocm10-samples-compat` and `rocm-cuda-migrate` (same
commands).

## P0 baseline result

Measured on the target: ROCm 10.0.0 / `gfx1201` / real CUDA 12.9 headers.

| Sample | C1 | C2 | C3 | Status |
|---|---|---|---|---|
| deviceQuery | PASS | PASS | PASS | rule-fixed |
| vectorAdd | PASS | PASS | PASS | rule-fixed |
| matrixMul | PASS | PASS | PASS | rule-fixed (profiler rule) |
| asyncAPI | PASS | PASS | PASS (exit status) | rule-fixed (profiler rule) |
| simpleStreams | PASS | PASS | PASS (exit status) | rule-fixed |
| scalarProd | PASS | PASS | PASS | rule-fixed |

`run-batch --batch P0_baseline` exits 0 and no evidence file records a
limitation. See
[docs/evidence/p0-baseline-rocm10-gfx1201.md](docs/evidence/p0-baseline-rocm10-gfx1201.md).
The earlier ROCm 7.2.4 / `gfx1100` run is archived in
[docs/evidence/p0-baseline-rocm7.2.4-gfx1100.md](docs/evidence/p0-baseline-rocm7.2.4-gfx1100.md)
and is interim-only.

## Pinned sample manifest

All **188** v12.9 C++ samples are recorded in:

```text
src/rocm10_samples_compat/data/cuda-samples-v12.9.json
```

The manifest includes:

- category and static difficulty tier;
- CUDA feature and library flags;
- host-only classification;
- initial priority batches;
- fixed ROCm/CUDA/Samples versions.

Priority batches:

```text
P0_baseline                    6
P1_semantics_and_business      6
P1_probes                      3
P2_remaining_basic           20
P3_semantic_and_library      72
P4_special                   53
P5_high_risk_rewrite         14
P6_deferred_platform         14
```

## Repository layout

```text
.
|-- .github/workflows/ci.yml
|-- pyproject.toml
|-- Makefile
|-- README.md
|-- AGENTS.md
|-- docs/
|   |-- api-and-sample-changes.md
|   |-- development-plan.md
|   |-- generic-conversion.md
|   |-- handoff-context.md
|   |-- machine-resources.md
|   `-- evidence/
|-- scripts/
|   |-- analyze_cuda_library_samples.py
|   |-- analyze_cuda_samples_versions.py
|   |-- check_rocm10.sh
|   |-- fetch_cuda_library_samples.sh
|   `-- fetch_cuda_samples.sh
|-- skills/
|   `-- cuda-samples-v12.9-rocm10/SKILL.md
|-- src/rocm10_samples_compat/
|   |-- checkout.py
|   |-- cli.py
|   |-- cuda_root.py
|   |-- environment.py
|   |-- hipify.py
|   |-- manifest.py
|   |-- pipeline.py
|   |-- process.py
|   |-- project.py
|   |-- redirects.py
|   |-- spec.py
|   |-- data/cuda-samples-v12.9.json
|   |-- data/specs/*.json
|   |-- overlay/rocm10_compat/helper_cuda.h
|   `-- rules/
`-- tests/
```

## What is not implemented

- sample adaptations beyond the P0 specs;
- automatic project build integration or CMakeLists rewriting for `convert`;
- per-target variants when one source file has incompatible compile options;
- Common helper overlay coverage beyond the P0 surface
  (`helper_cuda_drvapi.h`, `nvrtc_helper.h`, `helper_cusolver.h` are not provided);
- golden generation and performance comparison;
- support claims for samples outside the P0 batch or for arbitrary applications.

## Required machine resources

See [docs/machine-resources.md](docs/machine-resources.md).

Existing machines:

- AMD Radeon RX 9070 XT, `gfx1201`, 16 GB — **the ROCm 10.0.0 target**;
- AMD Radeon RX 7900 XTX, `gfx1100`, 24 GB — the earlier interim environment;
- NVIDIA GeForce RTX 4060, 8 GB.

Requested machines:

- NVIDIA GeForce RTX 5070 Ti, 16 GB.

## Boundaries

- All P0 evidence is now ROCm 10.0.0 / `gfx1201`; nothing here transfers to
  `gfx1100` or to another ROCm version.
- All six P0 samples reach C3. `simpleStreams` and `asyncAPI` rest on exit status
  only; the other four carry the sample's own success marker.
- Do not claim support for all 188 samples.
- Do not reuse v12.0 patches or results as v12.9 evidence.
- Do not introduce v13.5 samples into this baseline.
- No license has been selected for the standalone repository.

See [AGENTS.md](AGENTS.md), [development plan](docs/development-plan.md) and
[handoff context](docs/handoff-context.md) before continuing.
