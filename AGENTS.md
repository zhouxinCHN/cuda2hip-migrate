# Agent Handoff

Read these files before making changes:

1. `docs/handoff-context.md`
2. `docs/development-plan.md`
3. `skills/cuda-samples-v12.9-rocm10/SKILL.md`

## Fixed decisions

```text
Target ROCm:           10.0.0
CUDA Toolkit baseline: 12.9
CUDA Samples:          v12.9
Samples commit:        cab7c66b4fcf35b09ea8fbc4c3c1a9272e92d997
Primary deliverable:   skills
```

## Current status

- **This is a validation repository.** Its job was to prove the migration route
  works. Formal development will happen in a separate new repository; see
  `docs/development-plan.md` (v2).
- The v12.9 manifest contains all 188 C++ samples.
- Environment check and manifest CLI are implemented.
- The validation pipeline works: pre-HIPIFY include routing, hipify-clang runner,
  deterministic rules, Common helper overlay, C1/C2/C3 evidence capture.
- A project-level `convert` command is implemented for arbitrary CUDA source
  trees. It supports recursive layout preservation, optional explicit
  `compile_commands.json` options, automatic CMake configuration into
  `<output>/.tmp/<project>/cmake-build` when a root `CMakeLists.txt` exists,
  response-file expansion, deterministic redirects/rules, migration and change
  reports, and a generated CMake dependency target. It does not build or run
  the result; see `docs/generic-conversion.md`.
- All tool-owned temporary files are redirected under the user-supplied output
  directory: `.tmp/<unit>/hipify`, `.tmp/<unit>/tmpdir` and
  `.tmp/<unit>/cmake-build`, with `TMPDIR/TMP/TEMP` set for direct child
  processes. Sample hipcc/run stages use `.tmp/<sample>/`.
- Docker-target verification: 116 tests, including real HIPIFY + hipcc for
  `vectorAdd`, an include-directory CUDA header, and an auto-configured CMake
  CUDA project; P0 remains all C1/C2/C3.
- `make verify` and `make run-p0` exist. `verify-checkout --checkout` is fixed,
  and `primary_gfx_architecture()` now preserves rocminfo agent order, so the
  discrete `gfx1201` is selected before the iGPU on the target.
- The pinned checkout verifies as `cab7c66b` / `v12.9` and stays pristine.
- **P0 now has target-environment evidence**: ROCm 10.0.0, `gfx1201`, real CUDA
  12.9 headers, no recorded limitations. See
  `docs/evidence/p0-baseline-rocm10-gfx1201.md`. The earlier ROCm 7.2.4 /
  `gfx1100` run is archived as interim-only.
- Validation result on the target: **all six P0 samples reach C3**;
  `run-batch --batch P0_baseline` exits 0.
- Decided 2026-11-26 (supersedes the 2026-11-25 decision): `hipProfilerStart()`
  still returns 801 on ROCm 10.0.0 — measured under `rocprofv3` too — so the
  "wait for the target environment" reason is gone. `HIP-PROFILER-ROCTX-001`
  maps `hipProfilerStart/Stop` to `roctxProfilerResume/Pause`, which keeps the
  instrumentation contract. A no-op stub was rejected: the sample self-check
  passes with instrumentation disabled, so it would report PASS while measuring
  nothing.
- **Validation is not delivery.** No support claim for the 188 samples exists.
- `docs/api-and-sample-changes.md` holds the cross-version API analysis and the
  separate CUDALibrarySamples baseline (commit `692a738f`); it is analysis, not a
  scope change to the v12.9 migration.

## Commands

```bash
make test
make list
make verify                 # pinned checkout check
make run-p0                 # P0 batch, writes artifacts/<sample>/evidence.json
python3 -m rocm10_samples_compat convert --input DIR --output DIR
./scripts/fetch_cuda_samples.sh
./scripts/check_rocm10.sh
```

## Non-negotiable constraints

- Use CUDA Samples v12.9 only.
- Use ROCm 10.0.0 as target.
- Deliver reusable skills, deterministic rules, manifests and evidence.
- Do not use patch files as the migration mechanism.
- Do not run an AI model in the normal execution path.
- Do not modify the original cuda-samples checkout.
- Do not lower C1/C2/C3 to make a sample pass.
- Do not claim support beyond recorded evidence.
- Do not introduce v13.5 samples.
- Keep generated artifacts out of Git.

## Immediate next task

The generic converter is a developer preview, not a replacement for sample
acceptance. Remaining before expanding its scope:

1. Exercise `convert` against real multi-target projects and promote only
   generic, version-stable rules into HIPIFY.
2. Add sample specs and overlay coverage for P1; do not infer support for the
   other 182 samples.
3. Review the development plan's stage table and workload estimates against
   the now-measured ROCm 10.0.0 target and generic-conversion scope.
4. Formal development (engineering structure, contribution process, version
   control, CI, distribution) belongs to the new repository, not this one.

Do not add automatic builds, C1/C2/C3 claims or CMakeLists rewriting to
`convert` without tests and recorded evidence. The converter deliberately stops
at source conversion.
