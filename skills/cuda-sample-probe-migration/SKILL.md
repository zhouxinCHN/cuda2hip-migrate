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

Do not extrapolate this workflow into a claim that all cuda-samples or customer
projects can be migrated automatically.

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

Implemented rule:

```text
HIP-HOSTMALLOC-ARITY-001
Condition: hipHostMalloc has four arguments; argument 3 is 0 and argument 4 is
           hipHostMallocDefault
Action: remove argument 3
```

Pre-rule failure evidence:

```text
no matching function for call to 'hipHostMalloc'
requires at most 3 arguments, but 4 were provided
```

See `docs/evidence/probe_pinned_host_mem_pre_rule.md`.

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
