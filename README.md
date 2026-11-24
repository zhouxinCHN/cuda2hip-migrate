# cuda2hip-migrate

Deterministic post-HIPIFY processing and standalone CUDA probe validation for
ROCm.

This project is an early prototype. It proves the execution path and currently
contains one source rule; it is not a complete CUDA-to-HIP migration tool. The
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
```

Examples:

```bash
cuda2hip-migrate run --probe probe_malloc_memcpy
cuda2hip-migrate run \
  --probe probe_pinned_host_mem \
  --out /tmp/probe-pinned \
  --arch gfx1100
```

The architecture is taken from `--arch`, then `HIP_ARCH`, then `rocminfo`.

## Probes

| Probe | Coverage | Status |
|---|---|---|
| `probe_malloc_memcpy` | device allocation, synchronous H2D/D2H copy, kernel read/write | HIPIFY direct |
| `probe_pinned_host_mem` | pinned host allocation and device round trip | fixed by rule |
| `probe_stream_event` | two non-default streams, async copies, cross-stream event ordering | HIPIFY direct |

Planned probes are listed in the parent migration plan, not added ahead of
review. Add one probe at a time and stop for review before continuing.

## Rules

Current registry: `src/cuda2hip_migrate/rules/__init__.py`.

```text
HIP-HOSTMALLOC-ARITY-001
condition: hipHostMalloc with 4 args; arg 3 == 0; arg 4 == hipHostMallocDefault
action: remove argument 3
```

Rules:

- run only after HIPIFY;
- skip comments, strings and character literals;
- require exact conditions;
- are idempotent;
- record rule ID, line and before/after text in `rules.json`.

Rule tests are in `tests/test_hostmalloc_rule.py`.

## Acceptance

- **C1**: `hipcc` build succeeds.
- **C2**: probe exits with status 0.
- **C3**: probe emits `RESULT=PASS`.

A build alone is not treated as success. Performance is out of scope.

## Artifacts

Each run creates:

```text
<out>/
|-- source/<probe>.cu          # original packaged CUDA source
|-- hipified/<probe>.hip.cu    # untouched HIPIFY output
|-- adapted/<probe>.hip.cu     # rules applied; built source
|-- hipify.diff                # original -> HIPIFY, audit only
|-- rules.json                 # rule hits
|-- hipify.log
|-- build.log
|-- run.log
`-- <probe>                    # executable
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

- Only one deterministic rule is implemented.
- ROCm 10.0 has not been validated.
- `hipify-clang` may print `ptxas`/`nvlink` architecture diagnostics while still
  exiting successfully and producing valid output; the runner trusts the exit
  status and output file, and this must be rechecked when tools are upgraded.
- These probes do not prove performance, long-run stability, concurrency safety
  or full CUDA semantic equivalence.
- See [`docs/evidence/probe_pinned_host_mem_pre_rule.md`](docs/evidence/probe_pinned_host_mem_pre_rule.md)
  and [`skills/cuda-sample-probe-migration/SKILL.md`](skills/cuda-sample-probe-migration/SKILL.md).
