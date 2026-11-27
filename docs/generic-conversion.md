# Generic project conversion

Status: **developer preview**, implemented in `src/rocm10_samples_compat/project.py`.
It is validated for CUDA 12.9 source trees on ROCm 10.0.0 and is not a support
statement for arbitrary CUDA applications. See
[`evidence/generic-conversion-rocm10.md`](evidence/generic-conversion-rocm10.md)
for the measured checks.

## Scope

```text
input:  arbitrary local CUDA source tree
        optional compile_commands.json
output: converted HIP source tree + machine-readable report + build requirements

not included:
        compiling, running or C1/C2/C3 acceptance
        automatic CMakeLists.txt rewriting
        multi-configuration output for one source file
```

The sample runner (`run-sample`, `run-batch`) remains a separate acceptance
layer. `convert` never invokes a model and never writes to the input tree.

## Commands

The generic command is available under both installed entry-point names:

```bash
rocm-cuda-migrate convert --input ./cuda-app --output ../cuda-app-hip
```

or from the repository without installation:

```bash
PYTHONPATH=src python3 -m rocm10_samples_compat convert \
  --input ./cuda-app \
  --output ../cuda-app-hip \
  --cuda-root /usr/local/cuda-12.9 \
  --compile-commands ./build/compile_commands.json
```

If `--cuda-root` is omitted, the existing CUDA-root resolver is used and a
synthetic root may be materialised under `<output>/.migration/`.

### Compile database mode

This is the preferred mode for a real project:

```bash
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
rocm-cuda-migrate convert \
  --input . \
  --output ../app-hip \
  --compile-commands build/compile_commands.json
```

For each translation unit the converter reads its working directory, include
search paths, definitions, `-U` options, language standard and forced includes.
Paths inside the input tree are remapped to the corresponding converted file.

NVCC-only arguments such as `-gencode`, `-Xcompiler`, `-Xptxas`, `-o` and
architecture selection are not forwarded to hipify-clang. They are recorded in
`units[].ignored_arguments`. A source that appears more than once with
incompatible parse-relevant options stops the conversion: one output file
cannot represent both configurations.

Only `.cu` sources and files compiled by `nvcc` are translated in this mode.
Other compiler entries are reported but left untouched.

### Tree mode

Without a compile database, all translatable sources under the input are
selected and share the options supplied on the command line:

```bash
rocm-cuda-migrate convert \
  --input ./simple-app \
  --output ../simple-app-hip \
  --include ./include \
  --define FEATURE_X=1 \
  --source 'src/*.cu' \
  --exclude 'third_party/**'
```

Tree mode is a convenience for simple projects; it is not a substitute for
per-target compile options.

## Output layout

```text
<output>/
├── src/                      converted source, relative layout preserved
├── reports/
│   ├── migration.json        schema and environment report
│   ├── changes.jsonl         every redirect and rule edit
│   └── hipify.log            hipify-clang command/output per unit
├── rocm-migration.cmake      generated dependency target
└── .migration/
    ├── source/
    ├── preprocessed/
    └── hipified/
```

A CMake consumer can use the generated target:

```cmake
include("${MIGRATION_OUTPUT}/rocm-migration.cmake")
target_link_libraries(my_app PRIVATE rocm_cuda_migration_dependencies)
```

The target carries compatibility include directories, the CUDA-baseline
compile definition and libraries required by rules that actually fired (for
example `rocprofiler-sdk-roctx`). Project-specific build configuration still
belongs to the project.

## Behavior

- The input tree is read-only. Output inside the input, or input inside the
  output, is rejected.
- A non-empty output requires `--force`; `--force` replaces the whole output
  tree.
- `.git`, `.hg`, `.svn`, `__pycache__` and common build products are excluded by
  default. Additional `--exclude` patterns are allowed.
- Common-helper include routing is enabled only when the input actually
  includes that helper. An unsupported helper is a blocking finding, not a
  silent passthrough.
- A declared/active redirect that no longer matches is blocking.
- HIPIFY failure is blocking. The preprocessed source is restored to the output
  so no file silently disappears, and the command exits `2`.
- Local headers produced by hipify-clang are moved into place. A CUDA header
  reached through `-I` (which `--local-headers` does not translate) gets a
  standalone HIPIFY pass using the option sets of the units that can see it.
  Different translated content for the same header from different option sets
  is a blocking conflict.
- Deterministic rules run only after HIPIFY. Each change records file, rule id,
  line and before/after text.
- Residual CUDA includes, CUDA-looking identifiers and inline assembly are
  reported for review. These are warnings: static inspection cannot prove that
  a construct is or is not compiled out.
- Report generation has no wall-clock timestamp so repeated conversion of the
  same tree with the same output path is reproducible.

## Exit codes

| Code | Meaning |
|---:|---|
| 0 | conversion completed; review findings may still be present |
| 1 | input, output, environment or tool configuration error |
| 2 | blocking conversion issue; see `reports/migration.json` |

`convert` exit code `0` does not mean the output builds, runs, or is supported.

## Current limitations

- No project build or run step, and therefore no C1/C2/C3 claim.
- No automatic rewrite of project CMake files, RPATHs, library names or include
  layouts. The user must adapt build configuration to use `src/` and the
  generated dependency target.
- Standalone header translation is heuristic: a header needs a CUDA include or
  a recognized CUDA identifier. A header whose CUDA dependency is entirely
  indirect may therefore be left for the project build to reveal.
- No per-target duplicate output for one source compiled with different macros.
- CUDA host sources compiled by a non-NVCC compiler are not translated in
  compile-database mode.
- `helper_cuda_drvapi.h` is detected but its overlay is not implemented; the
  conversion is blocked rather than pretending it is handled.
- Numerical equivalence and performance are out of scope.
- Static residual findings are conservative and require manual review.

## Verification

Unit tests cover input safety, recursive layout, compile-database extraction,
argument filtering, redirect activation, rule dependency aggregation, blocked
recovery and report output:

```bash
make test
```

On a ROCm host with `hipify-clang`, `hipcc` and the pinned samples checkout, the
integration tests additionally convert, build and run `vectorAdd` and a project
whose CUDA header is reached through `-I`. On Docker:

```bash
docker exec ubuntu2404 bash -lc \
  'cd /home/mark/rocm10-cuda-samples-v12.9 && make test'
```
