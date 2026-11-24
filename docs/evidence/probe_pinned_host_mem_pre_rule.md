# probe_pinned_host_mem: pre-rule failure evidence

Date: 2026-11-24

Environment: ROCm 7.2.4, hipify-clang 22.0.0git, gfx1100

## Purpose

Demonstrate that the second probe triggers a post-HIPIFY compilation failure
instead of passing HIPIFY directly.

## Original CUDA call

```cuda
cudaMallocHost(&host_pinned, kBytes, 0);
```

## HIPIFY output

```cpp
hipHostMalloc(&host_pinned, kBytes, 0, hipHostMallocDefault);
```

## hipcc failure

```text
error: no matching function for call to 'hipHostMalloc'
note: candidate function template not viable: requires at most 3 arguments,
      but 4 were provided
```

The full failure artifacts were generated in a temporary directory and can be
recreated:

```text
/tmp/probe-pinned-before-rule/
|-- probe_pinned_host_mem.hip.cu
|-- hipify.log
|-- build.log
`-- probe
```

## Fix rule

```text
HIP-HOSTMALLOC-ARITY-001
Condition: hipHostMalloc has four arguments; argument 3 is 0 and argument 4 is
           hipHostMallocDefault
Action: remove argument 3
```

The rule is implemented in `src/cuda2hip_migrate/rules/hostmalloc.py`.
Positive, negative and idempotence tests are in
`tests/test_hostmalloc_rule.py`.

## Result after the rule

```text
[RULE] HIP-HOSTMALLOC-ARITY-001 line=35
hipHostMalloc(&host_pinned, kBytes, 0, hipHostMallocDefault)
-> hipHostMalloc(&host_pinned, kBytes, hipHostMallocDefault)
[RULES] applied=1
[C1] PASS
PROBE=pinned_host_mem device=AMD Radeon RX 7900 XTX bytes=65539 RESULT=PASS
[C2] PASS
[C3] PASS
```

## Boundaries

- The failure is reproduced with ROCm 7.2.4 and hipify-clang 22.0.0git.
- The rule must be revalidated after a HIPIFY or HIP upgrade to confirm that it
  is still necessary and still safe.
- The rule only changes the exact four-argument form. Calls with three
  arguments or other flags are left unchanged.
