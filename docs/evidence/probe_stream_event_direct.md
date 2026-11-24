# probe_stream_event: HIPIFY-direct evidence

Date: 2026-11-24

Environment: ROCm 7.2.4, hipify-clang 22.0.0git, gfx1100

## Purpose

Record that the third probe is translated correctly by HIPIFY without any
post-HIPIFY rule. No rule is added for this probe.

## Capability covered

- two explicitly created non-default streams (`cudaStreamCreate`);
- asynchronous H2D and D2H copies on a non-default stream (`cudaMemcpyAsync`);
- a kernel launch on a non-default stream;
- an event recorded on the copy stream (`cudaEventRecord`);
- a cross-stream dependency (`cudaStreamWaitEvent`);
- host-side completion (`cudaEventSynchronize`), then teardown.

## HIPIFY translation

`hipify-clang` exited with status 0 and produced
`hipified/probe_stream_event.hip.cu`. The translated symbols were:

```text
cudaStreamCreate           -> hipStreamCreate
cudaStreamWaitEvent        -> hipStreamWaitEvent
cudaEventCreateWithFlags   -> hipEventCreateWithFlags
cudaEventRecord            -> hipEventRecord
cudaEventSynchronize       -> hipEventSynchronize
cudaEventDestroy           -> hipEventDestroy
cudaStreamDestroy          -> hipStreamDestroy
cudaMemcpyAsync            -> hipMemcpyAsync
cudaMemcpyHostToDevice     -> hipMemcpyHostToDevice
cudaMemcpyDeviceToHost     -> hipMemcpyDeviceToHost
```

The three-argument `cudaStreamWaitEvent(stream, event, 0)` form is accepted by
HIP unchanged; no arity rule is required.

## Run result

```text
[RULES] applied=0
[C1] PASS
PROBE=stream_event device=AMD Radeon RX 7900 XTX elements=4099 streams=2 events=2 RESULT=PASS
[C2] PASS
[C3] PASS
```

Recreated with:

```bash
cuda2hip-migrate run --probe probe_stream_event --out /tmp/probe-stream-event
```

`rules.json` is `[]` and `adapted/` is byte-identical to `hipified/`, which is
the expected state for a HIPIFY-direct probe.

## Tool diagnostics

`hipify.log` contains the known non-fatal diagnostics:

```text
error: must pass in an explicit nvptx64 gpu architecture to 'ptxas'
error: must pass in an explicit nvptx64 gpu architecture to 'nvlink'
```

They were printed while `hipify-clang` still exited with status 0 and wrote
valid HIP output, matching the documented pitfall. The runner treats exit status
plus output-file existence as authoritative.

## Boundaries

- The probe shows that stream and event APIs translate and run correctly for one
  short, ordered sequence.
- It does not prove concurrency performance, multi-GPU behaviour, stream
  priority semantics, or overlap that depends on asynchronous pageable-memory
  handling.
- The event ordering is a functional check. A faulty `cudaStreamWaitEvent`
  translation is not guaranteed to surface as a mismatch on a single GPU.
- Revalidate after a HIPIFY or HIP upgrade.
