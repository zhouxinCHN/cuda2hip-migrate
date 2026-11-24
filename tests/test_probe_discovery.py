from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cuda2hip_migrate.runner import materialize_probe, probe_names, probe_source_text


class ProbeDiscoveryTests(unittest.TestCase):
    def test_packaged_probes_are_discoverable(self) -> None:
        names = probe_names()
        self.assertIn("probe_malloc_memcpy", names)
        self.assertIn("probe_pinned_host_mem", names)
        self.assertIn("probe_stream_event", names)
        self.assertIn("probe_managed_memory", names)
        self.assertIn("probe_kernel_dynamic_smem", names)
        self.assertEqual(names, tuple(sorted(names)))

    def test_dynamic_smem_probe_uses_extern_shared(self) -> None:
        source = probe_source_text("probe_kernel_dynamic_smem")
        self.assertIn("extern __shared__", source)
        self.assertIn("__syncthreads", source)
        self.assertIn("RESULT=PASS", source)

    def test_managed_memory_probe_uses_managed_allocation(self) -> None:
        source = probe_source_text("probe_managed_memory")
        self.assertIn("cudaMallocManaged", source)
        self.assertIn("cudaMemAttachGlobal", source)
        self.assertIn("RESULT=PASS", source)

    def test_stream_probe_covers_streams_and_events(self) -> None:
        source = probe_source_text("probe_stream_event")
        for symbol in (
            "cudaStreamCreate",
            "cudaStreamWaitEvent",
            "cudaEventRecord",
            "cudaEventSynchronize",
            "cudaMemcpyAsync",
        ):
            self.assertIn(symbol, source)
        self.assertIn("RESULT=PASS", source)

    def test_probe_source_is_non_empty_cuda(self) -> None:
        source = probe_source_text("probe_pinned_host_mem")
        self.assertIn("cudaMallocHost", source)
        self.assertIn("__global__", probe_source_text("probe_malloc_memcpy"))

    def test_unknown_probe_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unknown probe"):
            probe_source_text("does_not_exist")

    def test_materialize_probe_writes_expected_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "probe.cu"
            materialize_probe("probe_malloc_memcpy", destination)
            self.assertTrue(destination.is_file())
            self.assertIn("cudaMalloc", destination.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
