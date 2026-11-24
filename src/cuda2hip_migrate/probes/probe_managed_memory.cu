// Minimal CUDA probe: managed (unified) memory written by a kernel and read
// directly by the host. No explicit copy is performed, so the probe relies on
// the device-to-host migration semantics of cudaMallocManaged.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define PROBE_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t probe_err = (call);                                        \
        if (probe_err != cudaSuccess) {                                        \
            std::fprintf(stderr, "PROBE_ERROR %s:%d %s: %s (%d)\n", __FILE__,  \
                         __LINE__, #call, cudaGetErrorString(probe_err),       \
                         static_cast<int>(probe_err));                         \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

__global__ void fillManagedKernel(int *data, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        data[index] = index * 2 + 7;
    }
}

int main() {
    int device = 0;
    cudaDeviceProp properties{};
    PROBE_CHECK(cudaGetDevice(&device));
    PROBE_CHECK(cudaGetDeviceProperties(&properties, device));

    constexpr int kElementCount = 8192 + 5;
    constexpr int kThreadsPerBlock = 256;
    const size_t bytes = static_cast<size_t>(kElementCount) * sizeof(int);

    int *managed = nullptr;
    PROBE_CHECK(cudaMallocManaged(reinterpret_cast<void **>(&managed), bytes,
                                  cudaMemAttachGlobal));

    const int blocks = (kElementCount + kThreadsPerBlock - 1) / kThreadsPerBlock;
    fillManagedKernel<<<blocks, kThreadsPerBlock>>>(managed, kElementCount);
    PROBE_CHECK(cudaGetLastError());
    PROBE_CHECK(cudaDeviceSynchronize());

    // Host access without cudaMemcpy: this is the managed-memory contract.
    bool match = true;
    for (int index = 0; index < kElementCount; ++index) {
        if (managed[index] != index * 2 + 7) {
            match = false;
            break;
        }
    }

    PROBE_CHECK(cudaFree(managed));

    if (!match) {
        std::fprintf(stderr, "PROBE_ERROR managed memory output mismatch\n");
        return EXIT_FAILURE;
    }

    std::printf(
        "PROBE=managed_memory device=%s elements=%d bytes=%zu RESULT=PASS\n",
        properties.name, kElementCount, bytes);
    return EXIT_SUCCESS;
}
