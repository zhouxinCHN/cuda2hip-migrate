// Minimal CUDA probe: pinned host allocation plus H2D/D2H copies.
// This intentionally uses cudaMallocHost so the post-HIPIFY rule path is
// exercised instead of only validating a HIPIFY-direct probe.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

int main() {
    int device = 0;
    cudaDeviceProp properties{};
    PROBE_CHECK(cudaGetDevice(&device));
    PROBE_CHECK(cudaGetDeviceProperties(&properties, device));

    constexpr size_t kBytes = 64 * 1024 + 3;
    void *host_pinned = nullptr;
    void *device_buffer = nullptr;
    std::vector<unsigned char> host_result(kBytes, 0);
    std::vector<unsigned char> expected(kBytes, 0x5a);

    PROBE_CHECK(cudaMallocHost(&host_pinned, kBytes, 0));
    std::memset(host_pinned, 0x5a, kBytes);

    PROBE_CHECK(cudaMalloc(&device_buffer, kBytes));
    PROBE_CHECK(cudaMemcpy(device_buffer, host_pinned, kBytes,
                           cudaMemcpyHostToDevice));
    PROBE_CHECK(cudaMemcpy(host_result.data(), device_buffer, kBytes,
                           cudaMemcpyDeviceToHost));
    PROBE_CHECK(cudaFree(device_buffer));
    PROBE_CHECK(cudaFreeHost(host_pinned));

    if (std::memcmp(host_result.data(), expected.data(), kBytes) != 0) {
        std::fprintf(stderr, "PROBE_ERROR pinned host copy mismatch\n");
        return EXIT_FAILURE;
    }

    std::printf(
        "PROBE=pinned_host_mem device=%s bytes=%zu RESULT=PASS\n",
        properties.name, kBytes);
    return EXIT_SUCCESS;
}
