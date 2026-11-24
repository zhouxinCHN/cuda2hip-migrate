// Minimal CUDA probe: device allocation + synchronous H2D/D2H copies.
// Kept self-contained so it can be translated by HIPIFY without CUDA Samples
// helper headers or a sample-specific build system.

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

__global__ void copyKernel(const float *src, float *dst, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        dst[index] = src[index];
    }
}

int main() {
    int device = 0;
    cudaDeviceProp properties{};
    PROBE_CHECK(cudaGetDevice(&device));
    PROBE_CHECK(cudaGetDeviceProperties(&properties, device));

    constexpr int kElementCount = 1024 * 1024 + 17;
    constexpr int kThreadsPerBlock = 256;
    const size_t bytes = static_cast<size_t>(kElementCount) * sizeof(float);

    std::vector<float> host_source(kElementCount);
    std::vector<float> host_result(kElementCount, 0.0f);
    for (int i = 0; i < kElementCount; ++i) {
        host_source[i] = static_cast<float>(i % 1024) * 0.25f;
    }

    float *device_source = nullptr;
    float *device_result = nullptr;
    PROBE_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_source), bytes));
    PROBE_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_result), bytes));

    PROBE_CHECK(cudaMemcpy(device_source, host_source.data(), bytes,
                           cudaMemcpyHostToDevice));

    const int blocks = (kElementCount + kThreadsPerBlock - 1) / kThreadsPerBlock;
    copyKernel<<<blocks, kThreadsPerBlock>>>(device_source, device_result,
                                             kElementCount);
    PROBE_CHECK(cudaGetLastError());
    PROBE_CHECK(cudaDeviceSynchronize());

    PROBE_CHECK(cudaMemcpy(host_result.data(), device_result, bytes,
                           cudaMemcpyDeviceToHost));

    PROBE_CHECK(cudaFree(device_source));
    PROBE_CHECK(cudaFree(device_result));

    const bool match =
        std::memcmp(host_source.data(), host_result.data(), bytes) == 0;
    if (!match) {
        std::fprintf(stderr, "PROBE_ERROR output mismatch\n");
        return EXIT_FAILURE;
    }

    std::printf(
        "PROBE=malloc_memcpy device=%s elements=%d bytes=%zu RESULT=PASS\n",
        properties.name, kElementCount, bytes);
    return EXIT_SUCCESS;
}
