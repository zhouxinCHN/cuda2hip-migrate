// Minimal CUDA probe: non-default streams, async copies and event ordering.
// Two independent streams are connected by a recorded event so that
// cudaStreamWaitEvent, cudaEventRecord and the asynchronous memory API are all
// exercised without CUDA Samples helpers or a sample-specific build system.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
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

__global__ void transformKernel(const int *src, int *dst, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        dst[index] = src[index] * 3 + 1;
    }
}

int main() {
    int device = 0;
    cudaDeviceProp properties{};
    PROBE_CHECK(cudaGetDevice(&device));
    PROBE_CHECK(cudaGetDeviceProperties(&properties, device));

    constexpr int kElementCount = 4096 + 3;
    constexpr int kThreadsPerBlock = 256;
    const size_t bytes = static_cast<size_t>(kElementCount) * sizeof(int);

    std::vector<int> host_source(kElementCount);
    std::vector<int> host_result(kElementCount, 0);
    for (int i = 0; i < kElementCount; ++i) {
        host_source[i] = i;
    }

    int *device_source = nullptr;
    int *device_result = nullptr;
    cudaStream_t copy_stream = nullptr;
    cudaStream_t compute_stream = nullptr;
    cudaEvent_t copy_done = nullptr;
    cudaEvent_t compute_done = nullptr;

    PROBE_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_source), bytes));
    PROBE_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_result), bytes));
    PROBE_CHECK(cudaStreamCreate(&copy_stream));
    PROBE_CHECK(cudaStreamCreate(&compute_stream));
    PROBE_CHECK(cudaEventCreateWithFlags(&copy_done, cudaEventDisableTiming));
    PROBE_CHECK(cudaEventCreateWithFlags(&compute_done, cudaEventDisableTiming));

    PROBE_CHECK(cudaMemcpyAsync(device_source, host_source.data(), bytes,
                                cudaMemcpyHostToDevice, copy_stream));
    PROBE_CHECK(cudaEventRecord(copy_done, copy_stream));

    // The kernel stream must not start before the copy stream has finished.
    PROBE_CHECK(cudaStreamWaitEvent(compute_stream, copy_done, 0));

    const int blocks = (kElementCount + kThreadsPerBlock - 1) / kThreadsPerBlock;
    transformKernel<<<blocks, kThreadsPerBlock, 0, compute_stream>>>(
        device_source, device_result, kElementCount);
    PROBE_CHECK(cudaGetLastError());

    PROBE_CHECK(cudaMemcpyAsync(host_result.data(), device_result, bytes,
                                cudaMemcpyDeviceToHost, compute_stream));
    PROBE_CHECK(cudaEventRecord(compute_done, compute_stream));
    PROBE_CHECK(cudaEventSynchronize(compute_done));

    PROBE_CHECK(cudaEventDestroy(compute_done));
    PROBE_CHECK(cudaEventDestroy(copy_done));
    PROBE_CHECK(cudaStreamDestroy(compute_stream));
    PROBE_CHECK(cudaStreamDestroy(copy_stream));
    PROBE_CHECK(cudaFree(device_result));
    PROBE_CHECK(cudaFree(device_source));

    bool match = true;
    for (int i = 0; i < kElementCount; ++i) {
        if (host_result[i] != host_source[i] * 3 + 1) {
            match = false;
            break;
        }
    }
    if (!match) {
        std::fprintf(stderr, "PROBE_ERROR stream/event output mismatch\n");
        return EXIT_FAILURE;
    }

    std::printf(
        "PROBE=stream_event device=%s elements=%d streams=2 events=2 "
        "RESULT=PASS\n",
        properties.name, kElementCount);
    return EXIT_SUCCESS;
}
