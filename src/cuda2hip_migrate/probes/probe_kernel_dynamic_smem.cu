// Minimal CUDA probe: dynamic shared memory with a multi-block reduction.
// Each block stages its input in a dynamically sized shared array, reduces it
// tree-wise with __syncthreads and writes one partial sum per block.

#include <cuda_runtime.h>

#include <cmath>
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

__global__ void blockSumKernel(const float *src, float *block_sums, int count) {
    extern __shared__ float scratch[];
    const int tid = threadIdx.x;
    const int index = blockIdx.x * blockDim.x + tid;
    scratch[tid] = (index < count) ? src[index] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = scratch[0];
    }
}

int main() {
    int device = 0;
    cudaDeviceProp properties{};
    PROBE_CHECK(cudaGetDevice(&device));
    PROBE_CHECK(cudaGetDeviceProperties(&properties, device));

    constexpr int kElementCount = 4096 + 33;
    constexpr int kThreadsPerBlock = 256;
    const size_t bytes = static_cast<size_t>(kElementCount) * sizeof(float);
    const int blocks = (kElementCount + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const size_t shared_bytes = static_cast<size_t>(kThreadsPerBlock) * sizeof(float);

    std::vector<float> host_source(kElementCount);
    double host_sum = 0.0;
    for (int i = 0; i < kElementCount; ++i) {
        host_source[i] = static_cast<float>(i % 97) * 0.5f;
        host_sum += host_source[i];
    }
    std::vector<float> host_block_sums(blocks, 0.0f);

    float *device_source = nullptr;
    float *device_block_sums = nullptr;
    PROBE_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_source), bytes));
    PROBE_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_block_sums),
                           static_cast<size_t>(blocks) * sizeof(float)));
    PROBE_CHECK(cudaMemcpy(device_source, host_source.data(), bytes,
                           cudaMemcpyHostToDevice));

    blockSumKernel<<<blocks, kThreadsPerBlock, shared_bytes>>>(
        device_source, device_block_sums, kElementCount);
    PROBE_CHECK(cudaGetLastError());
    PROBE_CHECK(cudaDeviceSynchronize());

    PROBE_CHECK(cudaMemcpy(host_block_sums.data(), device_block_sums,
                           static_cast<size_t>(blocks) * sizeof(float),
                           cudaMemcpyDeviceToHost));

    double device_sum = 0.0;
    for (float partial : host_block_sums) {
        device_sum += static_cast<double>(partial);
    }

    PROBE_CHECK(cudaFree(device_block_sums));
    PROBE_CHECK(cudaFree(device_source));

    const double tolerance = 1e-3;
    if (std::fabs(device_sum - host_sum) > tolerance) {
        std::fprintf(stderr,
                     "PROBE_ERROR shared-memory reduction mismatch: "
                     "host=%.6f device=%.6f\n",
                     host_sum, device_sum);
        return EXIT_FAILURE;
    }

    std::printf(
        "PROBE=kernel_dynamic_smem device=%s blocks=%d threads=%d "
        "shared_bytes=%zu elements=%d RESULT=PASS\n",
        properties.name, blocks, kThreadsPerBlock, shared_bytes, kElementCount);
    return EXIT_SUCCESS;
}
