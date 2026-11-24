// Minimal CUDA probe: a linear texture object read through tex1Dfetch.
// Covers texture object creation, descriptor setup and device-side fetches,
// which are separate HIP surfaces from the runtime memory API.

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

__global__ void textureReadKernel(cudaTextureObject_t texture, float *dst,
                                  int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        dst[index] = tex1Dfetch<float>(texture, index);
    }
}

int main() {
    int device = 0;
    cudaDeviceProp properties{};
    PROBE_CHECK(cudaGetDevice(&device));
    PROBE_CHECK(cudaGetDeviceProperties(&properties, device));

    constexpr int kElementCount = 4096 + 17;
    constexpr int kThreadsPerBlock = 256;
    const size_t bytes = static_cast<size_t>(kElementCount) * sizeof(float);
    const int blocks = (kElementCount + kThreadsPerBlock - 1) / kThreadsPerBlock;

    std::vector<float> host_source(kElementCount);
    for (int i = 0; i < kElementCount; ++i) {
        host_source[i] = static_cast<float>(i % 61) * 0.25f;
    }
    std::vector<float> host_result(kElementCount, -1.0f);

    float *device_source = nullptr;
    float *device_result = nullptr;
    PROBE_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_source), bytes));
    PROBE_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_result), bytes));
    PROBE_CHECK(cudaMemcpy(device_source, host_source.data(), bytes,
                           cudaMemcpyHostToDevice));

    const cudaChannelFormatDesc channel = cudaCreateChannelDesc<float>();
    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypeLinear;
    resource.res.linear.devPtr = device_source;
    resource.res.linear.desc = channel;
    resource.res.linear.sizeInBytes = bytes;

    cudaTextureDesc texture_desc{};
    texture_desc.readMode = cudaReadModeElementType;

    cudaTextureObject_t texture = 0;
    PROBE_CHECK(cudaCreateTextureObject(&texture, &resource, &texture_desc,
                                        nullptr));

    textureReadKernel<<<blocks, kThreadsPerBlock>>>(texture, device_result,
                                                    kElementCount);
    PROBE_CHECK(cudaGetLastError());
    PROBE_CHECK(cudaDeviceSynchronize());

    PROBE_CHECK(cudaMemcpy(host_result.data(), device_result, bytes,
                           cudaMemcpyDeviceToHost));
    PROBE_CHECK(cudaDestroyTextureObject(texture));
    PROBE_CHECK(cudaFree(device_result));
    PROBE_CHECK(cudaFree(device_source));

    bool match = true;
    for (int i = 0; i < kElementCount; ++i) {
        if (host_result[i] != host_source[i]) {
            match = false;
            break;
        }
    }
    if (!match) {
        std::fprintf(stderr, "PROBE_ERROR texture fetch output mismatch\n");
        return EXIT_FAILURE;
    }

    std::printf(
        "PROBE=texture_object device=%s elements=%d resource=linear "
        "RESULT=PASS\n",
        properties.name, kElementCount);
    return EXIT_SUCCESS;
}
