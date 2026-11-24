// Minimal CUDA probe: cuBLAS single-precision GEMM on device buffers.
// Exercises the library layer rather than the runtime layer: the handle, the
// operation enum and the SGEMM call are the only library surface used.

#include <cublas_v2.h>
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

#define PROBE_CHECK_LIB(call)                                                  \
    do {                                                                       \
        cublasStatus_t probe_status = (call);                                  \
        if (probe_status != CUBLAS_STATUS_SUCCESS) {                           \
            std::fprintf(stderr, "PROBE_ERROR %s:%d %s: status %d\n", __FILE__, \
                         __LINE__, #call, static_cast<int>(probe_status));      \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

int main() {
    int device = 0;
    cudaDeviceProp properties{};
    PROBE_CHECK(cudaGetDevice(&device));
    PROBE_CHECK(cudaGetDeviceProperties(&properties, device));

    constexpr int kN = 64;
    const size_t elements = static_cast<size_t>(kN) * kN;
    const size_t bytes = elements * sizeof(float);

    // Column-major A and B, C = A * B.
    std::vector<float> host_a(elements);
    std::vector<float> host_b(elements);
    for (int column = 0; column < kN; ++column) {
        for (int row = 0; row < kN; ++row) {
            const size_t index = static_cast<size_t>(column) * kN + row;
            host_a[index] = static_cast<float>((row + 2 * column) % 7) * 0.25f;
            host_b[index] = static_cast<float>((3 * row + column) % 5) * 0.5f;
        }
    }

    std::vector<float> host_reference(elements, 0.0f);
    for (int column = 0; column < kN; ++column) {
        for (int row = 0; row < kN; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < kN; ++k) {
                sum += host_a[static_cast<size_t>(k) * kN + row] *
                       host_b[static_cast<size_t>(column) * kN + k];
            }
            host_reference[static_cast<size_t>(column) * kN + row] = sum;
        }
    }

    float *device_a = nullptr;
    float *device_b = nullptr;
    float *device_c = nullptr;
    PROBE_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_a), bytes));
    PROBE_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_b), bytes));
    PROBE_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_c), bytes));
    PROBE_CHECK(cudaMemcpy(device_a, host_a.data(), bytes, cudaMemcpyHostToDevice));
    PROBE_CHECK(cudaMemcpy(device_b, host_b.data(), bytes, cudaMemcpyHostToDevice));

    cublasHandle_t handle = nullptr;
    PROBE_CHECK_LIB(cublasCreate(&handle));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    PROBE_CHECK_LIB(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, kN, kN, kN,
                                &alpha, device_a, kN, device_b, kN, &beta,
                                device_c, kN));
    PROBE_CHECK(cudaDeviceSynchronize());
    PROBE_CHECK_LIB(cublasDestroy(handle));

    std::vector<float> host_result(elements, 0.0f);
    PROBE_CHECK(cudaMemcpy(host_result.data(), device_c, bytes,
                           cudaMemcpyDeviceToHost));
    PROBE_CHECK(cudaFree(device_c));
    PROBE_CHECK(cudaFree(device_b));
    PROBE_CHECK(cudaFree(device_a));

    bool match = true;
    for (size_t i = 0; i < elements; ++i) {
        const float tolerance =
            1e-2f * std::fmax(1.0f, std::fabs(host_reference[i]));
        if (std::fabs(host_result[i] - host_reference[i]) > tolerance) {
            match = false;
            break;
        }
    }
    if (!match) {
        std::fprintf(stderr, "PROBE_ERROR cuBLAS GEMM result mismatch\n");
        return EXIT_FAILURE;
    }

    std::printf(
        "PROBE=cublas_gemm device=%s order=column-major n=%d "
        "elements=%zu RESULT=PASS\n",
        properties.name, kN, elements);
    return EXIT_SUCCESS;
}
