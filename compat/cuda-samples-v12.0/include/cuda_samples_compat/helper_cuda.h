// Minimal HIP overlay for cuda-samples v12.0 Common/helper_cuda.h.
//
// Selected by a deterministic pre-HIPIFY include redirect. The original
// Common/helper_cuda.h is never edited and is never compiled in HIP builds.
// CPU-only Common headers (helper_string.h, helper_timer.h, helper_image.h,
// helper_functions.h) keep coming from the original checkout.
//
// Implemented so far:
//   checkCudaErrors(expr)      - error check that prints and exits
//   findCudaDevice(argc, argv) - honours -device=<n>, validates and selects it
//
// Not implemented on purpose: gpuGetMaxGflopsDeviceId-based selection,
// compute-capability checks, driver API helpers, cudaDeviceProp tuning.
//
// The header has two modes. hipcc defines __HIPCC__/__HIP__, so the HIP branch
// is what actually builds. hipify-clang parses the file in CUDA mode and only
// needs the surface to type-check, so the CUDA branch mirrors the same
// declarations against the CUDA runtime; it is never compiled into a binary.

#ifndef CUDA_SAMPLES_COMPAT_HELPER_CUDA_H
#define CUDA_SAMPLES_COMPAT_HELPER_CUDA_H

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "cuda_samples_compat/argparse.h"

// NVIDIA's helper_cuda.h includes the CPU-only helper_string.h and defines
// EXIT_WAIVED itself; keep both surfaces so samples that rely on them compile
// unchanged. helper_string.h keeps coming from the original Common checkout.
#include <helper_string.h>

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#if defined(__HIPCC__) || defined(__HIP__)

#include <hip/hip_runtime.h>

#define compat_error_t hipError_t
#define compat_device_prop_t hipDeviceProp_t
#define compat_success hipSuccess
#define compat_get_error_string hipGetErrorString
#define compat_get_device_count hipGetDeviceCount
#define compat_set_device hipSetDevice
#define compat_get_device_properties hipGetDeviceProperties

#else

// Parse-only branch for hipify-clang.
#include <cuda_runtime.h>

#define compat_error_t cudaError_t
#define compat_device_prop_t cudaDeviceProp
#define compat_success cudaSuccess
#define compat_get_error_string cudaGetErrorString
#define compat_get_device_count cudaGetDeviceCount
#define compat_set_device cudaSetDevice
#define compat_get_device_properties cudaGetDeviceProperties

#endif

namespace cuda_samples_compat {

inline void check_error(compat_error_t result, const char *func,
                        const char *file, int line) {
    if (result != compat_success) {
        std::fprintf(stderr, "HIP error at %s:%d code=%d(%s) \"%s\"\n", file,
                     line, static_cast<int>(result),
                     compat_get_error_string(result), func);
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace cuda_samples_compat

// Matches the NVIDIA helper's usage: checkCudaErrors(<runtime call>).
#define checkCudaErrors(val)                                                 \
    cuda_samples_compat::check_error((val), #val, __FILE__, __LINE__)

// Returns the selected device id and sets it as the current device.
inline int findCudaDevice(int argc, const char **argv) {
    int device = 0;
    const char *value = nullptr;
    if (cuda_samples_compat::find_flag_value(argc, argv, "device", &value)) {
        char *end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end == value || *end != '\0' || parsed < 0) {
            std::fprintf(stderr, "Invalid device id \"%s\"\n", value);
            std::exit(EXIT_FAILURE);
        }
        device = static_cast<int>(parsed);
    }

    int count = 0;
    checkCudaErrors(compat_get_device_count(&count));
    if (device >= count) {
        std::fprintf(stderr, "Device id %d is out of range (0..%d)\n", device,
                     count - 1);
        std::exit(EXIT_FAILURE);
    }

    checkCudaErrors(compat_set_device(device));
    compat_device_prop_t properties{};
    checkCudaErrors(compat_get_device_properties(&properties, device));
    std::printf("Device %d: \"%s\"\n", device, properties.name);
    return device;
}

// NVIDIA maps a compute capability to cores per SM. AMD architectures have no
// SM version, so this returns the SIMD width in lanes per compute unit (64 on
// GCN, RDNA and CDNA) as the closest hardware-level analogue. It is used by
// samples only for banner text and for a workload scale clamp, never for
// correctness; the value is deliberately conservative rather than a marketing
// shader count.
inline int _ConvertSMVer2Cores(int major, int minor) {
    (void)major;
    (void)minor;
    return 64;
}

// NVIDIA's version measures device GFLOPs by compute capability. The AMD proxy
// here is lanes x compute units x clock, used only to rank devices.
inline int gpuGetMaxGflopsDeviceId() {
    int count = 0;
    checkCudaErrors(compat_get_device_count(&count));
    if (count <= 0) {
        return -1;
    }

    int best_device = 0;
    long long best_score = -1;
    for (int device = 0; device < count; ++device) {
        compat_device_prop_t properties{};
        checkCudaErrors(compat_get_device_properties(&properties, device));
        const long long score = static_cast<long long>(properties.multiProcessorCount) *
                                _ConvertSMVer2Cores(properties.major, properties.minor) *
                                properties.clockRate;
        if (score > best_score) {
            best_score = score;
            best_device = device;
        }
    }

    checkCudaErrors(compat_set_device(best_device));
    return best_device;
}

// Validates one explicit device id, selects it and reports it.
inline int gpuDeviceInit(int dev_id) {
    int count = 0;
    checkCudaErrors(compat_get_device_count(&count));
    if (dev_id < 0 || dev_id >= count) {
        std::fprintf(stderr, ">> gpuDeviceInit (-device=%d) is not a valid device\n",
                     dev_id);
        return -1;
    }

    checkCudaErrors(compat_set_device(dev_id));
    compat_device_prop_t properties{};
    checkCudaErrors(compat_get_device_properties(&properties, dev_id));
    std::printf("gpuDeviceInit() HIP Device [%d]: \"%s\"\n", dev_id,
                properties.name);
    return dev_id;
}

#endif  // CUDA_SAMPLES_COMPAT_HELPER_CUDA_H
