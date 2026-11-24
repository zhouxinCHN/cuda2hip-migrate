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

#if defined(__HIPCC__) || defined(__HIP__)

#include <hip/hip_runtime.h>

#define compat_error_t hipError_t
#define compat_device_prop_t hipDeviceProp_t
#define compat_get_error_string hipGetErrorString
#define compat_get_device_count hipGetDeviceCount
#define compat_set_device hipSetDevice
#define compat_get_device_properties hipGetDeviceProperties

#else

// Parse-only branch for hipify-clang.
#include <cuda_runtime.h>

#define compat_error_t cudaError_t
#define compat_device_prop_t cudaDeviceProp
#define compat_get_error_string cudaGetErrorString
#define compat_get_device_count cudaGetDeviceCount
#define compat_set_device cudaSetDevice
#define compat_get_device_properties cudaGetDeviceProperties

#endif

namespace cuda_samples_compat {

inline void check_error(compat_error_t result, const char *func,
                        const char *file, int line) {
    if (result != 0) {
        std::fprintf(stderr, "HIP error at %s:%d code=%d(%s) \"%s\"\n", file,
                     line, static_cast<int>(result),
                     compat_get_error_string(result), func);
        std::exit(EXIT_FAILURE);
    }
}

inline bool find_flag_value(int argc, const char **argv, const char *name,
                            const char **value) {
    const std::size_t name_length = std::strlen(name);
    for (int i = 1; i < argc; ++i) {
        const char *argument = argv[i];
        if (argument == nullptr) {
            continue;
        }
        if (argument[0] == '-' && argument[1] == '-') {
            argument += 1;
        }
        if (std::strncmp(argument, name, name_length) != 0) {
            continue;
        }
        if (argument[name_length] != '=') {
            continue;
        }
        *value = argument + name_length + 1;
        return true;
    }
    return false;
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

#endif  // CUDA_SAMPLES_COMPAT_HELPER_CUDA_H
