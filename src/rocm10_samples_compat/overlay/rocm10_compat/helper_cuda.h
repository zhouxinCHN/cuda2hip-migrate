// ROCm 10 compatibility overlay for CUDA Samples v12.9 Common/helper_cuda.h
//
// This header is routed in *before* HIPIFY by PRE-HIPIFY-INCLUDE-REDIRECT.
// HIPIFY maps "#include <helper_cuda.h>" to "#include <hip/hip_runtime_api.h>",
// which removes every helper definition the sample calls. Routing first keeps
// the call sites intact and gives them HIP implementations with the same names
// and signatures, so the sample source itself needs no edits.
//
// Supported surface (measured against the P0 baseline):
//   check / checkCudaErrors / getLastError / getLastCudaError
//   _ConvertSMVer2Cores / _ConvertSMVer2ArchName
//   gpuDeviceInit / gpuGetMaxGflopsDeviceId / findCudaDevice
//   checkCudaCapabilities
//
// Anything outside this surface is intentionally not provided: a sample that
// needs more must fail at C1 with a clear missing-symbol error rather than be
// silently mis-adapted. Extend this file only with a test and evidence.

#ifndef ROCM10_COMPAT_HELPER_CUDA_H_
#define ROCM10_COMPAT_HELPER_CUDA_H_

// hipify-clang parses this header while the file is still CUDA. It defines
// __HIP_PLATFORM_NVIDIA__, so pulling in <hip/hip_runtime.h> there selects
// HIP's NVIDIA shim and fails against the CUDA headers of this machine.
// hipcc defines __HIPCC__ and never the NVIDIA platform macro, so the two
// parsers take different branches and only the build sees the real bodies.
#if defined(__HIPCC__)

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#else  // hipify-clang parse: declare the surface only, compile nothing

// hipify-clang re-parses its own output to collect statistics, and that output
// no longer includes a CUDA runtime header: cuda_runtime.h becomes
// hip/hip_runtime.h, which is not resolvable there. Unqualified max()/min() in
// host code (simpleStreams) then only sees clang's device-only CUDA overloads
// and the re-parse fails. Bringing the standard overloads into scope for the
// parse restores what the CUDA build gets. This branch is never compiled by
// hipcc, so it cannot change the built binary.
#include <algorithm>
using std::max;
using std::min;

// The CUDA-side call sites pass cudaError_t, an unscoped enum, so an int
// parameter accepts them. No body is needed for translation.
inline void check(int result, const char *const func, const char *const file,
                  int const line);

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

inline void getLastError();
inline void getLastCudaError(const char *errorMessage);
inline int _ConvertSMVer2Cores(int major, int minor);
inline const char *_ConvertSMVer2ArchName(int major, int minor);
inline int gpuDeviceInit(int devID);
inline int gpuGetMaxGflopsDeviceId();
inline int findCudaDevice(int argc, const char **argv);
inline bool checkCudaCapabilities(int major_version, int minor_version);

#endif  // __HIPCC__

#if defined(__HIPCC__)

// Error checking -------------------------------------------------------------

inline void check(hipError_t result, char const *const func,
                  const char *const file, int const line) {
  if (result != hipSuccess) {
    fprintf(stderr, "HIP error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), hipGetErrorName(result), func);
    fflush(stdout);
    fflush(stderr);
    exit(EXIT_FAILURE);
  }
}

// CUDA Samples name every runtime call as an argument; the name is kept so the
// sample text does not change between the CUDA and HIP builds.
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

inline void getLastError() { checkCudaErrors(hipGetLastError()); }

inline void getLastCudaError(const char *errorMessage) {
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "%s: HIP error at %s:%d code=%d(%s)\n", errorMessage, __FILE__,
            __LINE__, static_cast<unsigned int>(err), hipGetErrorName(err));
    exit(EXIT_FAILURE);
  }
}

// Architecture helpers -------------------------------------------------------
//
// The core-count table is copied from cuda-samples v12.9
// Common/helper_cuda.h so that printed values and device selection match the
// upstream CUDA build exactly. ROCm devices report a HIP major/minor that does
// not appear in the table; the upstream fallback is preserved.

inline int _ConvertSMVer2Cores(int major, int minor) {
  typedef struct {
    int SM;
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128},
      {0x52, 128}, {0x53, 128}, {0x60, 64},  {0x61, 128}, {0x62, 128},
      {0x70, 64},  {0x72, 64},  {0x75, 64},  {0x80, 64},  {0x86, 128},
      {0x87, 128}, {0x89, 128}, {0x90, 128}, {0xa0, 128}, {0xa1, 128},
      {0xc0, 128}, {-1, -1}};

  int index = 0;
  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }
    index++;
  }

  printf("MapSMtoCores for SM %d.%d is undefined."
         "  Default to use %d Cores/SM\n",
         major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}

// Upstream maps an SM version to an NVIDIA architecture name. A ROCm device has
// no such name, and printing an NVIDIA one would put a false statement into the
// run log, so unknown versions resolve to a neutral name instead.
inline const char *_ConvertSMVer2ArchName(int major, int minor) {
  typedef struct {
    int SM;
    const char *name;
  } sSMtoArchName;

  sSMtoArchName nGpuArchNameSM[] = {
      {0x30, "Kepler"},   {0x32, "Kepler"},   {0x35, "Kepler"},
      {0x37, "Kepler"},   {0x50, "Maxwell"},  {0x52, "Maxwell"},
      {0x53, "Maxwell"},  {0x60, "Pascal"},   {0x61, "Pascal"},
      {0x62, "Pascal"},   {0x70, "Volta"},    {0x72, "Xavier"},
      {0x75, "Turing"},   {0x80, "Ampere"},   {0x86, "Ampere"},
      {0x87, "Ampere"},   {0x89, "Ada"},      {0x90, "Hopper"},
      {0xa0, "Blackwell"}, {0xa1, "Blackwell"}, {0xc0, "Blackwell"},
      {-1, "Graphics Device"}};

  int index = 0;
  while (nGpuArchNameSM[index].SM != -1) {
    if (nGpuArchNameSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchNameSM[index].name;
    }
    index++;
  }

  return "Graphics Device";
}

// Device selection -----------------------------------------------------------

// Accept the flag forms used by cuda-samples: -device=N and --device=N.
// Returns -1 when the flag is absent.
inline int rocm10GetDeviceFlag(int argc, const char **argv) {
  const char *key = "device=";
  const size_t keyLength = strlen(key);

  for (int i = 1; i < argc; ++i) {
    const char *argument = argv[i];
    if (argument == NULL || argument[0] != '-') {
      continue;
    }
    const char *name = argument;
    while (*name == '-') {
      ++name;
    }
    if (name == argument) {
      continue;  // a single dash is not a long option
    }
    if (strncmp(name, key, keyLength) != 0) {
      continue;
    }
    return atoi(name + keyLength);
  }

  return -1;
}

inline int gpuDeviceInit(int devID) {
  int device_count;
  checkCudaErrors(hipGetDeviceCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr,
            "gpuDeviceInit() HIP error: "
            "no devices supporting HIP.\n");
    exit(EXIT_FAILURE);
  }

  if (devID < 0) {
    devID = 0;
  }

  if (devID > device_count - 1) {
    fprintf(stderr, "\n");
    fprintf(stderr, ">> %d HIP capable GPU device(s) detected. <<\n",
            device_count);
    fprintf(stderr,
            ">> gpuDeviceInit (-device=%d) is not a valid"
            " GPU device. <<\n",
            devID);
    fprintf(stderr, "\n");
    return -devID;
  }

  int computeMode = -1, major = 0, minor = 0;
  checkCudaErrors(
      hipDeviceGetAttribute(&computeMode, hipDeviceAttributeComputeMode, devID));
  checkCudaErrors(hipDeviceGetAttribute(&major,
                                        hipDeviceAttributeComputeCapabilityMajor,
                                        devID));
  checkCudaErrors(hipDeviceGetAttribute(&minor,
                                        hipDeviceAttributeComputeCapabilityMinor,
                                        devID));
  if (computeMode == hipComputeModeProhibited) {
    fprintf(stderr,
            "Error: device is running in <Compute Mode "
            "Prohibited>, no threads can use hipSetDevice().\n");
    return -1;
  }

  if (major < 1) {
    fprintf(stderr, "gpuDeviceInit(): GPU device does not support HIP.\n");
    exit(EXIT_FAILURE);
  }

  checkCudaErrors(hipSetDevice(devID));
  printf("gpuDeviceInit() HIP Device [%d]: \"%s\n", devID,
         _ConvertSMVer2ArchName(major, minor));

  return devID;
}

inline int gpuGetMaxGflopsDeviceId() {
  int current_device = 0, sm_per_multiproc = 0;
  int max_perf_device = 0;
  int device_count = 0;
  int devices_prohibited = 0;

  uint64_t max_compute_perf = 0;
  checkCudaErrors(hipGetDeviceCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr,
            "gpuGetMaxGflopsDeviceId() HIP error:"
            " no devices supporting HIP.\n");
    exit(EXIT_FAILURE);
  }

  current_device = 0;
  while (current_device < device_count) {
    int computeMode = -1, major = 0, minor = 0;
    checkCudaErrors(hipDeviceGetAttribute(
        &computeMode, hipDeviceAttributeComputeMode, current_device));
    checkCudaErrors(hipDeviceGetAttribute(
        &major, hipDeviceAttributeComputeCapabilityMajor, current_device));
    checkCudaErrors(hipDeviceGetAttribute(
        &minor, hipDeviceAttributeComputeCapabilityMinor, current_device));

    if (computeMode != hipComputeModeProhibited) {
      if (major == 9999 && minor == 9999) {
        sm_per_multiproc = 1;
      } else {
        sm_per_multiproc = _ConvertSMVer2Cores(major, minor);
      }
      int multiProcessorCount = 0, clockRate = 0;
      checkCudaErrors(hipDeviceGetAttribute(
          &multiProcessorCount, hipDeviceAttributeMultiprocessorCount,
          current_device));
      hipError_t result = hipDeviceGetAttribute(
          &clockRate, hipDeviceAttributeClockRate, current_device);
      if (result != hipSuccess) {
        if (result == hipErrorInvalidValue) {
          clockRate = 1;
        } else {
          fprintf(stderr, "HIP error at %s:%d code=%d(%s) \n", __FILE__,
                  __LINE__, static_cast<unsigned int>(result),
                  hipGetErrorName(result));
          exit(EXIT_FAILURE);
        }
      }
      uint64_t compute_perf =
          (uint64_t)multiProcessorCount * sm_per_multiproc * clockRate;

      if (compute_perf > max_compute_perf) {
        max_compute_perf = compute_perf;
        max_perf_device = current_device;
      }
    } else {
      devices_prohibited++;
    }

    ++current_device;
  }

  if (devices_prohibited == device_count) {
    fprintf(stderr,
            "gpuGetMaxGflopsDeviceId() HIP error:"
            " all devices have compute mode prohibited.\n");
    exit(EXIT_FAILURE);
  }

  return max_perf_device;
}

inline int findCudaDevice(int argc, const char **argv) {
  int devID = rocm10GetDeviceFlag(argc, argv);

  if (devID >= 0) {
    devID = gpuDeviceInit(devID);

    if (devID < 0) {
      printf("exiting...\n");
      exit(EXIT_FAILURE);
    }
  } else {
    devID = gpuGetMaxGflopsDeviceId();
    checkCudaErrors(hipSetDevice(devID));
    int major = 0, minor = 0;
    checkCudaErrors(hipDeviceGetAttribute(
        &major, hipDeviceAttributeComputeCapabilityMajor, devID));
    checkCudaErrors(hipDeviceGetAttribute(
        &minor, hipDeviceAttributeComputeCapabilityMinor, devID));
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID,
           _ConvertSMVer2ArchName(major, minor), major, minor);
  }

  return devID;
}

inline bool checkCudaCapabilities(int major_version, int minor_version) {
  int dev;
  int major = 0, minor = 0;

  checkCudaErrors(hipGetDevice(&dev));
  checkCudaErrors(hipDeviceGetAttribute(
      &major, hipDeviceAttributeComputeCapabilityMajor, dev));
  checkCudaErrors(hipDeviceGetAttribute(
      &minor, hipDeviceAttributeComputeCapabilityMinor, dev));

  if ((major > major_version) ||
      (major == major_version && minor >= minor_version)) {
    printf("  Device %d: <%16s >, Compute SM %d.%d detected\n", dev,
           _ConvertSMVer2ArchName(major, minor), major, minor);
    return true;
  } else {
    printf(
        "  No GPU device was found that can support "
        "HIP compute capability %d.%d.\n",
        major_version, minor_version);
    return false;
  }
}

#endif  // __HIPCC__

#endif  // ROCM10_COMPAT_HELPER_CUDA_H_
