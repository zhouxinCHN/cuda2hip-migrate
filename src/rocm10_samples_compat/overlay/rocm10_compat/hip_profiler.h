// ROCm 10 compatibility overlay for the deprecated CUDA profiler session API.
//
// hipProfilerStart / hipProfilerStop are declared HIP_DEPRECATED and documented
// to return hipErrorNotSupported; the symbols left in libamdhip64.so are hip_4.2
// ABI stubs. Measured on ROCm 10.0.0 (HIP 7.15.26333) they return 801 in every
// situation, including with rocprofv3 attached.
//
// HIP-PROFILER-ROCTX-001 rewrites the call sites to this wrapper. Names and
// signatures match the CUDA originals, so a sample reaches C1/C2/C3 unchanged
// while the instrumentation contract is preserved: rocTX pauses and resumes
// profiling for the whole process, exactly as cudaProfilerStart/Stop do.
//
// Measured on ROCm 10.0.0: roctxProfilerResume(0) / roctxProfilerPause(0)
// return 0 when no tool is attached and when the requested state already holds.
//
// Link requirement: the wrapper needs librocprofiler-sdk-roctx, which is why the
// samples that use it declare that library in their spec. Note that
// libROCTX64.so does not export these two functions - only
// librocprofiler-sdk-roctx.so does.

#ifndef ROCM10_COMPAT_HIP_PROFILER_H_
#define ROCM10_COMPAT_HIP_PROFILER_H_

#if defined(__HIPCC__)

#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>

namespace rocm10_compat {

// rocTX returns 0 on success. A non-zero value means the profiling tool refused
// or failed the request, which must not be reported as success.
inline hipError_t hipProfilerStart() {
  return roctxProfilerResume(0) == 0 ? hipSuccess : hipErrorUnknown;
}

inline hipError_t hipProfilerStop() {
  return roctxProfilerPause(0) == 0 ? hipSuccess : hipErrorUnknown;
}

}  // namespace rocm10_compat

#else  // hipify-clang parse: declare the surface only, compile nothing

namespace rocm10_compat {

inline int hipProfilerStart();
inline int hipProfilerStop();

}  // namespace rocm10_compat

#endif  // __HIPCC__

#endif  // ROCM10_COMPAT_HIP_PROFILER_H_
