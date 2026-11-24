// HIPIFY parse shim. Never compiled into a binary.
//
// hipify-clang parses CUDA sources with its own CUDA math shim, which declares
// max/min as __device__-only. nvcc exposes them to host code as well, so
// samples that call max(...) on the host (simpleMultiCopy does, for a workload
// clamp) fail to parse and HIPIFY produces no output at all.
//
// Pulling std::max/std::min into the global namespace gives the host call a
// viable overload, so the sample parses and HIPIFY can translate it. This file
// is passed to hipify-clang with -include and is deliberately not part of any
// build: hipcc's own headers already expose host max/min.
//
// A macro form (#define max(a, b) ...) was tried first and breaks libstdc++
// headers, so overload injection is used instead.

#ifndef CUDA_SAMPLES_COMPAT_HOST_MATH_PARSE_SHIM_H
#define CUDA_SAMPLES_COMPAT_HOST_MATH_PARSE_SHIM_H

#include <algorithm>

using std::max;
using std::min;

#endif  // CUDA_SAMPLES_COMPAT_HOST_MATH_PARSE_SHIM_H
