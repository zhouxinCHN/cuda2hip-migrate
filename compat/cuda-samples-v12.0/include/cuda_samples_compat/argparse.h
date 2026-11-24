// Host-only command-line parsing used by the cuda-samples overlay.
//
// Deliberately free of HIP and CUDA dependencies so it can be compiled and
// tested with a plain host compiler, without a GPU or a ROCm installation.
//
// Matches the NVIDIA helper convention: any number of leading '-' characters is
// stripped, then "<name>=<value>" is required. So -device=0, --device=0 and
// device=0 all name the same flag.

#ifndef CUDA_SAMPLES_COMPAT_ARGPARSE_H
#define CUDA_SAMPLES_COMPAT_ARGPARSE_H

#include <cstddef>
#include <cstring>

namespace cuda_samples_compat {

inline bool find_flag_value(int argc, const char **argv, const char *name,
                            const char **value) {
    const std::size_t name_length = std::strlen(name);
    for (int i = 1; i < argc; ++i) {
        const char *argument = argv[i];
        if (argument == nullptr) {
            continue;
        }
        while (*argument == '-') {
            ++argument;
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

#endif  // CUDA_SAMPLES_COMPAT_ARGPARSE_H
