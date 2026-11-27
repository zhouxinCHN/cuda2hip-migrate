# Evidence

Evidence is recorded per sample and per environment.

Required fields:

- CUDA Samples tag and commit;
- sample path;
- ROCm, HIP, driver and GPU architecture;
- compiler and optimization level;
- include directories and libraries;
- pre-HIPIFY redirects;
- post-HIPIFY rule hits;
- C1/C2/C3 result;
- success marker or golden comparison;
- known limitations;
- artifact path and log hashes.

Do not record a sample as supported without evidence. A blocked or unsupported
sample remains visible in the support matrix.
