#pragma once

#include <cstdint>
#include <string>
#include <vector>

enum class MontDispatch : uint8_t {
    PrivLegacy,
    PrivOpt,
    PrivUnroll,
    PrivLocal512,
    PrivOpt2Local512,
    PrivFipsMt,
    PrivFipsMtCs,
    Wg,
};

struct EcmMontSqrBenchKernel {
    const char* kernel_name;
    const char* path_label;
    bool is_mul;
    MontDispatch dispatch;
    uint32_t required_words; // 0 = match current limb count
    uint32_t mt_local_size;    // PrivFipsMt / PrivFipsMtCs
};

struct EcmMontSqrBuildManifest {
    std::vector<std::string> source_paths;
};

EcmMontSqrBuildManifest opencl_ecm_montsqr_build_manifest(uint32_t words, bool use_wg);

std::vector<EcmMontSqrBenchKernel> opencl_ecm_montsqr_mul_kernels(uint32_t words, bool use_wg);

std::vector<EcmMontSqrBenchKernel> opencl_ecm_montsqr_sqr_kernels(uint32_t words, bool use_wg);
