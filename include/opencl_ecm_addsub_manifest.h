#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Priority tier (1 = highest): ASM > LPT > manual unroll > auto unroll > fused > basic.
enum class EcmAddSubTier : int {
    Asm = 1,
    Lpt = 2,
    FusedUnrollManual = 3,
    FusedUnrollAuto = 4,
    Fused = 5,
    Basic = 6,
};

struct EcmAddSubBenchKernel {
    const char *kernel_name;   // OpenCL entry symbol
    const char *path_label;    // e.g. fused_unroll_asm_b16
    EcmAddSubTier tier;
    bool is_add;               // add-mod vs sub-mod
    bool amd_only;
    bool use_wg;               // requires local work-group (LPT)
    int lpt_chunk;             // 0 if N/A
    const char *compare_label; // vs reference for printed speedup line
    bool hot_inner_loop;       // arg5 = inner_iters; 1 enqueue per launch_repeat
};

struct EcmAddSubBuildManifest {
    std::vector<std::string> source_paths;
};

EcmAddSubBuildManifest opencl_ecm_addsub_build_manifest(uint32_t words, bool asm_enabled,
                                                        bool asm_b64_enabled);

std::vector<EcmAddSubBenchKernel> opencl_ecm_addsub_add_kernels(uint32_t words, bool asm_enabled,
                                                              bool asm_b64_enabled,
                                                              bool bench_unroll_only);

std::vector<EcmAddSubBenchKernel> opencl_ecm_addsub_sub_kernels(uint32_t words, bool asm_enabled,
                                                              bool asm_b64_enabled);
