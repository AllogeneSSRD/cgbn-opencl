#pragma once

#include <cstdio>
#include <cstdint>

// Keep in sync with ECM_ADDSUB_PATH_* in ecm_stage1.cl
enum {
    ECM_ADDSUB_PATH_FUSED = 0,
    ECM_ADDSUB_PATH_FUSED_UNROLL = 1,
    ECM_ADDSUB_PATH_FUSED_UNROLL_B32 = 2,
    ECM_ADDSUB_PATH_ASM_B32 = 3,
    ECM_ADDSUB_PATH_FUSED_UNROLL_B16 = 4,
    ECM_ADDSUB_PATH_ASM_B16 = 5,
    ECM_ADDSUB_PATH_UNROLL_128B = 6,
    ECM_ADDSUB_PATH_ASM_128B = 7,
    ECM_ADDSUB_PATH_UNROLL_192B = 8,
    ECM_ADDSUB_PATH_ASM_192B = 9,
    ECM_ADDSUB_PATH_UNROLL_256B = 10,
    ECM_ADDSUB_PATH_ASM_256B = 11,
    ECM_ADDSUB_PATH_UNROLL_384B = 12,
    ECM_ADDSUB_PATH_ASM_384B = 13,
};

int opencl_ecm_parse_addsub_path(const char *path);
const char *opencl_ecm_addsub_path_name(int path_id);
bool opencl_ecm_addsub_path_needs_asm_b32(int path_id);
bool opencl_ecm_addsub_path_needs_asm_b16(int path_id);
bool opencl_ecm_addsub_path_needs_addsub_bits(int path_id);

struct EcmPathContext;
struct EcmAddSubPathDescriptor;
const EcmAddSubPathDescriptor *opencl_ecm_resolve_addsub_add_path(const char *path,
                                                                  const EcmPathContext &ctx);
const EcmAddSubPathDescriptor *opencl_ecm_resolve_addsub_sub_path(const char *path,
                                                                  const EcmPathContext &ctx);

void opencl_ecm_print_available_kernels(FILE *out);
