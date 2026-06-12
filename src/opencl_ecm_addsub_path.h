#pragma once

#include <cstdint>
#include <stdio.h>

enum {
    ECM_ADDSUB_PATH_FUSED = 0,
    ECM_ADDSUB_PATH_FUSED_UNROLL = 1,
    ECM_ADDSUB_PATH_FUSED_UNROLL_B32 = 2,
    ECM_ADDSUB_PATH_ASM_B32 = 3,
    ECM_ADDSUB_PATH_FUSED_UNROLL_B16 = 4,
    ECM_ADDSUB_PATH_ASM_B16 = 5,
};

int opencl_ecm_parse_addsub_path(const char *path);
const char *opencl_ecm_addsub_path_name(int path_id);
bool opencl_ecm_addsub_path_needs_asm_b32(int path_id);
bool opencl_ecm_addsub_path_needs_asm_b16(int path_id);
struct EcmAddModPathDescriptor;
struct EcmSubModPathDescriptor;
const EcmAddModPathDescriptor *opencl_ecm_resolve_addsub_add_path(const char *path, uint32_t limbs,
                                                                  bool is_amd);
const EcmSubModPathDescriptor *opencl_ecm_resolve_addsub_sub_path(const char *path, uint32_t limbs,
                                                                  bool is_amd);
void opencl_ecm_print_available_kernels(FILE *out);
