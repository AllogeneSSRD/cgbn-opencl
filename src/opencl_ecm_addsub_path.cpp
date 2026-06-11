#include "opencl_ecm_addsub_path.h"

#include "opencl_ecm_mont_path.h"

#include <cstring>

int opencl_ecm_parse_addsub_path(const char *path) {
    if (path == nullptr || path[0] == '\0' || strcmp(path, "auto") == 0 ||
        strcmp(path, "default") == 0) {
        return -1;
    }
    if (strcmp(path, "fused") == 0) {
        return ECM_ADDSUB_PATH_FUSED;
    }
    if (strcmp(path, "fused_unroll") == 0) {
        return ECM_ADDSUB_PATH_FUSED_UNROLL;
    }
    if (strcmp(path, "fused_unroll_b32") == 0) {
        return ECM_ADDSUB_PATH_FUSED_UNROLL_B32;
    }
    if (strcmp(path, "asm_b32") == 0) {
        return ECM_ADDSUB_PATH_ASM_B32;
    }
    if (strcmp(path, "asm_b16") == 0 || strcmp(path, "fused_asm_b16") == 0) {
        return ECM_ADDSUB_PATH_ASM_B16;
    }
    if (strcmp(path, "fused_unroll_b16") == 0) {
        return ECM_ADDSUB_PATH_FUSED_UNROLL_B16;
    }
    if (strcmp(path, "fused_unroll_auto") == 0) {
        return ECM_ADDSUB_PATH_FUSED_UNROLL_B16;
    }
    return -2;
}

const char *opencl_ecm_addsub_path_name(int path_id) {
    switch (path_id) {
    case ECM_ADDSUB_PATH_FUSED:
        return "fused";
    case ECM_ADDSUB_PATH_FUSED_UNROLL:
        return "fused_unroll";
    case ECM_ADDSUB_PATH_FUSED_UNROLL_B32:
        return "fused_unroll_b32";
    case ECM_ADDSUB_PATH_ASM_B32:
        return "asm_b32";
    case ECM_ADDSUB_PATH_ASM_B16:
        return "asm_b16";
    case ECM_ADDSUB_PATH_FUSED_UNROLL_B16:
        return "fused_unroll_b16";
    default:
        return "unknown";
    }
}

bool opencl_ecm_addsub_path_needs_asm_b32(int path_id) {
    return path_id == ECM_ADDSUB_PATH_ASM_B32;
}

bool opencl_ecm_addsub_path_needs_asm_b16(int path_id) {
    return path_id == ECM_ADDSUB_PATH_ASM_B16;
}

int opencl_ecm_resolve_addsub_path(const char *path, uint32_t limbs, bool is_amd, bool is_add) {
    int parsed = opencl_ecm_parse_addsub_path(path);
    if (parsed >= 0) {
        return parsed;
    }
    if (parsed == -2) {
        return -2;
    }
    if (limbs == 128u) {
        if (is_amd) {
            return ECM_ADDSUB_PATH_ASM_B32;
        }
        return ECM_ADDSUB_PATH_FUSED_UNROLL_B32;
    }
    if (limbs == 16u) {
        if (is_add && is_amd) {
            return ECM_ADDSUB_PATH_ASM_B16;
        }
        if (!is_amd) {
            // Adreno: fused loop beats fused_unroll_b16 at typical gpucurves (1–512); see
            // Android/ECM/docs/DEV_ECM_ADDSUB_ANDROID.md
            return ECM_ADDSUB_PATH_FUSED;
        }
        return ECM_ADDSUB_PATH_FUSED_UNROLL_B16;
    }
    return ECM_ADDSUB_PATH_FUSED_UNROLL;
}

void opencl_ecm_print_available_kernels(FILE *out) {
    if (out == nullptr) {
        out = stdout;
    }
    fprintf(out, "ECM OpenCL kernels and paths\n\n");
    fprintf(out, "Stage1 main kernel:\n");
    fprintf(out, "  kernel_double_add\n\n");

    fprintf(out, "4096-bit Montgomery mul (--mul):\n");
    fprintf(out, "  default, unroll64_4096, unroll64_4096_mt2, fips4096, fips4096_mt8, fips4096_mt16\n\n");

    fprintf(out, "4096-bit Montgomery sqr (--sqr):\n");
    fprintf(out, "  default, unroll64_4096, unroll64_4096_mt2, fips4096, fips4096_mt8, fips4096_mt16\n\n");

    fprintf(out, "Add-mod (--add):\n");
    fprintf(out, "  default       512 AMD add: asm_b16, sub: fused_unroll_b16; 512 other: fused;\n");
    fprintf(out, "                4096 AMD: asm_b32;\n");
    fprintf(out, "                4096 other: fused_unroll_b32; else: fused_unroll\n");
    fprintf(out, "                else: fused_unroll (#pragma unroll on MAX_LIMBS)\n");
    fprintf(out, "  fused         loop (all sizes)\n");
    fprintf(out, "  fused_unroll  compile-time unroll (#pragma unroll 16/32/64 by MAX_LIMBS)\n");
    fprintf(out, "  asm_b16       512 AMDGCN asm (512 AMD default add-mod)\n");
    fprintf(out, "  fused_unroll_b16  512: 16-limb C unroll (high gpucurves on Adreno)\n");
    fprintf(out, "  fused_unroll_b32  4096: 4x32-limb C unroll (4096 non-AMD default)\n");
    fprintf(out, "  asm_b32       4096 AMDGCN asm (4096 AMD default)\n\n");

    fprintf(out, "Sub-mod (--sub):\n");
    fprintf(out, "  default       512 AMD add: asm_b16, sub: fused_unroll_b16; 512 other: fused;\n");
    fprintf(out, "                4096 AMD: asm_b32;\n");
    fprintf(out, "                4096 other: fused_unroll_b32; else: fused_unroll\n");
    fprintf(out, "                else: fused_unroll\n");
    fprintf(out, "  fused         loop (all sizes)\n");
    fprintf(out, "  fused_unroll  compile-time unroll\n");
    fprintf(out, "  fused_unroll_b16  512: 16-limb C unroll\n");
    fprintf(out, "  fused_unroll_b32  4096: 4x32-limb C unroll\n");
    fprintf(out, "  asm_b32       4096 AMDGCN asm\n");
}
