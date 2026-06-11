#include "opencl_ecm_mont_path.h"

#include "opencl_ecm_limb24.h"
#include "opencl_ecm_log.h"
#include "opencl_ecm_mont.h"

#include <cstdio>
#include <cstring>
namespace {

bool path_is_auto(const char *path) {
    return path == nullptr || path[0] == '\0' || strcmp(path, "auto") == 0 ||
           strcmp(path, "default") == 0;
}

bool path_requests_i24_u32_blsub(const char *path) {
    return path != nullptr &&
           (strcmp(path, "i24_u32_blsub") == 0 ||
            strcmp(path, "mont_mul_unroll_i24_u32_blsub") == 0 ||
            strcmp(path, "mont_sqr_unroll_i24_u32_blsub") == 0 ||
            strcmp(path, "i24_384_manual") == 0 ||
            strcmp(path, "mont_mul_unroll_i24_384_manual") == 0);
}

bool path_requests_i24_u32_branchy(const char *path) {
    return path != nullptr &&
           (strcmp(path, "i24_u32") == 0 || strcmp(path, "mont_mul_unroll_i24_u32") == 0 ||
            strcmp(path, "mont_sqr_unroll_i24_u32") == 0);
}

bool path_requests_i24_any(const char *path) {
    return path_requests_i24_u32_blsub(path) || path_requests_i24_u32_branchy(path);
}

bool path_requests_unroll512(const char *path) {
    return path != nullptr && (strcmp(path, "unroll_only_512") == 0 ||
                               strcmp(path, "mont_mul_priv_unroll_only_512") == 0);
}

bool path_requests_unroll32(const char *path) {
    return path != nullptr &&
           (strcmp(path, "unroll32") == 0 || strcmp(path, "mont_mul_priv_unroll32") == 0 ||
            strcmp(path, "mont_mul_stage1_unroll32") == 0);
}

bool path_requests_priv_opt(const char *path) {
    return path != nullptr &&
           (strcmp(path, "priv_opt") == 0 || strcmp(path, "mont_mul_priv_opt") == 0 ||
            strcmp(path, "mont_sqr_priv_opt") == 0 ||
            strcmp(path, "mont_mul_stage1_priv_opt") == 0);
}

bool path_requests_unroll384(const char *path) {
    return path != nullptr &&
           (strcmp(path, "unroll_only_384") == 0 ||
            strcmp(path, "mont_mul_priv_unroll_only_384") == 0 ||
            strcmp(path, "mont_sqr_priv_unroll_only_384") == 0);
}

} // namespace

bool opencl_ecm_stage1_should_use_i24(ecm_stage1_mont_mode mode, size_t n_bit_size, int verbose) {
    if (!opencl_ecm_stage1_mont_mode_uses_i24(mode)) {
        return false;
    }
    const uint32_t i24_limbs = ecm_limb24_stage1_limbs(n_bit_size);
    if (i24_limbs <= OPENCL_ECM_MAX_LIMBS) {
        return true;
    }
    ecm_ts_fprintf(stderr,
                   "Warning: i24 path needs %u limbs (N=%zu bits), host buffer limit is %u; "
                   "using 32-bit mont\n",
                   i24_limbs, n_bit_size, OPENCL_ECM_MAX_LIMBS);
    (void)verbose;
    return false;
}

int opencl_ecm_parse_mont4096_path(const char *path) {
    if (path == nullptr || path[0] == '\0' || strcmp(path, "auto") == 0 ||
        strcmp(path, "default") == 0) {
        return 0;
    }
    if (path_requests_i24_any(path) || path_requests_unroll512(path) ||
        path_requests_unroll32(path) || path_requests_unroll384(path) ||
        path_requests_priv_opt(path)) {
        return 0;
    }
    if (strcmp(path, "unroll64_4096") == 0) {
        return ECM_MONT4096_PATH_UNROLL64;
    }
    if (strcmp(path, "unroll64_4096_mt2") == 0) {
        return ECM_MONT4096_PATH_UNROLL64_MT2;
    }
    if (strcmp(path, "fips4096") == 0) {
        return ECM_MONT4096_PATH_FIPS4096;
    }
    if (strcmp(path, "fips4096_mt8") == 0) {
        return ECM_MONT4096_PATH_FIPS4096_MT8;
    }
    if (strcmp(path, "fips4096_mt16") == 0) {
        return ECM_MONT4096_PATH_FIPS4096_MT16;
    }
    return -1;
}

const char *opencl_ecm_mont4096_path_name(int path_id) {
    switch (path_id) {
    case ECM_MONT4096_PATH_UNROLL64:
        return "unroll64_4096";
    case ECM_MONT4096_PATH_UNROLL64_MT2:
        return "unroll64_4096_mt2";
    case ECM_MONT4096_PATH_FIPS4096:
        return "fips4096";
    case ECM_MONT4096_PATH_FIPS4096_MT8:
        return "fips4096_mt8";
    case ECM_MONT4096_PATH_FIPS4096_MT16:
        return "fips4096_mt16";
    default:
        return "unknown";
    }
}

int opencl_ecm_mont4096_coop_wg_size(int path_id) {
    switch (path_id) {
    case ECM_MONT4096_PATH_UNROLL64_MT2:
        return 2;
    case ECM_MONT4096_PATH_FIPS4096_MT8:
        return 8;
    case ECM_MONT4096_PATH_FIPS4096_MT16:
        return 16;
    default:
        return 1;
    }
}

int opencl_ecm_mont4096_coop_scratch_u32(int mul_path, int sqr_path) {
    int scratch = 0;
    if (mul_path == ECM_MONT4096_PATH_UNROLL64_MT2 ||
        sqr_path == ECM_MONT4096_PATH_UNROLL64_MT2) {
        scratch = 389;
    }
    if (mul_path >= ECM_MONT4096_PATH_FIPS4096_MT8 ||
        sqr_path >= ECM_MONT4096_PATH_FIPS4096_MT8) {
        scratch = 897;
    }
    return scratch;
}

bool opencl_ecm_mont4096_needs_fips4096(int mul_path, int sqr_path) {
    return mul_path >= ECM_MONT4096_PATH_FIPS4096 || sqr_path >= ECM_MONT4096_PATH_FIPS4096;
}

void opencl_ecm_mont4096_path_labels(int mul_path, int sqr_path, const char **mul_name,
                                     const char **sqr_name) {
    if (mul_name) {
        *mul_name = opencl_ecm_mont4096_path_name(mul_path);
    }
    if (sqr_name) {
        *sqr_name = opencl_ecm_mont4096_path_name(sqr_path);
    }
}

ecm_stage1_mont_mode opencl_ecm_resolve_stage1_mont_mode(const char *gpu_mul_path,
                                                         const char *gpu_sqr_path,
                                                         size_t n_bit_size) {
    if (path_requests_unroll512(gpu_mul_path) || path_requests_unroll512(gpu_sqr_path)) {
        return ECM_STAGE1_MONT_UNROLL512;
    }
    if (path_requests_unroll32(gpu_mul_path) || path_requests_unroll32(gpu_sqr_path)) {
        return ECM_STAGE1_MONT_UNROLL32;
    }
    if (path_requests_priv_opt(gpu_mul_path) || path_requests_priv_opt(gpu_sqr_path)) {
        return ECM_STAGE1_MONT_PRIV_OPT;
    }
    if (path_requests_unroll384(gpu_mul_path) || path_requests_unroll384(gpu_sqr_path)) {
        if (opencl_ecm_stage1_n_fits_unroll384(n_bit_size)) {
            return ECM_STAGE1_MONT_UNROLL384;
        }
        ecm_ts_fprintf(stderr,
                       "Warning: unroll_only_384 requires N+%zu<%zu bits (N<%zu), got %zu; "
                       "using unroll512\n",
                       ECM_STAGE1_MONT_CARRY_BITS, ECM_STAGE1_UNROLL384_MAX_BITS,
                       ECM_STAGE1_UNROLL384_MAX_BITS - ECM_STAGE1_MONT_CARRY_BITS, n_bit_size);
        return ECM_STAGE1_MONT_UNROLL512;
    }

    const bool mul_blsub = path_requests_i24_u32_blsub(gpu_mul_path);
    const bool sqr_blsub = path_requests_i24_u32_blsub(gpu_sqr_path);
    const bool mul_u32 = path_requests_i24_u32_branchy(gpu_mul_path);
    const bool sqr_u32 = path_requests_i24_u32_branchy(gpu_sqr_path);
    const bool both_auto = path_is_auto(gpu_mul_path) && path_is_auto(gpu_sqr_path);
    const bool auto_blsub = both_auto && n_bit_size < ECM_STAGE1_AUTO_I24_MAX_BITS;
    const bool auto_unroll384 =
        both_auto && n_bit_size >= ECM_STAGE1_AUTO_I24_MAX_BITS &&
        opencl_ecm_stage1_n_fits_unroll384(n_bit_size);

    if (mul_blsub || sqr_blsub || auto_blsub) {
        return ECM_STAGE1_MONT_I24_U32_BLSUB;
    }
    if (auto_unroll384) {
        return ECM_STAGE1_MONT_UNROLL384;
    }
    if (mul_u32 || sqr_u32) {
        return ECM_STAGE1_MONT_I24_U32;
    }
    return ECM_STAGE1_MONT_UNROLL512;
}

bool opencl_ecm_stage1_mont_mode_uses_i24(ecm_stage1_mont_mode mode) {
    return mode == ECM_STAGE1_MONT_I24_U32 || mode == ECM_STAGE1_MONT_I24_U32_BLSUB;
}

const char *opencl_ecm_stage1_mont_mode_name(ecm_stage1_mont_mode mode) {
    switch (mode) {
    case ECM_STAGE1_MONT_I24_U32:
        return "mont_mul_unroll_i24_u32";
    case ECM_STAGE1_MONT_I24_U32_BLSUB:
        return "mont_mul_unroll_i24_u32_blsub";
    case ECM_STAGE1_MONT_UNROLL32:
        return "mont_mul_stage1_unroll32";
    case ECM_STAGE1_MONT_UNROLL384:
        return "mont_mul_priv_unroll_only_384";
    case ECM_STAGE1_MONT_PRIV_OPT:
        return "mont_mul_stage1_priv_opt";
    default:
        return "mont_mul_priv_unroll_only_512";
    }
}

const char *opencl_ecm_stage1_mont_sqr_mode_name(ecm_stage1_mont_mode mode) {
    switch (mode) {
    case ECM_STAGE1_MONT_I24_U32:
        return "mont_sqr_unroll_i24_u32";
    case ECM_STAGE1_MONT_I24_U32_BLSUB:
        return "mont_sqr_unroll_i24_u32_blsub";
    case ECM_STAGE1_MONT_UNROLL32:
        return "mont_sqr_stage1_unroll32";
    case ECM_STAGE1_MONT_UNROLL384:
        return "mont_sqr_priv_unroll_only_384";
    case ECM_STAGE1_MONT_PRIV_OPT:
        return "mont_sqr_stage1_priv_opt";
    default:
        return "mont_sqr_priv_unroll_only_512";
    }
}
