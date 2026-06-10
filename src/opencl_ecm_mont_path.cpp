#include "opencl_ecm_mont_path.h"

#include <cstring>

namespace {

bool path_is_auto(const char *path) {
    return path == nullptr || path[0] == '\0' || strcmp(path, "auto") == 0 ||
           strcmp(path, "default") == 0;
}

bool path_requests_i24_384(const char *path) {
    return path != nullptr &&
           (strcmp(path, "i24_384_manual") == 0 ||
            strcmp(path, "mont_mul_unroll_i24_384_manual") == 0);
}

bool path_requests_unroll512(const char *path) {
    return path != nullptr && (strcmp(path, "unroll_only_512") == 0 ||
                               strcmp(path, "mont_mul_priv_unroll_only_512") == 0);
}

} // namespace

int opencl_ecm_parse_mont4096_path(const char *path) {
    if (path == nullptr || path[0] == '\0' || strcmp(path, "auto") == 0 ||
        strcmp(path, "default") == 0) {
        return 0;
    }
    if (path_requests_i24_384(path) || path_requests_unroll512(path)) {
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
    if (path_requests_i24_384(gpu_mul_path) || path_requests_i24_384(gpu_sqr_path)) {
        return ECM_STAGE1_MONT_I24_384;
    }
    if (path_is_auto(gpu_mul_path) && path_is_auto(gpu_sqr_path) && n_bit_size < 384u) {
        return ECM_STAGE1_MONT_I24_384;
    }
    return ECM_STAGE1_MONT_UNROLL512;
}

const char *opencl_ecm_stage1_mont_mode_name(ecm_stage1_mont_mode mode) {
    switch (mode) {
    case ECM_STAGE1_MONT_I24_384:
        return "mont_mul_priv_i24_u32_blsub";
    default:
        return "mont_mul_priv_unroll_only_512";
    }
}
