#include "opencl_ecm_mont_path.h"

#include "opencl_ecm_limb24.h"
#include "opencl_ecm_log.h"
#include "opencl_ecm_mont.h"
#include "opencl_ecm_path_registry.h"

#include <cstdio>

bool opencl_ecm_stage1_n_fits_unroll384(size_t n_bit_size) {
    return ecm_path_n_fits_unroll384(n_bit_size);
}

bool opencl_ecm_stage1_n_fits_unroll512_container(size_t n_bit_size) {
    return ecm_path_n_fits_unroll512_container(n_bit_size);
}

bool opencl_ecm_stage1_n_fits_i24_container(size_t n_bit_size) {
    return ecm_limb24_stage1_limbs(n_bit_size) <= OPENCL_ECM_MAX_LIMBS;
}

const EcmMontMulPathDescriptor *opencl_ecm_stage1_compatible_mont_fallback(size_t n_bit_size) {
    return opencl_ecm_resolve_stage1_mont_mul(nullptr, n_bit_size);
}

bool opencl_ecm_stage1_mont_mode_uses_i24(ecm_stage1_mont_mode mode) {
    return mode == ECM_STAGE1_MONT_I24_U32 || mode == ECM_STAGE1_MONT_I24_U32_BLSUB;
}

const char *opencl_ecm_mont_mul_cl_name(const EcmMontMulPathDescriptor *desc) {
    if (desc != nullptr && desc->cl_name != nullptr) {
        return desc->cl_name;
    }
    return "mont_mul_priv_unroll_only_512";
}

const char *opencl_ecm_mont_sqr_cl_name(const EcmMontSqrPathDescriptor *desc) {
    if (desc != nullptr && desc->cl_name != nullptr) {
        return desc->cl_name;
    }
    return "mont_sqr_priv_unroll_only_512";
}

bool opencl_ecm_stage1_should_use_i24(const EcmMontMulPathDescriptor *mul,
                                      const EcmMontSqrPathDescriptor *sqr, size_t n_bit_size,
                                      int verbose) {
    const bool mul_i24 = mul != nullptr && mul->stage1_use_i24;
    const bool sqr_i24 = sqr != nullptr && sqr->stage1_use_i24;
    if (!mul_i24 && !sqr_i24) {
        return false;
    }
    const uint32_t i24_limbs = ecm_limb24_stage1_limbs(n_bit_size);
    if (i24_limbs <= OPENCL_ECM_MAX_LIMBS) {
        return true;
    }
    ecm_ts_fprintf(stderr,
                   "Warning: i24 path needs %u limbs (N=%zu bits), host buffer limit is %u; "
                   "using 32-bit mont (%s)\n",
                   i24_limbs, n_bit_size, OPENCL_ECM_MAX_LIMBS,
                   opencl_ecm_mont_mul_cl_name(opencl_ecm_stage1_compatible_mont_fallback(
                       n_bit_size)));
    (void)verbose;
    return false;
}

int opencl_ecm_parse_mont4096_path(const char *path, size_t n_bit_size) {
    return ecm_mont_mul_4096_path_id(opencl_ecm_resolve_mont4096_mul(path, n_bit_size, nullptr));
}

const char *opencl_ecm_mont4096_path_name(int path_id) {
    return opencl_ecm_mont4096_mul_path_name(path_id);
}

void opencl_ecm_mont4096_path_labels(int mul_path, int sqr_path, const char **mul_name,
                                     const char **sqr_name) {
    if (mul_name) {
        *mul_name = opencl_ecm_mont4096_mul_path_name(mul_path);
    }
    if (sqr_name) {
        *sqr_name = opencl_ecm_mont4096_sqr_path_name(sqr_path);
    }
}

ecm_stage1_mont_mode opencl_ecm_resolve_stage1_mont_mode(const char *gpu_mul_path,
                                                         const char *gpu_sqr_path,
                                                         size_t n_bit_size) {
    (void)gpu_sqr_path;
    const EcmMontMulPathDescriptor *d =
        opencl_ecm_resolve_stage1_mont_mul(gpu_mul_path, n_bit_size);
    return d != nullptr ? static_cast<ecm_stage1_mont_mode>(d->variant_id)
                        : ECM_STAGE1_MONT_UNROLL512;
}

const char *opencl_ecm_stage1_mont_mode_name(ecm_stage1_mont_mode mode) {
    return opencl_ecm_mont_mul_cl_name(opencl_ecm_mont_mul_descriptor(mode));
}

const char *opencl_ecm_stage1_mont_sqr_mode_name(ecm_stage1_mont_mode mode) {
    return opencl_ecm_mont_sqr_cl_name(opencl_ecm_mont_sqr_descriptor(mode));
}
