#include "opencl_ecm_mont_path.h"

#include "opencl_ecm_log.h"
#include "opencl_ecm_mont.h"
#include "opencl_ecm_path_registry.h"

#include <cstring>

const EcmMontPathDescriptor *opencl_ecm_stage1_compatible_mont_fallback(size_t n_bit_size) {
    EcmPathContext ctx{};
    ctx.n_bit_size = n_bit_size;
    return opencl_ecm_resolve_mont_mul(nullptr, ctx, nullptr);
}

const char *opencl_ecm_mont_path_cl_name(const EcmMontPathDescriptor *desc,
                                          const char *fallback_cl_name) {
    if (desc != nullptr && desc->cl_name != nullptr) {
        return desc->cl_name;
    }
    return fallback_cl_name;
}

const char *opencl_ecm_mont_mul_cl_name(const EcmMontPathDescriptor *desc) {
    return opencl_ecm_mont_path_cl_name(desc, "mont_mul_priv_unroll_only_512");
}

const char *opencl_ecm_mont_sqr_cl_name(const EcmMontPathDescriptor *desc) {
    return opencl_ecm_mont_path_cl_name(desc, "mont_sqr_priv_unroll_only_512");
}

int opencl_ecm_parse_mont4096_path(const char *path, size_t n_bit_size) {
    EcmPathContext ctx{};
    ctx.n_bit_size = n_bit_size;
    ctx.container_limbs = static_cast<uint32_t>(ECM_PATH_4096_CONTAINER_BITS / 32u);
    const EcmMontPathDescriptor *desc = opencl_ecm_resolve_mont_mul(path, ctx, nullptr);
    if (desc == nullptr || !ecm_mont_path_is_4096_dedicated(desc)) {
        return ECM_MONT4096_PATH_UNROLL64;
    }
    return ecm_mont_descriptor_kernel_path(desc);
}

static ecm_stage1_mont_mode mont_desc_legacy_mode(const EcmMontPathDescriptor *d) {
    if (d == nullptr || d->id == nullptr) {
        return ECM_STAGE1_MONT_UNROLL512;
    }
    if (strcmp(d->id, "unroll_only_384") == 0) {
        return ECM_STAGE1_MONT_UNROLL384;
    }
    if (strcmp(d->id, "unroll_only_512") == 0) {
        return ECM_STAGE1_MONT_UNROLL512;
    }
    if (strcmp(d->id, "unroll32") == 0) {
        return ECM_STAGE1_MONT_UNROLL32;
    }
    if (strcmp(d->id, "priv_opt") == 0) {
        return ECM_STAGE1_MONT_PRIV_OPT;
    }
    return ECM_STAGE1_MONT_UNROLL512;
}

ecm_stage1_mont_mode opencl_ecm_resolve_stage1_mont_mode(const char *gpu_mul_path,
                                                         const char *gpu_sqr_path,
                                                         size_t n_bit_size) {
    (void)gpu_sqr_path;
    EcmPathContext ctx{};
    ctx.n_bit_size = n_bit_size;
    return mont_desc_legacy_mode(opencl_ecm_resolve_mont_mul(gpu_mul_path, ctx, nullptr));
}

const char *opencl_ecm_stage1_mont_mode_name(ecm_stage1_mont_mode mode) {
    return opencl_ecm_mont_mul_cl_name(opencl_ecm_mont_mul_descriptor(mode));
}

const char *opencl_ecm_stage1_mont_sqr_mode_name(ecm_stage1_mont_mode mode) {
    return opencl_ecm_mont_sqr_cl_name(opencl_ecm_mont_sqr_descriptor(mode));
}
