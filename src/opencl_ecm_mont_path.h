#pragma once

#include <cstddef>
#include <cstdint>

#include "opencl_ecm_mont.h"

struct EcmMontPathDescriptor;

/** Stage-1 Montgomery mul/sqr (<4096-bit, 512-bit CGBN container). */
enum ecm_stage1_mont_mode {
    ECM_STAGE1_MONT_UNROLL512 = 0,
    ECM_STAGE1_MONT_I24_U32 = 1,
    ECM_STAGE1_MONT_I24_U32_BLSUB = 2,
    ECM_STAGE1_MONT_UNROLL32 = 3,
    ECM_STAGE1_MONT_UNROLL384 = 4,
    ECM_STAGE1_MONT_PRIV_OPT = 5,
};

constexpr size_t ECM_STAGE1_AUTO_I24_BLSUB_MAX_BITS = 264u;
constexpr size_t ECM_STAGE1_AUTO_I24_MAX_BITS = ECM_STAGE1_AUTO_I24_BLSUB_MAX_BITS;
constexpr size_t ECM_STAGE1_MONT_CARRY_BITS = 6u;
constexpr size_t ECM_STAGE1_UNROLL384_MAX_BITS = 384u;
constexpr size_t ECM_STAGE1_UNROLL512_CONTAINER_BITS = 512u;

enum {
    ECM_MONT4096_PATH_UNROLL64 = 0,
    ECM_MONT4096_PATH_UNROLL64_MT2 = 1,
    ECM_MONT4096_PATH_FIPS4096 = 2,
    ECM_MONT4096_PATH_FIPS4096_MT8 = 3,
    ECM_MONT4096_PATH_FIPS4096_MT16 = 4,
};

/** Legacy int path_id for Android / CLI parsers. */
int opencl_ecm_parse_mont4096_path(const char *path, size_t n_bit_size);

const EcmMontPathDescriptor *opencl_ecm_stage1_compatible_mont_fallback(size_t n_bit_size);

const char *opencl_ecm_mont_path_cl_name(const EcmMontPathDescriptor *desc,
                                           const char *fallback_cl_name);
const char *opencl_ecm_mont_mul_cl_name(const EcmMontPathDescriptor *desc);
const char *opencl_ecm_mont_sqr_cl_name(const EcmMontPathDescriptor *desc);

bool opencl_ecm_stage1_should_use_i24(const EcmMontPathDescriptor *mul,
                                      const EcmMontPathDescriptor *sqr, size_t n_bit_size,
                                      int verbose);

bool opencl_ecm_stage1_n_fits_i24_container(size_t n_bit_size);

/** Legacy enum helpers (prefer descriptor resolve in opencl_ecm_path_registry.h). */
bool opencl_ecm_stage1_mont_mode_uses_i24(ecm_stage1_mont_mode mode);
const char *opencl_ecm_stage1_mont_mode_name(ecm_stage1_mont_mode mode);
const char *opencl_ecm_stage1_mont_sqr_mode_name(ecm_stage1_mont_mode mode);
ecm_stage1_mont_mode opencl_ecm_resolve_stage1_mont_mode(const char *gpu_mul_path,
                                                         const char *gpu_sqr_path,
                                                         size_t n_bit_size);
