#pragma once

#include <cstddef>
#include <cstdint>

#include "opencl_ecm_mont.h"

enum {
    ECM_MONT4096_PATH_UNROLL64 = 0,
    ECM_MONT4096_PATH_UNROLL64_MT2 = 1,
    ECM_MONT4096_PATH_FIPS4096 = 2,
    ECM_MONT4096_PATH_FIPS4096_MT8 = 3,
    ECM_MONT4096_PATH_FIPS4096_MT16 = 4,
};

int opencl_ecm_parse_mont4096_path(const char *path);
const char *opencl_ecm_mont4096_path_name(int path_id);
int opencl_ecm_mont4096_coop_wg_size(int path_id);
int opencl_ecm_mont4096_coop_scratch_u32(int mul_path, int sqr_path);
bool opencl_ecm_mont4096_needs_fips4096(int mul_path, int sqr_path);
void opencl_ecm_mont4096_path_labels(int mul_path, int sqr_path, const char **mul_name,
                                     const char **sqr_name);

/** Stage-1 Montgomery mul/sqr (<4096-bit, 512-bit CGBN container). */
enum ecm_stage1_mont_mode {
    ECM_STAGE1_MONT_UNROLL512 = 0,
    /** Same algorithm as mont_mul_unroll_i24_u32_body (private ABI in stage1). */
    ECM_STAGE1_MONT_I24_U32 = 1,
    /** Same algorithm as mont_mul_unroll_i24_u32_blsub_body; default for N < 384 auto. */
    ECM_STAGE1_MONT_I24_U32_BLSUB = 2,
    /** Generic mont_mul_stage1_unroll32 (any limb count). */
    ECM_STAGE1_MONT_UNROLL32 = 3,
    /** mont_mul_priv_unroll_only_384_body — 12 active 32-bit limbs; valid only if N+CARRY≤384. */
    ECM_STAGE1_MONT_UNROLL384 = 4,
    /** mont_mul_stage1_priv_opt — cached B + speculative subtract (generic fallback). */
    ECM_STAGE1_MONT_PRIV_OPT = 5,
};

/** Auto: N below this uses i24_u32_blsub when mul/sqr are both auto (beats unroll384 here). */
constexpr size_t ECM_STAGE1_AUTO_I24_BLSUB_MAX_BITS = 264u;
/** Legacy alias. */
constexpr size_t ECM_STAGE1_AUTO_I24_MAX_BITS = ECM_STAGE1_AUTO_I24_BLSUB_MAX_BITS;
constexpr size_t ECM_STAGE1_MONT_CARRY_BITS = 6u;
/** 12-limb CIOS width (32-bit limbs); Montgomery headroom uses CARRY_BITS. */
constexpr size_t ECM_STAGE1_UNROLL384_MAX_BITS = 384u;
/** 512-bit CGBN container for unroll512 fixed path. */
constexpr size_t ECM_STAGE1_UNROLL512_CONTAINER_BITS = 512u;

inline bool opencl_ecm_stage1_n_fits_unroll384(size_t n_bit_size) {
    return n_bit_size + ECM_STAGE1_MONT_CARRY_BITS < ECM_STAGE1_UNROLL384_MAX_BITS;
}

inline bool opencl_ecm_stage1_n_fits_unroll512_container(size_t n_bit_size) {
    return n_bit_size + ECM_STAGE1_MONT_CARRY_BITS <= ECM_STAGE1_UNROLL512_CONTAINER_BITS;
}

/** i24 stage-1 data fits host/OpenCL buffer (see OPENCL_ECM_MAX_LIMBS). */
bool opencl_ecm_stage1_n_fits_i24_container(size_t n_bit_size);
/** Generic fallback when no fixed-bit mont path applies (prefer i24_u32, else priv_opt). */
ecm_stage1_mont_mode opencl_ecm_stage1_compatible_mont_fallback(size_t n_bit_size);

/** Resolve mul/sqr path strings for stage1. Empty/null = auto. */
ecm_stage1_mont_mode opencl_ecm_resolve_stage1_mont_mode(const char *gpu_mul_path,
                                                         const char *gpu_sqr_path,
                                                         size_t n_bit_size);
bool opencl_ecm_stage1_mont_mode_uses_i24(ecm_stage1_mont_mode mode);
const char *opencl_ecm_stage1_mont_mode_name(ecm_stage1_mont_mode mode);
const char *opencl_ecm_stage1_mont_sqr_mode_name(ecm_stage1_mont_mode mode);
bool opencl_ecm_stage1_should_use_i24(ecm_stage1_mont_mode mode, size_t n_bit_size, int verbose);
