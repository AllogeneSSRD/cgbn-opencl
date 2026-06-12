#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "opencl_ecm_mont_path.h"

/** Bit-width context for Montgomery path n_fits callbacks. */
struct EcmPathContext {
    size_t n_bit_size;
    uint32_t limbs;
    bool is_amd;
};

/** Limb / vendor context for add-mod or sub-mod path selection. */
struct EcmAddSubPathContext {
    uint32_t limbs;
    bool is_amd;
};

enum EcmMontPathKind : uint8_t {
    ECM_MONT_PATH_STAGE1 = 0,
    ECM_MONT_PATH_4096 = 1,
};

/** One Montgomery mul path entry (independent from sqr). */
struct EcmMontMulPathDescriptor {
    EcmMontPathKind kind;
    int variant_id;
    const char *id;
    const char *display_name;
    const char *cl_name;
    bool dedicated;
    int auto_priority;
    const char *const *aliases;
    bool (*n_fits)(const EcmPathContext &ctx);
    int coop_wg_size;
    int coop_scratch_u32;
    bool needs_fips4096_cl;
    /** Stage-1 compile-time selector; nullptr = default limb dispatch (macro stays 0). */
    const char *stage1_force_macro;
    bool stage1_use_i24;
    bool stage1_i24_blsub;
};

/** One Montgomery sqr path entry (independent from mul). */
struct EcmMontSqrPathDescriptor {
    EcmMontPathKind kind;
    int variant_id;
    const char *id;
    const char *display_name;
    const char *cl_name;
    bool dedicated;
    int auto_priority;
    const char *const *aliases;
    bool (*n_fits)(const EcmPathContext &ctx);
    int coop_wg_size;
    int coop_scratch_u32;
    bool needs_fips4096_cl;
    const char *stage1_force_macro;
    bool stage1_use_i24;
    bool stage1_i24_blsub;
};

struct EcmAddModPathDescriptor {
    int path_id;
    const char *id;
    const char *display_name;
    const char *cl_name;
    int auto_priority;
    const char *const *aliases;
    bool (*limbs_fits)(const EcmAddSubPathContext &ctx);
    bool needs_asm_b32;
    bool needs_asm_b16;
};

struct EcmSubModPathDescriptor {
    int path_id;
    const char *id;
    const char *display_name;
    const char *cl_name;
    int auto_priority;
    const char *const *aliases;
    bool (*limbs_fits)(const EcmAddSubPathContext &ctx);
    bool needs_asm_b32;
    bool needs_asm_b16;
};

/** Resolved stage-1 kernel build plan: resolve() outputs feed this directly. */
struct EcmStage1KernelBuildPlan {
    uint32_t limbs;
    uint32_t tpi;
    int stage1_force_normalize;
    int add_mod_fused_unroll;
    const EcmMontMulPathDescriptor *mul;
    const EcmMontSqrPathDescriptor *sqr;
    const EcmMontMulPathDescriptor *mul_4096;
    const EcmMontSqrPathDescriptor *sqr_4096;
    const EcmAddModPathDescriptor *add;
    const EcmSubModPathDescriptor *sub;
    bool use_i24;
};

constexpr size_t ECM_PATH_4096_AUTO_MIN_BITS = 3072u;
constexpr size_t ECM_PATH_4096_CONTAINER_BITS = 4096u;

bool ecm_path_n_fits_unroll384(size_t n_bit_size);
bool ecm_path_n_fits_unroll512_container(size_t n_bit_size);
bool ecm_path_n_fits_4096_dedicated(size_t n_bit_size);
bool ecm_path_n_fits_4096_container(size_t n_bit_size);

bool opencl_ecm_path_is_auto(const char *path);

int ecm_mont_mul_4096_path_id(const EcmMontMulPathDescriptor *desc);
int ecm_mont_sqr_4096_path_id(const EcmMontSqrPathDescriptor *desc);

size_t opencl_ecm_mont_mul_registry_count();
const EcmMontMulPathDescriptor *opencl_ecm_mont_mul_registry_entry(size_t index);
const EcmMontMulPathDescriptor *opencl_ecm_mont_mul_descriptor(ecm_stage1_mont_mode mode);
const EcmMontMulPathDescriptor *opencl_ecm_mont4096_mul_descriptor(int path_id);

size_t opencl_ecm_mont_sqr_registry_count();
const EcmMontSqrPathDescriptor *opencl_ecm_mont_sqr_registry_entry(size_t index);
const EcmMontSqrPathDescriptor *opencl_ecm_mont_sqr_descriptor(ecm_stage1_mont_mode mode);
const EcmMontSqrPathDescriptor *opencl_ecm_mont4096_sqr_descriptor(int path_id);

/** Final stage-1 mul path; descriptor carries stage1_force_macro / i24 flags. */
const EcmMontMulPathDescriptor *opencl_ecm_resolve_stage1_mont_mul(const char *path,
                                                                   size_t n_bit_size);
const EcmMontSqrPathDescriptor *opencl_ecm_resolve_stage1_mont_sqr(const char *path,
                                                                    size_t n_bit_size);

/**
 * 4096-bit mul path when limbs==128. nullptr if path is stage1-only alias or unused.
 * Sets *unknown_path when the string is not a known alias (optional).
 */
const EcmMontMulPathDescriptor *opencl_ecm_resolve_mont4096_mul(const char *path,
                                                                size_t n_bit_size,
                                                                bool *unknown_path);
const EcmMontSqrPathDescriptor *opencl_ecm_resolve_mont4096_sqr(const char *path,
                                                               size_t n_bit_size,
                                                               bool *unknown_path);

int opencl_ecm_mont4096_coop_wg_size(int path_id);
int opencl_ecm_mont4096_coop_scratch_u32(int mul_path, int sqr_path);
bool opencl_ecm_mont4096_needs_fips4096(int mul_path, int sqr_path);
const char *opencl_ecm_mont4096_mul_path_name(int path_id);
const char *opencl_ecm_mont4096_sqr_path_name(int path_id);

size_t opencl_ecm_addmod_registry_count();
const EcmAddModPathDescriptor *opencl_ecm_addmod_registry_entry(size_t index);
const EcmAddModPathDescriptor *opencl_ecm_addmod_path_descriptor(int path_id);
const EcmAddModPathDescriptor *opencl_ecm_resolve_addmod_path(const char *path, uint32_t limbs,
                                                              bool is_amd);

size_t opencl_ecm_submod_registry_count();
const EcmSubModPathDescriptor *opencl_ecm_submod_registry_entry(size_t index);
const EcmSubModPathDescriptor *opencl_ecm_submod_path_descriptor(int path_id);
const EcmSubModPathDescriptor *opencl_ecm_resolve_submod_path(const char *path, uint32_t limbs,
                                                              bool is_amd);

EcmStage1KernelBuildPlan opencl_ecm_stage1_make_build_plan(
    uint32_t limbs, uint32_t tpi, const EcmMontMulPathDescriptor *mul,
    const EcmMontSqrPathDescriptor *sqr, const EcmMontMulPathDescriptor *mul_4096,
    const EcmMontSqrPathDescriptor *sqr_4096, const EcmAddModPathDescriptor *add,
    const EcmSubModPathDescriptor *sub, bool use_i24, int stage1_force_normalize,
    int add_mod_fused_unroll);

std::string opencl_ecm_stage1_generate_build_options(const EcmStage1KernelBuildPlan &plan);

bool opencl_ecm_stage1_build_plan_equal(const EcmStage1KernelBuildPlan &a,
                                        const EcmStage1KernelBuildPlan &b);

bool opencl_ecm_stage1_plan_use_i24_blsub(const EcmStage1KernelBuildPlan &plan);
