#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "opencl_ecm_mont_path.h"

/** Limb / vendor context for add-mod or sub-mod path selection. */
struct EcmAddSubPathContext {
    uint32_t limbs;
    bool is_amd;
};

enum EcmMontPathKind : uint8_t {
    ECM_MONT_PATH_STAGE1 = 0,
    ECM_MONT_PATH_4096 = 1,
};

enum EcmPathVendorFilter : int8_t {
    ECM_PATH_VENDOR_ANY = -1,
    ECM_PATH_VENDOR_NON_AMD = 0,
    ECM_PATH_VENDOR_AMD = 1,
};

/** Montgomery mul/sqr path entry; mul and sqr use separate registries. */
struct EcmMontPathDescriptor {
    EcmMontPathKind kind;
    int variant_id;
    const char *id;
    const char *display_name;
    const char *cl_name;
    bool dedicated;
    int auto_priority;
    const char *const *aliases;
    uint16_t min_n_bits;
    uint16_t max_n_bits;
    bool max_n_strict;
    uint16_t required_container_limbs;
    int coop_wg_size;
    int coop_scratch_u32;
    bool needs_fips4096_cl;
    const char *stage1_force_macro;
    bool stage1_use_i24;
    bool stage1_i24_blsub;
};

/** Add-mod / sub-mod path entry; add and sub use separate registries. */
struct EcmAddSubPathDescriptor {
    int path_id;
    const char *id;
    const char *display_name;
    const char *cl_name;
    int auto_priority;
    const char *const *aliases;
    uint32_t required_limbs;
    EcmPathVendorFilter vendor;
    bool needs_asm_b32;
    bool needs_asm_b16;
};

/** Resolved stage-1 kernel build plan: resolve() outputs feed this directly. */
struct EcmStage1KernelBuildPlan {
    uint32_t limbs;
    uint32_t tpi;
    int stage1_force_normalize;
    int add_mod_fused_unroll;
    const EcmMontPathDescriptor *mul;
    const EcmMontPathDescriptor *sqr;
    const EcmMontPathDescriptor *mul_4096;
    const EcmMontPathDescriptor *sqr_4096;
    const EcmAddSubPathDescriptor *add;
    const EcmAddSubPathDescriptor *sub;
    bool use_i24;
};

constexpr size_t ECM_PATH_4096_AUTO_MIN_BITS = 3072u;
constexpr size_t ECM_PATH_4096_CONTAINER_BITS = 4096u;

bool ecm_path_n_bit_fits(uint16_t min_n_bits, uint16_t max_n_bits, bool max_n_strict,
                         size_t n_bit_size);

bool ecm_mont_path_n_fits(const EcmMontPathDescriptor *desc, size_t n_bit_size);
bool ecm_mont_path_container_fits(const EcmMontPathDescriptor *desc, uint32_t limbs,
                                  size_t n_bit_size);
bool ecm_addsub_path_fits(const EcmAddSubPathDescriptor *desc, const EcmAddSubPathContext &ctx);

bool opencl_ecm_path_is_auto(const char *path);

int ecm_mont_4096_path_id(const EcmMontPathDescriptor *desc);

size_t opencl_ecm_mont_mul_registry_count();
const EcmMontPathDescriptor *opencl_ecm_mont_mul_registry_entry(size_t index);
const EcmMontPathDescriptor *opencl_ecm_mont_mul_descriptor(ecm_stage1_mont_mode mode);
const EcmMontPathDescriptor *opencl_ecm_mont4096_mul_descriptor(int path_id);

size_t opencl_ecm_mont_sqr_registry_count();
const EcmMontPathDescriptor *opencl_ecm_mont_sqr_registry_entry(size_t index);
const EcmMontPathDescriptor *opencl_ecm_mont_sqr_descriptor(ecm_stage1_mont_mode mode);
const EcmMontPathDescriptor *opencl_ecm_mont4096_sqr_descriptor(int path_id);

const EcmMontPathDescriptor *opencl_ecm_resolve_stage1_mont_mul(const char *path,
                                                              size_t n_bit_size);
const EcmMontPathDescriptor *opencl_ecm_resolve_stage1_mont_sqr(const char *path,
                                                                size_t n_bit_size);

const EcmMontPathDescriptor *opencl_ecm_resolve_mont4096_mul(const char *path, size_t n_bit_size,
                                                             bool *unknown_path);
const EcmMontPathDescriptor *opencl_ecm_resolve_mont4096_sqr(const char *path, size_t n_bit_size,
                                                             bool *unknown_path);

size_t opencl_ecm_addmod_registry_count();
const EcmAddSubPathDescriptor *opencl_ecm_addmod_registry_entry(size_t index);
const EcmAddSubPathDescriptor *opencl_ecm_addmod_path_descriptor(int path_id);
const EcmAddSubPathDescriptor *opencl_ecm_resolve_addmod_path(const char *path, uint32_t limbs,
                                                              bool is_amd);

size_t opencl_ecm_submod_registry_count();
const EcmAddSubPathDescriptor *opencl_ecm_submod_registry_entry(size_t index);
const EcmAddSubPathDescriptor *opencl_ecm_submod_path_descriptor(int path_id);
const EcmAddSubPathDescriptor *opencl_ecm_resolve_submod_path(const char *path, uint32_t limbs,
                                                              bool is_amd);

EcmStage1KernelBuildPlan opencl_ecm_stage1_make_build_plan(
    uint32_t limbs, uint32_t tpi, const EcmMontPathDescriptor *mul,
    const EcmMontPathDescriptor *sqr, const EcmMontPathDescriptor *mul_4096,
    const EcmMontPathDescriptor *sqr_4096, const EcmAddSubPathDescriptor *add,
    const EcmAddSubPathDescriptor *sub, bool use_i24, int stage1_force_normalize,
    int add_mod_fused_unroll);

std::string opencl_ecm_stage1_generate_build_options(const EcmStage1KernelBuildPlan &plan);

bool opencl_ecm_stage1_build_plan_equal(const EcmStage1KernelBuildPlan &a,
                                        const EcmStage1KernelBuildPlan &b);

bool opencl_ecm_stage1_plan_use_i24_blsub(const EcmStage1KernelBuildPlan &plan);
