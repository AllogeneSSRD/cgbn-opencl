#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "opencl_ecm_mont_path.h"

/** Host OS bitmask (runtime ∩ descriptor must be non-empty when descriptor mask ≠ ANY). */
enum EcmPathOs : uint32_t {
    ECM_OS_WINDOWS = 1u << 0,
    ECM_OS_ANDROID = 1u << 1,
    ECM_OS_LINUX = 1u << 2,
    ECM_OS_MACOS = 1u << 3,
};
constexpr uint32_t ECM_OS_ANY = 0xFFFFFFFFu;

/** OpenCL GPU vendor bitmask. */
enum EcmPathGpuVendor : uint32_t {
    ECM_GPU_AMD = 1u << 0,
    ECM_GPU_NVIDIA = 1u << 1,
    ECM_GPU_INTEL = 1u << 2,
    ECM_GPU_QUALCOMM = 1u << 3,
    ECM_GPU_HUAWEI = 1u << 4,
    ECM_GPU_APPLE = 1u << 5,
};
constexpr uint32_t ECM_GPU_ANY = 0xFFFFFFFFu;

/** Extra OpenCL translation units prepended at kernel build. */
enum EcmKernelInclude : uint32_t {
    ECM_KERNEL_INC_NONE = 0,
    ECM_KERNEL_INC_MONT_EXTENDED = 1u << 0,
    ECM_KERNEL_INC_MP_ASM_U32 = 1u << 1,
    ECM_KERNEL_INC_MP_ASM_U16 = 1u << 2,
    ECM_KERNEL_INC_ADDSUB_BITS = 1u << 3,
};

/** Runtime platform + container context for path selection. */
struct EcmPathContext {
    size_t n_bit_size;
    uint32_t container_limbs;
    uint32_t os_mask;
    uint32_t gpu_vendor_mask;
};

/**
 * Montgomery mul/sqr path (separate mul/sqr registries).
 * dedicated: fixed operator width = max_n_bits/32 (e.g. unroll384 → 12 limbs).
 * !dedicated: compatible; container must hold N+CARRY.
 */
struct EcmMontPathDescriptor {
    const char *id;
    const char *cl_name;
    const char *const *aliases;
    int8_t auto_priority;

    uint16_t min_n_bits;
    uint16_t max_n_bits;
    bool max_n_strict;
    bool dedicated;

    uint8_t coop_work_group_size;
    uint16_t local_scratch_u32;
    uint8_t cl_dispatch_id;
    uint32_t kernel_includes;
    const char *force_macro;
};

struct EcmAddSubPathDescriptor {
    int cl_dispatch_id;
    const char *id;
    const char *cl_name;
    const char *const *aliases;
    int8_t auto_priority;

    uint16_t max_n_bits;
    bool max_n_strict;
    uint16_t max_container_bits;
    uint32_t os_mask;
    uint32_t gpu_vendor_mask;
    uint32_t gpu_vendor_exclude_mask;
    uint32_t kernel_includes;
};

struct EcmStage1KernelBuildPlan {
    uint32_t limbs;
    uint32_t tpi;
    int stage1_force_normalize;
    int add_mod_fused_unroll;
    const EcmMontPathDescriptor *mul;
    const EcmMontPathDescriptor *sqr;
    const EcmAddSubPathDescriptor *add;
    const EcmAddSubPathDescriptor *sub;
};

constexpr size_t ECM_PATH_4096_AUTO_MIN_BITS = 3072u;
constexpr size_t ECM_PATH_4096_CONTAINER_BITS = 4096u;

bool ecm_path_n_bit_fits(uint16_t min_n_bits, uint16_t max_n_bits, bool max_n_strict,
                         size_t n_bit_size);

uint32_t ecm_path_host_os_mask();
uint32_t ecm_path_gpu_vendor_from_cl_vendor_string(const char *vendor_lower);

uint32_t ecm_mont_operator_limbs(const EcmMontPathDescriptor *desc);
bool ecm_mont_path_is_4096_dedicated(const EcmMontPathDescriptor *desc);
bool ecm_mont_path_fits(const EcmMontPathDescriptor *desc, size_t n_bit_size,
                        uint32_t container_limbs);
bool ecm_addsub_path_fits(const EcmAddSubPathDescriptor *desc, const EcmPathContext &ctx);

bool opencl_ecm_path_is_auto(const char *path);

const char *ecm_kernel_include_path(EcmKernelInclude include_bit);
uint32_t opencl_ecm_stage1_collect_kernel_includes(const EcmStage1KernelBuildPlan &plan);

size_t opencl_ecm_mont_mul_registry_count();
const EcmMontPathDescriptor *opencl_ecm_mont_mul_registry_entry(size_t index);
const EcmMontPathDescriptor *opencl_ecm_mont_mul_descriptor(ecm_stage1_mont_mode mode);

size_t opencl_ecm_mont_sqr_registry_count();
const EcmMontPathDescriptor *opencl_ecm_mont_sqr_registry_entry(size_t index);
const EcmMontPathDescriptor *opencl_ecm_mont_sqr_descriptor(ecm_stage1_mont_mode mode);

const EcmMontPathDescriptor *opencl_ecm_resolve_mont_mul(const char *path, size_t n_bit_size,
                                                         uint32_t container_limbs,
                                                         bool *unknown_path);
const EcmMontPathDescriptor *opencl_ecm_resolve_mont_sqr(const char *path, size_t n_bit_size,
                                                         uint32_t container_limbs,
                                                         bool *unknown_path);

size_t opencl_ecm_addmod_registry_count();
const EcmAddSubPathDescriptor *opencl_ecm_addmod_registry_entry(size_t index);
const EcmAddSubPathDescriptor *opencl_ecm_addmod_path_descriptor(int path_id);
const EcmAddSubPathDescriptor *opencl_ecm_resolve_addmod_path(const char *path,
                                                              const EcmPathContext &ctx);

size_t opencl_ecm_submod_registry_count();
const EcmAddSubPathDescriptor *opencl_ecm_submod_registry_entry(size_t index);
const EcmAddSubPathDescriptor *opencl_ecm_submod_path_descriptor(int path_id);
const EcmAddSubPathDescriptor *opencl_ecm_resolve_submod_path(const char *path,
                                                              const EcmPathContext &ctx);

EcmStage1KernelBuildPlan opencl_ecm_stage1_make_build_plan(
    uint32_t limbs, uint32_t tpi, const EcmMontPathDescriptor *mul,
    const EcmMontPathDescriptor *sqr, const EcmAddSubPathDescriptor *add,
    const EcmAddSubPathDescriptor *sub, int stage1_force_normalize, int add_mod_fused_unroll);

std::string opencl_ecm_stage1_generate_build_options(const EcmStage1KernelBuildPlan &plan);

bool opencl_ecm_stage1_build_plan_equal(const EcmStage1KernelBuildPlan &a,
                                        const EcmStage1KernelBuildPlan &b);
