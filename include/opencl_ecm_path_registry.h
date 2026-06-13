#pragma once

// ============================================================================
// ECM Stage-1 算子路径注册表（统一公共头）
//
// 本头文件是 Stage-1 GPU 算子（Montgomery mul/sqr 与 add/sub-mod）路径选择的唯一
// 对外接口。它合并了历史上分散在 mont_path / addsub_path 两个头文件的声明。
//
// 架构：去耦合 + 单一数据源
//   - 主内核 ecm_stage1.cl 只调用宏别名 mont_mul/mont_sqr/add_mod/sub_mod。
//   - Host 端用「描述符表 + 解析器」选定算子，通过 ECM_STAGE1_*_IMPL 宏注入绑定。
//   - 新增/删除算子只需改注册表一行 + 一个 .cl 文件。
//
// 开发者指南见 docs/DEV_OPERATOR_PATH_REGISTRY.md。
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "opencl_ecm_mont.h"  // 基础 mont 原语（兼容旧的传递包含）

// ---------------------------------------------------------------------------
// 平台 / 厂商位掩码
// ---------------------------------------------------------------------------

enum EcmPathOs : uint32_t {
    ECM_OS_WINDOWS = 1u << 0,
    ECM_OS_ANDROID = 1u << 1,
    ECM_OS_LINUX = 1u << 2,
    ECM_OS_MACOS = 1u << 3,
};
constexpr uint32_t ECM_OS_ANY = 0xFFFFFFFFu;

enum EcmPathGpuVendor : uint32_t {
    ECM_GPU_AMD = 1u << 0,
    ECM_GPU_NVIDIA = 1u << 1,
    ECM_GPU_INTEL = 1u << 2,
    ECM_GPU_QUALCOMM = 1u << 3,
    ECM_GPU_HUAWEI = 1u << 4,
    ECM_GPU_APPLE = 1u << 5,
};
constexpr uint32_t ECM_GPU_ANY = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------
// 编译期常量（历史上位于 opencl_ecm_mont_path.h）
// ---------------------------------------------------------------------------

constexpr size_t ECM_STAGE1_MONT_CARRY_BITS = 6u;
constexpr size_t ECM_STAGE1_UNROLL384_MAX_BITS = 384u;
constexpr size_t ECM_STAGE1_UNROLL512_CONTAINER_BITS = 512u;

constexpr size_t ECM_PATH_4096_AUTO_MIN_BITS = 3072u;
constexpr size_t ECM_PATH_4096_CONTAINER_BITS = 4096u;

// ---------------------------------------------------------------------------
// 运行期上下文
// ---------------------------------------------------------------------------

struct EcmPathContext {
    size_t n_bit_size;
    uint32_t container_limbs;
    uint32_t os_mask;
    uint32_t gpu_vendor_mask;
};

// ---------------------------------------------------------------------------
// 描述符（Montgomery 比 add/sub 多出 dedicated/coop/scratch 三个 4096 协作字段）
// ---------------------------------------------------------------------------

struct EcmMontPathDescriptor {
    const char *id;
    const char *cl_name;
    const char *const *aliases;
    const char *kernel_path;

    int8_t auto_priority;
    uint16_t min_n_bits;
    uint16_t max_n_bits;
    bool max_n_strict;
    uint16_t max_container_bits;

    uint32_t os_mask;
    uint32_t gpu_vendor_mask;
    uint32_t gpu_vendor_exclude_mask;

    bool dedicated;
    uint8_t coop_work_group_size;
    uint16_t local_scratch_u32;
};

struct EcmAddSubPathDescriptor {
    const char *id;
    const char *cl_name;
    const char *const *aliases;
    const char *kernel_path;

    int8_t auto_priority;
    uint16_t min_n_bits;
    uint16_t max_n_bits;
    bool max_n_strict;
    uint16_t max_container_bits;

    uint32_t os_mask;
    uint32_t gpu_vendor_mask;
    uint32_t gpu_vendor_exclude_mask;
};

// ---------------------------------------------------------------------------
// 旧版整型枚举（Android / CLI 解析器、coop 整型分发使用，保留兼容）
// ---------------------------------------------------------------------------

enum ecm_stage1_mont_mode {
    ECM_STAGE1_MONT_UNROLL512 = 0,
    ECM_STAGE1_MONT_UNROLL32 = 3,
    ECM_STAGE1_MONT_UNROLL384 = 4,
    ECM_STAGE1_MONT_PRIV_OPT = 5,
};

enum {
    ECM_MONT4096_PATH_UNROLL64 = 0,
    ECM_MONT4096_PATH_UNROLL64_MT2 = 1,
    ECM_MONT4096_PATH_FIPS4096 = 2,
    ECM_MONT4096_PATH_FIPS4096_MT8 = 3,
    ECM_MONT4096_PATH_FIPS4096_MT16 = 4,
};

enum {
    ECM_ADDSUB_PATH_FUSED = 0,
    ECM_ADDSUB_PATH_FUSED_UNROLL = 1,
    ECM_ADDSUB_PATH_FUSED_UNROLL_B32 = 2,
    ECM_ADDSUB_PATH_ASM_B32 = 3,
    ECM_ADDSUB_PATH_FUSED_UNROLL_B16 = 4,
    ECM_ADDSUB_PATH_ASM_B16 = 5,
    ECM_ADDSUB_PATH_UNROLL_128B = 6,
    ECM_ADDSUB_PATH_ASM_128B = 7,
    ECM_ADDSUB_PATH_UNROLL_192B = 8,
    ECM_ADDSUB_PATH_ASM_192B = 9,
    ECM_ADDSUB_PATH_UNROLL_256B = 10,
    ECM_ADDSUB_PATH_ASM_256B = 11,
    ECM_ADDSUB_PATH_UNROLL_384B = 12,
    ECM_ADDSUB_PATH_ASM_384B = 13,
};

// ---------------------------------------------------------------------------
// 构建计划
// ---------------------------------------------------------------------------

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

// ===========================================================================
// 通用谓词 / 工具
// ===========================================================================

bool ecm_path_n_bit_fits(uint16_t min_n_bits, uint16_t max_n_bits, bool max_n_strict,
                         size_t n_bit_size);

uint32_t ecm_path_host_os_mask();
uint32_t ecm_path_gpu_vendor_from_cl_vendor_string(const char *vendor_lower);

uint32_t ecm_mont_operator_limbs(const EcmMontPathDescriptor *desc);
bool ecm_mont_path_is_4096_dedicated(const EcmMontPathDescriptor *desc);
bool ecm_mont_path_fits(const EcmMontPathDescriptor *desc, const EcmPathContext &ctx);
bool ecm_addsub_path_fits(const EcmAddSubPathDescriptor *desc, const EcmPathContext &ctx);

bool opencl_ecm_path_is_auto(const char *path);

int ecm_addsub_descriptor_kernel_path(const EcmAddSubPathDescriptor *desc);
int ecm_mont_descriptor_kernel_path(const EcmMontPathDescriptor *desc);

// ===========================================================================
// 源码拼装
// ===========================================================================

std::vector<const char *> opencl_ecm_stage1_kernel_source_paths(const EcmStage1KernelBuildPlan &plan);

std::string opencl_ecm_stage1_assemble_kernel_source(
    const EcmStage1KernelBuildPlan &plan,
    const std::function<std::string(const char *rel_path)> &load_file);

// ===========================================================================
// Montgomery 注册表 / 解析
// ===========================================================================

size_t opencl_ecm_mont_mul_registry_count();
const EcmMontPathDescriptor *opencl_ecm_mont_mul_registry_entry(size_t index);
const EcmMontPathDescriptor *opencl_ecm_mont_mul_descriptor(ecm_stage1_mont_mode mode);

size_t opencl_ecm_mont_sqr_registry_count();
const EcmMontPathDescriptor *opencl_ecm_mont_sqr_registry_entry(size_t index);
const EcmMontPathDescriptor *opencl_ecm_mont_sqr_descriptor(ecm_stage1_mont_mode mode);

const EcmMontPathDescriptor *opencl_ecm_resolve_mont_mul(const char *path, const EcmPathContext &ctx,
                                                         bool *unknown_path);
const EcmMontPathDescriptor *opencl_ecm_resolve_mont_sqr(const char *path, const EcmPathContext &ctx,
                                                         bool *unknown_path);

// ===========================================================================
// Add/Sub-mod 注册表 / 解析
// ===========================================================================

size_t opencl_ecm_addmod_registry_count();
const EcmAddSubPathDescriptor *opencl_ecm_addmod_registry_entry(size_t index);
const EcmAddSubPathDescriptor *opencl_ecm_addmod_descriptor_by_kernel_path(int kernel_path);

size_t opencl_ecm_submod_registry_count();
const EcmAddSubPathDescriptor *opencl_ecm_submod_registry_entry(size_t index);
const EcmAddSubPathDescriptor *opencl_ecm_submod_descriptor_by_kernel_path(int kernel_path);

const EcmAddSubPathDescriptor *opencl_ecm_resolve_addmod_path(const char *path,
                                                              const EcmPathContext &ctx);
const EcmAddSubPathDescriptor *opencl_ecm_resolve_submod_path(const char *path,
                                                              const EcmPathContext &ctx);

// ===========================================================================
// 构建计划组装 / 选项
// ===========================================================================

EcmStage1KernelBuildPlan opencl_ecm_stage1_make_build_plan(
    uint32_t limbs, uint32_t tpi, const EcmMontPathDescriptor *mul,
    const EcmMontPathDescriptor *sqr, const EcmAddSubPathDescriptor *add,
    const EcmAddSubPathDescriptor *sub, int stage1_force_normalize, int add_mod_fused_unroll);

std::string opencl_ecm_stage1_generate_build_options(const EcmStage1KernelBuildPlan &plan);

bool opencl_ecm_stage1_build_plan_equal(const EcmStage1KernelBuildPlan &a,
                                        const EcmStage1KernelBuildPlan &b);

// ===========================================================================
// 兼容封装（历史 mont_path.h / addsub_path.h 接口）
// ===========================================================================

int opencl_ecm_parse_mont4096_path(const char *path, size_t n_bit_size);
const EcmMontPathDescriptor *opencl_ecm_stage1_compatible_mont_fallback(size_t n_bit_size);
const char *opencl_ecm_mont_path_cl_name(const EcmMontPathDescriptor *desc,
                                         const char *fallback_cl_name);
const char *opencl_ecm_mont_mul_cl_name(const EcmMontPathDescriptor *desc);
const char *opencl_ecm_mont_sqr_cl_name(const EcmMontPathDescriptor *desc);
const char *opencl_ecm_stage1_mont_mode_name(ecm_stage1_mont_mode mode);
const char *opencl_ecm_stage1_mont_sqr_mode_name(ecm_stage1_mont_mode mode);
ecm_stage1_mont_mode opencl_ecm_resolve_stage1_mont_mode(const char *gpu_mul_path,
                                                         const char *gpu_sqr_path,
                                                         size_t n_bit_size);

int opencl_ecm_parse_addsub_path(const char *path);
const char *opencl_ecm_addsub_path_name(int path_id);
bool opencl_ecm_addsub_path_needs_asm_b32(int path_id);
bool opencl_ecm_addsub_path_needs_asm_b16(int path_id);
bool opencl_ecm_addsub_path_needs_addsub_bits(int path_id);
const EcmAddSubPathDescriptor *opencl_ecm_resolve_addsub_add_path(const char *path,
                                                                  const EcmPathContext &ctx);
const EcmAddSubPathDescriptor *opencl_ecm_resolve_addsub_sub_path(const char *path,
                                                                  const EcmPathContext &ctx);
void opencl_ecm_print_available_kernels(FILE *out);
// (registry header end)
