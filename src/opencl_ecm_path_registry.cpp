#include "opencl_ecm_path_registry.h"

#include "opencl_ecm_addsub_path.h"
#include "opencl_ecm_log.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

bool ecm_mont_path_is_4096_dedicated(const EcmMontPathDescriptor *desc);

namespace {

bool alias_matches(const char *path, const char *alias) {
    return path != nullptr && alias != nullptr && strcmp(path, alias) == 0;
}

bool aliases_contain(const char *const *aliases, const char *path) {
    if (aliases == nullptr || path == nullptr) {
        return false;
    }
    for (const char *const *p = aliases; *p != nullptr; ++p) {
        if (alias_matches(path, *p)) {
            return true;
        }
    }
    return false;
}

constexpr uint16_t kMontNoMinN = 0;
constexpr uint16_t kMontNoMaxN = 0;
constexpr uint16_t kMontUnroll384MaxN = 384;
constexpr uint16_t kMontUnroll512MaxN = 512;
constexpr uint16_t kMont4096MinN = static_cast<uint16_t>(ECM_PATH_4096_AUTO_MIN_BITS);
constexpr uint16_t kMont4096MaxN = static_cast<uint16_t>(ECM_PATH_4096_CONTAINER_BITS);
constexpr uint32_t kContainer4096Limbs = 128u;

static const char *const kMulAliases_unroll384[] = {"unroll_only_384", "mont_mul_priv_unroll_only_384",
                                                    nullptr};
static const char *const kMulAliases_unroll512[] = {"unroll_only_512",
                                                    "mont_mul_priv_unroll_only_512", nullptr};
static const char *const kMulAliases_unroll64_4096[] = {"unroll64_4096", nullptr};
static const char *const kMulAliases_unroll64_4096_mt2[] = {"unroll64_4096_mt2", nullptr};
static const char *const kMulAliases_fips4096[] = {"fips4096", nullptr};
static const char *const kMulAliases_fips4096_mt8[] = {"fips4096_mt8", nullptr};
static const char *const kMulAliases_fips4096_mt16[] = {"fips4096_mt16", nullptr};
static const char *const kMulAliases_unroll32[] = {"unroll32", "mont_mul_priv_unroll32",
                                                   "mont_mul_stage1_unroll32", nullptr};
static const char *const kMulAliases_priv_opt[] = {"priv_opt", "mont_mul_priv_opt",
                                                   "mont_mul_stage1_priv_opt", nullptr};

constexpr EcmMontPathDescriptor kMontMulRegistry[] = {
    {"unroll_only_384", "mont_mul_unroll_384b", kMulAliases_unroll384,
     "mont_mul/mont_mul_unroll_384b.cl", 10, kMontNoMinN, kMontUnroll384MaxN, true, 0, ECM_OS_ANY,
     ECM_GPU_ANY, 0, true, 1, 0, nullptr},
    {"unroll_only_512", "mont_mul_unroll_512b", kMulAliases_unroll512,
     "mont_mul/mont_mul_unroll_512b.cl", 20, kMontNoMinN, kMontUnroll512MaxN, false, 0, ECM_OS_ANY,
     ECM_GPU_ANY, 0, true, 1, 0, nullptr},
    {"unroll64_4096", "mont_mul_unroll_4096b", kMulAliases_unroll64_4096,
     "mont_mul/mont_mul_unroll_4096b.cl", 21, kMont4096MinN, kMont4096MaxN, false,
     static_cast<uint16_t>(ECM_PATH_4096_CONTAINER_BITS), ECM_OS_ANY, ECM_GPU_ANY, 0, true, 1, 0,
     nullptr},
    {"unroll64_4096_mt2", "mont_mul_unroll_4096b_mt2", kMulAliases_unroll64_4096_mt2,
     "mont_mul/mont_mul_unroll_4096b_mt2.cl", 22, kMont4096MinN, kMont4096MaxN, false,
     static_cast<uint16_t>(ECM_PATH_4096_CONTAINER_BITS), ECM_OS_ANY, ECM_GPU_ANY, 0, true, 2, 389,
     nullptr},
    {"fips4096", "mont_mul_fips_4096b", kMulAliases_fips4096,
     "mont_mul/mont_mul_fips_4096b.cl", 23, kMont4096MinN, kMont4096MaxN, false,
     static_cast<uint16_t>(ECM_PATH_4096_CONTAINER_BITS), ECM_OS_ANY, ECM_GPU_ANY, 0, true, 1, 0,
     nullptr},
    {"fips4096_mt8", "mont_mul_fips_4096b", kMulAliases_fips4096_mt8,
     "mont_mul/mont_mul_fips_4096b.cl", 24, kMont4096MinN, kMont4096MaxN, false,
     static_cast<uint16_t>(ECM_PATH_4096_CONTAINER_BITS), ECM_OS_ANY, ECM_GPU_ANY, 0, true, 8,
     897, nullptr},
    {"fips4096_mt16", "mont_mul_fips_4096b", kMulAliases_fips4096_mt16,
     "mont_mul/mont_mul_fips_4096b.cl", 25, kMont4096MinN, kMont4096MaxN, false,
     static_cast<uint16_t>(ECM_PATH_4096_CONTAINER_BITS), ECM_OS_ANY, ECM_GPU_ANY, 0, true, 16,
     897, nullptr},
    {"unroll32", "mont_mul_unroll_32", kMulAliases_unroll32, "mont_mul/mont_mul_unroll_32.cl", -1,
     kMontNoMinN, kMontNoMaxN, false, 0, ECM_OS_ANY, ECM_GPU_ANY, 0, false, 1, 0, nullptr},
    {"priv_opt", "mont_mul_priv_opt", kMulAliases_priv_opt, "mont_mul/mont_mul_priv_opt.cl", 30,
     kMontNoMinN, kMontNoMaxN, false, 0, ECM_OS_ANY, ECM_GPU_ANY, 0, false, 1, 0, nullptr},
};

static const char *const kSqrAliases_unroll384[] = {"unroll_only_384", "mont_sqr_priv_unroll_only_384",
                                                    nullptr};
static const char *const kSqrAliases_unroll512[] = {"unroll_only_512",
                                                    "mont_sqr_priv_unroll_only_512", nullptr};
static const char *const kSqrAliases_unroll64_4096[] = {"unroll64_4096", nullptr};
static const char *const kSqrAliases_unroll64_4096_mt2[] = {"unroll64_4096_mt2", nullptr};
static const char *const kSqrAliases_fips4096[] = {"fips4096", nullptr};
static const char *const kSqrAliases_fips4096_mt8[] = {"fips4096_mt8", nullptr};
static const char *const kSqrAliases_fips4096_mt16[] = {"fips4096_mt16", nullptr};
static const char *const kSqrAliases_unroll32[] = {"unroll32", "mont_sqr_priv_unroll32",
                                                   "mont_sqr_stage1_unroll32", nullptr};
static const char *const kSqrAliases_priv_opt[] = {"priv_opt", "mont_sqr_priv_opt",
                                                   "mont_sqr_stage1_priv_opt", nullptr};

constexpr EcmMontPathDescriptor kMontSqrRegistry[] = {
    {"unroll_only_384", "mont_sqr_unroll_384b", kSqrAliases_unroll384,
     "mont_mul/mont_mul_unroll_384b.cl", 10, kMontNoMinN, kMontUnroll384MaxN, true, 0, ECM_OS_ANY,
     ECM_GPU_ANY, 0, true, 1, 0, nullptr},
    {"unroll_only_512", "mont_sqr_unroll_512b", kSqrAliases_unroll512,
     "mont_mul/mont_mul_unroll_512b.cl", 20, kMontNoMinN, kMontUnroll512MaxN, false, 0, ECM_OS_ANY,
     ECM_GPU_ANY, 0, true, 1, 0, nullptr},
    {"unroll64_4096", "mont_sqr_unroll_4096b", kSqrAliases_unroll64_4096,
     "mont_mul/mont_mul_unroll_4096b.cl", 21, kMont4096MinN, kMont4096MaxN, false,
     static_cast<uint16_t>(ECM_PATH_4096_CONTAINER_BITS), ECM_OS_ANY, ECM_GPU_ANY, 0, true, 1, 0,
     nullptr},
    {"unroll64_4096_mt2", "mont_sqr_unroll_4096b_mt2", kSqrAliases_unroll64_4096_mt2,
     "mont_mul/mont_mul_unroll_4096b_mt2.cl", 22, kMont4096MinN, kMont4096MaxN, false,
     static_cast<uint16_t>(ECM_PATH_4096_CONTAINER_BITS), ECM_OS_ANY, ECM_GPU_ANY, 0, true, 2, 389,
     nullptr},
    {"fips4096", "mont_sqr_fips_4096b", kSqrAliases_fips4096,
     "mont_mul/mont_mul_fips_4096b.cl", 23, kMont4096MinN, kMont4096MaxN, false,
     static_cast<uint16_t>(ECM_PATH_4096_CONTAINER_BITS), ECM_OS_ANY, ECM_GPU_ANY, 0, true, 1, 0,
     nullptr},
    {"fips4096_mt8", "mont_sqr_fips_4096b", kSqrAliases_fips4096_mt8,
     "mont_mul/mont_mul_fips_4096b.cl", 24, kMont4096MinN, kMont4096MaxN, false,
     static_cast<uint16_t>(ECM_PATH_4096_CONTAINER_BITS), ECM_OS_ANY, ECM_GPU_ANY, 0, true, 8, 897,
     nullptr},
    {"fips4096_mt16", "mont_sqr_fips_4096b", kSqrAliases_fips4096_mt16,
     "mont_mul/mont_mul_fips_4096b.cl", 25, kMont4096MinN, kMont4096MaxN, false,
     static_cast<uint16_t>(ECM_PATH_4096_CONTAINER_BITS), ECM_OS_ANY, ECM_GPU_ANY, 0, true, 16,
     897, nullptr},
    {"unroll32", "mont_sqr_unroll_32", kSqrAliases_unroll32, "mont_mul/mont_mul_unroll_32.cl", -1,
     kMontNoMinN, kMontNoMaxN, false, 0, ECM_OS_ANY, ECM_GPU_ANY, 0, false, 1, 0, nullptr},
    {"priv_opt", "mont_sqr_priv_opt", kSqrAliases_priv_opt, "mont_mul/mont_mul_priv_opt.cl", 30,
     kMontNoMinN, kMontNoMaxN, false, 0, ECM_OS_ANY, ECM_GPU_ANY, 0, false, 1, 0, nullptr},
};

static const char *const kAddAliases_asm_4096b[] = {"asm_4096b", "asm_b32", nullptr};
static const char *const kAddAliases_unroll_4096b[] = {"unroll_4096b", "fused_unroll_b32", nullptr};
static const char *const kAddAliases_asm_128b[] = {"asm_128b", nullptr};
static const char *const kAddAliases_unroll_128b[] = {"unroll_128b", nullptr};
static const char *const kAddAliases_asm_192b[] = {"asm_192b", nullptr};
static const char *const kAddAliases_unroll_192b[] = {"unroll_192b", nullptr};
static const char *const kAddAliases_asm_256b[] = {"asm_256b", nullptr};
static const char *const kAddAliases_unroll_256b[] = {"unroll_256b", nullptr};
static const char *const kAddAliases_asm_384b[] = {"asm_384b", nullptr};
static const char *const kAddAliases_unroll_384b[] = {"unroll_384b", nullptr};
static const char *const kAddAliases_asm_512b[] = {"asm_512b", "asm_b16", "fused_asm_b16", nullptr};
static const char *const kAddAliases_unroll_512b[] = {"unroll_512b", "fused_unroll_b16",
                                                      "fused_unroll_auto", nullptr};
static const char *const kAddAliases_fused[] = {"fused", nullptr};
static const char *const kAddAliases_fused_unroll[] = {"fused_unroll", nullptr};

static const char *const kSubAliases_asm_4096b[] = {"asm_4096b", "asm_b32", nullptr};
static const char *const kSubAliases_unroll_4096b[] = {"unroll_4096b", "fused_unroll_b32", nullptr};
static const char *const kSubAliases_asm_128b[] = {"asm_128b", nullptr};
static const char *const kSubAliases_unroll_128b[] = {"unroll_128b", nullptr};
static const char *const kSubAliases_asm_192b[] = {"asm_192b", nullptr};
static const char *const kSubAliases_unroll_192b[] = {"unroll_192b", nullptr};
static const char *const kSubAliases_asm_256b[] = {"asm_256b", nullptr};
static const char *const kSubAliases_unroll_256b[] = {"unroll_256b", nullptr};
static const char *const kSubAliases_asm_384b[] = {"asm_384b", nullptr};
static const char *const kSubAliases_unroll_384b[] = {"unroll_384b", nullptr};
static const char *const kSubAliases_asm_512b[] = {"asm_512b", "asm_b16", nullptr};
static const char *const kSubAliases_unroll_512b[] = {"unroll_512b", "fused_unroll_b16",
                                                        "fused_unroll_auto", nullptr};
static const char *const kSubAliases_fused[] = {"fused", nullptr};
static const char *const kSubAliases_fused_unroll[] = {"fused_unroll", nullptr};

constexpr uint16_t kAddSubNoMinN = 0;
constexpr uint16_t kAddSubNoMaxN = 0;
constexpr uint16_t kAddSub512Container = 512;
constexpr uint16_t kAddSub384MaxN = 378;

constexpr EcmAddSubPathDescriptor kAddModRegistry[] = {
    {"asm_4096b", "add_mod_asm_4096b", kAddAliases_asm_4096b, "add_mod/add_mod_asm_4096b.cl", 10,
     kAddSubNoMinN, kAddSubNoMaxN, false, static_cast<uint16_t>(ECM_PATH_4096_CONTAINER_BITS),
     ECM_OS_ANY, ECM_GPU_AMD, 0},
    {"unroll_4096b", "add_mod_unroll_4096b", kAddAliases_unroll_4096b,
     "add_mod/add_mod_unroll_4096b.cl", 11, kAddSubNoMinN, kAddSubNoMaxN, false,
     static_cast<uint16_t>(ECM_PATH_4096_CONTAINER_BITS), ECM_OS_ANY, 0, ECM_GPU_AMD},
    {"asm_128b", "add_mod_asm_128b", kAddAliases_asm_128b, "add_mod/add_mod_asm_128b.cl", 20,
     kAddSubNoMinN, 128, false, kAddSub512Container, ECM_OS_ANY, ECM_GPU_AMD, 0},
    {"unroll_128b", "add_mod_unroll_128b", kAddAliases_unroll_128b, "add_mod/add_mod_unroll_128b.cl",
     21, kAddSubNoMinN, 128, false, kAddSub512Container, ECM_OS_ANY, ECM_GPU_ANY, 0},
    {"asm_192b", "add_mod_asm_192b", kAddAliases_asm_192b, "add_mod/add_mod_asm_192b.cl", 22,
     kAddSubNoMinN, 192, false, kAddSub512Container, ECM_OS_ANY, ECM_GPU_AMD, 0},
    {"unroll_192b", "add_mod_unroll_192b", kAddAliases_unroll_192b, "add_mod/add_mod_unroll_192b.cl",
     23, kAddSubNoMinN, 192, false, kAddSub512Container, ECM_OS_ANY, ECM_GPU_ANY, 0},
    {"asm_256b", "add_mod_asm_256b", kAddAliases_asm_256b, "add_mod/add_mod_asm_256b.cl", 24,
     kAddSubNoMinN, 256, false, kAddSub512Container, ECM_OS_ANY, ECM_GPU_AMD, 0},
    {"unroll_256b", "add_mod_unroll_256b", kAddAliases_unroll_256b, "add_mod/add_mod_unroll_256b.cl",
     25, kAddSubNoMinN, 256, false, kAddSub512Container, ECM_OS_ANY, ECM_GPU_ANY, 0},
    {"asm_384b", "add_mod_asm_384b", kAddAliases_asm_384b, "add_mod/add_mod_asm_384b.cl", 26,
     kAddSubNoMinN, kAddSub384MaxN, false, kAddSub512Container, ECM_OS_ANY, ECM_GPU_AMD, 0},
    {"unroll_384b", "add_mod_unroll_384b", kAddAliases_unroll_384b, "add_mod/add_mod_unroll_384b.cl",
     27, kAddSubNoMinN, kAddSub384MaxN, false, kAddSub512Container, ECM_OS_ANY, ECM_GPU_ANY, 0},
    {"asm_512b", "add_mod_asm_512b", kAddAliases_asm_512b, "add_mod/add_mod_asm_512b.cl", 30,
     kAddSubNoMinN, kAddSubNoMaxN, false, kAddSub512Container, ECM_OS_ANY, ECM_GPU_AMD, 0},
    {"unroll_512b", "add_mod_unroll_512b", kAddAliases_unroll_512b, "add_mod/add_mod_unroll_512b.cl",
     31, kAddSubNoMinN, kAddSubNoMaxN, false, kAddSub512Container, ECM_OS_ANY, ECM_GPU_AMD, 0},
    {"fused", "add_mod_fused", kAddAliases_fused, "add_mod/add_mod_fused.cl", 32, kAddSubNoMinN,
     kAddSubNoMaxN, false, kAddSub512Container, ECM_OS_ANY, 0, ECM_GPU_AMD},
    {"fused_unroll", "add_mod_fused_unroll", kAddAliases_fused_unroll, "add_mod/add_mod_fused_unroll.cl",
     40, kAddSubNoMinN, kAddSubNoMaxN, false, 0, ECM_OS_ANY, ECM_GPU_ANY, 0},
};

constexpr EcmAddSubPathDescriptor kSubModRegistry[] = {
    {"asm_4096b", "sub_mod_asm_4096b", kSubAliases_asm_4096b, "sub_mod/sub_mod_asm_4096b.cl", 10,
     kAddSubNoMinN, kAddSubNoMaxN, false, static_cast<uint16_t>(ECM_PATH_4096_CONTAINER_BITS),
     ECM_OS_ANY, ECM_GPU_AMD, 0},
    {"unroll_4096b", "sub_mod_unroll_4096b", kSubAliases_unroll_4096b,
     "sub_mod/sub_mod_unroll_4096b.cl", 11, kAddSubNoMinN, kAddSubNoMaxN, false,
     static_cast<uint16_t>(ECM_PATH_4096_CONTAINER_BITS), ECM_OS_ANY, 0, ECM_GPU_AMD},
    {"asm_128b", "sub_mod_asm_128b", kSubAliases_asm_128b, "sub_mod/sub_mod_asm_128b.cl", 20,
     kAddSubNoMinN, 128, false, kAddSub512Container, ECM_OS_ANY, ECM_GPU_AMD, 0},
    {"unroll_128b", "sub_mod_unroll_128b", kSubAliases_unroll_128b, "sub_mod/sub_mod_unroll_128b.cl",
     21, kAddSubNoMinN, 128, false, kAddSub512Container, ECM_OS_ANY, ECM_GPU_ANY, 0},
    {"asm_192b", "sub_mod_asm_192b", kSubAliases_asm_192b, "sub_mod/sub_mod_asm_192b.cl", 22,
     kAddSubNoMinN, 192, false, kAddSub512Container, ECM_OS_ANY, ECM_GPU_AMD, 0},
    {"unroll_192b", "sub_mod_unroll_192b", kSubAliases_unroll_192b, "sub_mod/sub_mod_unroll_192b.cl",
     23, kAddSubNoMinN, 192, false, kAddSub512Container, ECM_OS_ANY, ECM_GPU_ANY, 0},
    {"asm_256b", "sub_mod_asm_256b", kSubAliases_asm_256b, "sub_mod/sub_mod_asm_256b.cl", 24,
     kAddSubNoMinN, 256, false, kAddSub512Container, ECM_OS_ANY, ECM_GPU_AMD, 0},
    {"unroll_256b", "sub_mod_unroll_256b", kSubAliases_unroll_256b, "sub_mod/sub_mod_unroll_256b.cl",
     25, kAddSubNoMinN, 256, false, kAddSub512Container, ECM_OS_ANY, ECM_GPU_ANY, 0},
    {"asm_384b", "sub_mod_asm_384b", kSubAliases_asm_384b, "sub_mod/sub_mod_asm_384b.cl", 26,
     kAddSubNoMinN, kAddSub384MaxN, false, kAddSub512Container, ECM_OS_ANY, ECM_GPU_AMD, 0},
    {"unroll_384b", "sub_mod_unroll_384b", kSubAliases_unroll_384b, "sub_mod/sub_mod_unroll_384b.cl",
     27, kAddSubNoMinN, kAddSub384MaxN, false, kAddSub512Container, ECM_OS_ANY, ECM_GPU_ANY, 0},
    {"asm_512b", "sub_mod_asm_512b", kSubAliases_asm_512b, "sub_mod/sub_mod_asm_512b.cl", 30,
     kAddSubNoMinN, kAddSubNoMaxN, false, kAddSub512Container, ECM_OS_ANY, ECM_GPU_AMD, 0},
    {"unroll_512b", "sub_mod_unroll_512b", kSubAliases_unroll_512b, "sub_mod/sub_mod_unroll_512b.cl",
     31, kAddSubNoMinN, kAddSubNoMaxN, false, kAddSub512Container, ECM_OS_ANY, ECM_GPU_AMD, 0},
    {"fused", "sub_mod_fused", kSubAliases_fused, "sub_mod/sub_mod_fused.cl", 32, kAddSubNoMinN,
     kAddSubNoMaxN, false, kAddSub512Container, ECM_OS_ANY, 0, ECM_GPU_AMD},
    {"fused_unroll", "sub_mod_fused_unroll", kSubAliases_fused_unroll, "sub_mod/sub_mod_fused_unroll.cl",
     40, kAddSubNoMinN, kAddSubNoMaxN, false, 0, ECM_OS_ANY, ECM_GPU_ANY, 0},
};

bool ecm_path_mask_fits(uint32_t required_mask, uint32_t exclude_mask, uint32_t runtime_mask) {
    if (required_mask != 0u && required_mask != ECM_OS_ANY && required_mask != ECM_GPU_ANY) {
        if ((runtime_mask & required_mask) == 0u) {
            return false;
        }
    }
    if (exclude_mask != 0u && (runtime_mask & exclude_mask) != 0u) {
        return false;
    }
    return true;
}

static constexpr const char *kEcmStage1CommonConfig = "common/stage1_config.h.cl";
static constexpr const char *kEcmStage1CommonMpPriv = "common/mp_priv.h.cl";
static constexpr const char *kEcmStage1LadderHelpers = "common/ladder_helpers.cl";
static constexpr const char *kEcmStage1AsmCommon = "common/asm_common.inc.cl";
static constexpr const char *kEcmStage1OperatorIface = "common/operator_iface.h.cl";
static constexpr const char *kEcmStage1Coop = "ecm_stage1_coop.cl";
static constexpr const char *kEcmStage1Entry = "ecm_stage1.cl";
static constexpr const char *kMontFips4096Kernel = "mont_mul/mont_mul_fips_4096b.cl";

bool mont_kernel_path_needs_fips4096(const char *kernel_path) {
    return kernel_path != nullptr && strcmp(kernel_path, kMontFips4096Kernel) == 0;
}

bool addsub_kernel_path_needs_asm_base(const char *kernel_path) {
    return kernel_path != nullptr && strstr(kernel_path, "_asm_") != nullptr;
}

bool plan_needs_asm_common(const EcmStage1KernelBuildPlan &plan) {
    return (plan.add != nullptr && addsub_kernel_path_needs_asm_base(plan.add->kernel_path)) ||
           (plan.sub != nullptr && addsub_kernel_path_needs_asm_base(plan.sub->kernel_path));
}

int stage1_coop_wg_for_plan(const EcmStage1KernelBuildPlan &plan) {
    int coop_wg = 1;
    if (plan.limbs == kContainer4096Limbs) {
        if (plan.mul != nullptr && ecm_mont_path_is_4096_dedicated(plan.mul)) {
            coop_wg = std::max(coop_wg, static_cast<int>(plan.mul->coop_work_group_size));
        }
        if (plan.sqr != nullptr && ecm_mont_path_is_4096_dedicated(plan.sqr)) {
            coop_wg = std::max(coop_wg, static_cast<int>(plan.sqr->coop_work_group_size));
        }
    }
    return coop_wg;
}

void append_unique_kernel_path(std::vector<const char *> &paths, const char *kernel_path) {
    if (kernel_path == nullptr || kernel_path[0] == '\0') {
        return;
    }
    for (const char *existing : paths) {
        if (strcmp(existing, kernel_path) == 0) {
            return;
        }
    }
    paths.push_back(kernel_path);
}

void append_impl_macro(std::string &out, const char *name, const char *symbol) {
    out += "#define ";
    out += name;
    out += " ";
    out += symbol;
    out += "\n";
}

int addsub_id_kernel_path(const char *id) {
    if (id == nullptr) {
        return ECM_ADDSUB_PATH_FUSED_UNROLL;
    }
    static const struct {
        const char *id;
        int path;
    } kMap[] = {
        {"fused", ECM_ADDSUB_PATH_FUSED},
        {"fused_unroll", ECM_ADDSUB_PATH_FUSED_UNROLL},
        {"unroll_4096b", ECM_ADDSUB_PATH_FUSED_UNROLL_B32},
        {"asm_4096b", ECM_ADDSUB_PATH_ASM_B32},
        {"unroll_512b", ECM_ADDSUB_PATH_FUSED_UNROLL_B16},
        {"asm_512b", ECM_ADDSUB_PATH_ASM_B16},
        {"unroll_128b", ECM_ADDSUB_PATH_UNROLL_128B},
        {"asm_128b", ECM_ADDSUB_PATH_ASM_128B},
        {"unroll_192b", ECM_ADDSUB_PATH_UNROLL_192B},
        {"asm_192b", ECM_ADDSUB_PATH_ASM_192B},
        {"unroll_256b", ECM_ADDSUB_PATH_UNROLL_256B},
        {"asm_256b", ECM_ADDSUB_PATH_ASM_256B},
        {"unroll_384b", ECM_ADDSUB_PATH_UNROLL_384B},
        {"asm_384b", ECM_ADDSUB_PATH_ASM_384B},
    };
    for (const auto &entry : kMap) {
        if (strcmp(id, entry.id) == 0) {
            return entry.path;
        }
    }
    return ECM_ADDSUB_PATH_FUSED_UNROLL;
}

int mont_id_kernel_path(const char *id) {
    if (id == nullptr) {
        return ECM_MONT4096_PATH_UNROLL64;
    }
    if (strcmp(id, "unroll64_4096_mt2") == 0) {
        return ECM_MONT4096_PATH_UNROLL64_MT2;
    }
    if (strcmp(id, "fips4096") == 0) {
        return ECM_MONT4096_PATH_FIPS4096;
    }
    if (strcmp(id, "fips4096_mt8") == 0) {
        return ECM_MONT4096_PATH_FIPS4096_MT8;
    }
    if (strcmp(id, "fips4096_mt16") == 0) {
        return ECM_MONT4096_PATH_FIPS4096_MT16;
    }
    return ECM_MONT4096_PATH_UNROLL64;
}

std::vector<const EcmMontPathDescriptor *> auto_sorted_mont(const EcmMontPathDescriptor *registry,
                                                            size_t count) {
    std::vector<const EcmMontPathDescriptor *> out;
    for (size_t i = 0; i < count; ++i) {
        if (registry[i].auto_priority < 0) {
            continue;
        }
        out.push_back(&registry[i]);
    }
    std::sort(out.begin(), out.end(),
              [](const EcmMontPathDescriptor *a, const EcmMontPathDescriptor *b) {
                  return a->auto_priority < b->auto_priority;
              });
    return out;
}

const EcmMontPathDescriptor *find_mont_by_id(const EcmMontPathDescriptor *registry, size_t count,
                                             const char *id) {
    if (id == nullptr) {
        return nullptr;
    }
    for (size_t i = 0; i < count; ++i) {
        if (registry[i].id != nullptr && strcmp(registry[i].id, id) == 0) {
            return &registry[i];
        }
    }
    return nullptr;
}

const EcmMontPathDescriptor *find_mont_legacy_mode(const EcmMontPathDescriptor *registry,
                                                   size_t count, ecm_stage1_mont_mode mode) {
    switch (mode) {
    case ECM_STAGE1_MONT_UNROLL512:
        return find_mont_by_id(registry, count, "unroll_only_512");
    case ECM_STAGE1_MONT_UNROLL32:
        return find_mont_by_id(registry, count, "unroll32");
    case ECM_STAGE1_MONT_UNROLL384:
        return find_mont_by_id(registry, count, "unroll_only_384");
    case ECM_STAGE1_MONT_PRIV_OPT:
        return find_mont_by_id(registry, count, "priv_opt");
    default:
        return nullptr;
    }
}

const EcmMontPathDescriptor *resolve_mont_side(const EcmMontPathDescriptor *registry,
                                               size_t count, const char *path,
                                               const EcmPathContext &ctx, bool *unknown_path) {
    if (unknown_path != nullptr) {
        *unknown_path = false;
    }
    const EcmMontPathDescriptor *priv_opt = find_mont_by_id(registry, count, "priv_opt");
    const EcmMontPathDescriptor *unroll512 = find_mont_by_id(registry, count, "unroll_only_512");

    if (opencl_ecm_path_is_auto(path)) {
        for (const EcmMontPathDescriptor *desc : auto_sorted_mont(registry, count)) {
            if (ecm_mont_path_fits(desc, ctx)) {
                return desc;
            }
        }
        return priv_opt != nullptr ? priv_opt : unroll512;
    }

    for (size_t i = 0; i < count; ++i) {
        const EcmMontPathDescriptor &desc = registry[i];
        if (!aliases_contain(desc.aliases, path)) {
            continue;
        }
        if (ecm_mont_path_fits(&desc, ctx)) {
            return &desc;
        }
        const int min_pri = desc.auto_priority >= 0 ? desc.auto_priority + 1 : 0;
        for (const EcmMontPathDescriptor *fb : auto_sorted_mont(registry, count)) {
            if (fb->auto_priority < min_pri) {
                continue;
            }
            if (ecm_mont_path_fits(fb, ctx)) {
                return fb;
            }
        }
        return priv_opt != nullptr ? priv_opt : unroll512;
    }

    if (unknown_path != nullptr) {
        *unknown_path = true;
    }
    return nullptr;
}

const EcmAddSubPathDescriptor *find_addsub_by_id(const EcmAddSubPathDescriptor *registry,
                                                 size_t count, const char *id) {
    if (id == nullptr) {
        return nullptr;
    }
    for (size_t i = 0; i < count; ++i) {
        if (registry[i].id != nullptr && strcmp(registry[i].id, id) == 0) {
            return &registry[i];
        }
    }
    return nullptr;
}

const EcmAddSubPathDescriptor *find_addsub_by_kernel_path(const EcmAddSubPathDescriptor *registry,
                                                          size_t count, int kernel_path) {
    for (size_t i = 0; i < count; ++i) {
        if (ecm_addsub_descriptor_kernel_path(&registry[i]) == kernel_path) {
            return &registry[i];
        }
    }
    return nullptr;
}

const EcmAddSubPathDescriptor *resolve_addsub_side(const EcmAddSubPathDescriptor *registry,
                                                   size_t count, const char *path,
                                                   const EcmPathContext &ctx) {
    if (!opencl_ecm_path_is_auto(path)) {
        for (size_t i = 0; i < count; ++i) {
            if (aliases_contain(registry[i].aliases, path)) {
                return &registry[i];
            }
        }
        return nullptr;
    }
    std::vector<const EcmAddSubPathDescriptor *> ordered;
    ordered.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        ordered.push_back(&registry[i]);
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const EcmAddSubPathDescriptor *a, const EcmAddSubPathDescriptor *b) {
                  return a->auto_priority < b->auto_priority;
              });
    for (const EcmAddSubPathDescriptor *desc : ordered) {
        if (ecm_addsub_path_fits(desc, ctx)) {
            return desc;
        }
    }
    return find_addsub_by_id(registry, count, "fused_unroll");
}

void append_define(std::string &opts, const char *macro, int value) {
    if (macro == nullptr || macro[0] == '\0') {
        return;
    }
    opts += " -D";
    opts += macro;
    opts += '=';
    opts += std::to_string(value);
}

int mont_kernel_path_for_plan(const EcmMontPathDescriptor *desc, uint32_t plan_limbs) {
    if (!ecm_mont_path_is_4096_dedicated(desc)) {
        return 0;
    }
    const uint32_t operator_limbs = ecm_mont_operator_limbs(desc);
    if (operator_limbs == 0u || plan_limbs != operator_limbs) {
        return 0;
    }
    return ecm_mont_descriptor_kernel_path(desc);
}

} // namespace

bool ecm_mont_path_is_4096_dedicated(const EcmMontPathDescriptor *desc) {
    return desc != nullptr && desc->dedicated &&
           desc->max_n_bits >= static_cast<uint16_t>(ECM_PATH_4096_AUTO_MIN_BITS);
}

bool ecm_path_n_bit_fits(uint16_t min_n_bits, uint16_t max_n_bits, bool max_n_strict,
                         size_t n_bit_size) {
    if (min_n_bits > 0 && n_bit_size < min_n_bits) {
        return false;
    }
    if (max_n_bits == 0) {
        return true;
    }
    const size_t n_eff = n_bit_size + ECM_STAGE1_MONT_CARRY_BITS;
    return max_n_strict ? (n_eff < max_n_bits) : (n_eff <= max_n_bits);
}

uint32_t ecm_path_host_os_mask() {
#if defined(_WIN32)
    return ECM_OS_WINDOWS;
#elif defined(__ANDROID__)
    return ECM_OS_ANDROID;
#elif defined(__APPLE__)
    return ECM_OS_MACOS;
#elif defined(__linux__)
    return ECM_OS_LINUX;
#else
    return ECM_OS_ANY;
#endif
}

uint32_t ecm_path_gpu_vendor_from_cl_vendor_string(const char *vendor_lower) {
    if (vendor_lower == nullptr || vendor_lower[0] == '\0') {
        return 0;
    }
    if (std::strstr(vendor_lower, "advanced micro devices") != nullptr ||
        std::strstr(vendor_lower, "amd") != nullptr) {
        return ECM_GPU_AMD;
    }
    if (std::strstr(vendor_lower, "nvidia") != nullptr) {
        return ECM_GPU_NVIDIA;
    }
    if (std::strstr(vendor_lower, "intel") != nullptr) {
        return ECM_GPU_INTEL;
    }
    if (std::strstr(vendor_lower, "qualcomm") != nullptr) {
        return ECM_GPU_QUALCOMM;
    }
    if (std::strstr(vendor_lower, "huawei") != nullptr ||
        std::strstr(vendor_lower, "hisilicon") != nullptr) {
        return ECM_GPU_HUAWEI;
    }
    if (std::strstr(vendor_lower, "apple") != nullptr) {
        return ECM_GPU_APPLE;
    }
    return 0;
}

uint32_t ecm_mont_operator_limbs(const EcmMontPathDescriptor *desc) {
    if (desc == nullptr || !desc->dedicated || desc->max_n_bits == 0) {
        return 0;
    }
    return (static_cast<uint32_t>(desc->max_n_bits) + 31u) / 32u;
}

bool ecm_mont_path_fits(const EcmMontPathDescriptor *desc, const EcmPathContext &ctx) {
    if (desc == nullptr) {
        return false;
    }
    if (!ecm_path_n_bit_fits(desc->min_n_bits, desc->max_n_bits, desc->max_n_strict,
                             ctx.n_bit_size)) {
        return false;
    }
    if (!ecm_path_mask_fits(desc->os_mask, 0, ctx.os_mask)) {
        return false;
    }
    if (!ecm_path_mask_fits(desc->gpu_vendor_mask, desc->gpu_vendor_exclude_mask,
                            ctx.gpu_vendor_mask)) {
        return false;
    }
    if (ctx.container_limbs == 0u) {
        return true;
    }
    const uint32_t container_bits = ctx.container_limbs * 32u;
    if (desc->max_container_bits > 0 && container_bits < desc->max_container_bits) {
        return false;
    }
    if (desc->dedicated && desc->max_n_bits > 0) {
        return container_bits >= desc->max_n_bits;
    }
    const size_t need_bits = ctx.n_bit_size + ECM_STAGE1_MONT_CARRY_BITS;
    return need_bits <= static_cast<size_t>(container_bits);
}

bool ecm_addsub_path_fits(const EcmAddSubPathDescriptor *desc, const EcmPathContext &ctx) {
    if (desc == nullptr) {
        return false;
    }
    if (!ecm_path_n_bit_fits(desc->min_n_bits, desc->max_n_bits, desc->max_n_strict,
                             ctx.n_bit_size)) {
        return false;
    }
    if (desc->max_container_bits > 0 &&
        ctx.container_limbs * 32u < desc->max_container_bits) {
        return false;
    }
    if (!ecm_path_mask_fits(desc->os_mask, 0, ctx.os_mask)) {
        return false;
    }
    if (!ecm_path_mask_fits(desc->gpu_vendor_mask, desc->gpu_vendor_exclude_mask,
                            ctx.gpu_vendor_mask)) {
        return false;
    }
    return true;
}

bool opencl_ecm_path_is_auto(const char *path) {
    return path == nullptr || path[0] == '\0' || strcmp(path, "auto") == 0 ||
           strcmp(path, "default") == 0;
}

int ecm_addsub_descriptor_kernel_path(const EcmAddSubPathDescriptor *desc) {
    return addsub_id_kernel_path(desc != nullptr ? desc->id : nullptr);
}

int ecm_mont_descriptor_kernel_path(const EcmMontPathDescriptor *desc) {
    return mont_id_kernel_path(desc != nullptr ? desc->id : nullptr);
}

std::vector<const char *> opencl_ecm_stage1_kernel_source_paths(const EcmStage1KernelBuildPlan &plan) {
    std::vector<const char *> paths;
    append_unique_kernel_path(paths, kEcmStage1CommonConfig);
    append_unique_kernel_path(paths, kEcmStage1CommonMpPriv);
    append_unique_kernel_path(paths, kEcmStage1LadderHelpers);
    if (plan_needs_asm_common(plan)) {
        append_unique_kernel_path(paths, kEcmStage1AsmCommon);
    }
    if (plan.mul != nullptr) {
        append_unique_kernel_path(paths, plan.mul->kernel_path);
    }
    if (plan.sqr != nullptr) {
        append_unique_kernel_path(paths, plan.sqr->kernel_path);
    }
    if (plan.add != nullptr) {
        append_unique_kernel_path(paths, plan.add->kernel_path);
    }
    if (plan.sub != nullptr) {
        append_unique_kernel_path(paths, plan.sub->kernel_path);
    }
    append_unique_kernel_path(paths, kEcmStage1OperatorIface);
    if (stage1_coop_wg_for_plan(plan) > 1) {
        append_unique_kernel_path(paths, kEcmStage1Coop);
    }
    append_unique_kernel_path(paths, kEcmStage1Entry);
    return paths;
}

std::string opencl_ecm_stage1_assemble_kernel_source(
    const EcmStage1KernelBuildPlan &plan,
    const std::function<std::string(const char *rel_path)> &load_file) {
    std::string source;
    source.reserve(65536);
    if (plan.mul == nullptr || plan.sqr == nullptr || plan.add == nullptr || plan.sub == nullptr) {
        return source;
    }
    append_impl_macro(source, "ECM_STAGE1_MUL_IMPL", plan.mul->cl_name);
    append_impl_macro(source, "ECM_STAGE1_SQR_IMPL", plan.sqr->cl_name);
    append_impl_macro(source, "ECM_STAGE1_ADD_IMPL", plan.add->cl_name);
    append_impl_macro(source, "ECM_STAGE1_SUB_IMPL", plan.sub->cl_name);
    source += "\n";

    const std::vector<const char *> paths = opencl_ecm_stage1_kernel_source_paths(plan);
    for (const char *rel_path : paths) {
        const std::string chunk = load_file(rel_path);
        if (chunk.empty()) {
            return std::string();
        }
        if (!source.empty() && source.back() != '\n') {
            source += "\n";
        }
        source += chunk;
        if (chunk.back() != '\n') {
            source += "\n";
        }
    }
    return source;
}

size_t opencl_ecm_mont_mul_registry_count() {
    return sizeof(kMontMulRegistry) / sizeof(kMontMulRegistry[0]);
}

const EcmMontPathDescriptor *opencl_ecm_mont_mul_registry_entry(size_t index) {
    if (index >= opencl_ecm_mont_mul_registry_count()) {
        return nullptr;
    }
    return &kMontMulRegistry[index];
}

const EcmMontPathDescriptor *opencl_ecm_mont_mul_descriptor(ecm_stage1_mont_mode mode) {
    return find_mont_legacy_mode(kMontMulRegistry, opencl_ecm_mont_mul_registry_count(), mode);
}

size_t opencl_ecm_mont_sqr_registry_count() {
    return sizeof(kMontSqrRegistry) / sizeof(kMontSqrRegistry[0]);
}

const EcmMontPathDescriptor *opencl_ecm_mont_sqr_registry_entry(size_t index) {
    if (index >= opencl_ecm_mont_sqr_registry_count()) {
        return nullptr;
    }
    return &kMontSqrRegistry[index];
}

const EcmMontPathDescriptor *opencl_ecm_mont_sqr_descriptor(ecm_stage1_mont_mode mode) {
    return find_mont_legacy_mode(kMontSqrRegistry, opencl_ecm_mont_sqr_registry_count(), mode);
}

const EcmMontPathDescriptor *opencl_ecm_resolve_mont_mul(const char *path, const EcmPathContext &ctx,
                                                         bool *unknown_path) {
    return resolve_mont_side(kMontMulRegistry, opencl_ecm_mont_mul_registry_count(), path, ctx,
                             unknown_path);
}

const EcmMontPathDescriptor *opencl_ecm_resolve_mont_sqr(const char *path, const EcmPathContext &ctx,
                                                         bool *unknown_path) {
    return resolve_mont_side(kMontSqrRegistry, opencl_ecm_mont_sqr_registry_count(), path, ctx,
                             unknown_path);
}

size_t opencl_ecm_addmod_registry_count() {
    return sizeof(kAddModRegistry) / sizeof(kAddModRegistry[0]);
}

const EcmAddSubPathDescriptor *opencl_ecm_addmod_registry_entry(size_t index) {
    if (index >= opencl_ecm_addmod_registry_count()) {
        return nullptr;
    }
    return &kAddModRegistry[index];
}

const EcmAddSubPathDescriptor *opencl_ecm_addmod_descriptor_by_kernel_path(int kernel_path) {
    return find_addsub_by_kernel_path(kAddModRegistry, opencl_ecm_addmod_registry_count(),
                                      kernel_path);
}

const EcmAddSubPathDescriptor *opencl_ecm_resolve_addmod_path(const char *path,
                                                              const EcmPathContext &ctx) {
    return resolve_addsub_side(kAddModRegistry, opencl_ecm_addmod_registry_count(), path, ctx);
}

size_t opencl_ecm_submod_registry_count() {
    return sizeof(kSubModRegistry) / sizeof(kSubModRegistry[0]);
}

const EcmAddSubPathDescriptor *opencl_ecm_submod_registry_entry(size_t index) {
    if (index >= opencl_ecm_submod_registry_count()) {
        return nullptr;
    }
    return &kSubModRegistry[index];
}

const EcmAddSubPathDescriptor *opencl_ecm_submod_descriptor_by_kernel_path(int kernel_path) {
    return find_addsub_by_kernel_path(kSubModRegistry, opencl_ecm_submod_registry_count(),
                                      kernel_path);
}

const EcmAddSubPathDescriptor *opencl_ecm_resolve_submod_path(const char *path,
                                                              const EcmPathContext &ctx) {
    return resolve_addsub_side(kSubModRegistry, opencl_ecm_submod_registry_count(), path, ctx);
}

EcmStage1KernelBuildPlan opencl_ecm_stage1_make_build_plan(
    uint32_t limbs, uint32_t tpi, const EcmMontPathDescriptor *mul,
    const EcmMontPathDescriptor *sqr, const EcmAddSubPathDescriptor *add,
    const EcmAddSubPathDescriptor *sub, int stage1_force_normalize, int add_mod_fused_unroll) {
    EcmStage1KernelBuildPlan plan{};
    plan.limbs = limbs;
    plan.tpi = tpi;
    plan.stage1_force_normalize = stage1_force_normalize;
    plan.add_mod_fused_unroll = add_mod_fused_unroll;
    plan.mul = mul;
    plan.sqr = sqr;
    plan.add = add;
    plan.sub = sub;
    return plan;
}

bool opencl_ecm_stage1_build_plan_equal(const EcmStage1KernelBuildPlan &a,
                                        const EcmStage1KernelBuildPlan &b) {
    return a.limbs == b.limbs && a.tpi == b.tpi &&
           a.stage1_force_normalize == b.stage1_force_normalize &&
           a.add_mod_fused_unroll == b.add_mod_fused_unroll && a.mul == b.mul && a.sqr == b.sqr &&
           a.add == b.add && a.sub == b.sub;
}

std::string opencl_ecm_stage1_generate_build_options(const EcmStage1KernelBuildPlan &plan) {
    std::string opts;
    opts.reserve(512);
    opts += "-DMAX_LIMBS=";
    opts += std::to_string(plan.limbs);
    opts += " -DTPI=";
    opts += std::to_string(plan.tpi);
    append_define(opts, "ECM_STAGE1_FORCE_NORMALIZE", plan.stage1_force_normalize);
    append_define(opts, "MP_ADD_MOD_FUSED_UNROLL", plan.add_mod_fused_unroll);
    append_define(opts, "ECM_STAGE1_KERNEL_REV", 14);
    append_define(opts, "MP_LIMB_BITS", 32);

    append_define(opts, "ECM_STAGE1_MUL_PATH", mont_kernel_path_for_plan(plan.mul, plan.limbs));
    append_define(opts, "ECM_STAGE1_SQR_PATH", mont_kernel_path_for_plan(plan.sqr, plan.limbs));

    const int coop_wg = stage1_coop_wg_for_plan(plan);
    int coop_scratch = 0;
    if (plan.limbs == kContainer4096Limbs) {
        if (plan.mul != nullptr && ecm_mont_path_is_4096_dedicated(plan.mul)) {
            coop_scratch = std::max(coop_scratch, static_cast<int>(plan.mul->local_scratch_u32));
        }
        if (plan.sqr != nullptr && ecm_mont_path_is_4096_dedicated(plan.sqr)) {
            coop_scratch = std::max(coop_scratch, static_cast<int>(plan.sqr->local_scratch_u32));
        }
    }
    const bool has_fips4096 =
        mont_kernel_path_needs_fips4096(plan.mul != nullptr ? plan.mul->kernel_path : nullptr) ||
        mont_kernel_path_needs_fips4096(plan.sqr != nullptr ? plan.sqr->kernel_path : nullptr);
    append_define(opts, "ECM_STAGE1_COOP_WG", coop_wg);
    append_define(opts, "ECM_STAGE1_COOP_SCRATCH_U32", coop_scratch);
    append_define(opts, "ECM_STAGE1_HAS_FIPS4096", has_fips4096 ? 1 : 0);

    return opts;
}
