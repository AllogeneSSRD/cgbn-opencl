#include "opencl_ecm_path_registry.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

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
constexpr uint16_t kContainer4096Bits = static_cast<uint16_t>(ECM_PATH_4096_CONTAINER_BITS);
constexpr uint32_t kContainer4096Limbs = 128u;

constexpr uint16_t kAddSubNoMinN = 0;
constexpr uint16_t kAddSubNoMaxN = 0;
constexpr uint16_t kAddSub512Container = 512;
constexpr uint16_t kAddSub384MaxN = 378;
constexpr uint16_t kAddSub512MaxN = 506;

#define ECM_MONT_ALIAS_TABLE(side, S)                                                              \
    static const char *const kMontAliases_##side##_unroll384[] = {                                 \
        "unroll_only_384", "mont_" S "_priv_unroll_only_384", nullptr};                            \
    static const char *const kMontAliases_##side##_unroll512[] = {                                 \
        "unroll_only_512", "mont_" S "_priv_unroll_only_512", nullptr};                            \
    static const char *const kMontAliases_##side##_unroll64_4096[] = {"unroll64_4096", nullptr};   \
    static const char *const kMontAliases_##side##_fips4096[] = {"fips4096", nullptr};             \
    static const char *const kMontAliases_##side##_fips4096_mt8[] = {"fips4096_mt8", nullptr};     \
    static const char *const kMontAliases_##side##_fips4096_mt16[] = {"fips4096_mt16", nullptr};   \
    static const char *const kMontAliases_##side##_unroll32[] = {                                  \
        "unroll32", "mont_" S "_priv_unroll32", "mont_" S "_stage1_unroll32", nullptr};           \
    static const char *const kMontAliases_##side##_priv_opt[] = {                                  \
        "priv_opt", "mont_" S "_priv_opt", "mont_" S "_stage1_priv_opt", nullptr};

ECM_MONT_ALIAS_TABLE(mul, "mul")
ECM_MONT_ALIAS_TABLE(sqr, "sqr")

#define ECM_MONT_OPERATORS(X)                                                                      \
    X(unroll_only_384, unroll_384b, unroll384, 10, kMontNoMinN, kMontUnroll384MaxN, true, 0,        \
      ECM_OS_ANY, ECM_GPU_ANY, 0, true, 1, 0)                                                       \
    X(unroll_only_512, unroll_512b, unroll512, 20, kMontNoMinN, kMontUnroll512MaxN, false, 0,       \
      ECM_OS_ANY, ECM_GPU_ANY, 0, true, 1, 0)                                                       \
    X(unroll64_4096, unroll_4096b, unroll64_4096, 21, kMont4096MinN, kMont4096MaxN, false,          \
      kContainer4096Bits, ECM_OS_ANY, ECM_GPU_ANY, 0, true, 1, 0)                                   \
    X(fips4096, fips_4096b, fips4096, 23, kMont4096MinN, kMont4096MaxN, false, kContainer4096Bits,  \
      ECM_OS_ANY, ECM_GPU_ANY, 0, true, 1, 0)                                                       \
    X(fips4096_mt8, fips_4096b, fips4096_mt8, 24, kMont4096MinN, kMont4096MaxN, false,              \
      kContainer4096Bits, ECM_OS_ANY, ECM_GPU_ANY, 0, true, 8, 897)                                 \
    X(fips4096_mt16, fips_4096b, fips4096_mt16, 25, kMont4096MinN, kMont4096MaxN, false,            \
      kContainer4096Bits, ECM_OS_ANY, ECM_GPU_ANY, 0, true, 16, 897)                                \
    X(unroll32, unroll_32, unroll32, -1, kMontNoMinN, kMontNoMaxN, false, 0, ECM_OS_ANY,            \
      ECM_GPU_ANY, 0, false, 1, 0)                                                                  \
    X(priv_opt, priv_opt, priv_opt, 30, kMontNoMinN, kMontNoMaxN, false, 0, ECM_OS_ANY,            \
      ECM_GPU_ANY, 0, false, 1, 0)

#define ECM_MONT_MUL_ROW(idt, stem, al, ...)                                                       \
    {#idt, "mont_mul_" #stem, kMontAliases_mul_##al, "mont_mul/mont_mul_" #stem ".cl", __VA_ARGS__},
#define ECM_MONT_SQR_ROW(idt, stem, al, ...)                                                       \
    {#idt, "mont_sqr_" #stem, kMontAliases_sqr_##al, "mont_mul/mont_mul_" #stem ".cl", __VA_ARGS__},

constexpr EcmMontPathDescriptor kMontMulRegistry[] = {ECM_MONT_OPERATORS(ECM_MONT_MUL_ROW)};
constexpr EcmMontPathDescriptor kMontSqrRegistry[] = {ECM_MONT_OPERATORS(ECM_MONT_SQR_ROW)};

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

#define ECM_ADDSUB_OPERATORS(X)                                                                    \
    X(asm_4096b, 28, kAddSubNoMinN, kAddSubNoMaxN, false, kContainer4096Bits, ECM_OS_ANY,           \
      ECM_GPU_AMD, 0)                                                                               \
    X(unroll_4096b, 29, kAddSubNoMinN, kAddSubNoMaxN, false, kContainer4096Bits, ECM_OS_ANY, 0,     \
      ECM_GPU_AMD)                                                                                  \
    X(asm_128b, 20, kAddSubNoMinN, 128, false, kAddSub512Container, ECM_OS_ANY, ECM_GPU_AMD, 0)     \
    X(unroll_128b, 21, kAddSubNoMinN, 128, false, kAddSub512Container, ECM_OS_ANY, ECM_GPU_ANY, 0)  \
    X(asm_192b, 22, kAddSubNoMinN, 192, false, kAddSub512Container, ECM_OS_ANY, ECM_GPU_AMD, 0)     \
    X(unroll_192b, 23, kAddSubNoMinN, 192, false, kAddSub512Container, ECM_OS_ANY, ECM_GPU_ANY, 0)  \
    X(asm_256b, 24, kAddSubNoMinN, 256, false, kAddSub512Container, ECM_OS_ANY, ECM_GPU_AMD, 0)     \
    X(unroll_256b, 25, kAddSubNoMinN, 256, false, kAddSub512Container, ECM_OS_ANY, ECM_GPU_ANY, 0)  \
    X(asm_384b, 26, kAddSubNoMinN, kAddSub384MaxN, false, kAddSub512Container, ECM_OS_ANY,          \
      ECM_GPU_AMD, 0)                                                                               \
    X(unroll_384b, 27, kAddSubNoMinN, kAddSub384MaxN, false, kAddSub512Container, ECM_OS_ANY,       \
      ECM_GPU_ANY, 0)                                                                               \
    X(asm_512b, 30, kAddSubNoMinN, kAddSub512MaxN, false, kAddSub512Container, ECM_OS_ANY,          \
      ECM_GPU_AMD, 0)                                                                               \
    X(unroll_512b, 31, kAddSubNoMinN, kAddSub512MaxN, false, kAddSub512Container, ECM_OS_ANY,       \
      ECM_GPU_AMD, 0)                                                                               \
    X(fused, 32, kAddSubNoMinN, kAddSubNoMaxN, false, kAddSub512Container, ECM_OS_ANY, 0,           \
      ECM_GPU_AMD)                                                                                  \
    X(fused_unroll, 40, kAddSubNoMinN, kAddSubNoMaxN, false, 0, ECM_OS_ANY, ECM_GPU_ANY, 0)

#define ECM_ADD_ROW(idt, ...)                                                                      \
    {#idt, "add_mod_" #idt, kAddAliases_##idt, "add_mod/add_mod_" #idt ".cl", __VA_ARGS__},
#define ECM_SUB_ROW(idt, ...)                                                                      \
    {#idt, "sub_mod_" #idt, kSubAliases_##idt, "sub_mod/sub_mod_" #idt ".cl", __VA_ARGS__},

constexpr EcmAddSubPathDescriptor kAddModRegistry[] = {ECM_ADDSUB_OPERATORS(ECM_ADD_ROW)};
constexpr EcmAddSubPathDescriptor kSubModRegistry[] = {ECM_ADDSUB_OPERATORS(ECM_SUB_ROW)};

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
static constexpr const char *kEcmStage1LadderHelpers = "common/ladder_helpers.h.cl";
static constexpr const char *kEcmStage1AsmCommon = "common/asm_common.h.cl";
static constexpr const char *kEcmStage1OperatorIface = "common/operator_iface.h.cl";
static constexpr const char *kEcmStage1Coop = "ecm_stage1_coop.cl";
static constexpr const char *kEcmStage1Entry = "ecm_stage1.cl";
static constexpr const char *kEcmStage1SpecialMult512b = "special_mult/special_mult_unroll_512b.cl";
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
        if (plan.mul != nullptr) {
            coop_wg = std::max(coop_wg, static_cast<int>(plan.mul->coop_work_group_size));
        }
        if (plan.sqr != nullptr) {
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

const EcmMontPathDescriptor *resolve_mont_side(const EcmMontPathDescriptor *registry, size_t count,
                                               const char *path, const EcmPathContext &ctx,
                                               bool *unknown_path) {
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
    if (desc == nullptr || !desc->dedicated) {
        return 0;
    }
    const uint32_t operator_limbs = ecm_mont_operator_limbs(desc);
    if (operator_limbs == 0u || plan_limbs != operator_limbs) {
        return 0;
    }
    return ecm_mont_descriptor_kernel_path(desc);
}

} // namespace


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
    if (desc->max_container_bits > 0 && ctx.container_limbs * 32u < desc->max_container_bits) {
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

std::vector<const char *> opencl_ecm_stage1_kernel_source_paths(
    const EcmStage1KernelBuildPlan &plan) {
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
    if (plan.limbs <= 16u) {
        append_unique_kernel_path(paths, kEcmStage1SpecialMult512b);
    }
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
        if (plan.mul != nullptr && plan.mul->coop_work_group_size > 1u) {
            coop_scratch = std::max(coop_scratch, static_cast<int>(plan.mul->local_scratch_u32));
        }
        if (plan.sqr != nullptr && plan.sqr->coop_work_group_size > 1u) {
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
    if (desc == nullptr || !desc->dedicated) {
        return ECM_MONT4096_PATH_UNROLL64;
    }
    return ecm_mont_descriptor_kernel_path(desc);
}

namespace {

ecm_stage1_mont_mode mont_desc_legacy_mode(const EcmMontPathDescriptor *d) {
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

} // namespace

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

int opencl_ecm_parse_addsub_path(const char *path) {
    if (opencl_ecm_path_is_auto(path)) {
        return -1;
    }
    for (size_t i = 0; i < opencl_ecm_addmod_registry_count(); ++i) {
        const EcmAddSubPathDescriptor *desc = opencl_ecm_addmod_registry_entry(i);
        if (desc != nullptr && aliases_contain(desc->aliases, path)) {
            return ecm_addsub_descriptor_kernel_path(desc);
        }
    }
    for (size_t i = 0; i < opencl_ecm_submod_registry_count(); ++i) {
        const EcmAddSubPathDescriptor *desc = opencl_ecm_submod_registry_entry(i);
        if (desc != nullptr && aliases_contain(desc->aliases, path)) {
            return ecm_addsub_descriptor_kernel_path(desc);
        }
    }
    return -2;
}

const char *opencl_ecm_addsub_path_name(int path_id) {
    const EcmAddSubPathDescriptor *add_d = opencl_ecm_addmod_descriptor_by_kernel_path(path_id);
    if (add_d != nullptr && add_d->cl_name != nullptr) {
        return add_d->cl_name;
    }
    const EcmAddSubPathDescriptor *sub_d = opencl_ecm_submod_descriptor_by_kernel_path(path_id);
    if (sub_d != nullptr && sub_d->cl_name != nullptr) {
        return sub_d->cl_name;
    }
    return "unknown";
}

bool opencl_ecm_addsub_path_needs_asm_b32(int path_id) {
    const EcmAddSubPathDescriptor *add_d = opencl_ecm_addmod_descriptor_by_kernel_path(path_id);
    if (add_d != nullptr && strcmp(add_d->id, "asm_4096b") == 0) {
        return true;
    }
    const EcmAddSubPathDescriptor *sub_d = opencl_ecm_submod_descriptor_by_kernel_path(path_id);
    return sub_d != nullptr && strcmp(sub_d->id, "asm_4096b") == 0;
}

bool opencl_ecm_addsub_path_needs_asm_b16(int path_id) {
    const EcmAddSubPathDescriptor *add_d = opencl_ecm_addmod_descriptor_by_kernel_path(path_id);
    if (add_d != nullptr && strcmp(add_d->id, "asm_512b") == 0) {
        return true;
    }
    const EcmAddSubPathDescriptor *sub_d = opencl_ecm_submod_descriptor_by_kernel_path(path_id);
    return sub_d != nullptr && strcmp(sub_d->id, "asm_512b") == 0;
}

bool opencl_ecm_addsub_path_needs_addsub_bits(int path_id) {
    static const char *const kBitsIds[] = {"asm_128b", "unroll_128b", "asm_192b", "unroll_192b",
                                           "asm_256b", "unroll_256b", "asm_384b", "unroll_384b"};
    const EcmAddSubPathDescriptor *add_d = opencl_ecm_addmod_descriptor_by_kernel_path(path_id);
    const EcmAddSubPathDescriptor *sub_d = opencl_ecm_submod_descriptor_by_kernel_path(path_id);
    for (const char *bid : kBitsIds) {
        if (add_d != nullptr && add_d->id != nullptr && strcmp(add_d->id, bid) == 0) {
            return true;
        }
        if (sub_d != nullptr && sub_d->id != nullptr && strcmp(sub_d->id, bid) == 0) {
            return true;
        }
    }
    return false;
}

const EcmAddSubPathDescriptor *opencl_ecm_resolve_addsub_add_path(const char *path,
                                                                  const EcmPathContext &ctx) {
    return opencl_ecm_resolve_addmod_path(path, ctx);
}

const EcmAddSubPathDescriptor *opencl_ecm_resolve_addsub_sub_path(const char *path,
                                                                  const EcmPathContext &ctx) {
    return opencl_ecm_resolve_submod_path(path, ctx);
}

void opencl_ecm_print_available_kernels(FILE *out) {
    if (out == nullptr) {
        out = stdout;
    }
    fprintf(out, "ECM OpenCL kernels: mul/sqr and add/sub paths are resolved independently.\n");
    fprintf(out, "See docs/DEV_OPERATOR_PATH_REGISTRY.md\n");
}
