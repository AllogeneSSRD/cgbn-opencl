#include "opencl_ecm_path_registry.h"

#include "opencl_ecm_addsub_path.h"
#include "opencl_ecm_log.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
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
static const char *const kMulAliases_i24_blsub[] = {
    "i24_u32_blsub", "mont_mul_unroll_i24_u32_blsub", "i24_384_manual",
    "mont_mul_unroll_i24_384_manual", nullptr};
static const char *const kMulAliases_i24_u32[] = {"i24_u32", "mont_mul_unroll_i24_u32", nullptr};

constexpr EcmMontPathDescriptor kMontMulRegistry[] = {
    {"unroll_only_384", "mont_mul_priv_unroll_only_384", kMulAliases_unroll384, 10, kMontNoMinN,
     kMontUnroll384MaxN, true, true, 1, 0, 0, ECM_KERNEL_INC_NONE,
     "ECM_STAGE1_MUL_FORCE_UNROLL384"},
    {"unroll_only_512", "mont_mul_priv_unroll_only_512", kMulAliases_unroll512, 20, kMontNoMinN,
     kMontUnroll512MaxN, false, true, 1, 0, 0, ECM_KERNEL_INC_NONE, nullptr},
    {"unroll64_4096", "mont_mul_stage1_unroll64_4096", kMulAliases_unroll64_4096, 21, kMont4096MinN,
     kMont4096MaxN, false, true, 1, 0, 0, ECM_KERNEL_INC_NONE, nullptr},
    {"unroll64_4096_mt2", "mont_mul_stage1_unroll64_4096_mt2", kMulAliases_unroll64_4096_mt2, 22,
     kMont4096MinN, kMont4096MaxN, false, true, 2, 389, 1, ECM_KERNEL_INC_NONE, nullptr},
    {"fips4096", "mont_mul_stage1_fips4096", kMulAliases_fips4096, 23, kMont4096MinN, kMont4096MaxN,
     false, true, 1, 0, 2, ECM_KERNEL_INC_MONT_EXTENDED, nullptr},
    {"fips4096_mt8", "mont_mul_stage1_fips4096_mt8", kMulAliases_fips4096_mt8, 24, kMont4096MinN,
     kMont4096MaxN, false, true, 8, 897, 3, ECM_KERNEL_INC_MONT_EXTENDED, nullptr},
    {"fips4096_mt16", "mont_mul_stage1_fips4096_mt16", kMulAliases_fips4096_mt16, 25,
     kMont4096MinN, kMont4096MaxN, false, true, 16, 897, 4, ECM_KERNEL_INC_MONT_EXTENDED,
     nullptr},
    {"unroll32", "mont_mul_stage1_unroll32", kMulAliases_unroll32, -1, kMontNoMinN, kMontNoMaxN,
     false, false, 1, 0, 0, ECM_KERNEL_INC_NONE, "ECM_STAGE1_MUL_FORCE_UNROLL32"},
    {"priv_opt", "mont_mul_stage1_priv_opt", kMulAliases_priv_opt, 30, kMontNoMinN, kMontNoMaxN,
     false, false, 1, 0, 0, ECM_KERNEL_INC_NONE, "ECM_STAGE1_MUL_FORCE_PRIV_OPT"},
    {"i24_u32_blsub", "mont_mul_unroll_i24_u32_blsub", kMulAliases_i24_blsub, -1, kMontNoMinN,
     kMontNoMaxN, false, false, 1, 0, 0, ECM_KERNEL_INC_NONE, nullptr},
    {"i24_u32", "mont_mul_unroll_i24_u32", kMulAliases_i24_u32, -1, kMontNoMinN, kMontNoMaxN,
     false, false, 1, 0, 0, ECM_KERNEL_INC_NONE, nullptr},
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
static const char *const kSqrAliases_i24_blsub[] = {"i24_u32_blsub", "mont_sqr_unroll_i24_u32_blsub",
                                                    nullptr};
static const char *const kSqrAliases_i24_u32[] = {"i24_u32", "mont_sqr_unroll_i24_u32", nullptr};

constexpr EcmMontPathDescriptor kMontSqrRegistry[] = {
    {"unroll_only_384", "mont_sqr_priv_unroll_only_384", kSqrAliases_unroll384, 10, kMontNoMinN,
     kMontUnroll384MaxN, true, true, 1, 0, 0, ECM_KERNEL_INC_NONE,
     "ECM_STAGE1_SQR_FORCE_UNROLL384"},
    {"unroll_only_512", "mont_sqr_priv_unroll_only_512", kSqrAliases_unroll512, 20, kMontNoMinN,
     kMontUnroll512MaxN, false, true, 1, 0, 0, ECM_KERNEL_INC_NONE, nullptr},
    {"unroll64_4096", "mont_sqr_stage1_unroll64_4096", kSqrAliases_unroll64_4096, 21, kMont4096MinN,
     kMont4096MaxN, false, true, 1, 0, 0, ECM_KERNEL_INC_NONE, nullptr},
    {"unroll64_4096_mt2", "mont_sqr_stage1_unroll64_4096_mt2", kSqrAliases_unroll64_4096_mt2, 22,
     kMont4096MinN, kMont4096MaxN, false, true, 2, 389, 1, ECM_KERNEL_INC_NONE, nullptr},
    {"fips4096", "mont_sqr_stage1_fips4096", kSqrAliases_fips4096, 23, kMont4096MinN, kMont4096MaxN,
     false, true, 1, 0, 2, ECM_KERNEL_INC_MONT_EXTENDED, nullptr},
    {"fips4096_mt8", "mont_sqr_stage1_fips4096_mt8", kSqrAliases_fips4096_mt8, 24, kMont4096MinN,
     kMont4096MaxN, false, true, 8, 897, 3, ECM_KERNEL_INC_MONT_EXTENDED, nullptr},
    {"fips4096_mt16", "mont_sqr_stage1_fips4096_mt16", kSqrAliases_fips4096_mt16, 25,
     kMont4096MinN, kMont4096MaxN, false, true, 16, 897, 4, ECM_KERNEL_INC_MONT_EXTENDED,
     nullptr},
    {"unroll32", "mont_sqr_stage1_unroll32", kSqrAliases_unroll32, -1, kMontNoMinN, kMontNoMaxN,
     false, false, 1, 0, 0, ECM_KERNEL_INC_NONE, "ECM_STAGE1_SQR_FORCE_UNROLL32"},
    {"priv_opt", "mont_sqr_stage1_priv_opt", kSqrAliases_priv_opt, 30, kMontNoMinN, kMontNoMaxN,
     false, false, 1, 0, 0, ECM_KERNEL_INC_NONE, "ECM_STAGE1_SQR_FORCE_PRIV_OPT"},
    {"i24_u32_blsub", "mont_sqr_unroll_i24_u32_blsub", kSqrAliases_i24_blsub, -1, kMontNoMinN,
     kMontNoMaxN, false, false, 1, 0, 0, ECM_KERNEL_INC_NONE, nullptr},
    {"i24_u32", "mont_sqr_unroll_i24_u32", kSqrAliases_i24_u32, -1, kMontNoMinN, kMontNoMaxN,
     false, false, 1, 0, 0, ECM_KERNEL_INC_NONE, nullptr},
};

static const char *const kAddAliases_fused[] = {"fused", nullptr};
static const char *const kAddAliases_fused_unroll[] = {"fused_unroll", nullptr};
static const char *const kAddAliases_fused_unroll_b32[] = {"fused_unroll_b32", nullptr};
static const char *const kAddAliases_asm_b32[] = {"asm_b32", nullptr};
static const char *const kAddAliases_asm_b16[] = {"asm_b16", "fused_asm_b16", nullptr};
static const char *const kAddAliases_fused_unroll_b16[] = {"fused_unroll_b16", "fused_unroll_auto",
                                                           nullptr};

static const char *const kSubAliases_fused[] = {"fused", nullptr};
static const char *const kSubAliases_fused_unroll[] = {"fused_unroll", nullptr};
static const char *const kSubAliases_fused_unroll_b32[] = {"fused_unroll_b32", nullptr};
static const char *const kSubAliases_fused_unroll_b16[] = {"fused_unroll_b16", "fused_unroll_auto",
                                                           nullptr};

constexpr EcmAddSubPathDescriptor kAddModRegistry[] = {
    {ECM_ADDSUB_PATH_ASM_B32, "asm_b32", "asm_b32", kAddAliases_asm_b32, 10,
     static_cast<uint16_t>(ECM_PATH_4096_CONTAINER_BITS), ECM_OS_ANY, ECM_GPU_AMD, 0,
     ECM_KERNEL_INC_MP_ASM_U32},
    {ECM_ADDSUB_PATH_FUSED_UNROLL_B32, "fused_unroll_b32", "fused_unroll_b32",
     kAddAliases_fused_unroll_b32, 11, static_cast<uint16_t>(ECM_PATH_4096_CONTAINER_BITS),
     ECM_OS_ANY, 0, ECM_GPU_AMD, ECM_KERNEL_INC_NONE},
    {ECM_ADDSUB_PATH_ASM_B16, "asm_b16", "asm_b16", kAddAliases_asm_b16, 20, 512, ECM_OS_ANY,
     ECM_GPU_AMD, 0, ECM_KERNEL_INC_MP_ASM_U16},
    {ECM_ADDSUB_PATH_FUSED, "fused", "fused", kAddAliases_fused, 21, 512, ECM_OS_ANY, 0,
     ECM_GPU_AMD, ECM_KERNEL_INC_NONE},
    {ECM_ADDSUB_PATH_FUSED_UNROLL_B16, "fused_unroll_b16", "fused_unroll_b16",
     kAddAliases_fused_unroll_b16, 22, 512, ECM_OS_ANY, ECM_GPU_AMD, 0, ECM_KERNEL_INC_NONE},
    {ECM_ADDSUB_PATH_FUSED_UNROLL, "fused_unroll", "fused_unroll", kAddAliases_fused_unroll, 30, 0,
     ECM_OS_ANY, ECM_GPU_ANY, 0, ECM_KERNEL_INC_NONE},
};

constexpr EcmAddSubPathDescriptor kSubModRegistry[] = {
    {ECM_ADDSUB_PATH_FUSED_UNROLL_B32, "fused_unroll_b32", "fused_unroll_b32",
     kSubAliases_fused_unroll_b32, 10, static_cast<uint16_t>(ECM_PATH_4096_CONTAINER_BITS),
     ECM_OS_ANY, ECM_GPU_AMD, 0, ECM_KERNEL_INC_NONE},
    {ECM_ADDSUB_PATH_FUSED_UNROLL_B32, "fused_unroll_b32", "fused_unroll_b32",
     kSubAliases_fused_unroll_b32, 11, static_cast<uint16_t>(ECM_PATH_4096_CONTAINER_BITS),
     ECM_OS_ANY, 0, ECM_GPU_AMD, ECM_KERNEL_INC_NONE},
    {ECM_ADDSUB_PATH_FUSED, "fused", "fused", kSubAliases_fused, 20, 512, ECM_OS_ANY, 0,
     ECM_GPU_AMD, ECM_KERNEL_INC_NONE},
    {ECM_ADDSUB_PATH_FUSED_UNROLL_B16, "fused_unroll_b16", "fused_unroll_b16",
     kSubAliases_fused_unroll_b16, 21, 512, ECM_OS_ANY, ECM_GPU_AMD, 0, ECM_KERNEL_INC_NONE},
    {ECM_ADDSUB_PATH_FUSED_UNROLL, "fused_unroll", "fused_unroll", kSubAliases_fused_unroll, 30, 0,
     ECM_OS_ANY, ECM_GPU_ANY, 0, ECM_KERNEL_INC_NONE},
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
    case ECM_STAGE1_MONT_I24_U32:
        return find_mont_by_id(registry, count, "i24_u32");
    case ECM_STAGE1_MONT_I24_U32_BLSUB:
        return find_mont_by_id(registry, count, "i24_u32_blsub");
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
                                               size_t count, const char *path, size_t n_bit_size,
                                               uint32_t container_limbs, bool *unknown_path) {
    if (unknown_path != nullptr) {
        *unknown_path = false;
    }
    const EcmMontPathDescriptor *priv_opt = find_mont_by_id(registry, count, "priv_opt");
    const EcmMontPathDescriptor *unroll512 = find_mont_by_id(registry, count, "unroll_only_512");

    if (opencl_ecm_path_is_auto(path)) {
        for (const EcmMontPathDescriptor *desc : auto_sorted_mont(registry, count)) {
            if (ecm_mont_path_fits(desc, n_bit_size, container_limbs)) {
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
        if (ecm_mont_path_fits(&desc, n_bit_size, container_limbs)) {
            return &desc;
        }
        const int min_pri = desc.auto_priority >= 0 ? desc.auto_priority + 1 : 0;
        for (const EcmMontPathDescriptor *fb : auto_sorted_mont(registry, count)) {
            if (fb->auto_priority < min_pri) {
                continue;
            }
            if (ecm_mont_path_fits(fb, n_bit_size, container_limbs)) {
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

const EcmAddSubPathDescriptor *find_addsub_dispatch_id(const EcmAddSubPathDescriptor *registry,
                                                       size_t count, int dispatch_id) {
    for (size_t i = 0; i < count; ++i) {
        if (registry[i].cl_dispatch_id == dispatch_id) {
            return &registry[i];
        }
    }
    return nullptr;
}

const EcmAddSubPathDescriptor *resolve_addsub_side(const EcmAddSubPathDescriptor *registry,
                                                 size_t count, const char *path,
                                                 const EcmPathContext &ctx, int default_dispatch_id) {
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
    return find_addsub_dispatch_id(registry, count, default_dispatch_id);
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

uint8_t mont_cl_dispatch_for_plan(const EcmMontPathDescriptor *desc, uint32_t plan_limbs) {
    if (!ecm_mont_path_is_4096_dedicated(desc)) {
        return 0;
    }
    const uint32_t operator_limbs = ecm_mont_operator_limbs(desc);
    if (operator_limbs == 0u || plan_limbs != operator_limbs) {
        return 0;
    }
    return desc->cl_dispatch_id;
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

bool ecm_mont_path_is_i24(const EcmMontPathDescriptor *desc) {
    return desc != nullptr && desc->id != nullptr && std::strncmp(desc->id, "i24", 3) == 0;
}

bool ecm_mont_path_fits(const EcmMontPathDescriptor *desc, size_t n_bit_size,
                        uint32_t container_limbs) {
    if (desc == nullptr) {
        return false;
    }
    if (!ecm_path_n_bit_fits(desc->min_n_bits, desc->max_n_bits, desc->max_n_strict, n_bit_size)) {
        return false;
    }
    if (container_limbs == 0u) {
        return true;
    }
    const uint32_t container_bits = container_limbs * 32u;
    if (desc->dedicated && desc->max_n_bits > 0) {
        return container_bits >= desc->max_n_bits;
    }
    const size_t need_bits = n_bit_size + ECM_STAGE1_MONT_CARRY_BITS;
    return need_bits <= static_cast<size_t>(container_bits);
}

bool ecm_addsub_path_fits(const EcmAddSubPathDescriptor *desc, const EcmPathContext &ctx) {
    if (desc == nullptr) {
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

const char *ecm_kernel_include_path(EcmKernelInclude include_bit) {
    switch (include_bit) {
    case ECM_KERNEL_INC_MONT_EXTENDED:
        return "cgbn/backends/opencl/kernels/ecm_stage1_mont4096_paths.cl";
    case ECM_KERNEL_INC_MP_ASM_U32:
        return "cgbn/backends/opencl/kernels/mp_addsub/stage1/asm_block32_stage1.cl";
    case ECM_KERNEL_INC_MP_ASM_U16:
        return "cgbn/backends/opencl/kernels/mp_addsub/stage1/asm_block16_stage1.cl";
    default:
        return nullptr;
    }
}

uint32_t opencl_ecm_stage1_collect_kernel_includes(const EcmStage1KernelBuildPlan &plan) {
    uint32_t mask = 0;
    if (plan.mul != nullptr) {
        mask |= plan.mul->kernel_includes;
    }
    if (plan.sqr != nullptr) {
        mask |= plan.sqr->kernel_includes;
    }
    if (plan.add != nullptr) {
        mask |= plan.add->kernel_includes;
    }
    if (plan.sub != nullptr) {
        mask |= plan.sub->kernel_includes;
    }
    return mask;
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

const EcmMontPathDescriptor *opencl_ecm_resolve_mont_mul(const char *path, size_t n_bit_size,
                                                         uint32_t container_limbs,
                                                         bool *unknown_path) {
    return resolve_mont_side(kMontMulRegistry, opencl_ecm_mont_mul_registry_count(), path,
                             n_bit_size, container_limbs, unknown_path);
}

const EcmMontPathDescriptor *opencl_ecm_resolve_mont_sqr(const char *path, size_t n_bit_size,
                                                         uint32_t container_limbs,
                                                         bool *unknown_path) {
    return resolve_mont_side(kMontSqrRegistry, opencl_ecm_mont_sqr_registry_count(), path,
                             n_bit_size, container_limbs, unknown_path);
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

const EcmAddSubPathDescriptor *opencl_ecm_addmod_path_descriptor(int path_id) {
    return find_addsub_dispatch_id(kAddModRegistry, opencl_ecm_addmod_registry_count(), path_id);
}

const EcmAddSubPathDescriptor *opencl_ecm_resolve_addmod_path(const char *path,
                                                              const EcmPathContext &ctx) {
    return resolve_addsub_side(kAddModRegistry, opencl_ecm_addmod_registry_count(), path, ctx,
                               ECM_ADDSUB_PATH_FUSED_UNROLL);
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

const EcmAddSubPathDescriptor *opencl_ecm_submod_path_descriptor(int path_id) {
    return find_addsub_dispatch_id(kSubModRegistry, opencl_ecm_submod_registry_count(), path_id);
}

const EcmAddSubPathDescriptor *opencl_ecm_resolve_submod_path(const char *path,
                                                              const EcmPathContext &ctx) {
    return resolve_addsub_side(kSubModRegistry, opencl_ecm_submod_registry_count(), path, ctx,
                               ECM_ADDSUB_PATH_FUSED_UNROLL);
}

EcmStage1KernelBuildPlan opencl_ecm_stage1_make_build_plan(
    uint32_t limbs, uint32_t tpi, const EcmMontPathDescriptor *mul,
    const EcmMontPathDescriptor *sqr, const EcmAddSubPathDescriptor *add,
    const EcmAddSubPathDescriptor *sub, bool use_i24, int stage1_force_normalize,
    int add_mod_fused_unroll) {
    EcmStage1KernelBuildPlan plan{};
    plan.limbs = limbs;
    plan.tpi = tpi;
    plan.stage1_force_normalize = stage1_force_normalize;
    plan.add_mod_fused_unroll = add_mod_fused_unroll;
    plan.mul = mul;
    plan.sqr = sqr;
    plan.add = add;
    plan.sub = sub;
    plan.use_i24 = use_i24;
    return plan;
}

bool opencl_ecm_stage1_plan_use_i24_blsub(const EcmStage1KernelBuildPlan &plan) {
    if (!plan.use_i24) {
        return false;
    }
    const auto uses_blsub = [](const EcmMontPathDescriptor *desc) {
        return desc != nullptr && desc->id != nullptr && std::strstr(desc->id, "blsub") != nullptr;
    };
    return uses_blsub(plan.mul) || uses_blsub(plan.sqr);
}

bool opencl_ecm_stage1_build_plan_equal(const EcmStage1KernelBuildPlan &a,
                                        const EcmStage1KernelBuildPlan &b) {
    return a.limbs == b.limbs && a.tpi == b.tpi &&
           a.stage1_force_normalize == b.stage1_force_normalize &&
           a.add_mod_fused_unroll == b.add_mod_fused_unroll && a.mul == b.mul && a.sqr == b.sqr &&
           a.add == b.add && a.sub == b.sub && a.use_i24 == b.use_i24;
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
    append_define(opts, "ECM_STAGE1_KERNEL_REV", 7);

    if (plan.use_i24) {
        append_define(opts, "ECM_STAGE1_USE_I24_384", 1);
        if (opencl_ecm_stage1_plan_use_i24_blsub(plan)) {
            append_define(opts, "ECM_STAGE1_I24_U32_BLSUB", 1);
        }
        append_define(opts, "MP_LIMB_BITS", 24);
    } else {
        append_define(opts, "MP_LIMB_BITS", 32);
        if (plan.mul != nullptr) {
            append_define(opts, plan.mul->force_macro, 1);
        }
        if (plan.sqr != nullptr) {
            append_define(opts, plan.sqr->force_macro, 1);
        }
    }

    append_define(opts, "ECM_STAGE1_MUL_PATH", mont_cl_dispatch_for_plan(plan.mul, plan.limbs));
    append_define(opts, "ECM_STAGE1_SQR_PATH", mont_cl_dispatch_for_plan(plan.sqr, plan.limbs));

    int coop_wg = 1;
    int coop_scratch = 0;
    if (plan.limbs == kContainer4096Limbs) {
        if (plan.mul != nullptr && ecm_mont_path_is_4096_dedicated(plan.mul)) {
            coop_wg = std::max(coop_wg, static_cast<int>(plan.mul->coop_work_group_size));
            coop_scratch = std::max(coop_scratch, static_cast<int>(plan.mul->local_scratch_u32));
        }
        if (plan.sqr != nullptr && ecm_mont_path_is_4096_dedicated(plan.sqr)) {
            coop_wg = std::max(coop_wg, static_cast<int>(plan.sqr->coop_work_group_size));
            coop_scratch = std::max(coop_scratch, static_cast<int>(plan.sqr->local_scratch_u32));
        }
    }
    const uint32_t kernel_includes = opencl_ecm_stage1_collect_kernel_includes(plan);
    append_define(opts, "ECM_STAGE1_COOP_WG", coop_wg);
    append_define(opts, "ECM_STAGE1_COOP_SCRATCH_U32", coop_scratch);
    append_define(opts, "ECM_STAGE1_HAS_FIPS4096",
                  (kernel_includes & ECM_KERNEL_INC_MONT_EXTENDED) != 0 ? 1 : 0);

    if (plan.add != nullptr) {
        append_define(opts, "ECM_STAGE1_ADDMOD_PATH", plan.add->cl_dispatch_id);
    }
    if (plan.sub != nullptr) {
        append_define(opts, "ECM_STAGE1_SUBMOD_PATH", plan.sub->cl_dispatch_id);
    }
    if ((kernel_includes & ECM_KERNEL_INC_MP_ASM_U32) != 0) {
        append_define(opts, "ECM_STAGE1_ASM_B32", 1);
    }
    if ((kernel_includes & ECM_KERNEL_INC_MP_ASM_U16) != 0) {
        append_define(opts, "ECM_STAGE1_ASM_B16", 1);
    }

    return opts;
}
