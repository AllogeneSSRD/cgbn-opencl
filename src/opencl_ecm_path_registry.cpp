#include "opencl_ecm_path_registry.h"

#include "opencl_ecm_addsub_path.h"
#include "opencl_ecm_log.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
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
constexpr uint16_t kMontContainer512Limbs = 16;

bool mont_path_is_4096_auto_priv_opt(size_t n_bit_size) {
    return ecm_path_n_bit_fits(kMont4096MinN, kMont4096MaxN, false, n_bit_size);
}

/* --- mul aliases --- */
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
    {ECM_MONT_PATH_STAGE1, ECM_STAGE1_MONT_UNROLL384, "unroll_only_384", "Unroll 384 mul",
     "mont_mul_priv_unroll_only_384", true, 10, kMulAliases_unroll384, kMontNoMinN,
     kMontUnroll384MaxN, true, kMontContainer512Limbs, 1, 0, false,
     "ECM_STAGE1_MUL_FORCE_UNROLL384", false, false},
    {ECM_MONT_PATH_STAGE1, ECM_STAGE1_MONT_UNROLL512, "unroll_only_512", "Unroll 512 mul",
     "mont_mul_priv_unroll_only_512", true, 20, kMulAliases_unroll512, kMontNoMinN,
     kMontUnroll512MaxN, false, 0, 1, 0, false, nullptr, false, false},
    {ECM_MONT_PATH_4096, ECM_MONT4096_PATH_UNROLL64, "unroll64_4096", "Unroll64 4096 mul",
     "mont_mul_stage1_unroll64_4096", true, 21, kMulAliases_unroll64_4096, kMont4096MinN,
     kMont4096MaxN, false, 0, 1, 0, false, nullptr, false, false},
    {ECM_MONT_PATH_4096, ECM_MONT4096_PATH_UNROLL64_MT2, "unroll64_4096_mt2", "Unroll64 4096 MT2 mul",
     "mont_mul_stage1_unroll64_4096_mt2", true, 22, kMulAliases_unroll64_4096_mt2, kMont4096MinN,
     kMont4096MaxN, false, 0, 2, 389, false, nullptr, false, false},
    {ECM_MONT_PATH_4096, ECM_MONT4096_PATH_FIPS4096, "fips4096", "FIPS4096 mul",
     "mont_mul_stage1_fips4096", true, 23, kMulAliases_fips4096, kMont4096MinN, kMont4096MaxN,
     false, 0, 1, 0, true, nullptr, false, false},
    {ECM_MONT_PATH_4096, ECM_MONT4096_PATH_FIPS4096_MT8, "fips4096_mt8", "FIPS4096 MT8 mul",
     "mont_mul_stage1_fips4096_mt8", true, 24, kMulAliases_fips4096_mt8, kMont4096MinN,
     kMont4096MaxN, false, 0, 8, 897, true, nullptr, false, false},
    {ECM_MONT_PATH_4096, ECM_MONT4096_PATH_FIPS4096_MT16, "fips4096_mt16", "FIPS4096 MT16 mul",
     "mont_mul_stage1_fips4096_mt16", true, 25, kMulAliases_fips4096_mt16, kMont4096MinN,
     kMont4096MaxN, false, 0, 16, 897, true, nullptr, false, false},
    {ECM_MONT_PATH_STAGE1, ECM_STAGE1_MONT_UNROLL32, "unroll32", "Generic unroll32 mul",
     "mont_mul_stage1_unroll32", false, -1, kMulAliases_unroll32, kMontNoMinN, kMontNoMaxN, false,
     0, 1, 0, false, "ECM_STAGE1_MUL_FORCE_UNROLL32", false, false},
    {ECM_MONT_PATH_STAGE1, ECM_STAGE1_MONT_PRIV_OPT, "priv_opt", "Private opt mul",
     "mont_mul_stage1_priv_opt", false, 30, kMulAliases_priv_opt, kMontNoMinN, kMontNoMaxN, false,
     0, 1, 0, false, "ECM_STAGE1_MUL_FORCE_PRIV_OPT", false, false},
    {ECM_MONT_PATH_STAGE1, ECM_STAGE1_MONT_I24_U32_BLSUB, "i24_u32_blsub", "i24 blsub mul",
     "mont_mul_unroll_i24_u32_blsub", true, -1, kMulAliases_i24_blsub, kMontNoMinN, kMontNoMaxN,
     false, 0, 1, 0, false, nullptr, true, true},
    {ECM_MONT_PATH_STAGE1, ECM_STAGE1_MONT_I24_U32, "i24_u32", "i24 u32 mul",
     "mont_mul_unroll_i24_u32", true, -1, kMulAliases_i24_u32, kMontNoMinN, kMontNoMaxN, false, 0,
     1, 0, false, nullptr, true, false},
};

/* --- sqr aliases --- */
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
    {ECM_MONT_PATH_STAGE1, ECM_STAGE1_MONT_UNROLL384, "unroll_only_384", "Unroll 384 sqr",
     "mont_sqr_priv_unroll_only_384", true, 10, kSqrAliases_unroll384, kMontNoMinN,
     kMontUnroll384MaxN, true, kMontContainer512Limbs, 1, 0, false,
     "ECM_STAGE1_SQR_FORCE_UNROLL384", false, false},
    {ECM_MONT_PATH_STAGE1, ECM_STAGE1_MONT_UNROLL512, "unroll_only_512", "Unroll 512 sqr",
     "mont_sqr_priv_unroll_only_512", true, 20, kSqrAliases_unroll512, kMontNoMinN,
     kMontUnroll512MaxN, false, 0, 1, 0, false, nullptr, false, false},
    {ECM_MONT_PATH_4096, ECM_MONT4096_PATH_UNROLL64, "unroll64_4096", "Unroll64 4096 sqr",
     "mont_sqr_stage1_unroll64_4096", true, 21, kSqrAliases_unroll64_4096, kMont4096MinN,
     kMont4096MaxN, false, 0, 1, 0, false, nullptr, false, false},
    {ECM_MONT_PATH_4096, ECM_MONT4096_PATH_UNROLL64_MT2, "unroll64_4096_mt2", "Unroll64 4096 MT2 sqr",
     "mont_sqr_stage1_unroll64_4096_mt2", true, 22, kSqrAliases_unroll64_4096_mt2, kMont4096MinN,
     kMont4096MaxN, false, 0, 2, 389, false, nullptr, false, false},
    {ECM_MONT_PATH_4096, ECM_MONT4096_PATH_FIPS4096, "fips4096", "FIPS4096 sqr",
     "mont_sqr_stage1_fips4096", true, 23, kSqrAliases_fips4096, kMont4096MinN, kMont4096MaxN,
     false, 0, 1, 0, true, nullptr, false, false},
    {ECM_MONT_PATH_4096, ECM_MONT4096_PATH_FIPS4096_MT8, "fips4096_mt8", "FIPS4096 MT8 sqr",
     "mont_sqr_stage1_fips4096_mt8", true, 24, kSqrAliases_fips4096_mt8, kMont4096MinN,
     kMont4096MaxN, false, 0, 8, 897, true, nullptr, false, false},
    {ECM_MONT_PATH_4096, ECM_MONT4096_PATH_FIPS4096_MT16, "fips4096_mt16", "FIPS4096 MT16 sqr",
     "mont_sqr_stage1_fips4096_mt16", true, 25, kSqrAliases_fips4096_mt16, kMont4096MinN,
     kMont4096MaxN, false, 0, 16, 897, true, nullptr, false, false},
    {ECM_MONT_PATH_STAGE1, ECM_STAGE1_MONT_UNROLL32, "unroll32", "Generic unroll32 sqr",
     "mont_sqr_stage1_unroll32", false, -1, kSqrAliases_unroll32, kMontNoMinN, kMontNoMaxN, false,
     0, 1, 0, false, "ECM_STAGE1_SQR_FORCE_UNROLL32", false, false},
    {ECM_MONT_PATH_STAGE1, ECM_STAGE1_MONT_PRIV_OPT, "priv_opt", "Private opt sqr",
     "mont_sqr_stage1_priv_opt", false, 30, kSqrAliases_priv_opt, kMontNoMinN, kMontNoMaxN, false,
     0, 1, 0, false, "ECM_STAGE1_SQR_FORCE_PRIV_OPT", false, false},
    {ECM_MONT_PATH_STAGE1, ECM_STAGE1_MONT_I24_U32_BLSUB, "i24_u32_blsub", "i24 blsub sqr",
     "mont_sqr_unroll_i24_u32_blsub", true, -1, kSqrAliases_i24_blsub, kMontNoMinN, kMontNoMaxN,
     false, 0, 1, 0, false, nullptr, true, true},
    {ECM_MONT_PATH_STAGE1, ECM_STAGE1_MONT_I24_U32, "i24_u32", "i24 u32 sqr",
     "mont_sqr_unroll_i24_u32", true, -1, kSqrAliases_i24_u32, kMontNoMinN, kMontNoMaxN, false, 0,
     1, 0, false, nullptr, true, false},
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
static const char *const kSubAliases_asm_b32[] = {"asm_b32", nullptr};
static const char *const kSubAliases_fused_unroll_b16[] = {"fused_unroll_b16", "fused_unroll_auto",
                                                           nullptr};

constexpr EcmAddSubPathDescriptor kAddModRegistry[] = {
    {ECM_ADDSUB_PATH_ASM_B32, "asm_b32", "ASM b32 add (4096 AMD)", "asm_b32", 10, kAddAliases_asm_b32,
     128u, ECM_PATH_VENDOR_AMD, true, false},
    {ECM_ADDSUB_PATH_FUSED_UNROLL_B32, "fused_unroll_b32", "Fused unroll b32 add (4096)",
     "fused_unroll_b32", 11, kAddAliases_fused_unroll_b32, 128u, ECM_PATH_VENDOR_NON_AMD, false,
     false},
    {ECM_ADDSUB_PATH_ASM_B16, "asm_b16", "ASM b16 add (512 AMD)", "asm_b16", 20, kAddAliases_asm_b16,
     16u, ECM_PATH_VENDOR_AMD, false, true},
    {ECM_ADDSUB_PATH_FUSED, "fused", "Fused add (512 Adreno)", "fused", 21, kAddAliases_fused, 16u,
     ECM_PATH_VENDOR_NON_AMD, false, false},
    {ECM_ADDSUB_PATH_FUSED_UNROLL_B16, "fused_unroll_b16", "Fused unroll b16 add (512 AMD)",
     "fused_unroll_b16", 22, kAddAliases_fused_unroll_b16, 16u, ECM_PATH_VENDOR_AMD, false, false},
    {ECM_ADDSUB_PATH_FUSED_UNROLL, "fused_unroll", "Fused unroll add (generic)", "fused_unroll", 30,
     kAddAliases_fused_unroll, 0u, ECM_PATH_VENDOR_ANY, false, false},
};

constexpr EcmAddSubPathDescriptor kSubModRegistry[] = {
    {ECM_ADDSUB_PATH_FUSED_UNROLL_B32, "fused_unroll_b32", "Fused unroll b32 sub (4096 AMD)",
     "fused_unroll_b32", 10, kSubAliases_fused_unroll_b32, 128u, ECM_PATH_VENDOR_AMD, false, false},
    {ECM_ADDSUB_PATH_FUSED_UNROLL_B32, "fused_unroll_b32", "Fused unroll b32 sub (4096)",
     "fused_unroll_b32", 11, kSubAliases_fused_unroll_b32, 128u, ECM_PATH_VENDOR_NON_AMD, false,
     false},
    {ECM_ADDSUB_PATH_FUSED, "fused", "Fused sub (512 Adreno)", "fused", 20, kSubAliases_fused, 16u,
     ECM_PATH_VENDOR_NON_AMD, false, false},
    {ECM_ADDSUB_PATH_FUSED_UNROLL_B16, "fused_unroll_b16", "Fused unroll b16 sub (512 AMD)",
     "fused_unroll_b16", 21, kSubAliases_fused_unroll_b16, 16u, ECM_PATH_VENDOR_AMD, false, false},
    {ECM_ADDSUB_PATH_FUSED_UNROLL, "fused_unroll", "Fused unroll sub (generic)", "fused_unroll", 30,
     kSubAliases_fused_unroll, 0u, ECM_PATH_VENDOR_ANY, false, false},
};

std::vector<const EcmMontPathDescriptor *> auto_sorted_stage1(const EcmMontPathDescriptor *registry,
                                                              size_t count) {
    std::vector<const EcmMontPathDescriptor *> out;
    for (size_t i = 0; i < count; ++i) {
        if (registry[i].kind != ECM_MONT_PATH_STAGE1 || registry[i].auto_priority < 0) {
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

std::vector<const EcmMontPathDescriptor *> auto_sorted_4096(const EcmMontPathDescriptor *registry,
                                                            size_t count) {
    std::vector<const EcmMontPathDescriptor *> out;
    for (size_t i = 0; i < count; ++i) {
        if (registry[i].kind != ECM_MONT_PATH_4096 || registry[i].auto_priority < 0) {
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

const EcmMontPathDescriptor *find_4096(const EcmMontPathDescriptor *registry, size_t count,
                                       int path_id) {
    for (size_t i = 0; i < count; ++i) {
        if (registry[i].kind == ECM_MONT_PATH_4096 && registry[i].variant_id == path_id) {
            return &registry[i];
        }
    }
    return nullptr;
}

const EcmMontPathDescriptor *find_stage1(const EcmMontPathDescriptor *registry, size_t count,
                                         ecm_stage1_mont_mode mode) {
    for (size_t i = 0; i < count; ++i) {
        if (registry[i].kind == ECM_MONT_PATH_STAGE1 &&
            registry[i].variant_id == static_cast<int>(mode)) {
            return &registry[i];
        }
    }
    return nullptr;
}

const EcmMontPathDescriptor *resolve_stage1_side(const EcmMontPathDescriptor *registry,
                                                 size_t count, const char *path,
                                                 size_t n_bit_size,
                                                 const EcmMontPathDescriptor *priv_opt_fallback) {
    const EcmMontPathDescriptor *unroll512_fallback =
        find_stage1(registry, count, ECM_STAGE1_MONT_UNROLL512);
    if (opencl_ecm_path_is_auto(path)) {
        if (mont_path_is_4096_auto_priv_opt(n_bit_size)) {
            return priv_opt_fallback;
        }
        for (const EcmMontPathDescriptor *desc : auto_sorted_stage1(registry, count)) {
            if (ecm_mont_path_n_fits(desc, n_bit_size)) {
                return desc;
            }
        }
        return priv_opt_fallback;
    }
    for (size_t i = 0; i < count; ++i) {
        const EcmMontPathDescriptor &desc = registry[i];
        if (desc.kind != ECM_MONT_PATH_STAGE1) {
            continue;
        }
        if (!aliases_contain(desc.aliases, path)) {
            continue;
        }
        if (!ecm_mont_path_n_fits(&desc, n_bit_size)) {
            const int min_pri = desc.auto_priority >= 0 ? desc.auto_priority + 1 : 0;
            for (const EcmMontPathDescriptor *fb : auto_sorted_stage1(registry, count)) {
                if (fb->auto_priority < min_pri) {
                    continue;
                }
                if (ecm_mont_path_n_fits(fb, n_bit_size)) {
                    return fb;
                }
            }
            return priv_opt_fallback;
        }
        return &desc;
    }
    return unroll512_fallback != nullptr ? unroll512_fallback : priv_opt_fallback;
}

const EcmMontPathDescriptor *resolve_4096_side(const EcmMontPathDescriptor *registry, size_t count,
                                               const char *path, size_t n_bit_size,
                                               bool *unknown_path) {
    if (unknown_path != nullptr) {
        *unknown_path = false;
    }
    const EcmMontPathDescriptor *default_unroll64 =
        find_4096(registry, count, ECM_MONT4096_PATH_UNROLL64);
    if (opencl_ecm_path_is_auto(path)) {
        for (const EcmMontPathDescriptor *desc : auto_sorted_4096(registry, count)) {
            if (!desc->dedicated) {
                continue;
            }
            if (ecm_mont_path_n_fits(desc, n_bit_size)) {
                return desc;
            }
        }
        return default_unroll64;
    }
    for (size_t i = 0; i < count; ++i) {
        const EcmMontPathDescriptor &stage1_desc = registry[i];
        if (stage1_desc.kind == ECM_MONT_PATH_STAGE1 &&
            aliases_contain(stage1_desc.aliases, path)) {
            return nullptr;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        const EcmMontPathDescriptor &desc = registry[i];
        if (desc.kind != ECM_MONT_PATH_4096) {
            continue;
        }
        if (!aliases_contain(desc.aliases, path)) {
            continue;
        }
        if (!ecm_mont_path_n_fits(&desc, n_bit_size)) {
            const int min_pri = desc.auto_priority >= 0 ? desc.auto_priority + 1 : 21;
            for (const EcmMontPathDescriptor *fb : auto_sorted_4096(registry, count)) {
                if (fb->auto_priority < min_pri) {
                    continue;
                }
                if (ecm_mont_path_n_fits(fb, n_bit_size)) {
                    return fb;
                }
            }
            return default_unroll64;
        }
        return &desc;
    }
    if (unknown_path != nullptr) {
        *unknown_path = true;
    }
    return nullptr;
}

const EcmAddSubPathDescriptor *find_addsub_path_id(const EcmAddSubPathDescriptor *registry,
                                                   size_t count, int path_id) {
    for (size_t i = 0; i < count; ++i) {
        if (registry[i].path_id == path_id) {
            return &registry[i];
        }
    }
    return nullptr;
}

const EcmAddSubPathDescriptor *resolve_addsub_side(const EcmAddSubPathDescriptor *registry,
                                                   size_t count, const char *path,
                                                   const EcmAddSubPathContext &ctx,
                                                   int default_path_id) {
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
    return find_addsub_path_id(registry, count, default_path_id);
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

bool ecm_mont_path_n_fits(const EcmMontPathDescriptor *desc, size_t n_bit_size) {
    if (desc == nullptr) {
        return false;
    }
    return ecm_path_n_bit_fits(desc->min_n_bits, desc->max_n_bits, desc->max_n_strict, n_bit_size);
}

bool ecm_mont_path_container_fits(const EcmMontPathDescriptor *desc, uint32_t limbs,
                                  size_t n_bit_size) {
    if (!ecm_mont_path_n_fits(desc, n_bit_size)) {
        return false;
    }
    return desc->required_container_limbs == 0 || limbs == desc->required_container_limbs;
}

bool ecm_addsub_path_fits(const EcmAddSubPathDescriptor *desc, const EcmAddSubPathContext &ctx) {
    if (desc == nullptr) {
        return false;
    }
    if (desc->required_limbs != 0 && ctx.limbs != desc->required_limbs) {
        return false;
    }
    if (desc->vendor == ECM_PATH_VENDOR_AMD && !ctx.is_amd) {
        return false;
    }
    if (desc->vendor == ECM_PATH_VENDOR_NON_AMD && ctx.is_amd) {
        return false;
    }
    return true;
}

bool opencl_ecm_path_is_auto(const char *path) {
    return path == nullptr || path[0] == '\0' || strcmp(path, "auto") == 0 ||
           strcmp(path, "default") == 0;
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
    return find_stage1(kMontMulRegistry, opencl_ecm_mont_mul_registry_count(), mode);
}

const EcmMontPathDescriptor *opencl_ecm_mont4096_mul_descriptor(int path_id) {
    return find_4096(kMontMulRegistry, opencl_ecm_mont_mul_registry_count(), path_id);
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
    return find_stage1(kMontSqrRegistry, opencl_ecm_mont_sqr_registry_count(), mode);
}

const EcmMontPathDescriptor *opencl_ecm_mont4096_sqr_descriptor(int path_id) {
    return find_4096(kMontSqrRegistry, opencl_ecm_mont_sqr_registry_count(), path_id);
}

int ecm_mont_4096_path_id(const EcmMontPathDescriptor *desc) {
    return (desc != nullptr && desc->kind == ECM_MONT_PATH_4096) ? desc->variant_id : 0;
}

const EcmMontPathDescriptor *opencl_ecm_resolve_stage1_mont_mul(const char *path,
                                                                size_t n_bit_size) {
    return resolve_stage1_side(kMontMulRegistry, opencl_ecm_mont_mul_registry_count(), path,
                               n_bit_size,
                               opencl_ecm_mont_mul_descriptor(ECM_STAGE1_MONT_PRIV_OPT));
}

const EcmMontPathDescriptor *opencl_ecm_resolve_stage1_mont_sqr(const char *path,
                                                              size_t n_bit_size) {
    return resolve_stage1_side(kMontSqrRegistry, opencl_ecm_mont_sqr_registry_count(), path,
                               n_bit_size,
                               opencl_ecm_mont_sqr_descriptor(ECM_STAGE1_MONT_PRIV_OPT));
}

const EcmMontPathDescriptor *opencl_ecm_resolve_mont4096_mul(const char *path, size_t n_bit_size,
                                                             bool *unknown_path) {
    return resolve_4096_side(kMontMulRegistry, opencl_ecm_mont_mul_registry_count(), path,
                             n_bit_size, unknown_path);
}

const EcmMontPathDescriptor *opencl_ecm_resolve_mont4096_sqr(const char *path, size_t n_bit_size,
                                                           bool *unknown_path) {
    return resolve_4096_side(kMontSqrRegistry, opencl_ecm_mont_sqr_registry_count(), path,
                             n_bit_size, unknown_path);
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
    return find_addsub_path_id(kAddModRegistry, opencl_ecm_addmod_registry_count(), path_id);
}

const EcmAddSubPathDescriptor *opencl_ecm_resolve_addmod_path(const char *path, uint32_t limbs,
                                                              bool is_amd) {
    const EcmAddSubPathContext ctx{limbs, is_amd};
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
    return find_addsub_path_id(kSubModRegistry, opencl_ecm_submod_registry_count(), path_id);
}

const EcmAddSubPathDescriptor *opencl_ecm_resolve_submod_path(const char *path, uint32_t limbs,
                                                              bool is_amd) {
    const EcmAddSubPathContext ctx{limbs, is_amd};
    return resolve_addsub_side(kSubModRegistry, opencl_ecm_submod_registry_count(), path, ctx,
                               ECM_ADDSUB_PATH_FUSED_UNROLL);
}

EcmStage1KernelBuildPlan opencl_ecm_stage1_make_build_plan(
    uint32_t limbs, uint32_t tpi, const EcmMontPathDescriptor *mul,
    const EcmMontPathDescriptor *sqr, const EcmMontPathDescriptor *mul_4096,
    const EcmMontPathDescriptor *sqr_4096, const EcmAddSubPathDescriptor *add,
    const EcmAddSubPathDescriptor *sub, bool use_i24, int stage1_force_normalize,
    int add_mod_fused_unroll) {
    EcmStage1KernelBuildPlan plan{};
    plan.limbs = limbs;
    plan.tpi = tpi;
    plan.stage1_force_normalize = stage1_force_normalize;
    plan.add_mod_fused_unroll = add_mod_fused_unroll;
    plan.mul = mul;
    plan.sqr = sqr;
    plan.mul_4096 = mul_4096;
    plan.sqr_4096 = sqr_4096;
    plan.add = add;
    plan.sub = sub;
    plan.use_i24 = use_i24;
    return plan;
}

bool opencl_ecm_stage1_plan_use_i24_blsub(const EcmStage1KernelBuildPlan &plan) {
    if (!plan.use_i24) {
        return false;
    }
    return (plan.mul != nullptr && plan.mul->stage1_i24_blsub) ||
           (plan.sqr != nullptr && plan.sqr->stage1_i24_blsub);
}

bool opencl_ecm_stage1_build_plan_equal(const EcmStage1KernelBuildPlan &a,
                                        const EcmStage1KernelBuildPlan &b) {
    return a.limbs == b.limbs && a.tpi == b.tpi &&
           a.stage1_force_normalize == b.stage1_force_normalize &&
           a.add_mod_fused_unroll == b.add_mod_fused_unroll && a.mul == b.mul && a.sqr == b.sqr &&
           a.mul_4096 == b.mul_4096 && a.sqr_4096 == b.sqr_4096 &&
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
    append_define(opts, "ECM_STAGE1_KERNEL_REV", 6);

    if (plan.use_i24) {
        append_define(opts, "ECM_STAGE1_USE_I24_384", 1);
        if (opencl_ecm_stage1_plan_use_i24_blsub(plan)) {
            append_define(opts, "ECM_STAGE1_I24_U32_BLSUB", 1);
        }
        append_define(opts, "MP_LIMB_BITS", 24);
    } else {
        append_define(opts, "MP_LIMB_BITS", 32);
        if (plan.mul != nullptr) {
            append_define(opts, plan.mul->stage1_force_macro, 1);
        }
        if (plan.sqr != nullptr) {
            append_define(opts, plan.sqr->stage1_force_macro, 1);
        }
    }

    append_define(opts, "ECM_STAGE1_MUL_PATH", ecm_mont_4096_path_id(plan.mul_4096));
    append_define(opts, "ECM_STAGE1_SQR_PATH", ecm_mont_4096_path_id(plan.sqr_4096));

    int coop_wg = 1;
    int coop_scratch = 0;
    bool has_fips4096 = false;
    if (plan.limbs == 128u) {
        if (plan.mul_4096 != nullptr) {
            coop_wg = std::max(coop_wg, plan.mul_4096->coop_wg_size);
            coop_scratch = std::max(coop_scratch, plan.mul_4096->coop_scratch_u32);
            has_fips4096 = has_fips4096 || plan.mul_4096->needs_fips4096_cl;
        }
        if (plan.sqr_4096 != nullptr) {
            coop_wg = std::max(coop_wg, plan.sqr_4096->coop_wg_size);
            coop_scratch = std::max(coop_scratch, plan.sqr_4096->coop_scratch_u32);
            has_fips4096 = has_fips4096 || plan.sqr_4096->needs_fips4096_cl;
        }
    }
    append_define(opts, "ECM_STAGE1_COOP_WG", coop_wg);
    append_define(opts, "ECM_STAGE1_COOP_SCRATCH_U32", coop_scratch);
    append_define(opts, "ECM_STAGE1_HAS_FIPS4096", has_fips4096 ? 1 : 0);

    if (plan.add != nullptr) {
        append_define(opts, "ECM_STAGE1_ADDMOD_PATH", plan.add->path_id);
    }
    if (plan.sub != nullptr) {
        append_define(opts, "ECM_STAGE1_SUBMOD_PATH", plan.sub->path_id);
    }

    const bool needs_asm_b32 =
        (plan.add != nullptr && plan.add->needs_asm_b32) ||
        (plan.sub != nullptr && plan.sub->needs_asm_b32);
    const bool needs_asm_b16 = plan.add != nullptr && plan.add->needs_asm_b16;
    if (needs_asm_b32) {
        append_define(opts, "ECM_STAGE1_ASM_B32", 1);
    }
    if (needs_asm_b16) {
        append_define(opts, "ECM_STAGE1_ASM_B16", 1);
    }

    return opts;
}
