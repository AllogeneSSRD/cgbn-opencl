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

EcmPathContext make_n_ctx(size_t n_bit_size) {
    EcmPathContext ctx{};
    ctx.n_bit_size = n_bit_size;
    return ctx;
}

bool n_fits_unroll384_ctx(const EcmPathContext &ctx) {
    return ecm_path_n_fits_unroll384(ctx.n_bit_size);
}

bool n_fits_unroll512_ctx(const EcmPathContext &ctx) {
    return ecm_path_n_fits_unroll512_container(ctx.n_bit_size);
}

bool n_fits_4096_dedicated_ctx(const EcmPathContext &ctx) {
    return ecm_path_n_fits_4096_dedicated(ctx.n_bit_size);
}

bool n_fits_always_ctx(const EcmPathContext & /*ctx*/) {
    return true;
}

bool addmod_limbs_128_amd(const EcmAddSubPathContext &ctx) {
    return ctx.limbs == 128u && ctx.is_amd;
}

bool addmod_limbs_128_non_amd(const EcmAddSubPathContext &ctx) {
    return ctx.limbs == 128u && !ctx.is_amd;
}

bool addmod_limbs_16_amd(const EcmAddSubPathContext &ctx) {
    return ctx.limbs == 16u && ctx.is_amd;
}

bool addmod_limbs_16_non_amd(const EcmAddSubPathContext &ctx) {
    return ctx.limbs == 16u && !ctx.is_amd;
}

bool addmod_limbs_always(const EcmAddSubPathContext & /*ctx*/) {
    return true;
}

bool submod_limbs_128_amd(const EcmAddSubPathContext &ctx) {
    return ctx.limbs == 128u && ctx.is_amd;
}

bool submod_limbs_128_non_amd(const EcmAddSubPathContext &ctx) {
    return ctx.limbs == 128u && !ctx.is_amd;
}

bool submod_limbs_16_amd(const EcmAddSubPathContext &ctx) {
    return ctx.limbs == 16u && ctx.is_amd;
}

bool submod_limbs_16_non_amd(const EcmAddSubPathContext &ctx) {
    return ctx.limbs == 16u && !ctx.is_amd;
}

bool submod_limbs_always(const EcmAddSubPathContext & /*ctx*/) {
    return true;
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

constexpr EcmMontMulPathDescriptor kMontMulRegistry[] = {
    {ECM_MONT_PATH_STAGE1, ECM_STAGE1_MONT_UNROLL384, "unroll_only_384", "Unroll 384 mul",
     "mont_mul_priv_unroll_only_384", true, 10, kMulAliases_unroll384, n_fits_unroll384_ctx, 1, 0,
     false, "ECM_STAGE1_MUL_FORCE_UNROLL384", false, false},
    {ECM_MONT_PATH_STAGE1, ECM_STAGE1_MONT_UNROLL512, "unroll_only_512", "Unroll 512 mul",
     "mont_mul_priv_unroll_only_512", true, 20, kMulAliases_unroll512, n_fits_unroll512_ctx, 1, 0,
     false, nullptr, false, false},
    {ECM_MONT_PATH_4096, ECM_MONT4096_PATH_UNROLL64, "unroll64_4096", "Unroll64 4096 mul",
     "mont_mul_stage1_unroll64_4096", true, 21, kMulAliases_unroll64_4096, n_fits_4096_dedicated_ctx,
     1, 0, false, nullptr, false, false},
    {ECM_MONT_PATH_4096, ECM_MONT4096_PATH_UNROLL64_MT2, "unroll64_4096_mt2", "Unroll64 4096 MT2 mul",
     "mont_mul_stage1_unroll64_4096_mt2", true, 22, kMulAliases_unroll64_4096_mt2,
     n_fits_4096_dedicated_ctx, 2, 389, false, nullptr, false, false},
    {ECM_MONT_PATH_4096, ECM_MONT4096_PATH_FIPS4096, "fips4096", "FIPS4096 mul",
     "mont_mul_stage1_fips4096", true, 23, kMulAliases_fips4096, n_fits_4096_dedicated_ctx, 1, 0,
     true, nullptr, false, false},
    {ECM_MONT_PATH_4096, ECM_MONT4096_PATH_FIPS4096_MT8, "fips4096_mt8", "FIPS4096 MT8 mul",
     "mont_mul_stage1_fips4096_mt8", true, 24, kMulAliases_fips4096_mt8, n_fits_4096_dedicated_ctx,
     8, 897, true, nullptr, false, false},
    {ECM_MONT_PATH_4096, ECM_MONT4096_PATH_FIPS4096_MT16, "fips4096_mt16", "FIPS4096 MT16 mul",
     "mont_mul_stage1_fips4096_mt16", true, 25, kMulAliases_fips4096_mt16,
     n_fits_4096_dedicated_ctx, 16, 897, true, nullptr, false, false},
    {ECM_MONT_PATH_STAGE1, ECM_STAGE1_MONT_UNROLL32, "unroll32", "Generic unroll32 mul",
     "mont_mul_stage1_unroll32", false, -1, kMulAliases_unroll32, n_fits_always_ctx, 1, 0, false,
     "ECM_STAGE1_MUL_FORCE_UNROLL32", false, false},
    {ECM_MONT_PATH_STAGE1, ECM_STAGE1_MONT_PRIV_OPT, "priv_opt", "Private opt mul",
     "mont_mul_stage1_priv_opt", false, 30, kMulAliases_priv_opt, n_fits_always_ctx, 1, 0, false,
     "ECM_STAGE1_MUL_FORCE_PRIV_OPT", false, false},
    {ECM_MONT_PATH_STAGE1, ECM_STAGE1_MONT_I24_U32_BLSUB, "i24_u32_blsub", "i24 blsub mul",
     "mont_mul_unroll_i24_u32_blsub", true, -1, kMulAliases_i24_blsub, n_fits_always_ctx, 1, 0,
     false, nullptr, true, true},
    {ECM_MONT_PATH_STAGE1, ECM_STAGE1_MONT_I24_U32, "i24_u32", "i24 u32 mul",
     "mont_mul_unroll_i24_u32", true, -1, kMulAliases_i24_u32, n_fits_always_ctx, 1, 0, false,
     nullptr, true, false},
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

constexpr EcmMontSqrPathDescriptor kMontSqrRegistry[] = {
    {ECM_MONT_PATH_STAGE1, ECM_STAGE1_MONT_UNROLL384, "unroll_only_384", "Unroll 384 sqr",
     "mont_sqr_priv_unroll_only_384", true, 10, kSqrAliases_unroll384, n_fits_unroll384_ctx, 1, 0,
     false, "ECM_STAGE1_SQR_FORCE_UNROLL384", false, false},
    {ECM_MONT_PATH_STAGE1, ECM_STAGE1_MONT_UNROLL512, "unroll_only_512", "Unroll 512 sqr",
     "mont_sqr_priv_unroll_only_512", true, 20, kSqrAliases_unroll512, n_fits_unroll512_ctx, 1, 0,
     false, nullptr, false, false},
    {ECM_MONT_PATH_4096, ECM_MONT4096_PATH_UNROLL64, "unroll64_4096", "Unroll64 4096 sqr",
     "mont_sqr_stage1_unroll64_4096", true, 21, kSqrAliases_unroll64_4096, n_fits_4096_dedicated_ctx,
     1, 0, false, nullptr, false, false},
    {ECM_MONT_PATH_4096, ECM_MONT4096_PATH_UNROLL64_MT2, "unroll64_4096_mt2", "Unroll64 4096 MT2 sqr",
     "mont_sqr_stage1_unroll64_4096_mt2", true, 22, kSqrAliases_unroll64_4096_mt2,
     n_fits_4096_dedicated_ctx, 2, 389, false, nullptr, false, false},
    {ECM_MONT_PATH_4096, ECM_MONT4096_PATH_FIPS4096, "fips4096", "FIPS4096 sqr",
     "mont_sqr_stage1_fips4096", true, 23, kSqrAliases_fips4096, n_fits_4096_dedicated_ctx, 1, 0,
     true, nullptr, false, false},
    {ECM_MONT_PATH_4096, ECM_MONT4096_PATH_FIPS4096_MT8, "fips4096_mt8", "FIPS4096 MT8 sqr",
     "mont_sqr_stage1_fips4096_mt8", true, 24, kSqrAliases_fips4096_mt8, n_fits_4096_dedicated_ctx,
     8, 897, true, nullptr, false, false},
    {ECM_MONT_PATH_4096, ECM_MONT4096_PATH_FIPS4096_MT16, "fips4096_mt16", "FIPS4096 MT16 sqr",
     "mont_sqr_stage1_fips4096_mt16", true, 25, kSqrAliases_fips4096_mt16,
     n_fits_4096_dedicated_ctx, 16, 897, true, nullptr, false, false},
    {ECM_MONT_PATH_STAGE1, ECM_STAGE1_MONT_UNROLL32, "unroll32", "Generic unroll32 sqr",
     "mont_sqr_stage1_unroll32", false, -1, kSqrAliases_unroll32, n_fits_always_ctx, 1, 0, false,
     "ECM_STAGE1_SQR_FORCE_UNROLL32", false, false},
    {ECM_MONT_PATH_STAGE1, ECM_STAGE1_MONT_PRIV_OPT, "priv_opt", "Private opt sqr",
     "mont_sqr_stage1_priv_opt", false, 30, kSqrAliases_priv_opt, n_fits_always_ctx, 1, 0, false,
     "ECM_STAGE1_SQR_FORCE_PRIV_OPT", false, false},
    {ECM_MONT_PATH_STAGE1, ECM_STAGE1_MONT_I24_U32_BLSUB, "i24_u32_blsub", "i24 blsub sqr",
     "mont_sqr_unroll_i24_u32_blsub", true, -1, kSqrAliases_i24_blsub, n_fits_always_ctx, 1, 0,
     false, nullptr, true, true},
    {ECM_MONT_PATH_STAGE1, ECM_STAGE1_MONT_I24_U32, "i24_u32", "i24 u32 sqr",
     "mont_sqr_unroll_i24_u32", true, -1, kSqrAliases_i24_u32, n_fits_always_ctx, 1, 0, false,
     nullptr, true, false},
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

constexpr EcmAddModPathDescriptor kAddModRegistry[] = {
    {ECM_ADDSUB_PATH_ASM_B32, "asm_b32", "ASM b32 add (4096 AMD)", "asm_b32", 10, kAddAliases_asm_b32,
     addmod_limbs_128_amd, true, false},
    {ECM_ADDSUB_PATH_FUSED_UNROLL_B32, "fused_unroll_b32", "Fused unroll b32 add (4096)", "fused_unroll_b32",
     11, kAddAliases_fused_unroll_b32, addmod_limbs_128_non_amd, false, false},
    {ECM_ADDSUB_PATH_ASM_B16, "asm_b16", "ASM b16 add (512 AMD)", "asm_b16", 20, kAddAliases_asm_b16,
     addmod_limbs_16_amd, false, true},
    {ECM_ADDSUB_PATH_FUSED, "fused", "Fused add (512 Adreno)", "fused", 21, kAddAliases_fused,
     addmod_limbs_16_non_amd, false, false},
    {ECM_ADDSUB_PATH_FUSED_UNROLL_B16, "fused_unroll_b16", "Fused unroll b16 add (512 AMD)",
     "fused_unroll_b16", 22, kAddAliases_fused_unroll_b16, addmod_limbs_16_amd, false, false},
    {ECM_ADDSUB_PATH_FUSED_UNROLL, "fused_unroll", "Fused unroll add (generic)", "fused_unroll", 30,
     kAddAliases_fused_unroll, addmod_limbs_always, false, false},
};

constexpr EcmSubModPathDescriptor kSubModRegistry[] = {
    {ECM_ADDSUB_PATH_FUSED_UNROLL_B32, "fused_unroll_b32", "Fused unroll b32 sub (4096 AMD)",
     "fused_unroll_b32", 10, kSubAliases_fused_unroll_b32, submod_limbs_128_amd, false, false},
    {ECM_ADDSUB_PATH_FUSED_UNROLL_B32, "fused_unroll_b32", "Fused unroll b32 sub (4096)", "fused_unroll_b32",
     11, kSubAliases_fused_unroll_b32, submod_limbs_128_non_amd, false, false},
    {ECM_ADDSUB_PATH_FUSED, "fused", "Fused sub (512 Adreno)", "fused", 20, kSubAliases_fused,
     submod_limbs_16_non_amd, false, false},
    {ECM_ADDSUB_PATH_FUSED_UNROLL_B16, "fused_unroll_b16", "Fused unroll b16 sub (512 AMD)",
     "fused_unroll_b16", 21, kSubAliases_fused_unroll_b16, submod_limbs_16_amd, false, false},
    {ECM_ADDSUB_PATH_FUSED_UNROLL, "fused_unroll", "Fused unroll sub (generic)", "fused_unroll", 30,
     kSubAliases_fused_unroll, submod_limbs_always, false, false},
};

template <typename Desc>
std::vector<const Desc *> auto_sorted_stage1(const Desc *registry, size_t count) {
    std::vector<const Desc *> out;
    for (size_t i = 0; i < count; ++i) {
        if (registry[i].kind != ECM_MONT_PATH_STAGE1 || registry[i].auto_priority < 0) {
            continue;
        }
        out.push_back(&registry[i]);
    }
    std::sort(out.begin(), out.end(), [](const Desc *a, const Desc *b) {
        return a->auto_priority < b->auto_priority;
    });
    return out;
}

template <typename Desc>
std::vector<const Desc *> auto_sorted_4096(const Desc *registry, size_t count) {
    std::vector<const Desc *> out;
    for (size_t i = 0; i < count; ++i) {
        if (registry[i].kind != ECM_MONT_PATH_4096 || registry[i].auto_priority < 0) {
            continue;
        }
        out.push_back(&registry[i]);
    }
    std::sort(out.begin(), out.end(), [](const Desc *a, const Desc *b) {
        return a->auto_priority < b->auto_priority;
    });
    return out;
}

template <typename Desc>
const Desc *find_4096(const Desc *registry, size_t count, int path_id) {
    for (size_t i = 0; i < count; ++i) {
        if (registry[i].kind == ECM_MONT_PATH_4096 && registry[i].variant_id == path_id) {
            return &registry[i];
        }
    }
    return nullptr;
}

template <typename Desc>
const Desc *find_stage1(const Desc *registry, size_t count, ecm_stage1_mont_mode mode) {
    for (size_t i = 0; i < count; ++i) {
        if (registry[i].kind == ECM_MONT_PATH_STAGE1 &&
            registry[i].variant_id == static_cast<int>(mode)) {
            return &registry[i];
        }
    }
    return nullptr;
}

template <typename Desc>
const Desc *resolve_stage1_side(const Desc *registry, size_t count, const char *path,
                                size_t n_bit_size, const Desc *priv_opt_fallback) {
    const EcmPathContext ctx = make_n_ctx(n_bit_size);
    const Desc *unroll512_fallback =
        find_stage1(registry, count, ECM_STAGE1_MONT_UNROLL512);
    if (opencl_ecm_path_is_auto(path)) {
        if (ecm_path_n_fits_4096_dedicated(n_bit_size)) {
            return priv_opt_fallback;
        }
        for (const Desc *desc : auto_sorted_stage1(registry, count)) {
            if (desc->n_fits != nullptr && desc->n_fits(ctx)) {
                return desc;
            }
        }
        return priv_opt_fallback;
    }
    for (size_t i = 0; i < count; ++i) {
        const Desc &desc = registry[i];
        if (desc.kind != ECM_MONT_PATH_STAGE1) {
            continue;
        }
        if (!aliases_contain(desc.aliases, path)) {
            continue;
        }
        if (desc.n_fits != nullptr && !desc.n_fits(ctx)) {
            const int min_pri = desc.auto_priority >= 0 ? desc.auto_priority + 1 : 0;
            for (const Desc *fb : auto_sorted_stage1(registry, count)) {
                if (fb->auto_priority < min_pri) {
                    continue;
                }
                if (fb->n_fits != nullptr && fb->n_fits(ctx)) {
                    return fb;
                }
            }
            return priv_opt_fallback;
        }
        return &desc;
    }
    return unroll512_fallback != nullptr ? unroll512_fallback : priv_opt_fallback;
}

template <typename Desc>
const Desc *resolve_4096_side(const Desc *registry, size_t count, const char *path,
                               size_t n_bit_size, bool *unknown_path) {
    if (unknown_path != nullptr) {
        *unknown_path = false;
    }
    const EcmPathContext ctx = make_n_ctx(n_bit_size);
    const Desc *default_unroll64 = find_4096(registry, count, ECM_MONT4096_PATH_UNROLL64);
    if (opencl_ecm_path_is_auto(path)) {
        for (const Desc *desc : auto_sorted_4096(registry, count)) {
            if (!desc->dedicated) {
                continue;
            }
            if (desc->n_fits != nullptr && desc->n_fits(ctx)) {
                return desc;
            }
        }
        return default_unroll64;
    }
    for (size_t i = 0; i < count; ++i) {
        const Desc &stage1_desc = registry[i];
        if (stage1_desc.kind == ECM_MONT_PATH_STAGE1 &&
            aliases_contain(stage1_desc.aliases, path)) {
            return nullptr;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        const Desc &desc = registry[i];
        if (desc.kind != ECM_MONT_PATH_4096) {
            continue;
        }
        if (!aliases_contain(desc.aliases, path)) {
            continue;
        }
        if (desc.n_fits != nullptr && !desc.n_fits(ctx)) {
            const int min_pri = desc.auto_priority >= 0 ? desc.auto_priority + 1 : 21;
            for (const Desc *fb : auto_sorted_4096(registry, count)) {
                if (fb->auto_priority < min_pri) {
                    continue;
                }
                if (fb->n_fits != nullptr && fb->n_fits(ctx)) {
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

bool ecm_path_n_fits_unroll384(size_t n_bit_size) {
    return n_bit_size + ECM_STAGE1_MONT_CARRY_BITS < ECM_STAGE1_UNROLL384_MAX_BITS;
}

bool ecm_path_n_fits_unroll512_container(size_t n_bit_size) {
    return n_bit_size + ECM_STAGE1_MONT_CARRY_BITS <= ECM_STAGE1_UNROLL512_CONTAINER_BITS;
}

bool ecm_path_n_fits_4096_dedicated(size_t n_bit_size) {
    return n_bit_size >= ECM_PATH_4096_AUTO_MIN_BITS &&
           n_bit_size + ECM_STAGE1_MONT_CARRY_BITS <= ECM_PATH_4096_CONTAINER_BITS;
}

bool ecm_path_n_fits_4096_container(size_t n_bit_size) {
    return n_bit_size + ECM_STAGE1_MONT_CARRY_BITS <= ECM_PATH_4096_CONTAINER_BITS;
}

bool opencl_ecm_path_is_auto(const char *path) {
    return path == nullptr || path[0] == '\0' || strcmp(path, "auto") == 0 ||
           strcmp(path, "default") == 0;
}

size_t opencl_ecm_mont_mul_registry_count() {
    return sizeof(kMontMulRegistry) / sizeof(kMontMulRegistry[0]);
}

const EcmMontMulPathDescriptor *opencl_ecm_mont_mul_registry_entry(size_t index) {
    if (index >= opencl_ecm_mont_mul_registry_count()) {
        return nullptr;
    }
    return &kMontMulRegistry[index];
}

const EcmMontMulPathDescriptor *opencl_ecm_mont_mul_descriptor(ecm_stage1_mont_mode mode) {
    return find_stage1(kMontMulRegistry, opencl_ecm_mont_mul_registry_count(), mode);
}

const EcmMontMulPathDescriptor *opencl_ecm_mont4096_mul_descriptor(int path_id) {
    return find_4096(kMontMulRegistry, opencl_ecm_mont_mul_registry_count(), path_id);
}

size_t opencl_ecm_mont_sqr_registry_count() {
    return sizeof(kMontSqrRegistry) / sizeof(kMontSqrRegistry[0]);
}

const EcmMontSqrPathDescriptor *opencl_ecm_mont_sqr_registry_entry(size_t index) {
    if (index >= opencl_ecm_mont_sqr_registry_count()) {
        return nullptr;
    }
    return &kMontSqrRegistry[index];
}

const EcmMontSqrPathDescriptor *opencl_ecm_mont_sqr_descriptor(ecm_stage1_mont_mode mode) {
    return find_stage1(kMontSqrRegistry, opencl_ecm_mont_sqr_registry_count(), mode);
}

const EcmMontSqrPathDescriptor *opencl_ecm_mont4096_sqr_descriptor(int path_id) {
    return find_4096(kMontSqrRegistry, opencl_ecm_mont_sqr_registry_count(), path_id);
}

int ecm_mont_mul_4096_path_id(const EcmMontMulPathDescriptor *desc) {
    return (desc != nullptr && desc->kind == ECM_MONT_PATH_4096) ? desc->variant_id : 0;
}

int ecm_mont_sqr_4096_path_id(const EcmMontSqrPathDescriptor *desc) {
    return (desc != nullptr && desc->kind == ECM_MONT_PATH_4096) ? desc->variant_id : 0;
}

const EcmMontMulPathDescriptor *opencl_ecm_resolve_stage1_mont_mul(const char *path,
                                                                   size_t n_bit_size) {
    const EcmMontMulPathDescriptor *priv_opt =
        opencl_ecm_mont_mul_descriptor(ECM_STAGE1_MONT_PRIV_OPT);
    return resolve_stage1_side(kMontMulRegistry, opencl_ecm_mont_mul_registry_count(), path,
                               n_bit_size, priv_opt);
}

const EcmMontSqrPathDescriptor *opencl_ecm_resolve_stage1_mont_sqr(const char *path,
                                                                     size_t n_bit_size) {
    const EcmMontSqrPathDescriptor *priv_opt =
        opencl_ecm_mont_sqr_descriptor(ECM_STAGE1_MONT_PRIV_OPT);
    return resolve_stage1_side(kMontSqrRegistry, opencl_ecm_mont_sqr_registry_count(), path,
                               n_bit_size, priv_opt);
}

const EcmMontMulPathDescriptor *opencl_ecm_resolve_mont4096_mul(const char *path,
                                                               size_t n_bit_size,
                                                               bool *unknown_path) {
    return resolve_4096_side(kMontMulRegistry, opencl_ecm_mont_mul_registry_count(), path,
                             n_bit_size, unknown_path);
}

const EcmMontSqrPathDescriptor *opencl_ecm_resolve_mont4096_sqr(const char *path,
                                                                size_t n_bit_size,
                                                                bool *unknown_path) {
    return resolve_4096_side(kMontSqrRegistry, opencl_ecm_mont_sqr_registry_count(), path,
                             n_bit_size, unknown_path);
}

int opencl_ecm_mont4096_coop_wg_size(int path_id) {
    const EcmMontMulPathDescriptor *d = opencl_ecm_mont4096_mul_descriptor(path_id);
    if (d != nullptr) {
        return d->coop_wg_size;
    }
    const EcmMontSqrPathDescriptor *s = opencl_ecm_mont4096_sqr_descriptor(path_id);
    return s != nullptr ? s->coop_wg_size : 1;
}

int opencl_ecm_mont4096_coop_scratch_u32(int mul_path, int sqr_path) {
    int scratch = 0;
    const EcmMontMulPathDescriptor *mul_d = opencl_ecm_mont4096_mul_descriptor(mul_path);
    const EcmMontSqrPathDescriptor *sqr_d = opencl_ecm_mont4096_sqr_descriptor(sqr_path);
    if (mul_d != nullptr && mul_d->coop_scratch_u32 > scratch) {
        scratch = mul_d->coop_scratch_u32;
    }
    if (sqr_d != nullptr && sqr_d->coop_scratch_u32 > scratch) {
        scratch = sqr_d->coop_scratch_u32;
    }
    return scratch;
}

bool opencl_ecm_mont4096_needs_fips4096(int mul_path, int sqr_path) {
    const EcmMontMulPathDescriptor *mul_d = opencl_ecm_mont4096_mul_descriptor(mul_path);
    const EcmMontSqrPathDescriptor *sqr_d = opencl_ecm_mont4096_sqr_descriptor(sqr_path);
    return (mul_d != nullptr && mul_d->needs_fips4096_cl) ||
           (sqr_d != nullptr && sqr_d->needs_fips4096_cl);
}

const char *opencl_ecm_mont4096_mul_path_name(int path_id) {
    const EcmMontMulPathDescriptor *d = opencl_ecm_mont4096_mul_descriptor(path_id);
    return d != nullptr && d->id != nullptr ? d->id : "unknown";
}

const char *opencl_ecm_mont4096_sqr_path_name(int path_id) {
    const EcmMontSqrPathDescriptor *d = opencl_ecm_mont4096_sqr_descriptor(path_id);
    return d != nullptr && d->id != nullptr ? d->id : "unknown";
}

size_t opencl_ecm_addmod_registry_count() {
    return sizeof(kAddModRegistry) / sizeof(kAddModRegistry[0]);
}

const EcmAddModPathDescriptor *opencl_ecm_addmod_registry_entry(size_t index) {
    if (index >= opencl_ecm_addmod_registry_count()) {
        return nullptr;
    }
    return &kAddModRegistry[index];
}

const EcmAddModPathDescriptor *opencl_ecm_addmod_path_descriptor(int path_id) {
    for (const EcmAddModPathDescriptor &d : kAddModRegistry) {
        if (d.path_id == path_id) {
            return &d;
        }
    }
    return nullptr;
}

const EcmAddModPathDescriptor *opencl_ecm_resolve_addmod_path(const char *path, uint32_t limbs,
                                                              bool is_amd) {
    EcmAddSubPathContext ctx{};
    ctx.limbs = limbs;
    ctx.is_amd = is_amd;
    if (!opencl_ecm_path_is_auto(path)) {
        for (size_t i = 0; i < opencl_ecm_addmod_registry_count(); ++i) {
            const EcmAddModPathDescriptor *desc = opencl_ecm_addmod_registry_entry(i);
            if (desc != nullptr && aliases_contain(desc->aliases, path)) {
                return desc;
            }
        }
        return nullptr;
    }
    std::vector<const EcmAddModPathDescriptor *> ordered;
    for (size_t i = 0; i < opencl_ecm_addmod_registry_count(); ++i) {
        ordered.push_back(&kAddModRegistry[i]);
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const EcmAddModPathDescriptor *a, const EcmAddModPathDescriptor *b) {
                  return a->auto_priority < b->auto_priority;
              });
    for (const EcmAddModPathDescriptor *desc : ordered) {
        if (desc->limbs_fits != nullptr && desc->limbs_fits(ctx)) {
            return desc;
        }
    }
    return opencl_ecm_addmod_path_descriptor(ECM_ADDSUB_PATH_FUSED_UNROLL);
}

size_t opencl_ecm_submod_registry_count() {
    return sizeof(kSubModRegistry) / sizeof(kSubModRegistry[0]);
}

const EcmSubModPathDescriptor *opencl_ecm_submod_registry_entry(size_t index) {
    if (index >= opencl_ecm_submod_registry_count()) {
        return nullptr;
    }
    return &kSubModRegistry[index];
}

const EcmSubModPathDescriptor *opencl_ecm_submod_path_descriptor(int path_id) {
    for (const EcmSubModPathDescriptor &d : kSubModRegistry) {
        if (d.path_id == path_id) {
            return &d;
        }
    }
    return nullptr;
}

const EcmSubModPathDescriptor *opencl_ecm_resolve_submod_path(const char *path, uint32_t limbs,
                                                            bool is_amd) {
    EcmAddSubPathContext ctx{};
    ctx.limbs = limbs;
    ctx.is_amd = is_amd;
    if (!opencl_ecm_path_is_auto(path)) {
        for (size_t i = 0; i < opencl_ecm_submod_registry_count(); ++i) {
            const EcmSubModPathDescriptor *desc = opencl_ecm_submod_registry_entry(i);
            if (desc != nullptr && aliases_contain(desc->aliases, path)) {
                return desc;
            }
        }
        return nullptr;
    }
    std::vector<const EcmSubModPathDescriptor *> ordered;
    for (size_t i = 0; i < opencl_ecm_submod_registry_count(); ++i) {
        ordered.push_back(&kSubModRegistry[i]);
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const EcmSubModPathDescriptor *a, const EcmSubModPathDescriptor *b) {
                  return a->auto_priority < b->auto_priority;
              });
    for (const EcmSubModPathDescriptor *desc : ordered) {
        if (desc->limbs_fits != nullptr && desc->limbs_fits(ctx)) {
            return desc;
        }
    }
    return opencl_ecm_submod_path_descriptor(ECM_ADDSUB_PATH_FUSED_UNROLL);
}

EcmStage1KernelBuildPlan opencl_ecm_stage1_make_build_plan(
    uint32_t limbs, uint32_t tpi, const EcmMontMulPathDescriptor *mul,
    const EcmMontSqrPathDescriptor *sqr, const EcmMontMulPathDescriptor *mul_4096,
    const EcmMontSqrPathDescriptor *sqr_4096, const EcmAddModPathDescriptor *add,
    const EcmSubModPathDescriptor *sub, bool use_i24, int stage1_force_normalize,
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

    const int mul_4096_id = ecm_mont_mul_4096_path_id(plan.mul_4096);
    const int sqr_4096_id = ecm_mont_sqr_4096_path_id(plan.sqr_4096);
    append_define(opts, "ECM_STAGE1_MUL_PATH", mul_4096_id);
    append_define(opts, "ECM_STAGE1_SQR_PATH", sqr_4096_id);

    const int coop_wg =
        (plan.limbs == 128u)
            ? std::max(opencl_ecm_mont4096_coop_wg_size(mul_4096_id),
                       opencl_ecm_mont4096_coop_wg_size(sqr_4096_id))
            : 1;
    const int coop_scratch =
        (plan.limbs == 128u)
            ? opencl_ecm_mont4096_coop_scratch_u32(mul_4096_id, sqr_4096_id)
            : 0;
    const int has_fips4096 =
        (plan.limbs == 128u)
            ? (opencl_ecm_mont4096_needs_fips4096(mul_4096_id, sqr_4096_id) ? 1 : 0)
            : 0;
    append_define(opts, "ECM_STAGE1_COOP_WG", coop_wg);
    append_define(opts, "ECM_STAGE1_COOP_SCRATCH_U32", coop_scratch);
    append_define(opts, "ECM_STAGE1_HAS_FIPS4096", has_fips4096);

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
