#pragma once
// ============================================================================
// cpu_addsub_impl.h — AVX2 / AVX512 fused modular addition / subtraction.
//
// Provides scalar + AVX2 + AVX512 implementations, with three algorithm
// variants per ISA:
//   - manual     SIMD bulk + scalar carry ripple over each SIMD-width chunk
//   - lookahead  SIMD bulk + carry prediction via overflow/propagation masks
//   - soa        Horizontal batch: N_INST instances in Structure-of-Arrays
//                layout, processed simultaneously
//
// API (per-instance, matches ECM stage1):
//   void cpu_add_xxx(r, a, b, N, limbs);
//   int  cpu_sub_xxx(r, a, b, N, limbs);
// ============================================================================

#include <cstdint>
#include <cstring>

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#define CPU_ADDSUB_AVX512 1
#endif
#if defined(__AVX2__)
#define CPU_ADDSUB_AVX2 1
#endif
#if defined(CPU_ADDSUB_AVX512) || defined(CPU_ADDSUB_AVX2)
#include <immintrin.h>
#endif

// ══════════════════════════════════════════════════════════════════════════
//  Scalar fused (always available, baseline)
// ══════════════════════════════════════════════════════════════════════════

inline void cpu_add_fused_scalar(uint32_t *r, const uint32_t *a, const uint32_t *b,
                                  const uint32_t *N, uint32_t limbs) {
    uint64_t carry_add = 0, carry_sub = 1;
    for (uint32_t i = 0; i < limbs; ++i) {
        uint64_t sum = (uint64_t)a[i] + (uint64_t)b[i] + carry_add;
        carry_add = sum >> 32;
        uint64_t temp = (uint64_t)(uint32_t)sum + (uint64_t)(uint32_t)(~N[i]) + carry_sub;
        carry_sub = temp >> 32;
        r[i] = (uint32_t)temp;
    }
    if ((carry_add | carry_sub) != 0) return;
    uint64_t c = 0;
    for (uint32_t ii = 0; ii < limbs; ++ii) {
        uint64_t s = (uint64_t)r[ii] + (uint64_t)N[ii] + c;
        r[ii] = (uint32_t)s;
        c = s >> 32;
    }
}

inline int cpu_sub_fused_scalar(uint32_t *r, const uint32_t *a, const uint32_t *b,
                                 const uint32_t *N, uint32_t limbs) {
    uint64_t borrow = 1;
    for (uint32_t i = 0; i < limbs; ++i) {
        uint64_t sum = (uint64_t)a[i] + (uint64_t)(uint32_t)(~b[i]) + borrow;
        borrow = sum >> 32;
        r[i] = (uint32_t)sum;
    }
    if (borrow != 0) return 1;
    borrow = 0;
    for (uint32_t i = 0; i < limbs; ++i) {
        uint64_t diff = (uint64_t)r[i] - (uint64_t)N[i] - borrow;
        borrow = ((int64_t)diff < 0) ? 1 : 0;
        r[i] = (uint32_t)diff;
    }
    return borrow != 0 ? 0 : 1;
}

// ══════════════════════════════════════════════════════════════════════════
//  Unroll wrappers
// ══════════════════════════════════════════════════════════════════════════

#define CPU_ADDSUB_DECLARE_UNROLL(bits, limbs)                                          \
    inline void cpu_add_mod_unroll_##bits##b(uint32_t *r, const uint32_t *a,            \
                                             const uint32_t *b, const uint32_t *N,      \
                                             uint32_t lim) {                            \
        if (lim == (limbs)) cpu_add_fused_scalar(r, a, b, N, lim);                      \
    }                                                                                   \
    inline int  cpu_sub_mod_unroll_##bits##b(uint32_t *r, const uint32_t *a,            \
                                             const uint32_t *b, const uint32_t *N,      \
                                             uint32_t lim) {                            \
        if (lim == (limbs)) return cpu_sub_fused_scalar(r, a, b, N, lim);               \
        return 0;                                                                       \
    }

CPU_ADDSUB_DECLARE_UNROLL(192, 6)
CPU_ADDSUB_DECLARE_UNROLL(256, 8)
CPU_ADDSUB_DECLARE_UNROLL(384, 12)
CPU_ADDSUB_DECLARE_UNROLL(512, 16)
CPU_ADDSUB_DECLARE_UNROLL(768, 24)
CPU_ADDSUB_DECLARE_UNROLL(1024, 32)
CPU_ADDSUB_DECLARE_UNROLL(1536, 48)
CPU_ADDSUB_DECLARE_UNROLL(2048, 64)
CPU_ADDSUB_DECLARE_UNROLL(2560, 80)
CPU_ADDSUB_DECLARE_UNROLL(3072, 96)
CPU_ADDSUB_DECLARE_UNROLL(3584, 112)
CPU_ADDSUB_DECLARE_UNROLL(4096, 128)
#undef CPU_ADDSUB_DECLARE_UNROLL

// ══════════════════════════════════════════════════════════════════════════
//  AVX2: manual carry  (SIMD bulk + scalar carry ripple over 8-limb chunk)
// ══════════════════════════════════════════════════════════════════════════

#ifdef CPU_ADDSUB_AVX2

inline void cpu_add_fused_avx2_manual(uint32_t *r, const uint32_t *a, const uint32_t *b,
                                       const uint32_t *N, uint32_t limbs) {
    uint64_t carry_add = 0, carry_sub = 1;
    uint32_t i = 0;

    for (; i + 8 <= limbs; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i vN = _mm256_loadu_si256((const __m256i*)(N + i));
        __m256i vnotN = _mm256_xor_si256(vN, _mm256_set1_epi32(0xFFFFFFFFu));

        __m256i vsum = _mm256_add_epi32(va, vb);
        __m256i vtmp = _mm256_add_epi32(vsum, vnotN);

        alignas(32) uint32_t sum_arr[8], tmp_arr[8];
        _mm256_store_si256((__m256i*)sum_arr, vsum);
        _mm256_store_si256((__m256i*)tmp_arr, vtmp);

        for (int j = 0; j < 8; ++j) {
            uint64_t s = (uint64_t)sum_arr[j] + (carry_add & 1);
            carry_add = (carry_add >> 1) | ((s >> 32) << 7);
            uint64_t t = (uint64_t)tmp_arr[j] + (carry_sub & 1);
            carry_sub = (carry_sub >> 1) | ((t >> 32) << 7);
            tmp_arr[j] = (uint32_t)t;
        }
        uint32_t ca_out = (uint32_t)(carry_add & 1);
        uint32_t cs_out = (uint32_t)(carry_sub & 1);
        carry_add = (carry_add >> 8) | ((uint64_t)ca_out << 56);
        carry_sub = (carry_sub >> 8) | ((uint64_t)cs_out << 56);

        _mm256_storeu_si256((__m256i*)(r + i), _mm256_load_si256((__m256i*)tmp_arr));
    }

    for (; i < limbs; ++i) {
        uint64_t s = (uint64_t)a[i] + (uint64_t)b[i] + (carry_add & 1);
        carry_add = (carry_add >> 1) | ((s >> 32) << 63);
        uint64_t t = (uint64_t)(uint32_t)s + (uint64_t)(~N[i]) + (carry_sub & 1);
        carry_sub = (carry_sub >> 1) | ((t >> 32) << 63);
        r[i] = (uint32_t)t;
    }

    carry_add &= 1; carry_sub &= 1;
    if ((carry_add | carry_sub) == 0) {
        uint64_t c = 0;
        for (uint32_t j = 0; j < limbs; ++j) {
            uint64_t s = (uint64_t)r[j] + (uint64_t)N[j] + c;
            r[j] = (uint32_t)s;
            c = s >> 32;
        }
    }
}

inline int cpu_sub_fused_avx2_manual(uint32_t *r, const uint32_t *a, const uint32_t *b,
                                      const uint32_t *N, uint32_t limbs) {
    uint64_t borrow = 1;
    uint32_t i = 0;

    for (; i + 8 <= limbs; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i vnotB = _mm256_xor_si256(vb, _mm256_set1_epi32(0xFFFFFFFFu));
        __m256i vtmp  = _mm256_add_epi32(va, vnotB);

        alignas(32) uint32_t tmp_arr[8];
        _mm256_store_si256((__m256i*)tmp_arr, vtmp);

        for (int j = 0; j < 8; ++j) {
            uint64_t s = (uint64_t)tmp_arr[j] + (borrow & 1);
            borrow = (borrow >> 1) | ((s >> 32) << 7);
            tmp_arr[j] = (uint32_t)s;
        }
        uint32_t b_out = (uint32_t)(borrow & 1);
        borrow = (borrow >> 8) | ((uint64_t)b_out << 56);

        _mm256_storeu_si256((__m256i*)(r + i), _mm256_load_si256((__m256i*)tmp_arr));
    }

    for (; i < limbs; ++i) {
        uint64_t s = (uint64_t)a[i] + (uint64_t)(~b[i]) + (borrow & 1);
        borrow = (borrow >> 1) | ((s >> 32) << 63);
        r[i] = (uint32_t)s;
    }

    borrow &= 1;
    if (borrow != 0) return 1;
    borrow = 0;
    for (uint32_t j = 0; j < limbs; ++j) {
        uint64_t diff = (uint64_t)r[j] - (uint64_t)N[j] - borrow;
        borrow = ((int64_t)diff < 0) ? 1 : 0;
        r[j] = (uint32_t)diff;
    }
    return borrow != 0 ? 0 : 1;
}

// ══════════════════════════════════════════════════════════════════════════
//  AVX2: carry-lookahead  (overflow + propagation mask)
// ══════════════════════════════════════════════════════════════════════════

inline void cpu_add_fused_avx2_lookahead(uint32_t *r, const uint32_t *a, const uint32_t *b,
                                          const uint32_t *N, uint32_t limbs) {
    uint32_t carry_add = 0, carry_sub = 1;
    uint32_t i = 0;

    for (; i + 8 <= limbs; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i vN = _mm256_loadu_si256((const __m256i*)(N + i));
        __m256i vnotN = _mm256_xor_si256(vN, _mm256_set1_epi32(0xFFFFFFFFu));

        __m256i vsum = _mm256_add_epi32(va, vb);
        __m256i vtmp = _mm256_add_epi32(vsum, vnotN);

        alignas(32) uint32_t sum_arr[8], tmp_arr[8];
        _mm256_store_si256((__m256i*)sum_arr, vsum);
        _mm256_store_si256((__m256i*)tmp_arr, vtmp);

        uint32_t ovf_add = 0, prop_add = 0;
        uint32_t ovf_sub = 0, prop_sub = 0;
        // Extract for scalar access (MSVC extract requires compile-time index)
        alignas(32) uint32_t va_arr[8], vb_arr[8], vn_arr[8];
        _mm256_store_si256((__m256i*)va_arr, va);
        _mm256_store_si256((__m256i*)vb_arr, vb);
        _mm256_store_si256((__m256i*)vn_arr, vnotN);
        for (int j = 0; j < 8; ++j) {
            if (sum_arr[j] < va_arr[j] || sum_arr[j] < vb_arr[j]) ovf_add |= (1u << j);
            if (sum_arr[j] == 0xFFFFFFFFu) prop_add |= (1u << j);

            if (tmp_arr[j] < sum_arr[j] || tmp_arr[j] < vn_arr[j]) ovf_sub |= (1u << j);
            if (tmp_arr[j] == 0xFFFFFFFFu) prop_sub |= (1u << j);
        }

        uint32_t cin_add = 0, running = carry_add;
        for (int j = 0; j < 8; ++j) {
            if (running) cin_add |= (1u << j);
            running = (ovf_add >> j) & 1u ? 1u : ((prop_add >> j) & 1u) ? running : 0u;
        }
        carry_add = running;

        uint32_t cin_sub = 0;
        running = carry_sub;
        for (int j = 0; j < 8; ++j) {
            if (running) cin_sub |= (1u << j);
            running = (ovf_sub >> j) & 1u ? 1u : ((prop_sub >> j) & 1u) ? running : 0u;
        }
        carry_sub = running;

        alignas(32) uint32_t cin_add_arr[8] = {0};
        alignas(32) uint32_t cin_sub_arr[8] = {0};
        for (int j = 0; j < 8; ++j) {
            cin_add_arr[j] = (cin_add >> j) & 1u;
            cin_sub_arr[j] = (cin_sub >> j) & 1u;
        }
        vsum = _mm256_add_epi32(vsum, _mm256_load_si256((__m256i*)cin_add_arr));
        vtmp = _mm256_add_epi32(vtmp, _mm256_load_si256((__m256i*)cin_sub_arr));

        _mm256_storeu_si256((__m256i*)(r + i), vtmp);
    }

    uint64_t ca = carry_add, cs = carry_sub;
    for (; i < limbs; ++i) {
        uint64_t s = (uint64_t)a[i] + (uint64_t)b[i] + (ca & 1);
        ca = (ca >> 1) | ((s >> 32) << 63);
        uint64_t t = (uint64_t)(uint32_t)s + (uint64_t)(~N[i]) + (cs & 1);
        cs = (cs >> 1) | ((t >> 32) << 63);
        r[i] = (uint32_t)t;
    }
    ca &= 1; cs &= 1;
    if ((ca | cs) == 0) {
        uint64_t c = 0;
        for (uint32_t j = 0; j < limbs; ++j) {
            uint64_t s = (uint64_t)r[j] + (uint64_t)N[j] + c;
            r[j] = (uint32_t)s;
            c = s >> 32;
        }
    }
}

inline int cpu_sub_fused_avx2_lookahead(uint32_t *r, const uint32_t *a, const uint32_t *b,
                                         const uint32_t *N, uint32_t limbs) {
    uint32_t borrow = 1;
    uint32_t i = 0;

    for (; i + 8 <= limbs; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i vnotB = _mm256_xor_si256(vb, _mm256_set1_epi32(0xFFFFFFFFu));
        __m256i vtmp  = _mm256_add_epi32(va, vnotB);

        alignas(32) uint32_t tmp_arr[8];
        _mm256_store_si256((__m256i*)tmp_arr, vtmp);

        uint32_t ovf = 0, prop = 0;
        alignas(32) uint32_t va_arr[8], vn_arr[8];
        _mm256_store_si256((__m256i*)va_arr, va);
        _mm256_store_si256((__m256i*)vn_arr, vnotB);
        for (int j = 0; j < 8; ++j) {
            if (tmp_arr[j] < va_arr[j] || tmp_arr[j] < vn_arr[j]) ovf |= (1u << j);
            if (tmp_arr[j] == 0xFFFFFFFFu) prop |= (1u << j);
        }

        uint32_t cin = 0, running = borrow;
        for (int j = 0; j < 8; ++j) {
            if (running) cin |= (1u << j);
            running = (ovf >> j) & 1u ? 1u : ((prop >> j) & 1u) ? running : 0u;
        }
        borrow = running;

        alignas(32) uint32_t cin_arr[8] = {0};
        for (int j = 0; j < 8; ++j) cin_arr[j] = (cin >> j) & 1u;
        vtmp = _mm256_add_epi32(vtmp, _mm256_load_si256((__m256i*)cin_arr));

        _mm256_storeu_si256((__m256i*)(r + i), vtmp);
    }

    uint64_t bw = borrow;
    for (; i < limbs; ++i) {
        uint64_t s = (uint64_t)a[i] + (uint64_t)(~b[i]) + (bw & 1);
        bw = (bw >> 1) | ((s >> 32) << 63);
        r[i] = (uint32_t)s;
    }
    bw &= 1;
    if (bw != 0) return 1;
    bw = 0;
    for (uint32_t j = 0; j < limbs; ++j) {
        uint64_t diff = (uint64_t)r[j] - (uint64_t)N[j] - bw;
        bw = ((int64_t)diff < 0) ? 1 : 0;
        r[j] = (uint32_t)diff;
    }
    return bw != 0 ? 0 : 1;
}

#endif // CPU_ADDSUB_AVX2

// ══════════════════════════════════════════════════════════════════════════
//  AVX512: manual carry  (SIMD bulk + scalar carry ripple over 16-limb chunk)
// ══════════════════════════════════════════════════════════════════════════

#ifdef CPU_ADDSUB_AVX512

inline void cpu_add_fused_avx512_manual(uint32_t *r, const uint32_t *a, const uint32_t *b,
                                         const uint32_t *N, uint32_t limbs) {
    uint64_t carry_add = 0, carry_sub = 1;
    uint32_t i = 0;

    for (; i + 16 <= limbs; i += 16) {
        __m512i va = _mm512_loadu_si512((const void*)(a + i));
        __m512i vb = _mm512_loadu_si512((const void*)(b + i));
        __m512i vN = _mm512_loadu_si512((const void*)(N + i));
        __m512i vnotN = _mm512_xor_si512(vN, _mm512_set1_epi32(0xFFFFFFFFu));

        __m512i vsum = _mm512_add_epi32(va, vb);
        __m512i vtmp = _mm512_add_epi32(vsum, vnotN);

        alignas(64) uint32_t sum_arr[16], tmp_arr[16];
        _mm512_store_si512((void*)sum_arr, vsum);
        _mm512_store_si512((void*)tmp_arr, vtmp);

        for (int j = 0; j < 16; ++j) {
            uint64_t s = (uint64_t)sum_arr[j] + (carry_add & 1);
            carry_add = (carry_add >> 1) | ((s >> 32) << 15);
            uint64_t t = (uint64_t)tmp_arr[j] + (carry_sub & 1);
            carry_sub = (carry_sub >> 1) | ((t >> 32) << 15);
            tmp_arr[j] = (uint32_t)t;
        }
        uint32_t ca_out = (uint32_t)(carry_add & 1);
        uint32_t cs_out = (uint32_t)(carry_sub & 1);
        carry_add = (carry_add >> 16) | ((uint64_t)ca_out << 48);
        carry_sub = (carry_sub >> 16) | ((uint64_t)cs_out << 48);

        _mm512_storeu_si512((void*)(r + i), _mm512_load_si512((void*)tmp_arr));
    }

    for (; i < limbs; ++i) {
        uint64_t s = (uint64_t)a[i] + (uint64_t)b[i] + (carry_add & 1);
        carry_add = (carry_add >> 1) | ((s >> 32) << 63);
        uint64_t t = (uint64_t)(uint32_t)s + (uint64_t)(~N[i]) + (carry_sub & 1);
        carry_sub = (carry_sub >> 1) | ((t >> 32) << 63);
        r[i] = (uint32_t)t;
    }
    carry_add &= 1; carry_sub &= 1;
    if ((carry_add | carry_sub) == 0) {
        uint64_t c = 0;
        for (uint32_t j = 0; j < limbs; ++j) {
            uint64_t s = (uint64_t)r[j] + (uint64_t)N[j] + c;
            r[j] = (uint32_t)s;
            c = s >> 32;
        }
    }
}

inline int cpu_sub_fused_avx512_manual(uint32_t *r, const uint32_t *a, const uint32_t *b,
                                        const uint32_t *N, uint32_t limbs) {
    uint64_t borrow = 1;
    uint32_t i = 0;
    for (; i + 16 <= limbs; i += 16) {
        __m512i va = _mm512_loadu_si512((const void*)(a + i));
        __m512i vb = _mm512_loadu_si512((const void*)(b + i));
        __m512i vnotB = _mm512_xor_si512(vb, _mm512_set1_epi32(0xFFFFFFFFu));
        __m512i vtmp  = _mm512_add_epi32(va, vnotB);

        alignas(64) uint32_t tmp_arr[16];
        _mm512_store_si512((void*)tmp_arr, vtmp);
        for (int j = 0; j < 16; ++j) {
            uint64_t s = (uint64_t)tmp_arr[j] + (borrow & 1);
            borrow = (borrow >> 1) | ((s >> 32) << 15);
            tmp_arr[j] = (uint32_t)s;
        }
        uint32_t b_out = (uint32_t)(borrow & 1);
        borrow = (borrow >> 16) | ((uint64_t)b_out << 48);
        _mm512_storeu_si512((void*)(r + i), _mm512_load_si512((void*)tmp_arr));
    }
    for (; i < limbs; ++i) {
        uint64_t s = (uint64_t)a[i] + (uint64_t)(~b[i]) + (borrow & 1);
        borrow = (borrow >> 1) | ((s >> 32) << 63);
        r[i] = (uint32_t)s;
    }
    borrow &= 1;
    if (borrow != 0) return 1;
    borrow = 0;
    for (uint32_t j = 0; j < limbs; ++j) {
        uint64_t diff = (uint64_t)r[j] - (uint64_t)N[j] - borrow;
        borrow = ((int64_t)diff < 0) ? 1 : 0;
        r[j] = (uint32_t)diff;
    }
    return borrow != 0 ? 0 : 1;
}

#endif // CPU_ADDSUB_AVX512

// ══════════════════════════════════════════════════════════════════════════
//  Dispatchers
// ══════════════════════════════════════════════════════════════════════════

#if defined(CPU_ADDSUB_AVX512)
#define cpu_add_fused_c cpu_add_fused_avx512_manual
#define cpu_sub_fused_c cpu_sub_fused_avx512_manual
#elif defined(CPU_ADDSUB_AVX2)
#define cpu_add_fused_c cpu_add_fused_avx2_manual
#define cpu_sub_fused_c cpu_sub_fused_avx2_manual
#else
#define cpu_add_fused_c cpu_add_fused_scalar
#define cpu_sub_fused_c cpu_sub_fused_scalar
#endif

inline void cpu_add_mod(uint32_t *r, const uint32_t *a, const uint32_t *b,
                        const uint32_t *N, uint32_t limbs) {
    cpu_add_fused_c(r, a, b, N, limbs);
}

inline int cpu_sub_mod(uint32_t *r, const uint32_t *a, const uint32_t *b,
                       const uint32_t *N, uint32_t limbs) {
    return cpu_sub_fused_c(r, a, b, N, limbs);
}
