#pragma once
// ============================================================================
// cpu_addsub_impl.h — fused modular addition / subtraction.
//
// Provides scalar + AVX2/AVX512 SoA (Structure-of-Arrays) implementations.
//
// Scalar API (AoS layout, one instance):
//   void cpu_add_fused_scalar(r, a, b, N, limbs);
//   int  cpu_sub_fused_scalar(r, a, b, N, limbs);
//
// SoA API (K independent instances sharing N):
//   arr[limb * K + instance] layout, K=8 (AVX2) or 16 (AVX512)
//   void cpu_add_fused_avx2_soa(r_soa, a_soa, b_soa, N, limbs);
//   void cpu_sub_fused_avx2_soa(r_soa, a_soa, b_soa, N, limbs);
//
//   void cpu_add_fused_k_soa(r_soa, a_soa, b_soa, N, limbs, K);
//   void cpu_sub_fused_k_soa(r_soa, a_soa, b_soa, N, limbs, K);
//     K must be 8 (AVX2) or 16 (AVX512).  AVX512 path only compiled when
//     __AVX512F__ && __AVX512DQ__ is defined.
//
// The fused algorithm: add + conditional subtract in a single pass,
// per-lane carry chains.  Conditional subtract uses predicated execution
// (no lane divergence).  All K instances share the same modulus N.
//
// Historical: vertical SIMD (manual & lookahead) was evaluated at 0.48x/0.20x
// scalar and abandoned.  See docs/DEV_CPU_ADDSUB_AVX.md.
// ============================================================================

#include <cstdint>
#include <cstring>
#include <cstdio>

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
//  Scalar fused (baseline)
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
//  SoA fused add/sub (dispatched by K)
// ══════════════════════════════════════════════════════════════════════════

#ifdef CPU_ADDSUB_AVX2

// unsigned greater-than for 256-bit (AVX2 lacks _mm256_cmpgt_epu32)
static inline __m256i _mm256_cmpgt_epu32_impl(__m256i a, __m256i b) {
    __m256i flip = _mm256_set1_epi32(0x80000000u);
    return _mm256_cmpgt_epi32(_mm256_xor_si256(a, flip), _mm256_xor_si256(b, flip));
}

// ── Predicated conditional add-N (r += N where mask != 0) ───────────
// mask: -1 (0xFF..FF) or 0 per lane.  Runs a single scalar pass across
// limbs for the lanes that need it (acceptable because conditional sub
// is rare and at the end of the fused loop).
static inline void _add_n_predicated_256(uint32_t *r, const uint32_t *N,
                                          uint32_t limbs, __m256i vmask) {
    __m256i vones = _mm256_set1_epi32(0xFFFFFFFFu);
    __m256i vneed = _mm256_xor_si256(vmask, vones); // invert: -1 where sub needed
    alignas(32) uint32_t need[8];
    _mm256_store_si256((__m256i*)need, vneed);

    // Per-lane carry for N-addition (very short chain, ~4 limbs typical)
    uint32_t carry[8] = {0};
    for (uint32_t i = 0; i < limbs; ++i) {
        uint32_t *rp = r + i * 8;
        uint32_t ni = N[i];
        for (int j = 0; j < 8; ++j) {
            if (need[j]) {
                uint64_t s = (uint64_t)rp[j] + (uint64_t)ni + (uint64_t)carry[j];
                rp[j] = (uint32_t)s;
                carry[j] = (uint32_t)(s >> 32);
            }
        }
        // Add carry to next limb (wrap around for the final carry)
        if (i + 1 < limbs) {
            uint32_t *next_rp = r + (i + 1) * 8;
            for (int j = 0; j < 8; ++j) {
                if (need[j]) {
                    uint64_t s2 = (uint64_t)next_rp[j] + (uint64_t)carry[j];
                    next_rp[j] = (uint32_t)s2;
                    carry[j] = (uint32_t)(s2 >> 32);
                }
            }
        }
    }
}

// ── AVX2 SoA fused add (8 instances) ────────────────────────────────

inline void cpu_add_fused_avx2_soa(uint32_t *r, const uint32_t *a,
                                    const uint32_t *b, const uint32_t *N,
                                    uint32_t limbs) {
    __m256i vones = _mm256_set1_epi32(0xFFFFFFFFu);
    __m256i vzero = _mm256_setzero_si256();
    __m256i vcarry_add = vzero;   // -1 where carry=1, 0 where carry=0
    __m256i vcarry_sub = vones;   // borrow-in starts at 1

    for (uint32_t i = 0; i < limbs; ++i) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i * 8));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i * 8));
        __m256i vN = _mm256_set1_epi32(N[i]);
        __m256i vnotN = _mm256_xor_si256(vN, vones);

        // Add chain
        __m256i vsum0 = _mm256_add_epi32(va, vb);
        __m256i vovf_ab = _mm256_cmpgt_epu32_impl(va, vsum0);
        __m256i vsum = _mm256_sub_epi32(vsum0, vcarry_add); // sub(-1)=add 1
        __m256i vovf_carry = _mm256_and_si256(vcarry_add, _mm256_cmpeq_epi32(vsum0, vones));
        vcarry_add = _mm256_or_si256(vovf_ab, vovf_carry);

        // Sub chain
        __m256i vtmp0 = _mm256_add_epi32(vsum, vnotN);
        __m256i vovf_sub0 = _mm256_cmpgt_epu32_impl(vsum, vtmp0);
        __m256i vtmp = _mm256_sub_epi32(vtmp0, vcarry_sub);
        __m256i vovf_carry_sub = _mm256_and_si256(vcarry_sub, _mm256_cmpeq_epi32(vtmp0, vones));
        vcarry_sub = _mm256_or_si256(vovf_sub0, vovf_carry_sub);

        _mm256_storeu_si256((__m256i*)(r + i * 8), vtmp);
    }

    // Conditional add-N for lanes with (carry_add|carry_sub)==0
    __m256i vmask = _mm256_or_si256(vcarry_add, vcarry_sub);
    _add_n_predicated_256(r, N, limbs, vmask);
}

// ── AVX2 SoA fused sub (8 instances) ────────────────────────────────

inline void cpu_sub_fused_avx2_soa(uint32_t *r, const uint32_t *a,
                                    const uint32_t *b, const uint32_t *N,
                                    uint32_t limbs) {
    __m256i vones = _mm256_set1_epi32(0xFFFFFFFFu);
    __m256i vborrow = vones;  // -1 where borrow=1

    for (uint32_t i = 0; i < limbs; ++i) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i * 8));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i * 8));
        __m256i vnotB = _mm256_xor_si256(vb, vones);

        __m256i vsum0 = _mm256_add_epi32(va, vnotB);
        __m256i vovf = _mm256_cmpgt_epu32_impl(va, vsum0);
        __m256i vsum = _mm256_sub_epi32(vsum0, vborrow);
        __m256i vovf_b = _mm256_and_si256(vborrow, _mm256_cmpeq_epi32(vsum0, vones));
        vborrow = _mm256_or_si256(vovf, vovf_b);

        _mm256_storeu_si256((__m256i*)(r + i * 8), vsum);
    }

    // Conditional sub-N: lanes with borrow==0 need to subtract N
    // vneed_sub = ~vborrow (inverted)
    __m256i vneed_sub = _mm256_xor_si256(vborrow, vones);
    alignas(32) uint32_t need[8];
    _mm256_store_si256((__m256i*)need, vneed_sub);

    uint32_t borrow[8] = {0};
    for (uint32_t i = 0; i < limbs; ++i) {
        uint32_t *rp = r + i * 8;
        uint32_t ni = N[i];
        for (int j = 0; j < 8; ++j) {
            if (need[j]) {
                uint64_t d = (uint64_t)rp[j] - (uint64_t)ni - (uint64_t)borrow[j];
                borrow[j] = ((int64_t)d < 0) ? 1u : 0u;
                rp[j] = (uint32_t)d;
            }
        }
    }
}

#endif // CPU_ADDSUB_AVX2

// ══════════════════════════════════════════════════════════════════════════
//  AVX512 SoA fused add/sub  (K=16 instances)
// ══════════════════════════════════════════════════════════════════════════

#ifdef CPU_ADDSUB_AVX512

// unsigned gt, returns __m512i (-1 or 0 per lane) via single vpmovm2d
static inline __m512i _mm512_cmpgt_epu32_m512i(__m512i a, __m512i b) {
    __m512i flip = _mm512_set1_epi32(0x80000000u);
    return _mm512_movm_epi32(
        _mm512_cmpgt_epi32_mask(
            _mm512_xor_si512(a, flip), _mm512_xor_si512(b, flip)));
}
// cmpeq → returns __m512i mask
static inline __m512i _mm512_cmpeq_epi32_m512i(__m512i a, __m512i b) {
    return _mm512_movm_epi32(_mm512_cmpeq_epi32_mask(a, b));
}

inline void cpu_add_fused_avx512_soa(uint32_t *r, const uint32_t *a,
                                      const uint32_t *b, const uint32_t *N,
                                      uint32_t limbs) {
    __m512i vones = _mm512_set1_epi32(0xFFFFFFFFu);
    __m512i vzero = _mm512_setzero_si512();
    __m512i vcarry_add = vzero;
    __m512i vcarry_sub = vones;

    for (uint32_t i = 0; i < limbs; ++i) {
        __m512i va = _mm512_loadu_si512((const void*)(a + i * 16));
        __m512i vb = _mm512_loadu_si512((const void*)(b + i * 16));
        __m512i vN = _mm512_set1_epi32(N[i]);
        __m512i vnotN = _mm512_xor_si512(vN, vones);

        // Add chain
        __m512i vsum0 = _mm512_add_epi32(va, vb);
        __m512i vovf_ab = _mm512_cmpgt_epu32_m512i(va, vsum0);
        __m512i vsum = _mm512_sub_epi32(vsum0, vcarry_add);
        __m512i vovf_carry = _mm512_and_si512(vcarry_add, _mm512_cmpeq_epi32_m512i(vsum0, vones));
        vcarry_add = _mm512_or_si512(vovf_ab, vovf_carry);

        // Sub chain
        __m512i vtmp0 = _mm512_add_epi32(vsum, vnotN);
        __m512i vovf_sub0 = _mm512_cmpgt_epu32_m512i(vsum, vtmp0);
        __m512i vtmp = _mm512_sub_epi32(vtmp0, vcarry_sub);
        __m512i vovf_carry_sub = _mm512_and_si512(vcarry_sub, _mm512_cmpeq_epi32_m512i(vtmp0, vones));
        vcarry_sub = _mm512_or_si512(vovf_sub0, vovf_carry_sub);

        _mm512_storeu_si512((void*)(r + i * 16), vtmp);
    }

    // Conditional add-N (rare, scalar pass)
    __m512i vmask = _mm512_or_si512(vcarry_add, vcarry_sub);
    alignas(64) uint32_t need[16];
    _mm512_store_si512((void*)need, _mm512_xor_si512(vmask, vones));
    uint32_t carry[16] = {0};
    for (uint32_t i = 0; i < limbs; ++i) {
        uint32_t *rp = r + i * 16;
        uint32_t ni = N[i];
        for (int j = 0; j < 16; ++j) {
            if (need[j]) {
                uint64_t s = (uint64_t)rp[j] + (uint64_t)ni + (uint64_t)carry[j];
                rp[j] = (uint32_t)s;
                carry[j] = (uint32_t)(s >> 32);
            }
        }
    }
}

inline void cpu_sub_fused_avx512_soa(uint32_t *r, const uint32_t *a,
                                      const uint32_t *b, const uint32_t *N,
                                      uint32_t limbs) {
    __m512i vones = _mm512_set1_epi32(0xFFFFFFFFu);
    __m512i vborrow = vones;

    for (uint32_t i = 0; i < limbs; ++i) {
        __m512i va = _mm512_loadu_si512((const void*)(a + i * 16));
        __m512i vb = _mm512_loadu_si512((const void*)(b + i * 16));
        __m512i vnotB = _mm512_xor_si512(vb, vones);

        __m512i vsum0 = _mm512_add_epi32(va, vnotB);
        __m512i vovf = _mm512_cmpgt_epu32_m512i(va, vsum0);
        __m512i vsum = _mm512_sub_epi32(vsum0, vborrow);
        __m512i vovf_b = _mm512_and_si512(vborrow, _mm512_cmpeq_epi32_m512i(vsum0, vones));
        vborrow = _mm512_or_si512(vovf, vovf_b);

        _mm512_storeu_si512((void*)(r + i * 16), vsum);
    }

    // Conditional sub-N
    alignas(64) uint32_t need[16];
    _mm512_store_si512((void*)need, _mm512_xor_si512(vborrow, vones));
    uint32_t borrow[16] = {0};
    for (uint32_t i = 0; i < limbs; ++i) {
        uint32_t *rp = r + i * 16;
        uint32_t ni = N[i];
        for (int j = 0; j < 16; ++j) {
            if (need[j]) {
                uint64_t d = (uint64_t)rp[j] - (uint64_t)ni - (uint64_t)borrow[j];
                borrow[j] = ((int64_t)d < 0) ? 1u : 0u;
                rp[j] = (uint32_t)d;
            }
        }
    }
}

#endif // CPU_ADDSUB_AVX512

// ══════════════════════════════════════════════════════════════════════════
//  Type-erased SoA dispatch (K = 8 or 16)
// ══════════════════════════════════════════════════════════════════════════

inline void cpu_add_fused_k_soa(uint32_t *r, const uint32_t *a,
                                 const uint32_t *b, const uint32_t *N,
                                 uint32_t limbs, int K) {
#ifdef CPU_ADDSUB_AVX512
    if (K == 16) { cpu_add_fused_avx512_soa(r, a, b, N, limbs); return; }
#endif
#ifdef CPU_ADDSUB_AVX2
    if (K == 8)  { cpu_add_fused_avx2_soa(r, a, b, N, limbs); return; }
#endif
    // Fallback: scalar per instance (should not happen in normal use)
    for (int inst = 0; inst < K; ++inst) {
        cpu_add_fused_scalar(r + inst * limbs, a + inst * limbs, b + inst * limbs, N, limbs);
    }
}

inline void cpu_sub_fused_k_soa(uint32_t *r, const uint32_t *a,
                                 const uint32_t *b, const uint32_t *N,
                                 uint32_t limbs, int K) {
#ifdef CPU_ADDSUB_AVX512
    if (K == 16) { cpu_sub_fused_avx512_soa(r, a, b, N, limbs); return; }
#endif
#ifdef CPU_ADDSUB_AVX2
    if (K == 8)  { cpu_sub_fused_avx2_soa(r, a, b, N, limbs); return; }
#endif
    for (int inst = 0; inst < K; ++inst) {
        cpu_sub_fused_scalar(r + inst * limbs, a + inst * limbs, b + inst * limbs, N, limbs);
    }
}

// ══════════════════════════════════════════════════════════════════════════
//  Convenience aliases (AoS scalar)
// ══════════════════════════════════════════════════════════════════════════

inline void cpu_add_fused_c(uint32_t *r, const uint32_t *a, const uint32_t *b,
                            const uint32_t *N, uint32_t limbs) {
    cpu_add_fused_scalar(r, a, b, N, limbs);
}

inline int cpu_sub_fused_c(uint32_t *r, const uint32_t *a, const uint32_t *b,
                           const uint32_t *N, uint32_t limbs) {
    return cpu_sub_fused_scalar(r, a, b, N, limbs);
}
