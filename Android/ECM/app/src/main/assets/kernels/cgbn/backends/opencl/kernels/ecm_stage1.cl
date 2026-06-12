// OpenCL ECM Stage 1 — Montgomery ladder (double_add_v2), ported from test/cgbn_stage1.cu

#ifndef MAX_LIMBS
#define MAX_LIMBS 64
#endif

#ifndef TPI
#define TPI 8
#endif

#ifndef ECM_STAGE1_FORCE_NORMALIZE
#define ECM_STAGE1_FORCE_NORMALIZE 0
#endif

#ifndef ECM_STAGE1_MUL_PATH
#define ECM_STAGE1_MUL_PATH 0
#endif

#ifndef ECM_STAGE1_SQR_PATH
#define ECM_STAGE1_SQR_PATH 0
#endif

#ifndef ECM_STAGE1_COOP_WG
#define ECM_STAGE1_COOP_WG 0
#endif

#ifndef ECM_STAGE1_COOP_SCRATCH_U32
#define ECM_STAGE1_COOP_SCRATCH_U32 0
#endif

#ifndef ECM_STAGE1_USE_I24_384
#define ECM_STAGE1_USE_I24_384 0
#endif

#ifndef ECM_STAGE1_I24_U32_BLSUB
#define ECM_STAGE1_I24_U32_BLSUB 0
#endif

#ifndef ECM_STAGE1_MUL_FORCE_UNROLL32
#define ECM_STAGE1_MUL_FORCE_UNROLL32 0
#endif

#ifndef ECM_STAGE1_MUL_FORCE_UNROLL384
#define ECM_STAGE1_MUL_FORCE_UNROLL384 0
#endif

#ifndef ECM_STAGE1_MUL_FORCE_PRIV_OPT
#define ECM_STAGE1_MUL_FORCE_PRIV_OPT 0
#endif

#ifndef ECM_STAGE1_SQR_FORCE_UNROLL32
#define ECM_STAGE1_SQR_FORCE_UNROLL32 0
#endif

#ifndef ECM_STAGE1_SQR_FORCE_UNROLL384
#define ECM_STAGE1_SQR_FORCE_UNROLL384 0
#endif

#ifndef ECM_STAGE1_SQR_FORCE_PRIV_OPT
#define ECM_STAGE1_SQR_FORCE_PRIV_OPT 0
#endif

#ifndef ECM_STAGE1_384_LIMBS
#define ECM_STAGE1_384_LIMBS 12u
#endif

#ifndef ECM_STAGE1_512_CONTAINER_LIMBS
#define ECM_STAGE1_512_CONTAINER_LIMBS 16u
#endif

// Path ids: 0=unroll64_4096, 1=unroll64_4096_mt2, 2=fips4096, 3=fips4096_mt8, 4=fips4096_mt16
#if ECM_STAGE1_COOP_WG > 1
#define ECM_STAGE1_USE_COOP_WG 1
#else
#define ECM_STAGE1_USE_COOP_WG 0
#endif

#define MONT_FIXED_4096_LIMBS 128u
#define ECM_STAGE1_MT2_LOCAL_U32 (MONT_FIXED_4096_LIMBS + 2u + MONT_FIXED_4096_LIMBS + MONT_FIXED_4096_LIMBS + 3u)

// Stage1-private Montgomery variants (private pointer ABI).
static inline void mont_mul_stage1_unroll_only_512(uint *out, const uint *a, const uint *b,
                                                   const uint *N, uint np0) {
    uint t[16u + 2u];
    #pragma unroll
    for (uint i = 0u; i < 18u; ++i) t[i] = 0u;
    uint B[16u];
    #pragma unroll
    for (uint j = 0u; j < 16u; ++j) B[j] = b[j];
    #pragma unroll
    for (uint i = 0u; i < 16u; ++i) {
        uint ai = a[i];
        ulong carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < 16u; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong top = (ulong)t[16u] + carry;
        t[16u] = (uint)top;
        t[17u] = (uint)(top >> 32);
        uint m = (uint)((ulong)t[0] * (ulong)np0);
        carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < 16u; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
            if (j > 0u) t[j - 1u] = (uint)uv;
            carry = uv >> 32;
        }
        top = (ulong)t[16u] + carry;
        t[15u] = (uint)top;
        top = (ulong)t[17u] + (top >> 32);
        t[16u] = (uint)top;
        t[17u] = (uint)(top >> 32);
    }
    ulong borrow = 0ul;
    uint D[16u];
    #pragma unroll
    for (uint i = 0u; i < 16u; ++i) {
        ulong tv = (ulong)t[i], nv = (ulong)N[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }
    uint need_sub = (t[16u] != 0u || t[17u] != 0u) ? 1u : 0u;
    need_sub = (borrow == 0u) ? 1u : need_sub;
    uint mask = 0u - need_sub;
    #pragma unroll
    for (uint i = 0u; i < 16u; ++i) out[i] = (D[i] & mask) | (t[i] & ~mask);
}

// 384-bit CIOS: 12 active 32-bit limbs in a 16-limb private layout.
// Valid only when N + CARRY_BITS < 384 (host: opencl_ecm_stage1_n_fits_unroll384).
static inline void mont_mul_stage1_unroll_only_384(uint *out, const uint *a, const uint *b,
                                                   const uint *N, uint np0) {
    uint t[ECM_STAGE1_384_LIMBS + 2u];
    #pragma unroll
    for (uint i = 0u; i < ECM_STAGE1_384_LIMBS + 2u; ++i) {
        t[i] = 0u;
    }
    uint B[ECM_STAGE1_512_CONTAINER_LIMBS];
    #pragma unroll
    for (uint j = 0u; j < ECM_STAGE1_512_CONTAINER_LIMBS; ++j) {
        B[j] = b[j];
    }

    #pragma unroll
    for (uint i = 0u; i < ECM_STAGE1_384_LIMBS; ++i) {
        uint ai = a[i];
        ulong carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < ECM_STAGE1_384_LIMBS; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong top = (ulong)t[ECM_STAGE1_384_LIMBS] + carry;
        t[ECM_STAGE1_384_LIMBS] = (uint)top;
        t[ECM_STAGE1_384_LIMBS + 1u] = (uint)(top >> 32);

        uint m = (uint)((ulong)t[0] * (ulong)np0);
        carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < ECM_STAGE1_384_LIMBS; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
            if (j > 0u) {
                t[j - 1u] = (uint)uv;
            }
            carry = uv >> 32;
        }
        top = (ulong)t[ECM_STAGE1_384_LIMBS] + carry;
        t[ECM_STAGE1_384_LIMBS - 1u] = (uint)top;
        top = (ulong)t[ECM_STAGE1_384_LIMBS + 1u] + (top >> 32);
        t[ECM_STAGE1_384_LIMBS] = (uint)top;
        t[ECM_STAGE1_384_LIMBS + 1u] = (uint)(top >> 32);
    }

    ulong borrow = 0ul;
    uint D[ECM_STAGE1_384_LIMBS];
    #pragma unroll
    for (uint i = 0u; i < ECM_STAGE1_384_LIMBS; ++i) {
        ulong tv = (ulong)t[i];
        ulong nv = (ulong)N[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }

    uint any_high =
        (t[ECM_STAGE1_384_LIMBS] | t[ECM_STAGE1_384_LIMBS + 1u]) != 0u;
    uint need_sub = any_high | (borrow == 0u);
    uint mask = 0u - need_sub;
    #pragma unroll
    for (uint i = 0u; i < ECM_STAGE1_384_LIMBS; ++i) {
        out[i] = (D[i] & mask) | (t[i] & ~mask);
    }
    #pragma unroll
    for (uint i = ECM_STAGE1_384_LIMBS; i < ECM_STAGE1_512_CONTAINER_LIMBS; ++i) {
        out[i] = 0u;
    }
}

static inline void mont_mul_stage1_unroll64_4096(uint *out, const uint *a, const uint *b,
                                                  const uint *N, uint np0) {
    uint t[128u + 2u];
    for (uint i = 0u; i < 130u; ++i) t[i] = 0u;
    uint B[128u];
    for (uint j = 0u; j < 128u; ++j) B[j] = b[j];
    for (uint i = 0u; i < 128u; ++i) {
        uint ai = a[i];
        ulong carry = 0ul;
        #pragma unroll 64
        for (uint j = 0u; j < 128u; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong top = (ulong)t[128u] + carry;
        t[128u] = (uint)top;
        t[129u] = (uint)(top >> 32);
        uint m = (uint)((ulong)t[0] * (ulong)np0);
        ulong uv0 = (ulong)t[0] + (ulong)m * (ulong)N[0];
        carry = uv0 >> 32;
        #pragma unroll 64
        for (uint j = 1u; j < 128u; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
            t[j - 1u] = (uint)uv;
            carry = uv >> 32;
        }
        top = (ulong)t[128u] + carry;
        t[127u] = (uint)top;
        top = (ulong)t[129u] + (top >> 32);
        t[128u] = (uint)top;
        t[129u] = (uint)(top >> 32);
    }
    ulong borrow = 0ul;
    uint D[128u];
    for (uint i = 0u; i < 128u; ++i) {
        ulong tv = (ulong)t[i], nv = (ulong)N[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }
    uint need_sub = (t[128u] != 0u || t[129u] != 0u) ? 1u : 0u;
    need_sub = (borrow == 0u) ? 1u : need_sub;
    uint mask = 0u - need_sub;
    for (uint i = 0u; i < 128u; ++i) out[i] = (D[i] & mask) | (t[i] & ~mask);
}

static inline void mont_mul_stage1_unroll64_4096_mt2_local(
    __local uint *out,
    __local const uint *a,
    __local const uint *b,
    __local const uint *N,
    uint np0,
    __local uint *local_mem,
    uint lid)
{
    __local uint *t = local_mem;
    __local uint *B = t + (MONT_FIXED_4096_LIMBS + 2u);
    __local uint *D = B + MONT_FIXED_4096_LIMBS;
    __local uint *meta = D + MONT_FIXED_4096_LIMBS;
    const uint half_words = MONT_FIXED_4096_LIMBS / 2u;
    const uint j_begin = lid * half_words;
    const uint j_end = j_begin + half_words;

    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS + 2u; ++i) {
            t[i] = 0u;
        }
    }

    #pragma unroll 64
    for (uint j = j_begin; j < j_end; ++j) {
        B[j] = b[j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        uint ai = a[i];

        if (lid == 0u) {
            ulong carry = 0ul;
            #pragma unroll 64
            for (uint j = 0u; j < half_words; ++j) {
                ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
                t[j] = (uint)uv;
                carry = uv >> 32;
            }
            meta[0] = (uint)carry;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid == 1u) {
            ulong carry = (ulong)meta[0];
            #pragma unroll 64
            for (uint j = half_words; j < MONT_FIXED_4096_LIMBS; ++j) {
                ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
                t[j] = (uint)uv;
                carry = uv >> 32;
            }
            ulong top = (ulong)t[MONT_FIXED_4096_LIMBS] + carry;
            t[MONT_FIXED_4096_LIMBS] = (uint)top;
            t[MONT_FIXED_4096_LIMBS + 1u] = (uint)(top >> 32);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid == 0u) {
            uint m = (uint)((ulong)t[0] * (ulong)np0);
            ulong uv0 = (ulong)t[0] + (ulong)m * (ulong)N[0];
            ulong carry = uv0 >> 32;
            #pragma unroll 64
            for (uint j = 1u; j < half_words; ++j) {
                ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
                t[j - 1u] = (uint)uv;
                carry = uv >> 32;
            }
            meta[0] = (uint)carry;
            meta[1] = m;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid == 1u) {
            uint m = meta[1];
            ulong carry = (ulong)meta[0];
            #pragma unroll 64
            for (uint j = half_words; j < MONT_FIXED_4096_LIMBS; ++j) {
                ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
                t[j - 1u] = (uint)uv;
                carry = uv >> 32;
            }
            ulong top = (ulong)t[MONT_FIXED_4096_LIMBS] + carry;
            t[MONT_FIXED_4096_LIMBS - 1u] = (uint)top;
            top = (ulong)t[MONT_FIXED_4096_LIMBS + 1u] + (top >> 32);
            t[MONT_FIXED_4096_LIMBS] = (uint)top;
            t[MONT_FIXED_4096_LIMBS + 1u] = (uint)(top >> 32);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0u) {
        ulong borrow = 0ul;
        #pragma unroll 64
        for (uint i = 0u; i < half_words; ++i) {
            ulong tv = (ulong)t[i];
            ulong nv = (ulong)N[i];
            ulong w = tv - nv - borrow;
            D[i] = (uint)w;
            borrow = (tv < nv + borrow) ? 1ul : 0ul;
        }
        meta[0] = (uint)borrow;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 1u) {
        ulong borrow = (ulong)meta[0];
        #pragma unroll 64
        for (uint i = half_words; i < MONT_FIXED_4096_LIMBS; ++i) {
            ulong tv = (ulong)t[i];
            ulong nv = (ulong)N[i];
            ulong w = tv - nv - borrow;
            D[i] = (uint)w;
            borrow = (tv < nv + borrow) ? 1ul : 0ul;
        }
        uint need_sub = (t[MONT_FIXED_4096_LIMBS] != 0u || t[MONT_FIXED_4096_LIMBS + 1u] != 0u) ? 1u : 0u;
        need_sub = (borrow == 0u) ? 1u : need_sub;
        meta[2] = need_sub;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint mask = 0u - meta[2];
    #pragma unroll 64
    for (uint i = j_begin; i < j_end; ++i) {
        out[i] = (D[i] & mask) | (t[i] & ~mask);
    }
}

// Generic Montgomery mul: cached B + speculative final subtract (mont_mul_priv_opt_core ABI).
static inline void mont_mul_stage1_priv_opt(uint *out, const uint *a, const uint *b,
                                            const uint *N, uint np0, uint limbs) {
    if (limbs == 0u || limbs > MAX_LIMBS) {
        return;
    }

    uint t[MAX_LIMBS + 2u];
    for (uint i = 0u; i < limbs + 2u; ++i) {
        t[i] = 0u;
    }

    uint B[MAX_LIMBS];
    for (uint j = 0u; j < limbs; ++j) {
        B[j] = b[j];
    }

    for (uint i = 0u; i < limbs; ++i) {
        uint ai = a[i];

        ulong carry = 0ul;
        for (uint j = 0u; j < limbs; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong top = (ulong)t[limbs] + carry;
        t[limbs] = (uint)top;
        t[limbs + 1u] = (uint)(top >> 32);

        uint m = (uint)((ulong)t[0] * (ulong)np0);
        carry = 0ul;
        for (uint j = 0u; j < limbs; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
            if (j > 0u) {
                t[j - 1u] = (uint)uv;
            }
            carry = uv >> 32;
        }
        top = (ulong)t[limbs] + carry;
        t[limbs - 1u] = (uint)top;
        top = (ulong)t[limbs + 1u] + (top >> 32);
        t[limbs] = (uint)top;
        t[limbs + 1u] = (uint)(top >> 32);
    }

    ulong borrow = 0ul;
    uint D[MAX_LIMBS];
    for (uint i = 0u; i < limbs; ++i) {
        ulong tv = (ulong)t[i];
        ulong nv = (ulong)N[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }

    uint need_sub = (t[limbs] != 0u || t[limbs + 1u] != 0u) ? 1u : 0u;
    need_sub = (borrow == 0u) ? 1u : need_sub;
    uint mask = 0u - need_sub;

    for (uint i = 0u; i < limbs; ++i) {
        out[i] = (D[i] & mask) | (t[i] & ~mask);
    }
}

static inline void mont_mul_stage1_unroll32(uint *out, const uint *a, const uint *b,
                                            const uint *N, uint np0, uint limbs) {
    uint t[MAX_LIMBS + 2u];
    for (uint i = 0u; i < limbs + 2u; ++i) t[i] = 0u;
    uint B[MAX_LIMBS];
    for (uint j = 0u; j < limbs; ++j) B[j] = b[j];
    for (uint i = 0u; i < limbs; ++i) {
        uint ai = a[i];
        ulong carry = 0ul;
        #pragma unroll 32
        for (uint j = 0u; j < limbs; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong top = (ulong)t[limbs] + carry;
        t[limbs] = (uint)top;
        t[limbs + 1u] = (uint)(top >> 32);
        uint m = (uint)((ulong)t[0] * (ulong)np0);
        carry = 0ul;
        #pragma unroll 32
        for (uint j = 0u; j < limbs; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
            if (j > 0u) t[j - 1u] = (uint)uv;
            carry = uv >> 32;
        }
        top = (ulong)t[limbs] + carry;
        t[limbs - 1u] = (uint)top;
        top = (ulong)t[limbs + 1u] + (top >> 32);
        t[limbs] = (uint)top;
        t[limbs + 1u] = (uint)(top >> 32);
    }
    ulong borrow = 0ul;
    uint D[MAX_LIMBS];
    for (uint i = 0u; i < limbs; ++i) {
        ulong tv = (ulong)t[i], nv = (ulong)N[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }
    uint need_sub = (t[limbs] != 0u || t[limbs + 1u] != 0u) ? 1u : 0u;
    need_sub = (borrow == 0u) ? 1u : need_sub;
    uint mask = 0u - need_sub;
    for (uint i = 0u; i < limbs; ++i) out[i] = (D[i] & mask) | (t[i] & ~mask);
}

#if ECM_STAGE1_USE_I24_384
static inline void mont_mul_stage1_i24(uint *out, const uint *a, const uint *b, const uint *N,
                                       uint np0) {
#if ECM_STAGE1_I24_U32_BLSUB
    mont_mul_unroll_i24_u32_blsub_priv_body(out, a, b, N, np0);
#else
    mont_mul_unroll_i24_u32_priv_body(out, a, b, N, np0);
#endif
}

static inline void mont_sqr_stage1_i24(uint *out, const uint *a, const uint *N, uint np0) {
#if ECM_STAGE1_I24_U32_BLSUB
    mont_sqr_unroll_i24_u32_blsub_priv_body(out, a, N, np0);
#else
    mont_sqr_unroll_i24_u32_priv_body(out, a, N, np0);
#endif
}
#endif

// Default stage1 montgomery selector:
// - 512-bit: use fixed unroll-only path
// - 4096-bit: use fixed unroll64 path
// - others: use generic priv_opt path (bench: faster than unroll32 on Adreno)
static inline void mont_mul_stage1(uint *out, const uint *a, const uint *b,
                                   const uint *N, uint np0, uint limbs) {
#if ECM_STAGE1_USE_I24_384
    if (limbs == MAX_LIMBS) {
        mont_mul_stage1_i24(out, a, b, N, np0);
        return;
    }
#endif
#if ECM_STAGE1_MUL_FORCE_UNROLL32
    mont_mul_stage1_unroll32(out, a, b, N, np0, limbs);
    return;
#endif
#if ECM_STAGE1_MUL_FORCE_PRIV_OPT
    mont_mul_stage1_priv_opt(out, a, b, N, np0, limbs);
    return;
#endif
    if (limbs == 16u) {
#if ECM_STAGE1_MUL_FORCE_UNROLL384
        mont_mul_stage1_unroll_only_384(out, a, b, N, np0);
#else
        mont_mul_stage1_unroll_only_512(out, a, b, N, np0);
#endif
    } else if (limbs == 128u) {
#if ECM_STAGE1_MUL_PATH == 2
        mont_mul_stage1_fips4096(out, a, b, N, np0);
#else
        mont_mul_stage1_unroll64_4096(out, a, b, N, np0);
#endif
    } else {
        mont_mul_stage1_priv_opt(out, a, b, N, np0, limbs);
    }
}

static inline void mont_sqr_stage1(uint *out, const uint *a,
                                   const uint *N, uint np0, uint limbs) {
#if ECM_STAGE1_USE_I24_384
    if (limbs == MAX_LIMBS) {
        mont_sqr_stage1_i24(out, a, N, np0);
        return;
    }
#endif
#if ECM_STAGE1_SQR_FORCE_UNROLL32
    mont_mul_stage1_unroll32(out, a, a, N, np0, limbs);
    return;
#endif
#if ECM_STAGE1_SQR_FORCE_PRIV_OPT
    mont_mul_stage1_priv_opt(out, a, a, N, np0, limbs);
    return;
#endif
    if (limbs == 16u) {
#if ECM_STAGE1_SQR_FORCE_UNROLL384
        mont_mul_stage1_unroll_only_384(out, a, a, N, np0);
#else
        mont_mul_stage1_unroll_only_512(out, a, a, N, np0);
#endif
    } else if (limbs == 128u) {
#if ECM_STAGE1_SQR_PATH == 2
        mont_mul_stage1_fips4096(out, a, a, N, np0);
#else
        mont_mul_stage1_unroll64_4096(out, a, a, N, np0);
#endif
    } else {
        mont_mul_stage1_priv_opt(out, a, a, N, np0, limbs);
    }
}

// ---------------------------------------------------------------------------
// Private multi-limb helpers (one curve per work-item)
//
// Naming convention:
//   `mp_*` means "multi-precision integer primitive". These are the low-level
//   bignum building blocks used by higher-level curve operations.
//
// Why keep the `mp_` prefix:
// 1) Distinguish bignum operators from point/curve operators (double_add_v2, etc.).
// 2) Make call-sites read like arithmetic formulas over Z and Z/NZ.
// 3) Avoid ambiguity with OpenCL scalar/vector add/sub intrinsics.
// ---------------------------------------------------------------------------

static inline void mp_copy(uint *dst, const uint *src, uint limbs) {
    for (uint i = 0u; i < limbs; ++i) {
        dst[i] = src[i];
    }
}

static inline void mp_zero(uint *dst, uint limbs) {
    for (uint i = 0u; i < limbs; ++i) {
        dst[i] = 0u;
    }
}

static inline int mp_ge(const uint *a, const uint *N, uint limbs) {
    for (int i = (int)limbs - 1; i >= 0; --i) {
        if (a[(uint)i] > N[(uint)i]) return 1;
        if (a[(uint)i] < N[(uint)i]) return 0;
    }
    return 1;
}

static inline void mp_sub_n(uint *r, const uint *a, const uint *N, uint limbs) {
    ulong borrow = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong av = (ulong)a[i];
        ulong nv = (ulong)N[i];
        ulong w = av - nv - borrow;
        r[i] = (uint)w;
        borrow = (av < nv + borrow) ? 1ul : 0ul;
    }
}

static inline uint mp_add_n(uint *r, const uint *a, const uint *b, uint limbs) {
    ulong carry = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong sum = (ulong)a[i] + (ulong)b[i] + carry;
        r[i] = (uint)sum;
        carry = sum >> 32;
    }
    return (uint)carry;
}

// Modular add over Z/NZ:
//   r = a + b (mod N)
//
// Implementation idea:
//   1) Compute raw limb sum S = a + b in base 2^32.
//   2) If S overflowed (carry != 0) OR S >= N, subtract N once.
// This is valid because a,b are already reduced, so S is in [0, 2N-2].
#ifndef MP_ADD_MOD_FUSED_UNROLL
#define MP_ADD_MOD_FUSED_UNROLL 2
#endif

#ifndef ECM_STAGE1_ADDMOD_PATH
#define ECM_STAGE1_ADDMOD_PATH 2
#endif
#ifndef ECM_STAGE1_SUBMOD_PATH
#define ECM_STAGE1_SUBMOD_PATH 1
#endif
#ifndef ECM_STAGE1_ASM_B32
#define ECM_STAGE1_ASM_B32 0
#endif
#ifndef ECM_STAGE1_ASM_B16
#define ECM_STAGE1_ASM_B16 0
#endif

#define ECM_ADDSUB_PATH_FUSED 0
#define ECM_ADDSUB_PATH_FUSED_UNROLL 1
#define ECM_ADDSUB_PATH_FUSED_UNROLL_B32 2
#define ECM_ADDSUB_PATH_ASM_B32 3
#define ECM_ADDSUB_PATH_FUSED_UNROLL_B16 4
#define ECM_ADDSUB_PATH_ASM_B16 5

// Hint for compile-time limb unroll (MAX_LIMBS is fixed per kernel build).
#if MAX_LIMBS <= 16
#define ECM_ADDSUB_UNROLL_HINT 16
#elif MAX_LIMBS <= 32
#define ECM_ADDSUB_UNROLL_HINT 32
#elif MAX_LIMBS <= 64
#define ECM_ADDSUB_UNROLL_HINT 64
#else
#define ECM_ADDSUB_UNROLL_HINT 32
#endif

#if ECM_STAGE1_USE_I24_384
#ifndef MONT_I24_LIMB_MASK
#define MONT_I24_LIMB_MASK 0xFFFFFFu
#endif

static inline void mp_add_mod_fused_unroll_i24(uint *r, const uint *a, const uint *b,
                                               const uint *N) {
    const uint mask = MONT_I24_LIMB_MASK;
    ulong carry_add = 0ul;
    ulong carry_sub = 1ul;
    #pragma unroll
    for (uint i = 0u; i < MAX_LIMBS; ++i) {
        ulong sum = (ulong)(a[i] & mask) + (ulong)(b[i] & mask) + carry_add;
        carry_add = sum >> 24;
        ulong temp = (ulong)(uint)(sum & mask) + (ulong)(~(N[i] & mask) & mask) + carry_sub;
        carry_sub = temp >> 24;
        r[i] = (uint)(temp & mask);
    }
    if ((carry_add | carry_sub) != 0ul) {
        return;
    }
    ulong c = 0ul;
    #pragma unroll
    for (uint i = 0u; i < MAX_LIMBS; ++i) {
        ulong s = (ulong)(r[i] & mask) + (ulong)(N[i] & mask) + c;
        r[i] = (uint)(s & mask);
        c = s >> 24;
    }
}

static inline int mp_sub_mod_fused_unroll_i24(uint *r, const uint *a, const uint *b,
                                              const uint *N) {
    const uint mask = MONT_I24_LIMB_MASK;
    ulong br = 0ul;
    #pragma unroll
    for (uint i = 0u; i < MAX_LIMBS; ++i) {
        ulong av = (ulong)(a[i] & mask);
        ulong bv = (ulong)(b[i] & mask);
        ulong w = av - bv - br;
        r[i] = (uint)(w & mask);
        br = (av < bv + br) ? 1ul : 0ul;
    }
    if (br != 0ul) {
        ulong c = 0ul;
        #pragma unroll
        for (uint i = 0u; i < MAX_LIMBS; ++i) {
            ulong s = (ulong)(r[i] & mask) + (ulong)(N[i] & mask) + c;
            r[i] = (uint)(s & mask);
            c = s >> 24;
        }
        return 1;
    }
    return 0;
}

static inline int mp_ge_i24(const uint *a, const uint *N, uint limbs) {
    const uint mask = MONT_I24_LIMB_MASK;
    for (int i = (int)limbs - 1; i >= 0; --i) {
        const uint av = a[(uint)i] & mask;
        const uint nv = N[(uint)i] & mask;
        if (av > nv) {
            return 1;
        }
        if (av < nv) {
            return 0;
        }
    }
    return 1;
}

static inline void mp_sub_n_i24(uint *r, const uint *a, const uint *N, uint limbs) {
    const uint mask = MONT_I24_LIMB_MASK;
    ulong borrow = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        const ulong av = (ulong)(a[i] & mask);
        const ulong nv = (ulong)(N[i] & mask);
        const ulong w = av - nv - borrow;
        r[i] = (uint)(w & mask);
        borrow = (av < nv + borrow) ? 1ul : 0ul;
    }
}
#endif

// Generic fused add-mod: full compile-time unroll for MAX_LIMBS (421b, 512b, etc.).
static inline void mp_add_mod_fused_unroll(uint *r, const uint *a, const uint *b, const uint *N) {
    ulong carry_add = 0ul;
    ulong carry_sub = 1ul;
    #pragma unroll ECM_ADDSUB_UNROLL_HINT
    for (uint i = 0u; i < MAX_LIMBS; ++i) {
        ulong sum = (ulong)a[i] + (ulong)b[i] + carry_add;
        carry_add = sum >> 32;
        ulong temp = (ulong)(uint)sum + (ulong)(~N[i]) + carry_sub;
        carry_sub = temp >> 32;
        r[i] = (uint)temp;
    }
    if ((carry_add | carry_sub) != 0ul) {
        return;
    }
    ulong c = 0ul;
    #pragma unroll ECM_ADDSUB_UNROLL_HINT
    for (uint i = 0u; i < MAX_LIMBS; ++i) {
        ulong s = (ulong)r[i] + (ulong)N[i] + c;
        r[i] = (uint)s;
        c = s >> 32;
    }
}

static inline int mp_sub_mod_fused_unroll(uint *r, const uint *a, const uint *b, const uint *N) {
    ulong br = 0ul;
    #pragma unroll ECM_ADDSUB_UNROLL_HINT
    for (uint i = 0u; i < MAX_LIMBS; ++i) {
        ulong av = (ulong)a[i];
        ulong bv = (ulong)b[i];
        ulong w = av - bv - br;
        r[i] = (uint)w;
        br = (av < bv + br) ? 1ul : 0ul;
    }
    if (br != 0ul) {
        ulong c = 0ul;
        #pragma unroll ECM_ADDSUB_UNROLL_HINT
        for (uint i = 0u; i < MAX_LIMBS; ++i) {
            ulong s = (ulong)r[i] + (ulong)N[i] + c;
            r[i] = (uint)s;
            c = s >> 32;
        }
        return 1;
    }
    return 0;
}

// 512-bit alias (16 limbs): same as mp_add_mod_fused_unroll when MAX_LIMBS==16.
static inline void mp_add_mod_fused_unroll_b16_512(uint *r, const uint *a, const uint *b,
                                                     const uint *N) {
    mp_add_mod_fused_unroll(r, a, b, N);
}

static inline int mp_sub_mod_fused_unroll_b16_512(uint *r, const uint *a, const uint *b,
                                                  const uint *N) {
    return mp_sub_mod_fused_unroll(r, a, b, N);
}

// 4 blocks x 32 limbs, matching the b32 unroll structure.
static inline void mp_add_mod_fused_unroll_b32_4096(uint *r, const uint *a, const uint *b, const uint *N) {
    ulong carry_add = 0ul;
    ulong carry_sub = 1ul;

    #pragma unroll
    for (uint blk = 0u; blk < 4u; ++blk) {
        uint off = blk * 32u;
        #pragma unroll 32
        for (uint j = 0u; j < 32u; ++j) {
            uint i = off + j;
            ulong sum = (ulong)a[i] + (ulong)b[i] + carry_add;
            carry_add = sum >> 32;
            ulong temp = (ulong)(uint)sum + (ulong)(~N[i]) + carry_sub;
            carry_sub = temp >> 32;
            r[i] = (uint)temp;
        }
    }

    if ((carry_add | carry_sub) != 0ul) {
        return;
    }
    ulong c = 0ul;
    #pragma unroll 32
    for (uint i = 0u; i < 128u; ++i) {
        ulong s = (ulong)r[i] + (ulong)N[i] + c;
        r[i] = (uint)s;
        c = s >> 32;
    }
}

static inline int mp_sub_mod_fused_unroll_b32_4096(uint *r, const uint *a, const uint *b,
                                                   const uint *N) {
    ulong br = 0ul;
    #pragma unroll
    for (uint blk = 0u; blk < 4u; ++blk) {
        uint off = blk * 32u;
        #pragma unroll 32
        for (uint j = 0u; j < 32u; ++j) {
            uint i = off + j;
            ulong av = (ulong)a[i];
            ulong bv = (ulong)b[i];
            ulong w = av - bv - br;
            r[i] = (uint)w;
            br = (av < bv + br) ? 1ul : 0ul;
        }
    }
    if (br != 0ul) {
        ulong c = 0ul;
        #pragma unroll 32
        for (uint i = 0u; i < 128u; ++i) {
            ulong s = (ulong)r[i] + (ulong)N[i] + c;
            r[i] = (uint)s;
            c = s >> 32;
        }
        return 1;
    }
    return 0;
}

#if ECM_STAGE1_ASM_B32 && defined(__AMDGCN__)
static inline void mp_add_mod_asm_b32_4096(uint *r, const uint *a, const uint *b, const uint *N) {
    uint ca = 0u, cs = 1u;
    #pragma unroll
    for (uint blk = 0u; blk < 4u; ++blk) {
        uint off = blk * 32u;
        asm_fused_block32_priv(a + off, b + off, N + off, r + off, ca, cs, &ca, &cs);
    }
    if ((ca | cs) == 0u) {
        ulong c = 0ul;
        #pragma unroll 32
        for (uint i = 0u; i < 128u; ++i) {
            ulong s = (ulong)r[i] + (ulong)N[i] + c;
            r[i] = (uint)s;
            c = s >> 32;
        }
    }
}

static inline int mp_sub_mod_asm_b32_4096(uint *r, const uint *a, const uint *b, const uint *N) {
    uint br = 0u;
    #pragma unroll
    for (uint blk = 0u; blk < 4u; ++blk) {
        uint off = blk * 32u;
        asm_sub_fused_block32_priv(a + off, b + off, N + off, r + off, br, &br);
    }
    if (br != 0u) {
        ulong c = 0ul;
        #pragma unroll 32
        for (uint i = 0u; i < 128u; ++i) {
            ulong s = (ulong)r[i] + (ulong)N[i] + c;
            r[i] = (uint)s;
            c = s >> 32;
        }
        return 1;
    }
    return 0;
}
#endif

#if ECM_STAGE1_ASM_B16 && defined(__AMDGCN__)
static inline void mp_add_mod_asm_b16_512(uint *r, const uint *a, const uint *b, const uint *N) {
    uint ca = 0u, cs = 1u;
    asm_fused_block16_priv(a, b, N, r, ca, cs, &ca, &cs);
}
#endif

static inline void mp_add_mod(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {
#if ECM_STAGE1_USE_I24_384
    if (limbs == MAX_LIMBS) {
        mp_add_mod_fused_unroll_i24(r, a, b, N);
        return;
    }
#endif
    if (limbs == 128u) {
#if ECM_STAGE1_ADDMOD_PATH == ECM_ADDSUB_PATH_ASM_B32
#if ECM_STAGE1_ASM_B32 && defined(__AMDGCN__)
        mp_add_mod_asm_b32_4096(r, a, b, N);
        return;
#else
        mp_add_mod_fused_unroll_b32_4096(r, a, b, N);
        return;
#endif
#elif ECM_STAGE1_ADDMOD_PATH >= ECM_ADDSUB_PATH_FUSED_UNROLL
        mp_add_mod_fused_unroll_b32_4096(r, a, b, N);
        return;
#endif
    }
#if ECM_STAGE1_ADDMOD_PATH == ECM_ADDSUB_PATH_ASM_B16
#if ECM_STAGE1_ASM_B16 && defined(__AMDGCN__)
    if (limbs == 16u) {
        mp_add_mod_asm_b16_512(r, a, b, N);
        return;
    }
#else
    if (limbs == 16u) {
        mp_add_mod_fused_unroll_b16_512(r, a, b, N);
        return;
    }
#endif
#elif ECM_STAGE1_ADDMOD_PATH >= ECM_ADDSUB_PATH_FUSED_UNROLL_B16
    if (limbs == 16u) {
        mp_add_mod_fused_unroll_b16_512(r, a, b, N);
        return;
    }
#endif
#if ECM_STAGE1_ADDMOD_PATH >= ECM_ADDSUB_PATH_FUSED_UNROLL
    if (limbs == MAX_LIMBS) {
        mp_add_mod_fused_unroll(r, a, b, N);
        return;
    }
#endif
    ulong carry_add = 0ul;
    ulong carry_sub = 1ul;
#if MP_ADD_MOD_FUSED_UNROLL == 2
    uint j = 0u;
    for (; j + 1u < limbs; j += 2u) {
        ulong sum0 = (ulong)a[j] + (ulong)b[j] + carry_add;
        carry_add = sum0 >> 32;
        ulong temp0 = (ulong)(uint)sum0 + (ulong)(~N[j]) + carry_sub;
        carry_sub = temp0 >> 32;
        r[j] = (uint)temp0;

        ulong sum1 = (ulong)a[j + 1u] + (ulong)b[j + 1u] + carry_add;
        carry_add = sum1 >> 32;
        ulong temp1 = (ulong)(uint)sum1 + (ulong)(~N[j + 1u]) + carry_sub;
        carry_sub = temp1 >> 32;
        r[j + 1u] = (uint)temp1;
    }
    if (limbs & 1u) {
        ulong sum = (ulong)a[j] + (ulong)b[j] + carry_add;
        carry_add = sum >> 32;
        ulong temp = (ulong)(uint)sum + (ulong)(~N[j]) + carry_sub;
        carry_sub = temp >> 32;
        r[j] = (uint)temp;
    }
#else
    for (uint i = 0u; i < limbs; ++i) {
        ulong sum = (ulong)a[i] + (ulong)b[i] + carry_add;
        carry_add = sum >> 32;
        ulong temp = (ulong)(uint)sum + (ulong)(~N[i]) + carry_sub;
        carry_sub = temp >> 32;
        r[i] = (uint)temp;
    }
#endif
    if ((carry_add | carry_sub) != 0ul) {
        return;
    }
    ulong c = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong s = (ulong)r[i] + (ulong)N[i] + c;
        r[i] = (uint)s;
        c = s >> 32;
    }
}

// Returns 1 if borrow (a < b)
static inline int mp_sub_mod(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {
#if ECM_STAGE1_USE_I24_384
    if (limbs == MAX_LIMBS) {
        return mp_sub_mod_fused_unroll_i24(r, a, b, N);
    }
#endif
    if (limbs == 128u) {
#if ECM_STAGE1_SUBMOD_PATH == ECM_ADDSUB_PATH_ASM_B32
#if ECM_STAGE1_ASM_B32 && defined(__AMDGCN__)
        return mp_sub_mod_asm_b32_4096(r, a, b, N);
#else
        return mp_sub_mod_fused_unroll_b32_4096(r, a, b, N);
#endif
#elif ECM_STAGE1_SUBMOD_PATH >= ECM_ADDSUB_PATH_FUSED_UNROLL
        return mp_sub_mod_fused_unroll_b32_4096(r, a, b, N);
#endif
    }
#if ECM_STAGE1_SUBMOD_PATH >= ECM_ADDSUB_PATH_FUSED_UNROLL_B16
    if (limbs == 16u) {
        return mp_sub_mod_fused_unroll_b16_512(r, a, b, N);
    }
#endif
#if ECM_STAGE1_SUBMOD_PATH >= ECM_ADDSUB_PATH_FUSED_UNROLL
    if (limbs == MAX_LIMBS) {
        return mp_sub_mod_fused_unroll(r, a, b, N);
    }
#endif
    ulong borrow = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong av = (ulong)a[i];
        ulong bv = (ulong)b[i];
        ulong w = av - bv - borrow;
        r[i] = (uint)w;
        borrow = (av < bv + borrow) ? 1ul : 0ul;
    }
    if (borrow) {
        // For subtraction underflow, add modulus without modular reduction.
        (void)mp_add_n(r, r, N, limbs);
        return 1;
    }
    return 0;
}

#if ECM_STAGE1_USE_I24_384
static inline uint mp_add_n_i24(uint *r, const uint *a, const uint *b, uint limbs) {
    const uint mask = MONT_I24_LIMB_MASK;
    ulong carry = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong sum = (ulong)(a[i] & mask) + (ulong)(b[i] & mask) + carry;
        r[i] = (uint)(sum & mask);
        carry = sum >> 24;
    }
    return (uint)carry;
}

static inline void shift_right_24_limbs(uint *r, uint limbs) {
    for (uint i = 0u; i + 1u < limbs; ++i) {
        r[i] = r[i + 1u];
    }
    r[limbs - 1u] = 0u;
}

static inline uint mul_ui24_limbs(uint *r, uint m, uint limbs) {
    const uint mask = MONT_I24_LIMB_MASK;
    ulong carry = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong prod = mont_i24_mul_full(r[i] & mask, m) + carry;
        r[i] = (uint)(prod & mask);
        carry = prod >> 24;
    }
    return (uint)carry;
}

static inline void special_mult_ui24(uint *r, uint m, const uint *N, uint np0, uint limbs) {
    uint carry_t1 = mul_ui24_limbs(r, m, limbs);
    const uint t1_0 = r[0] & MONT_I24_LIMB_MASK;
    const uint q = mul24(t1_0, np0);

    uint temp[MAX_LIMBS];
    mp_copy(temp, N, limbs);
    const uint carry_t2 = mul_ui24_limbs(temp, q, limbs);

    shift_right_24_limbs(r, limbs);
    shift_right_24_limbs(temp, limbs);
    r[limbs - 1u] = carry_t1;
    temp[limbs - 1u] = carry_t2;

    int carry_q = (int)mp_add_n_i24(r, r, temp, limbs);
    if (t1_0 != 0u) {
        ulong c = 1ul;
        for (uint i = 0u; i < limbs; ++i) {
            ulong sum = (ulong)(r[i] & MONT_I24_LIMB_MASK) + c;
            r[i] = (uint)(sum & MONT_I24_LIMB_MASK);
            c = sum >> 24;
        }
        carry_q += (int)c;
    }
    if (carry_q > 0) {
        mp_sub_n_i24(r, r, N, limbs);
    }
    if (mp_ge_i24(r, N, limbs)) {
        mp_sub_n_i24(r, r, N, limbs);
    }
}

static inline void mp_shift_left_1_mod_i24(uint *r, const uint *a, const uint *N, uint limbs) {
    const uint mask = MONT_I24_LIMB_MASK;
    uint carry = 0u;
    for (uint i = 0u; i < limbs; ++i) {
        const uint old = a[i] & mask;
        r[i] = ((old << 1) | carry) & mask;
        carry = old >> 23;
    }
    if (carry != 0u || mp_ge_i24(r, N, limbs)) {
        mp_sub_n_i24(r, r, N, limbs);
    }
}
#endif

static inline uint mul_ui32_limbs(uint *r, uint m, uint limbs);
static inline void shift_right_32_limbs(uint *r, uint limbs);
static inline void special_mult_ui32(uint *r, uint m, const uint *N, uint np0, uint limbs);

static inline void special_mult_stage1(uint *r, uint m, const uint *N, uint np0, uint limbs) {
#if ECM_STAGE1_USE_I24_384
    if (limbs == MAX_LIMBS) {
        special_mult_ui24(r, m, N, np0, limbs);
        return;
    }
#endif
    special_mult_ui32(r, m, N, np0, limbs);
}

static inline void mp_shift_left_1_mod(uint *r, const uint *a, const uint *N, uint limbs) {
#if ECM_STAGE1_USE_I24_384
    if (limbs == MAX_LIMBS) {
        mp_shift_left_1_mod_i24(r, a, N, limbs);
        return;
    }
#endif
    uint carry = 0u;
    for (uint i = 0u; i < limbs; ++i) {
        uint old = a[i];
        r[i] = (old << 1) | carry;
        carry = old >> 31;
    }
    if (carry || mp_ge(r, N, limbs)) {
        mp_sub_n(r, r, N, limbs);
    }
}

static inline void mont_normalize(uint *r, const uint *N, uint limbs) {
#if ECM_STAGE1_USE_I24_384
    if (limbs == MAX_LIMBS) {
        if (mp_ge_i24(r, N, limbs)) {
            mp_sub_n_i24(r, r, N, limbs);
        }
        return;
    }
#endif
    if (mp_ge(r, N, limbs)) {
        mp_sub_n(r, r, N, limbs);
    }
}

static inline void maybe_mont_normalize(uint *r, const uint *N, uint limbs) {
#if ECM_STAGE1_FORCE_NORMALIZE
    mont_normalize(r, N, limbs);
#else
    (void)r;
    (void)N;
    (void)limbs;
#endif
}

// r <- low limbs of (r * m); returns overflow limb above r
static inline uint mul_ui32_limbs(uint *r, uint m, uint limbs) {
    ulong carry = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong prod = (ulong)r[i] * (ulong)m + carry;
        r[i] = (uint)prod;
        carry = prod >> 32;
    }
    return (uint)carry;
}

static inline void shift_right_32_limbs(uint *r, uint limbs) {
    for (uint i = 0u; i + 1u < limbs; ++i) {
        r[i] = r[i + 1u];
    }
    r[limbs - 1u] = 0u;
}

// (r * m) / 2^32 mod N — ported from CUDA curve_t::special_mult_ui32
static inline void special_mult_ui32(uint *r, uint m, const uint *N, uint np0, uint limbs) {
    uint carry_t1 = mul_ui32_limbs(r, m, limbs);
    uint t1_0 = r[0];
    uint q = (uint)((ulong)t1_0 * (ulong)np0);

    uint temp[MAX_LIMBS];
    mp_copy(temp, N, limbs);
    uint carry_t2 = mul_ui32_limbs(temp, q, limbs);

    shift_right_32_limbs(r, limbs);
    shift_right_32_limbs(temp, limbs);
    r[limbs - 1u] = carry_t1;
    temp[limbs - 1u] = carry_t2;

    {
        int carry_q = (int)mp_add_n(r, r, temp, limbs);
        if (t1_0 != 0u) {
            uint carry1 = 1u;
            for (uint i = 0u; i < limbs && carry1 != 0u; ++i) {
                ulong sum = (ulong)r[i] + (ulong)carry1;
                r[i] = (uint)sum;
                carry1 = (uint)(sum >> 32);
            }
            carry_q += (int)carry1;
        }
        if (carry_q > 0) {
            mp_sub_n(r, r, N, limbs);
        }
        if (mp_ge(r, N, limbs)) {
            mp_sub_n(r, r, N, limbs);
        }
    }
}

#if ECM_STAGE1_USE_COOP_WG
static inline void mont_mul_stage1_unroll64_4096_local(__local uint *out, __local const uint *a,
                                                       __local const uint *b, __local const uint *N,
                                                       uint np0) {
    uint t[128u + 2u];
    for (uint i = 0u; i < 130u; ++i) t[i] = 0u;
    uint B[128u];
    for (uint j = 0u; j < 128u; ++j) B[j] = b[j];
    for (uint i = 0u; i < 128u; ++i) {
        uint ai = a[i];
        ulong carry = 0ul;
        #pragma unroll 64
        for (uint j = 0u; j < 128u; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong top = (ulong)t[128u] + carry;
        t[128u] = (uint)top;
        t[129u] = (uint)(top >> 32);
        uint m = (uint)((ulong)t[0] * (ulong)np0);
        ulong uv0 = (ulong)t[0] + (ulong)m * (ulong)N[0];
        carry = uv0 >> 32;
        #pragma unroll 64
        for (uint j = 1u; j < 128u; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
            t[j - 1u] = (uint)uv;
            carry = uv >> 32;
        }
        top = (ulong)t[128u] + carry;
        t[127u] = (uint)top;
        top = (ulong)t[129u] + (top >> 32);
        t[128u] = (uint)top;
        t[129u] = (uint)(top >> 32);
    }
    ulong borrow = 0ul;
    uint D[128u];
    for (uint i = 0u; i < 128u; ++i) {
        ulong tv = (ulong)t[i], nv = (ulong)N[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }
    uint need_sub = (t[128u] != 0u || t[129u] != 0u) ? 1u : 0u;
    need_sub = (borrow == 0u) ? 1u : need_sub;
    uint mask = 0u - need_sub;
    for (uint i = 0u; i < 128u; ++i) out[i] = (D[i] & mask) | (t[i] & ~mask);
}

static inline void mp_add_mod_fused_unroll_b32_4096_local(__local uint *r, __local const uint *a,
                                                          __local const uint *b,
                                                          __local const uint *N) {
    ulong carry_add = 0ul;
    ulong carry_sub = 1ul;
    #pragma unroll
    for (uint blk = 0u; blk < 4u; ++blk) {
        uint off = blk * 32u;
        #pragma unroll 32
        for (uint j = 0u; j < 32u; ++j) {
            uint i = off + j;
            ulong sum = (ulong)a[i] + (ulong)b[i] + carry_add;
            carry_add = sum >> 32;
            ulong temp = (ulong)(uint)sum + (ulong)(~N[i]) + carry_sub;
            carry_sub = temp >> 32;
            r[i] = (uint)temp;
        }
    }
    if ((carry_add | carry_sub) != 0ul) {
        return;
    }
    ulong c = 0ul;
    #pragma unroll 32
    for (uint i = 0u; i < 128u; ++i) {
        ulong s = (ulong)r[i] + (ulong)N[i] + c;
        r[i] = (uint)s;
        c = s >> 32;
    }
}

static inline void mp_add_mod_local(__local uint *r, __local const uint *a, __local const uint *b,
                                    __local const uint *N, uint limbs) {
    if (limbs == 128u) {
#if ECM_STAGE1_ADDMOD_PATH == ECM_ADDSUB_PATH_ASM_B32
#if ECM_STAGE1_ASM_B32 && defined(__AMDGCN__)
        mp_add_mod_asm_b32_4096(r, a, b, N);
        (void)limbs;
        return;
#else
        mp_add_mod_fused_unroll_b32_4096_local(r, a, b, N);
        (void)limbs;
        return;
#endif
#elif ECM_STAGE1_ADDMOD_PATH >= ECM_ADDSUB_PATH_FUSED_UNROLL
        mp_add_mod_fused_unroll_b32_4096_local(r, a, b, N);
        (void)limbs;
        return;
#endif
    }
#if ECM_STAGE1_ADDMOD_PATH == ECM_ADDSUB_PATH_ASM_B16
#if ECM_STAGE1_ASM_B16 && defined(__AMDGCN__)
    if (limbs == 16u) {
        mp_add_mod_asm_b16_512(r, a, b, N);
        (void)limbs;
        return;
    }
#else
    if (limbs == 16u) {
        mp_add_mod_fused_unroll_b16_512(r, a, b, N);
        (void)limbs;
        return;
    }
#endif
#elif ECM_STAGE1_ADDMOD_PATH >= ECM_ADDSUB_PATH_FUSED_UNROLL_B16
    if (limbs == 16u) {
        mp_add_mod_fused_unroll_b16_512(r, a, b, N);
        (void)limbs;
        return;
    }
#endif
#if ECM_STAGE1_ADDMOD_PATH >= ECM_ADDSUB_PATH_FUSED_UNROLL
    if (limbs == MAX_LIMBS) {
        mp_add_mod_fused_unroll(r, a, b, N);
        (void)limbs;
        return;
    }
#endif
    mp_add_mod(r, a, b, N, limbs);
}

static inline uint mp_add_n_local(__local uint *r, __local const uint *a, __local const uint *b,
                                  uint limbs) {
    ulong carry = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong sum = (ulong)a[i] + (ulong)b[i] + carry;
        r[i] = (uint)sum;
        carry = sum >> 32;
    }
    return (uint)carry;
}

static inline void mp_sub_n_local(__local uint *r, __local const uint *a, __local const uint *N,
                                  uint limbs) {
    ulong borrow = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong av = (ulong)a[i];
        ulong nv = (ulong)N[i];
        ulong w = av - nv - borrow;
        r[i] = (uint)w;
        borrow = (av < nv + borrow) ? 1ul : 0ul;
    }
}

static inline int mp_sub_mod_local(__local uint *r, __local const uint *a, __local const uint *b,
                                     __local const uint *N, uint limbs) {
    if (limbs == 128u) {
#if ECM_STAGE1_SUBMOD_PATH == ECM_ADDSUB_PATH_ASM_B32
#if ECM_STAGE1_ASM_B32 && defined(__AMDGCN__)
        return mp_sub_mod_asm_b32_4096(r, a, b, N);
#else
        return mp_sub_mod_fused_unroll_b32_4096(r, a, b, N);
#endif
#elif ECM_STAGE1_SUBMOD_PATH >= ECM_ADDSUB_PATH_FUSED_UNROLL
        return mp_sub_mod_fused_unroll_b32_4096(r, a, b, N);
#endif
    }
#if ECM_STAGE1_SUBMOD_PATH >= ECM_ADDSUB_PATH_FUSED_UNROLL_B16
    if (limbs == 16u) {
        return mp_sub_mod_fused_unroll_b16_512(r, a, b, N);
    }
#endif
#if ECM_STAGE1_SUBMOD_PATH >= ECM_ADDSUB_PATH_FUSED_UNROLL
    if (limbs == MAX_LIMBS) {
        return mp_sub_mod_fused_unroll(r, a, b, N);
    }
#endif
    ulong borrow = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong av = (ulong)a[i];
        ulong bv = (ulong)b[i];
        ulong w = av - bv - borrow;
        r[i] = (uint)w;
        borrow = (av < bv + borrow) ? 1ul : 0ul;
    }
    if (borrow) {
        (void)mp_add_n_local(r, r, N, limbs);
        return 1;
    }
    return 0;
}

static inline void mp_copy_local(__local uint *dst, __local const uint *src, uint limbs) {
    for (uint i = 0u; i < limbs; ++i) {
        dst[i] = src[i];
    }
}

static inline int mp_ge_local(__local const uint *a, __local const uint *N, uint limbs) {
    for (int i = (int)limbs - 1; i >= 0; --i) {
        if (a[(uint)i] > N[(uint)i]) return 1;
        if (a[(uint)i] < N[(uint)i]) return 0;
    }
    return 1;
}

static inline void mont_normalize_local(__local uint *r, __local const uint *N, uint limbs) {
    if (mp_ge_local(r, N, limbs)) {
        mp_sub_n_local(r, r, N, limbs);
    }
}

static inline void maybe_mont_normalize_local(__local uint *r, __local const uint *N, uint limbs) {
#if ECM_STAGE1_FORCE_NORMALIZE
    mont_normalize_local(r, N, limbs);
#else
    (void)r;
    (void)N;
    (void)limbs;
#endif
}

static inline uint mul_ui32_limbs_local(__local uint *r, uint m, uint limbs) {
    ulong carry = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong prod = (ulong)r[i] * (ulong)m + carry;
        r[i] = (uint)prod;
        carry = prod >> 32;
    }
    return (uint)carry;
}

static inline void shift_right_32_limbs_local(__local uint *r, uint limbs) {
    for (uint i = 0u; i + 1u < limbs; ++i) {
        r[i] = r[i + 1u];
    }
    r[limbs - 1u] = 0u;
}

static inline void special_mult_ui32_local(__local uint *r, uint m, __local const uint *N,
                                           uint np0, uint limbs) {
    uint carry_t1 = mul_ui32_limbs_local(r, m, limbs);
    uint t1_0 = r[0];
    uint q = (uint)((ulong)t1_0 * (ulong)np0);
    uint temp[MAX_LIMBS];
    for (uint i = 0u; i < limbs; ++i) {
        temp[i] = N[i];
    }
    uint carry_t2 = mul_ui32_limbs(temp, q, limbs);
    shift_right_32_limbs_local(r, limbs);
    shift_right_32_limbs(temp, limbs);
    r[limbs - 1u] = carry_t1;
    temp[limbs - 1u] = carry_t2;
    {
        uint carry = 0u;
        for (uint i = 0u; i < limbs; ++i) {
            ulong sum = (ulong)r[i] + (ulong)temp[i] + (ulong)carry;
            r[i] = (uint)sum;
            carry = (uint)(sum >> 32);
        }
        int carry_q = (int)carry;
        if (t1_0 != 0u) {
            carry = 0u;
            for (uint i = 0u; i < limbs; ++i) {
                ulong sum = (ulong)r[i] + (ulong)carry;
                r[i] = (uint)sum;
                carry = (uint)(sum >> 32);
            }
            carry_q += (int)carry;
        }
        if (carry_q > 0) {
            mp_sub_n_local(r, r, N, limbs);
        }
        if (mp_ge_local(r, N, limbs)) {
            mp_sub_n_local(r, r, N, limbs);
        }
    }
}

static inline void mp_shift_left_1_mod_local(__local uint *r, __local const uint *a,
                                             __local const uint *N, uint limbs) {
    uint carry = 0u;
    for (uint i = 0u; i < limbs; ++i) {
        uint old = a[i];
        r[i] = (old << 1) | carry;
        carry = old >> 31;
    }
    if (carry || mp_ge_local(r, N, limbs)) {
        mp_sub_n_local(r, r, N, limbs);
    }
}
#endif

#if ECM_STAGE1_USE_COOP_WG
static inline void mont_mul_stage1_coop(
    uint *out, const uint *a, const uint *b, const uint *N, __local const uint *N_loc,
    uint np0, __local uint *op_a, __local uint *op_b, __local uint *op_out,
    __local uint *mont_scratch, uint lid)
{
#if ECM_STAGE1_MUL_PATH == 1
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            op_a[i] = a[i];
            op_b[i] = b[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mont_mul_stage1_unroll64_4096_mt2_local(op_out, op_a, op_b, N_loc, np0, mont_scratch, lid);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            out[i] = op_out[i];
        }
    }
#elif ECM_STAGE1_MUL_PATH == 2
    if (lid == 0u) {
        mont_mul_stage1_fips4096(out, a, b, N, np0);
    }
#elif ECM_STAGE1_MUL_PATH == 3
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            op_a[i] = a[i];
            op_b[i] = b[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mont_mul_stage1_fips4096_mtn_local(op_out, op_a, op_b, N_loc, np0, mont_scratch, lid, 8u);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            out[i] = op_out[i];
        }
    }
#elif ECM_STAGE1_MUL_PATH == 4
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            op_a[i] = a[i];
            op_b[i] = b[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mont_mul_stage1_fips4096_mtn_local(op_out, op_a, op_b, N_loc, np0, mont_scratch, lid, 16u);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            out[i] = op_out[i];
        }
    }
#else
    if (lid == 0u) {
        mont_mul_stage1_unroll64_4096(out, a, b, N, np0);
    }
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
}

static inline void mont_sqr_stage1_coop(
    uint *out, const uint *a, const uint *N, __local const uint *N_loc, uint np0,
    __local uint *op_a, __local uint *op_b, __local uint *op_out, __local uint *mont_scratch,
    uint lid)
{
#if ECM_STAGE1_SQR_PATH == 1
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            op_a[i] = a[i];
            op_b[i] = a[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mont_mul_stage1_unroll64_4096_mt2_local(op_out, op_a, op_b, N_loc, np0, mont_scratch, lid);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            out[i] = op_out[i];
        }
    }
#elif ECM_STAGE1_SQR_PATH == 2
    if (lid == 0u) {
        mont_mul_stage1_fips4096(out, a, a, N, np0);
    }
#elif ECM_STAGE1_SQR_PATH == 3
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            op_a[i] = a[i];
            op_b[i] = a[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mont_mul_stage1_fips4096_mtn_local(op_out, op_a, op_b, N_loc, np0, mont_scratch, lid, 8u);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            out[i] = op_out[i];
        }
    }
#elif ECM_STAGE1_SQR_PATH == 4
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            op_a[i] = a[i];
            op_b[i] = a[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mont_mul_stage1_fips4096_mtn_local(op_out, op_a, op_b, N_loc, np0, mont_scratch, lid, 16u);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            out[i] = op_out[i];
        }
    }
#else
    if (lid == 0u) {
        mont_mul_stage1_unroll64_4096(out, a, a, N, np0);
    }
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
}
#endif

// Simultaneous double-and-add (CUDA curve_t::double_add_v2)
//
// Note: each call executes a fixed "operator mix" and is the performance hotspot:
//   - mp_add_mod:      4 calls
//   - mp_sub_mod:      4 calls
//   - mont_mul_priv:   4 calls
//   - mont_sqr_priv:   4 calls
//   - mont_normalize:  8 calls
//   - special_mult_ui32: 1 call
//   - mp_shift_left_1_mod: 1 call
static inline void double_add_v2(
    uint *q, uint *u, uint *w, uint *v,
    uint d, const uint *N, uint np0, uint limbs)
{
    uint t[MAX_LIMBS], CB[MAX_LIMBS], DA[MAX_LIMBS], AA[MAX_LIMBS], BB[MAX_LIMBS];
    uint K[MAX_LIMBS], dK[MAX_LIMBS];

    mp_add_mod(t, v, w, N, limbs);
    (void)mp_sub_mod(v, v, w, N, limbs);

    mp_add_mod(w, u, q, N, limbs);
    (void)mp_sub_mod(u, u, q, N, limbs);

    mont_mul_stage1(CB, t, u, N, np0, limbs);
    maybe_mont_normalize(CB, N, limbs);
    mont_mul_stage1(DA, v, w, N, np0, limbs);
    maybe_mont_normalize(DA, N, limbs);

    mont_sqr_stage1(AA, w, N, np0, limbs);
    mont_sqr_stage1(BB, u, N, np0, limbs);
    maybe_mont_normalize(AA, N, limbs);
    maybe_mont_normalize(BB, N, limbs);

    mont_mul_stage1(q, AA, BB, N, np0, limbs);
    maybe_mont_normalize(q, N, limbs);

    (void)mp_sub_mod(K, AA, BB, N, limbs);

    mp_copy(dK, K, limbs);
    special_mult_stage1(dK, d, N, np0, limbs);

    mp_add_mod(u, BB, dK, N, limbs);
    mont_mul_stage1(u, K, u, N, np0, limbs);
    maybe_mont_normalize(u, N, limbs);

    mp_add_mod(w, DA, CB, N, limbs);
    (void)mp_sub_mod(v, DA, CB, N, limbs);

    mont_sqr_stage1(w, w, N, np0, limbs);
    maybe_mont_normalize(w, N, limbs);
    mont_sqr_stage1(v, v, N, np0, limbs);
    maybe_mont_normalize(v, N, limbs);
    mp_shift_left_1_mod(v, v, N, limbs);
}

static inline void swap_limbs(uint *a, uint *b, uint limbs) {
    for (uint i = 0u; i < limbs; ++i) {
        uint tmp = a[i];
        a[i] = b[i];
        b[i] = tmp;
    }
}

#if ECM_STAGE1_USE_COOP_WG
static inline void double_add_v2_coop(
    uint *q, uint *u, uint *w, uint *v,
    uint d, const uint *N, uint np0, uint limbs,
    __local const uint *N_loc,
    __local uint *op_a, __local uint *op_b, __local uint *op_out,
    __local uint *mont_scratch, uint lid)
{
    uint t[MAX_LIMBS], CB[MAX_LIMBS], DA[MAX_LIMBS], AA[MAX_LIMBS], BB[MAX_LIMBS];
    uint K[MAX_LIMBS], dK[MAX_LIMBS];

    if (lid == 0u) {
        mp_add_mod(t, v, w, N, limbs);
        (void)mp_sub_mod(v, v, w, N, limbs);
        mp_add_mod(w, u, q, N, limbs);
        (void)mp_sub_mod(u, u, q, N, limbs);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    mont_mul_stage1_coop(CB, t, u, N, N_loc, np0, op_a, op_b, op_out, mont_scratch, lid);
    if (lid == 0u) {
        maybe_mont_normalize(CB, N, limbs);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    mont_mul_stage1_coop(DA, v, w, N, N_loc, np0, op_a, op_b, op_out, mont_scratch, lid);
    if (lid == 0u) {
        maybe_mont_normalize(DA, N, limbs);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    mont_sqr_stage1_coop(AA, w, N, N_loc, np0, op_a, op_b, op_out, mont_scratch, lid);
    mont_sqr_stage1_coop(BB, u, N, N_loc, np0, op_a, op_b, op_out, mont_scratch, lid);
    if (lid == 0u) {
        maybe_mont_normalize(AA, N, limbs);
        maybe_mont_normalize(BB, N, limbs);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    mont_mul_stage1_coop(q, AA, BB, N, N_loc, np0, op_a, op_b, op_out, mont_scratch, lid);
    if (lid == 0u) {
        maybe_mont_normalize(q, N, limbs);
        (void)mp_sub_mod(K, AA, BB, N, limbs);
        mp_copy(dK, K, limbs);
        special_mult_stage1(dK, d, N, np0, limbs);
        mp_add_mod(u, BB, dK, N, limbs);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    mont_mul_stage1_coop(u, K, u, N, N_loc, np0, op_a, op_b, op_out, mont_scratch, lid);
    if (lid == 0u) {
        maybe_mont_normalize(u, N, limbs);
        mp_add_mod(w, DA, CB, N, limbs);
        (void)mp_sub_mod(v, DA, CB, N, limbs);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    mont_sqr_stage1_coop(w, w, N, N_loc, np0, op_a, op_b, op_out, mont_scratch, lid);
    mont_sqr_stage1_coop(v, v, N, N_loc, np0, op_a, op_b, op_out, mont_scratch, lid);
    if (lid == 0u) {
        maybe_mont_normalize(w, N, limbs);
        maybe_mont_normalize(v, N, limbs);
        mp_shift_left_1_mod(v, v, N, limbs);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

static inline void run_double_add_instance_mt2_wg(
    uint instance_i,
    __global const uint *s_bits,
    ulong s_num_bits,
    ulong s_bits_start,
    ulong s_bits_interval,
    __global uint *data,
    uint sigma_0,
    uint np0,
    uint limbs,
    __local uint *N_loc,
    __local uint *op_a,
    __local uint *op_b,
    __local uint *op_out,
    __local uint *mont_scratch,
    uint lid)
{
    const uint base = instance_i * 5u * limbs;
    uint N[MAX_LIMBS];
    uint aX[MAX_LIMBS], aZ[MAX_LIMBS], bX[MAX_LIMBS], bZ[MAX_LIMBS];
    const uint half_limbs = limbs / 2u;
    const uint j_begin = lid * half_limbs;
    const uint j_end = j_begin + half_limbs;

    #pragma unroll 64
    for (uint i = j_begin; i < j_end; ++i) {
        N_loc[i] = data[base + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0u) {
        for (uint i = 0u; i < limbs; ++i) {
            N[i] = data[base + i];
            aX[i] = data[base + limbs + i];
            aZ[i] = data[base + 2u * limbs + i];
            bX[i] = data[base + 3u * limbs + i];
            bZ[i] = data[base + 4u * limbs + i];
        }
    }

    const uint d = sigma_0 + instance_i;
    int swapped = 0;
    ulong s_end = s_bits_start + s_bits_interval;
    if (s_end > s_num_bits) {
        s_end = s_num_bits;
    }

    for (ulong b = s_bits_start; b < s_end; ++b) {
        ulong nth = s_num_bits - 1ul - b;
        uint limb_idx = (uint)(nth >> 5);
        uint bit_idx = (uint)(nth & 31ul);
        int bit = (int)((s_bits[limb_idx] >> bit_idx) & 1u);

        if (lid == 0u && bit != swapped) {
            swapped = !swapped;
            swap_limbs(aX, bX, limbs);
            swap_limbs(aZ, bZ, limbs);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        double_add_v2_coop(aX, aZ, bX, bZ, d, N, np0, limbs, N_loc, op_a, op_b, op_out,
                           mont_scratch, lid);
    }

    if (lid == 0u && swapped) {
        swap_limbs(aX, bX, limbs);
        swap_limbs(aZ, bZ, limbs);
    }

    if (lid == 0u) {
        for (uint i = 0u; i < limbs; ++i) {
            data[base + limbs + i] = aX[i];
            data[base + 2u * limbs + i] = aZ[i];
            data[base + 3u * limbs + i] = bX[i];
            data[base + 4u * limbs + i] = bZ[i];
        }
    }
}
#endif

// ---------------------------------------------------------------------------
// Main kernel — mirrors CUDA kernel_double_add
// data layout per curve (5 * limbs uint32): N, aX, aZ, bX, bZ
// ---------------------------------------------------------------------------

static inline void run_double_add_instance(
    uint instance_i,
    __global const uint *s_bits,
    ulong s_num_bits,
    ulong s_bits_start,
    ulong s_bits_interval,
    __global uint *data,
    uint sigma_0,
    uint np0,
    uint limbs)
{
    uint base = instance_i * 5u * limbs;
    uint N[MAX_LIMBS];
    uint aX[MAX_LIMBS], aZ[MAX_LIMBS], bX[MAX_LIMBS], bZ[MAX_LIMBS];

    for (uint i = 0u; i < limbs; ++i) {
        N[i] = data[base + i];
        aX[i] = data[base + limbs + i];
        aZ[i] = data[base + 2u * limbs + i];
        bX[i] = data[base + 3u * limbs + i];
        bZ[i] = data[base + 4u * limbs + i];
    }

    uint d = sigma_0 + instance_i;
    int swapped = 0;

    ulong s_end = s_bits_start + s_bits_interval;
    if (s_end > s_num_bits) {
        s_end = s_num_bits;
    }

    for (ulong b = s_bits_start; b < s_end; ++b) {
        ulong nth = s_num_bits - 1ul - b;
        uint limb_idx = (uint)(nth >> 5);
        uint bit_idx = (uint)(nth & 31ul);
        int bit = (int)((s_bits[limb_idx] >> bit_idx) & 1u);

        if (bit != swapped) {
            swapped = !swapped;
            swap_limbs(aX, bX, limbs);
            swap_limbs(aZ, bZ, limbs);
        }
        double_add_v2(aX, aZ, bX, bZ, d, N, np0, limbs);
    }

    if (swapped) {
        swap_limbs(aX, bX, limbs);
        swap_limbs(aZ, bZ, limbs);
    }

    for (uint i = 0u; i < limbs; ++i) {
        data[base + limbs + i] = aX[i];
        data[base + 2u * limbs + i] = aZ[i];
        data[base + 3u * limbs + i] = bX[i];
        data[base + 4u * limbs + i] = bZ[i];
    }
}

#if ECM_STAGE1_USE_COOP_WG
__kernel __attribute__((reqd_work_group_size(ECM_STAGE1_COOP_WG, 1, 1)))
#else
__kernel
#endif
void kernel_double_add(
    __global const uint *s_bits,
    ulong s_num_bits,
    ulong s_bits_start,
    ulong s_bits_interval,
    __global uint *data,
    uint count,
    uint sigma_0,
    uint np0,
    uint limbs)
{
#if ECM_STAGE1_USE_COOP_WG
    __local uint N_local[MONT_FIXED_4096_LIMBS];
    __local uint mont_op_a[MONT_FIXED_4096_LIMBS];
    __local uint mont_op_b[MONT_FIXED_4096_LIMBS];
    __local uint mont_op_out[MONT_FIXED_4096_LIMBS];
    __local uint mont_scratch[ECM_STAGE1_COOP_SCRATCH_U32];
    const uint instance_i = get_group_id(0);
    const uint lid = get_local_id(0);
    if (instance_i >= count) {
        return;
    }
    if (limbs != MONT_FIXED_4096_LIMBS) {
        return;
    }
    run_double_add_instance_mt2_wg(instance_i, s_bits, s_num_bits, s_bits_start, s_bits_interval,
                                   data, sigma_0, np0, limbs, N_local, mont_op_a, mont_op_b,
                                   mont_op_out, mont_scratch, lid);
#else
    uint instance_i = get_global_id(0);
    if (instance_i >= count) {
        return;
    }
    if (limbs == 0u || limbs > MAX_LIMBS) {
        return;
    }
    run_double_add_instance(instance_i, s_bits, s_num_bits, s_bits_start, s_bits_interval, data,
                            sigma_0, np0, limbs);
#endif
}
