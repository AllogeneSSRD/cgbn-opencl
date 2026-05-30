// Optimized private Montgomery mul: N/np0 in __constant, B cached, speculative final subtract.

#pragma once

#ifndef MAX_LIMBS
#define MAX_LIMBS 128
#endif

#define MONT_OPT2_FIXED_LIMBS 16u
#define MONT_FIXED_4096_LIMBS 128u

static inline void mont_mul_priv_opt_core(__global uint *out, __global const uint *a,
                                          __global const uint *b, __constant uint *n,
                                          uint base, uint np0, uint limbs) {
    if (limbs == 0u || limbs > MAX_LIMBS) {
        return;
    }

    uint t[MAX_LIMBS + 2];
    for (uint i = 0u; i < limbs + 2u; ++i) {
        t[i] = 0u;
    }

    uint B[MAX_LIMBS];
    for (uint j = 0u; j < limbs; ++j) {
        B[j] = b[base + j];
    }

    for (uint i = 0u; i < limbs; ++i) {
        uint ai = a[base + i];

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
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)n[j] + carry;
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
        ulong nv = (ulong)n[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }

    uint need_sub = (t[limbs] != 0u || t[limbs + 1u] != 0u) ? 1u : 0u;
    need_sub = (borrow == 0u) ? 1u : need_sub;
    uint mask = 0u - need_sub;

    for (uint i = 0u; i < limbs; ++i) {
        out[base + i] = (D[i] & mask) | (t[i] & ~mask);
    }
}

static inline void mont_sqr_priv_opt_core(__global uint *out, __global const uint *a,
                                          __constant uint *n, uint base, uint np0, uint limbs) {
    if (limbs == 0u || limbs > MAX_LIMBS) {
        return;
    }

    uint t[MAX_LIMBS + 2];
    for (uint i = 0u; i < limbs + 2u; ++i) {
        t[i] = 0u;
    }

    uint B[MAX_LIMBS];
    for (uint j = 0u; j < limbs; ++j) {
        B[j] = a[base + j];
    }

    for (uint i = 0u; i < limbs; ++i) {
        uint ai = a[base + i];

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
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)n[j] + carry;
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
        ulong nv = (ulong)n[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }

    uint need_sub = (t[limbs] != 0u || t[limbs + 1u] != 0u) ? 1u : 0u;
    need_sub = (borrow == 0u) ? 1u : need_sub;
    uint mask = 0u - need_sub;

    for (uint i = 0u; i < limbs; ++i) {
        out[base + i] = (D[i] & mask) | (t[i] & ~mask);
    }
}

__kernel void cgbn_mont_mul_opt(__global const uint *a, __global const uint *b, __constant uint *n,
                                __global uint *out, __constant uint *np0_ptr, uint limbs) {
    uint idx = get_global_id(0);
    uint base = idx * limbs;
    uint np0 = np0_ptr[0];
    mont_mul_priv_opt_core(out, a, b, n, base, np0, limbs);
}

__kernel void cgbn_mont_sqr_opt(__global const uint *a, __constant uint *n, __global uint *out,
                                __constant uint *np0_ptr, uint limbs) {
    uint idx = get_global_id(0);
    uint base = idx * limbs;
    uint np0 = np0_ptr[0];
    mont_sqr_priv_opt_core(out, a, n, base, np0, limbs);
}

// Fixed 512-bit opt2: local-memory cached B/N and fully unrolled loops.
// local_mem layout (per work-group):
//   B_cache[local_size][16] + N_cache[local_size][16]
static inline void mont_mul_priv_opt2_512_local_body(
    __global uint *out,
    __global const uint *a,
    __global const uint *b,
    __constant uint *n,
    uint base,
    uint np0,
    __local uint *local_mem,
    uint lid,
    uint lsize)
{
    __local uint *B_cache = local_mem;
    __local uint *N_cache = B_cache + lsize * MONT_OPT2_FIXED_LIMBS;
    __local uint *B = B_cache + lid * MONT_OPT2_FIXED_LIMBS;
    __local uint *N = N_cache + lid * MONT_OPT2_FIXED_LIMBS;

    #pragma unroll
    for (uint j = 0u; j < MONT_OPT2_FIXED_LIMBS; ++j) {
        B[j] = b[base + j];
        N[j] = n[j];
    }

    uint t[MONT_OPT2_FIXED_LIMBS + 2u];
    #pragma unroll
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS + 2u; ++i) {
        t[i] = 0u;
    }

    #pragma unroll
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        uint ai = a[base + i];

        ulong carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < MONT_OPT2_FIXED_LIMBS; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong top = (ulong)t[MONT_OPT2_FIXED_LIMBS] + carry;
        t[MONT_OPT2_FIXED_LIMBS] = (uint)top;
        t[MONT_OPT2_FIXED_LIMBS + 1u] = (uint)(top >> 32);

        uint m = (uint)((ulong)t[0] * (ulong)np0);
        carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < MONT_OPT2_FIXED_LIMBS; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
            if (j > 0u) {
                t[j - 1u] = (uint)uv;
            }
            carry = uv >> 32;
        }
        top = (ulong)t[MONT_OPT2_FIXED_LIMBS] + carry;
        t[MONT_OPT2_FIXED_LIMBS - 1u] = (uint)top;
        top = (ulong)t[MONT_OPT2_FIXED_LIMBS + 1u] + (top >> 32);
        t[MONT_OPT2_FIXED_LIMBS] = (uint)top;
        t[MONT_OPT2_FIXED_LIMBS + 1u] = (uint)(top >> 32);
    }

    ulong borrow = 0ul;
    uint D[MONT_OPT2_FIXED_LIMBS];
    #pragma unroll
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        ulong tv = (ulong)t[i];
        ulong nv = (ulong)N[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }

    uint need_sub = (t[MONT_OPT2_FIXED_LIMBS] != 0u || t[MONT_OPT2_FIXED_LIMBS + 1u] != 0u) ? 1u : 0u;
    need_sub = (borrow == 0u) ? 1u : need_sub;
    uint mask = 0u - need_sub;

    #pragma unroll
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        out[base + i] = (D[i] & mask) | (t[i] & ~mask);
    }
}

static inline void mont_sqr_priv_opt2_512_local_body(
    __global uint *out,
    __global const uint *a,
    __constant uint *n,
    uint base,
    uint np0,
    __local uint *local_mem,
    uint lid,
    uint lsize)
{
    __local uint *B_cache = local_mem;
    __local uint *N_cache = B_cache + lsize * MONT_OPT2_FIXED_LIMBS;
    __local uint *B = B_cache + lid * MONT_OPT2_FIXED_LIMBS;
    __local uint *N = N_cache + lid * MONT_OPT2_FIXED_LIMBS;

    #pragma unroll
    for (uint j = 0u; j < MONT_OPT2_FIXED_LIMBS; ++j) {
        uint av = a[base + j];
        B[j] = av;
        N[j] = n[j];
    }

    uint t[MONT_OPT2_FIXED_LIMBS + 2u];
    #pragma unroll
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS + 2u; ++i) {
        t[i] = 0u;
    }

    #pragma unroll
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        uint ai = a[base + i];

        ulong carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < MONT_OPT2_FIXED_LIMBS; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong top = (ulong)t[MONT_OPT2_FIXED_LIMBS] + carry;
        t[MONT_OPT2_FIXED_LIMBS] = (uint)top;
        t[MONT_OPT2_FIXED_LIMBS + 1u] = (uint)(top >> 32);

        uint m = (uint)((ulong)t[0] * (ulong)np0);
        carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < MONT_OPT2_FIXED_LIMBS; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
            if (j > 0u) {
                t[j - 1u] = (uint)uv;
            }
            carry = uv >> 32;
        }
        top = (ulong)t[MONT_OPT2_FIXED_LIMBS] + carry;
        t[MONT_OPT2_FIXED_LIMBS - 1u] = (uint)top;
        top = (ulong)t[MONT_OPT2_FIXED_LIMBS + 1u] + (top >> 32);
        t[MONT_OPT2_FIXED_LIMBS] = (uint)top;
        t[MONT_OPT2_FIXED_LIMBS + 1u] = (uint)(top >> 32);
    }

    ulong borrow = 0ul;
    uint D[MONT_OPT2_FIXED_LIMBS];
    #pragma unroll
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        ulong tv = (ulong)t[i];
        ulong nv = (ulong)N[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }

    uint need_sub = (t[MONT_OPT2_FIXED_LIMBS] != 0u || t[MONT_OPT2_FIXED_LIMBS + 1u] != 0u) ? 1u : 0u;
    need_sub = (borrow == 0u) ? 1u : need_sub;
    uint mask = 0u - need_sub;

    #pragma unroll
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        out[base + i] = (D[i] & mask) | (t[i] & ~mask);
    }
}

#define FIPS512_T_WORDS (2u * MONT_OPT2_FIXED_LIMBS + 2u)
#define FIPS512_P_WORDS (MONT_OPT2_FIXED_LIMBS + 1u)
#define FIPS512_MT_PROD_WORDS (2u * MONT_OPT2_FIXED_LIMBS)

// Koç FIPS Algorithm 1 (ACNS 2003): CSA value = t*2^(2w) + u*2^w + v.
static inline void fips512_csa_add_priv(uint *t, uint *u, uint *v, ulong prod) {
    ulong val = (ulong)(*v) + prod;
    *v = (uint)val;
    val = (val >> 32) + (ulong)(*u);
    *u = (uint)val;
    val = (val >> 32) + (ulong)(*t);
    *t = (uint)val;
}

static inline void fips512_csa_shift_priv(uint *t, uint *u, uint *v) {
    uint tv = *t;
    uint uv = *u;
    *v = uv;
    *u = tv;
    *t = 0u;
}

static inline void fips512_finalize_p_priv(__global uint *out, const uint P[FIPS512_P_WORDS],
                                           __constant uint *n, uint base) {
    uint ps = P[MONT_OPT2_FIXED_LIMBS];
    ulong borrow = 0ul;
    uint D[MONT_OPT2_FIXED_LIMBS];
    #pragma unroll
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        ulong tv = (ulong)P[i];
        ulong nv = (ulong)n[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }
    uint any_high = (ps != 0u) ? 1u : 0u;
    uint need_sub = any_high | ((borrow == 0u) ? 1u : 0u);
    uint mask = 0u - need_sub;
    #pragma unroll
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        out[base + i] = (D[i] & mask) | (P[i] & ~mask);
    }
}

static inline void fips512_finalize_p_local(__global uint *out, __local uint *P, __constant uint *n,
                                            uint base) {
    uint ps = P[MONT_OPT2_FIXED_LIMBS];
    ulong borrow = 0ul;
    uint D[MONT_OPT2_FIXED_LIMBS];
    #pragma unroll
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        ulong tv = (ulong)P[i];
        ulong nv = (ulong)n[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }
    uint any_high = (ps != 0u) ? 1u : 0u;
    uint need_sub = any_high | ((borrow == 0u) ? 1u : 0u);
    uint mask = 0u - need_sub;
    #pragma unroll
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        out[base + i] = (D[i] & mask) | (P[i] & ~mask);
    }
}

static inline void fips512_carry_ripple_local(__local uint *t, uint start, ulong carry) {
    uint pos = start;
    while (carry != 0ul && pos < FIPS512_T_WORDS) {
        ulong uv = (ulong)t[pos] + carry;
        t[pos] = (uint)uv;
        carry = uv >> 32;
        pos++;
    }
}

static inline void fips512_redc_step_priv(uint t[FIPS512_T_WORDS], __constant uint *n, uint np0) {
    uint m = (uint)((ulong)t[0] * (ulong)np0);
    ulong uv0 = (ulong)t[0] + (ulong)m * (ulong)n[0];
    ulong carry = uv0 >> 32;
    #pragma unroll
    for (uint j = 1u; j < MONT_OPT2_FIXED_LIMBS; ++j) {
        ulong uv = (ulong)t[j] + (ulong)m * (ulong)n[j] + carry;
        t[j - 1u] = (uint)uv;
        carry = uv >> 32;
    }
    ulong top = (ulong)t[MONT_OPT2_FIXED_LIMBS] + carry;
    t[MONT_OPT2_FIXED_LIMBS - 1u] = (uint)top;
    top = (ulong)t[MONT_OPT2_FIXED_LIMBS + 1u] + (top >> 32);
    t[MONT_OPT2_FIXED_LIMBS] = (uint)top;
    t[MONT_OPT2_FIXED_LIMBS + 1u] = (uint)(top >> 32);
}

static inline void fips512_redc_step_local(__local uint *t, __constant uint *n, uint np0) {
    uint m = (uint)((ulong)t[0] * (ulong)np0);
    ulong uv0 = (ulong)t[0] + (ulong)m * (ulong)n[0];
    ulong carry = uv0 >> 32;
    #pragma unroll
    for (uint j = 1u; j < MONT_OPT2_FIXED_LIMBS; ++j) {
        ulong uv = (ulong)t[j] + (ulong)m * (ulong)n[j] + carry;
        t[j - 1u] = (uint)uv;
        carry = uv >> 32;
    }
    ulong top = (ulong)t[MONT_OPT2_FIXED_LIMBS] + carry;
    t[MONT_OPT2_FIXED_LIMBS - 1u] = (uint)top;
    top = (ulong)t[MONT_OPT2_FIXED_LIMBS + 1u] + (top >> 32);
    t[MONT_OPT2_FIXED_LIMBS] = (uint)top;
    t[MONT_OPT2_FIXED_LIMBS + 1u] = (uint)(top >> 32);
}

// SOS Montgomery reduction on a 2n-limb buffer in local memory.
static inline void mont_redc_cios_512_local(__local uint *tt, __constant uint *n, uint np0) {
    #pragma unroll
    for (uint iter = 0u; iter < MONT_OPT2_FIXED_LIMBS; ++iter) {
        uint m = (uint)((ulong)tt[0] * (ulong)np0);
        ulong carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < 2u * MONT_OPT2_FIXED_LIMBS + 2u; ++j) {
            ulong add = (j < MONT_OPT2_FIXED_LIMBS) ? (ulong)m * (ulong)n[j] : 0ul;
            ulong uv = (ulong)tt[j] + add + carry;
            if (j > 0u) {
                tt[j - 1u] = (uint)uv;
            }
            carry = uv >> 32;
        }
        tt[2u * MONT_OPT2_FIXED_LIMBS + 1u] = (uint)carry;
    }
}

// Single-lane FIPS (Koç Algorithm 1).
static inline void mont_mul_priv_fips512_body(__global uint *out, __global const uint *a,
                                               __global const uint *b, __constant uint *n,
                                               uint base, uint np0) {
    uint A[MONT_OPT2_FIXED_LIMBS];
    uint B[MONT_OPT2_FIXED_LIMBS];
    #pragma unroll
    for (uint j = 0u; j < MONT_OPT2_FIXED_LIMBS; ++j) {
        A[j] = a[base + j];
        B[j] = b[base + j];
    }

    uint P[FIPS512_P_WORDS];
    uint t = 0u;
    uint u = 0u;
    uint v = 0u;

    #pragma unroll
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        #pragma unroll
        for (uint j = 0u; j < i; ++j) {
            fips512_csa_add_priv(&t, &u, &v, (ulong)A[j] * (ulong)B[i - j]);
            fips512_csa_add_priv(&t, &u, &v, (ulong)P[j] * (ulong)n[i - j]);
        }
        fips512_csa_add_priv(&t, &u, &v, (ulong)A[i] * (ulong)B[0]);
        uint pi = (uint)((ulong)v * (ulong)np0);
        fips512_csa_add_priv(&t, &u, &v, (ulong)pi * (ulong)n[0]);
        P[i] = pi;
        fips512_csa_shift_priv(&t, &u, &v);
    }

    #pragma unroll
    for (uint i = MONT_OPT2_FIXED_LIMBS; i < 2u * MONT_OPT2_FIXED_LIMBS; ++i) {
        #pragma unroll
        for (uint j = i - MONT_OPT2_FIXED_LIMBS + 1u; j < MONT_OPT2_FIXED_LIMBS; ++j) {
            fips512_csa_add_priv(&t, &u, &v, (ulong)A[j] * (ulong)B[i - j]);
            fips512_csa_add_priv(&t, &u, &v, (ulong)P[j] * (ulong)n[i - j]);
        }
        P[i - MONT_OPT2_FIXED_LIMBS] = v;
        fips512_csa_shift_priv(&t, &u, &v);
    }
    P[MONT_OPT2_FIXED_LIMBS] = v;

    fips512_finalize_p_priv(out, P, n, base);
}

static inline void mont_sqr_priv_fips512_body(__global uint *out, __global const uint *a,
                                               __constant uint *n, uint base, uint np0) {
    mont_mul_priv_fips512_body(out, a, a, n, base, np0);
}

// FIPS mtN: parallel inner j-loop products, lane 0 runs CSA + shift.
static inline void mont_mul_priv_fips512_mtn_body(__global uint *out, __global const uint *a,
                                                   __global const uint *b, __constant uint *n,
                                                   uint base, uint np0, __local uint *local_mem,
                                                   uint lid, uint mt) {
    __local uint *A = local_mem;
    __local uint *B = A + MONT_OPT2_FIXED_LIMBS;
    __local ulong *prods = (__local ulong *)(B + MONT_OPT2_FIXED_LIMBS);
    __local uint *P = (__local uint *)(prods + FIPS512_MT_PROD_WORDS);

    #pragma unroll
    for (uint j = lid; j < MONT_OPT2_FIXED_LIMBS; j += mt) {
        A[j] = a[base + j];
        B[j] = b[base + j];
    }
    if (lid == 0u) {
        #pragma unroll
        for (uint i = 0u; i < FIPS512_P_WORDS; ++i) {
            P[i] = 0u;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint t = 0u;
    uint u = 0u;
    uint v = 0u;

    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        for (uint j = lid; j < i; j += mt) {
            prods[2u * j] = (ulong)A[j] * (ulong)B[i - j];
            prods[2u * j + 1u] = (ulong)P[j] * (ulong)n[i - j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid == 0u) {
            for (uint j = 0u; j < i; ++j) {
                fips512_csa_add_priv(&t, &u, &v, prods[2u * j]);
                fips512_csa_add_priv(&t, &u, &v, prods[2u * j + 1u]);
            }
            fips512_csa_add_priv(&t, &u, &v, (ulong)A[i] * (ulong)B[0]);
            uint pi = (uint)((ulong)v * (ulong)np0);
            fips512_csa_add_priv(&t, &u, &v, (ulong)pi * (ulong)n[0]);
            P[i] = pi;
            fips512_csa_shift_priv(&t, &u, &v);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint i = MONT_OPT2_FIXED_LIMBS; i < 2u * MONT_OPT2_FIXED_LIMBS; ++i) {
        for (uint j = i - MONT_OPT2_FIXED_LIMBS + 1u + lid; j < MONT_OPT2_FIXED_LIMBS; j += mt) {
            prods[2u * j] = (ulong)A[j] * (ulong)B[i - j];
            prods[2u * j + 1u] = (ulong)P[j] * (ulong)n[i - j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid == 0u) {
            for (uint j = i - MONT_OPT2_FIXED_LIMBS + 1u; j < MONT_OPT2_FIXED_LIMBS; ++j) {
                fips512_csa_add_priv(&t, &u, &v, prods[2u * j]);
                fips512_csa_add_priv(&t, &u, &v, prods[2u * j + 1u]);
            }
            P[i - MONT_OPT2_FIXED_LIMBS] = v;
            fips512_csa_shift_priv(&t, &u, &v);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0u) {
        P[MONT_OPT2_FIXED_LIMBS] = v;
        fips512_finalize_p_local(out, P, n, base);
    }
}

static inline void mont_mul_priv_fips512_mt4_body(__global uint *out, __global const uint *a,
                                                   __global const uint *b, __constant uint *n,
                                                   uint base, uint np0, __local uint *local_mem,
                                                   uint lid) {
    mont_mul_priv_fips512_mtn_body(out, a, b, n, base, np0, local_mem, lid, 4u);
}

static inline void mont_mul_priv_fips512_mt8_body(__global uint *out, __global const uint *a,
                                                   __global const uint *b, __constant uint *n,
                                                   uint base, uint np0, __local uint *local_mem,
                                                   uint lid) {
    mont_mul_priv_fips512_mtn_body(out, a, b, n, base, np0, local_mem, lid, 8u);
}

static inline void mont_mul_priv_fips512_mt16_body(__global uint *out, __global const uint *a,
                                                    __global const uint *b, __constant uint *n,
                                                    uint base, uint np0, __local uint *local_mem,
                                                    uint lid) {
    mont_mul_priv_fips512_mtn_body(out, a, b, n, base, np0, local_mem, lid, 16u);
}

static inline void mont_sqr_priv_fips512_mt4_body(__global uint *out, __global const uint *a,
                                                  __constant uint *n, uint base, uint np0,
                                                  __local uint *local_mem, uint lid) {
    mont_mul_priv_fips512_mt4_body(out, a, a, n, base, np0, local_mem, lid);
}

static inline void mont_sqr_priv_fips512_mt8_body(__global uint *out, __global const uint *a,
                                                  __constant uint *n, uint base, uint np0,
                                                  __local uint *local_mem, uint lid) {
    mont_mul_priv_fips512_mt8_body(out, a, a, n, base, np0, local_mem, lid);
}

static inline void mont_sqr_priv_fips512_mt16_body(__global uint *out, __global const uint *a,
                                                    __constant uint *n, uint base, uint np0,
                                                    __local uint *local_mem, uint lid) {
    mont_mul_priv_fips512_mt16_body(out, a, a, n, base, np0, local_mem, lid);
}

static inline void fips512_finalize_t_local(__global uint *out, __local uint *t, __constant uint *n,
                                            uint base) {
    ulong borrow = 0ul;
    uint D[MONT_OPT2_FIXED_LIMBS];
    #pragma unroll
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        ulong tv = (ulong)t[i];
        ulong nv = (ulong)n[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }
    uint need_sub = (t[MONT_OPT2_FIXED_LIMBS] != 0u || t[MONT_OPT2_FIXED_LIMBS + 1u] != 0u) ? 1u : 0u;
    need_sub = (borrow == 0u) ? 1u : need_sub;
    uint mask = 0u - need_sub;
    #pragma unroll
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        out[base + i] = (D[i] & mask) | (t[i] & ~mask);
    }
}

// Carry-save parallel product grid: each lane accumulates a subset of (i,j) pairs,
// tree-merge limb arrays, then sequential CIOS REDC + finalize.
static inline void mont_mul_priv_fips512_mtn_cs_body(__global uint *out, __global const uint *a,
                                                      __global const uint *b, __constant uint *n,
                                                      uint base, uint np0, __local uint *local_mem,
                                                      uint lid, uint mt) {
    __local uint *A = local_mem;
    __local uint *B = A + MONT_OPT2_FIXED_LIMBS;
    __local uint *parts = B + MONT_OPT2_FIXED_LIMBS;
    __local uint *t = parts + mt * FIPS512_T_WORDS;
    const uint tile = lid * FIPS512_T_WORDS;

    #pragma unroll
    for (uint j = lid; j < MONT_OPT2_FIXED_LIMBS; j += mt) {
        A[j] = a[base + j];
        B[j] = b[base + j];
    }
    for (uint idx = 0u; idx < FIPS512_T_WORDS; ++idx) {
        parts[tile + idx] = 0u;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint t_priv[FIPS512_T_WORDS];
    #pragma unroll
    for (uint idx = 0u; idx < FIPS512_T_WORDS; ++idx) {
        t_priv[idx] = 0u;
    }

    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        for (uint j = lid; j < MONT_OPT2_FIXED_LIMBS; j += mt) {
            uint k = i + j;
            ulong prod = (ulong)A[i] * (ulong)B[j];
            ulong uv = (ulong)t_priv[k] + (prod & 0xFFFFFFFFul);
            t_priv[k] = (uint)uv;
            uv = (ulong)t_priv[k + 1u] + (prod >> 32) + (uv >> 32);
            t_priv[k + 1u] = (uint)uv;
            uint pos = k + 2u;
            ulong carry = uv >> 32;
            while (carry != 0ul && pos < FIPS512_T_WORDS) {
                uv = (ulong)t_priv[pos] + carry;
                t_priv[pos] = (uint)uv;
                carry = uv >> 32;
                pos++;
            }
        }
    }

    for (uint idx = 0u; idx < FIPS512_T_WORDS; ++idx) {
        parts[tile + idx] = t_priv[idx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = mt >> 1; stride > 0u; stride >>= 1) {
        if (lid < stride) {
            uint src = (lid + stride) * FIPS512_T_WORDS;
            uint dst = lid * FIPS512_T_WORDS;
            ulong carry = 0ul;
            uint pos = 0u;
            for (; pos < FIPS512_T_WORDS; ++pos) {
                ulong uv = (ulong)parts[dst + pos] + (ulong)parts[src + pos] + carry;
                parts[dst + pos] = (uint)uv;
                carry = uv >> 32;
            }
            while (carry != 0ul && pos < FIPS512_T_WORDS) {
                ulong uv = (ulong)parts[dst + pos] + carry;
                parts[dst + pos] = (uint)uv;
                carry = uv >> 32;
                pos++;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0u) {
        #pragma unroll
        for (uint idx = 0u; idx < FIPS512_T_WORDS; ++idx) {
            t[idx] = parts[idx];
        }

        mont_redc_cios_512_local(t, n, np0);
        fips512_finalize_t_local(out, t, n, base);
    }
}

static inline void mont_mul_priv_fips512_mt8_cs_body(__global uint *out, __global const uint *a,
                                                      __global const uint *b, __constant uint *n,
                                                      uint base, uint np0, __local uint *local_mem,
                                                      uint lid) {
    mont_mul_priv_fips512_mtn_cs_body(out, a, b, n, base, np0, local_mem, lid, 8u);
}

static inline void mont_mul_priv_fips512_mt16_cs_body(__global uint *out, __global const uint *a,
                                                       __global const uint *b, __constant uint *n,
                                                       uint base, uint np0, __local uint *local_mem,
                                                       uint lid) {
    mont_mul_priv_fips512_mtn_cs_body(out, a, b, n, base, np0, local_mem, lid, 16u);
}

static inline void mont_sqr_priv_fips512_mt8_cs_body(__global uint *out, __global const uint *a,
                                                      __constant uint *n, uint base, uint np0,
                                                      __local uint *local_mem, uint lid) {
    mont_mul_priv_fips512_mt8_cs_body(out, a, a, n, base, np0, local_mem, lid);
}

static inline void mont_sqr_priv_fips512_mt16_cs_body(__global uint *out, __global const uint *a,
                                                       __constant uint *n, uint base, uint np0,
                                                       __local uint *local_mem, uint lid) {
    mont_mul_priv_fips512_mt16_cs_body(out, a, a, n, base, np0, local_mem, lid);
}

// - unroll_only: fixed 16 limbs + unrolled loops, private B/N (no local memory)
// - local_only:  fixed 16 limbs + local B/N, but no explicit unroll pragmas
static inline void mont_mul_priv_unroll_only_512_body(
    __global uint *out,
    __global const uint *a,
    __global const uint *b,
    __constant uint *n,
    uint base,
    uint np0)
{
    uint t[MONT_OPT2_FIXED_LIMBS + 2u];
    #pragma unroll
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS + 2u; ++i) {
        t[i] = 0u;
    }
    uint B[MONT_OPT2_FIXED_LIMBS];
    uint N[MONT_OPT2_FIXED_LIMBS];
    #pragma unroll
    for (uint j = 0u; j < MONT_OPT2_FIXED_LIMBS; ++j) {
        B[j] = b[base + j];
        N[j] = n[j];
    }

    #pragma unroll
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        uint ai = a[base + i];
        ulong carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < MONT_OPT2_FIXED_LIMBS; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong top = (ulong)t[MONT_OPT2_FIXED_LIMBS] + carry;
        t[MONT_OPT2_FIXED_LIMBS] = (uint)top;
        t[MONT_OPT2_FIXED_LIMBS + 1u] = (uint)(top >> 32);

        uint m = (uint)((ulong)t[0] * (ulong)np0);
        carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < MONT_OPT2_FIXED_LIMBS; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
            if (j > 0u) {
                t[j - 1u] = (uint)uv;
            }
            carry = uv >> 32;
        }
        top = (ulong)t[MONT_OPT2_FIXED_LIMBS] + carry;
        t[MONT_OPT2_FIXED_LIMBS - 1u] = (uint)top;
        top = (ulong)t[MONT_OPT2_FIXED_LIMBS + 1u] + (top >> 32);
        t[MONT_OPT2_FIXED_LIMBS] = (uint)top;
        t[MONT_OPT2_FIXED_LIMBS + 1u] = (uint)(top >> 32);
    }

    ulong borrow = 0ul;
    uint D[MONT_OPT2_FIXED_LIMBS];
    #pragma unroll
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        ulong tv = (ulong)t[i];
        ulong nv = (ulong)N[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }

    // uint need_sub = (t[MONT_OPT2_FIXED_LIMBS] != 0u || t[MONT_OPT2_FIXED_LIMBS + 1u] != 0u) ? 1u : 0u;
    // need_sub = (borrow == 0u) ? 1u : need_sub;
    // uint mask = 0u - need_sub;
    uint any_high = (t[MONT_OPT2_FIXED_LIMBS] | t[MONT_OPT2_FIXED_LIMBS + 1u]) != 0u;    // 0 或 1
    uint no_borrow = (borrow == 0u);          // 0 或 1
    uint need_sub = any_high | no_borrow;     // 需要减法 -> 1
    uint mask = 0u - need_sub;                // 0xFFFFFFFF 或 0

    #pragma unroll
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        out[base + i] = (D[i] & mask) | (t[i] & ~mask);
    }
}

// GMP sqr_basecase (n=16): off-diagonal triangle in tp, double + diagonal -> rp[2n].
static inline void sqr_basecase_512_priv(const uint A[MONT_OPT2_FIXED_LIMBS], uint rp[32]) {
    uint tp[32];
    #pragma unroll
    for (uint i = 0u; i < 32u; ++i) {
        tp[i] = 0u;
    }

    ulong carry = 0ul;
    #pragma unroll
    for (uint j = 1u; j < MONT_OPT2_FIXED_LIMBS; ++j) {
        ulong uv = (ulong)A[j] * (ulong)A[0] + carry;
        tp[j - 1u] = (uint)uv;
        carry = uv >> 32;
    }
    tp[MONT_OPT2_FIXED_LIMBS - 1u] = (uint)carry;

    #pragma unroll
    for (uint i = 2u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        carry = 0ul;
        #pragma unroll
        for (uint k = 0u; k < MONT_OPT2_FIXED_LIMBS - i; ++k) {
            uint idx = 2u * i - 2u + k;
            ulong uv = (ulong)tp[idx] + (ulong)A[i + k] * (ulong)A[i - 1u] + carry;
            tp[idx] = (uint)uv;
            carry = uv >> 32;
        }
        tp[MONT_OPT2_FIXED_LIMBS + i - 2u] = (uint)carry;
    }

    #pragma unroll
    for (uint i = 0u; i < 32u; ++i) {
        rp[i] = 0u;
    }

    #pragma unroll
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        ulong sq = (ulong)A[i] * (ulong)A[i];
        rp[2u * i] = (uint)sq;
        rp[2u * i + 1u] = (uint)(sq >> 32);
    }

    ulong dcarry = 0ul;
    ulong acarry = 0ul;
    #pragma unroll
    for (uint i = 0u; i < 2u * MONT_OPT2_FIXED_LIMBS - 2u; ++i) {
        ulong dbl = ((ulong)tp[i] << 1) + dcarry;
        dcarry = dbl >> 32;
        ulong uv = (ulong)rp[i + 1u] + (dbl & 0xFFFFFFFFul) + acarry;
        rp[i + 1u] = (uint)uv;
        acarry = uv >> 32;
    }
    rp[2u * MONT_OPT2_FIXED_LIMBS - 1u] += (uint)(acarry + dcarry);
}

// SOS Montgomery reduction: 2n-limb product -> n limbs (one limb per outer step).
static inline void mont_redc_cios_512_priv(uint tt[34], __constant uint *n, uint np0) {
    #pragma unroll
    for (uint iter = 0u; iter < MONT_OPT2_FIXED_LIMBS; ++iter) {
        uint m = (uint)((ulong)tt[0] * (ulong)np0);
        ulong carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < 2u * MONT_OPT2_FIXED_LIMBS + 2u; ++j) {
            ulong add = (j < MONT_OPT2_FIXED_LIMBS) ? (ulong)m * (ulong)n[j] : 0ul;
            ulong uv = (ulong)tt[j] + add + carry;
            if (j > 0u) {
                tt[j - 1u] = (uint)uv;
            }
            carry = uv >> 32;
        }
        tt[2u * MONT_OPT2_FIXED_LIMBS + 1u] = (uint)carry;
    }
}

static inline void mont_sqr_priv_unroll_only_512_body(
    __global uint *out,
    __global const uint *a,
    __constant uint *n,
    uint base,
    uint np0)
{
    uint A[MONT_OPT2_FIXED_LIMBS];
    uint N[MONT_OPT2_FIXED_LIMBS];
    #pragma unroll
    for (uint j = 0u; j < MONT_OPT2_FIXED_LIMBS; ++j) {
        A[j] = a[base + j];
        N[j] = n[j];
    }

    uint prod[32];
    sqr_basecase_512_priv(A, prod);

    uint tt[34];
    #pragma unroll
    for (uint i = 0u; i < 32u; ++i) {
        tt[i] = prod[i];
    }
    tt[32] = 0u;
    tt[33] = 0u;
    mont_redc_cios_512_priv(tt, n, np0);

    ulong borrow = 0ul;
    uint D[MONT_OPT2_FIXED_LIMBS];
    #pragma unroll
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        ulong tv = (ulong)tt[i];
        ulong nv = (ulong)N[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }

    uint any_high = (tt[MONT_OPT2_FIXED_LIMBS] | tt[MONT_OPT2_FIXED_LIMBS + 1u]) != 0u;
    uint no_borrow = (borrow == 0u);
    uint need_sub = any_high | no_borrow;
    uint mask = 0u - need_sub;
    #pragma unroll
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        out[base + i] = (D[i] & mask) | (tt[i] & ~mask);
    }
}

// Legacy CIOS sqr (= mul with B=A); kept for A/B reference.
static inline void mont_sqr_priv_unroll_only_512_mul_body(
    __global uint *out,
    __global const uint *a,
    __constant uint *n,
    uint base,
    uint np0)
{
    uint t[MONT_OPT2_FIXED_LIMBS + 2u];
    #pragma unroll
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS + 2u; ++i) {
        t[i] = 0u;
    }
    uint B[MONT_OPT2_FIXED_LIMBS];
    uint N[MONT_OPT2_FIXED_LIMBS];
    #pragma unroll
    for (uint j = 0u; j < MONT_OPT2_FIXED_LIMBS; ++j) {
        B[j] = a[base + j];
        N[j] = n[j];
    }

    #pragma unroll
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        uint ai = a[base + i];
        ulong carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < MONT_OPT2_FIXED_LIMBS; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong top = (ulong)t[MONT_OPT2_FIXED_LIMBS] + carry;
        t[MONT_OPT2_FIXED_LIMBS] = (uint)top;
        t[MONT_OPT2_FIXED_LIMBS + 1u] = (uint)(top >> 32);

        uint m = (uint)((ulong)t[0] * (ulong)np0);
        carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < MONT_OPT2_FIXED_LIMBS; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
            if (j > 0u) {
                t[j - 1u] = (uint)uv;
            }
            carry = uv >> 32;
        }
        top = (ulong)t[MONT_OPT2_FIXED_LIMBS] + carry;
        t[MONT_OPT2_FIXED_LIMBS - 1u] = (uint)top;
        top = (ulong)t[MONT_OPT2_FIXED_LIMBS + 1u] + (top >> 32);
        t[MONT_OPT2_FIXED_LIMBS] = (uint)top;
        t[MONT_OPT2_FIXED_LIMBS + 1u] = (uint)(top >> 32);
    }

    ulong borrow = 0ul;
    uint D[MONT_OPT2_FIXED_LIMBS];
    #pragma unroll
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        ulong tv = (ulong)t[i];
        ulong nv = (ulong)N[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }

    uint need_sub = (t[MONT_OPT2_FIXED_LIMBS] != 0u || t[MONT_OPT2_FIXED_LIMBS + 1u] != 0u) ? 1u : 0u;
    need_sub = (borrow == 0u) ? 1u : need_sub;
    uint mask = 0u - need_sub;
    #pragma unroll
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        out[base + i] = (D[i] & mask) | (t[i] & ~mask);
    }
}

static inline void mont_mul_priv_local_only_512_body(
    __global uint *out,
    __global const uint *a,
    __global const uint *b,
    __constant uint *n,
    uint base,
    uint np0,
    __local uint *local_mem,
    uint lid,
    uint lsize)
{
    __local uint *B_cache = local_mem;
    __local uint *N_cache = B_cache + lsize * MONT_OPT2_FIXED_LIMBS;
    __local uint *B = B_cache + lid * MONT_OPT2_FIXED_LIMBS;
    __local uint *N = N_cache + lid * MONT_OPT2_FIXED_LIMBS;

    for (uint j = 0u; j < MONT_OPT2_FIXED_LIMBS; ++j) {
        B[j] = b[base + j];
        N[j] = n[j];
    }

    uint t[MONT_OPT2_FIXED_LIMBS + 2u];
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS + 2u; ++i) {
        t[i] = 0u;
    }

    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        uint ai = a[base + i];
        ulong carry = 0ul;
        for (uint j = 0u; j < MONT_OPT2_FIXED_LIMBS; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong top = (ulong)t[MONT_OPT2_FIXED_LIMBS] + carry;
        t[MONT_OPT2_FIXED_LIMBS] = (uint)top;
        t[MONT_OPT2_FIXED_LIMBS + 1u] = (uint)(top >> 32);

        uint m = (uint)((ulong)t[0] * (ulong)np0);
        carry = 0ul;
        for (uint j = 0u; j < MONT_OPT2_FIXED_LIMBS; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
            if (j > 0u) {
                t[j - 1u] = (uint)uv;
            }
            carry = uv >> 32;
        }
        top = (ulong)t[MONT_OPT2_FIXED_LIMBS] + carry;
        t[MONT_OPT2_FIXED_LIMBS - 1u] = (uint)top;
        top = (ulong)t[MONT_OPT2_FIXED_LIMBS + 1u] + (top >> 32);
        t[MONT_OPT2_FIXED_LIMBS] = (uint)top;
        t[MONT_OPT2_FIXED_LIMBS + 1u] = (uint)(top >> 32);
    }

    ulong borrow = 0ul;
    uint D[MONT_OPT2_FIXED_LIMBS];
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        ulong tv = (ulong)t[i];
        ulong nv = (ulong)N[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }

    uint need_sub = (t[MONT_OPT2_FIXED_LIMBS] != 0u || t[MONT_OPT2_FIXED_LIMBS + 1u] != 0u) ? 1u : 0u;
    need_sub = (borrow == 0u) ? 1u : need_sub;
    uint mask = 0u - need_sub;
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        out[base + i] = (D[i] & mask) | (t[i] & ~mask);
    }
}

static inline void mont_sqr_priv_local_only_512_body(
    __global uint *out,
    __global const uint *a,
    __constant uint *n,
    uint base,
    uint np0,
    __local uint *local_mem,
    uint lid,
    uint lsize)
{
    __local uint *B_cache = local_mem;
    __local uint *N_cache = B_cache + lsize * MONT_OPT2_FIXED_LIMBS;
    __local uint *B = B_cache + lid * MONT_OPT2_FIXED_LIMBS;
    __local uint *N = N_cache + lid * MONT_OPT2_FIXED_LIMBS;

    for (uint j = 0u; j < MONT_OPT2_FIXED_LIMBS; ++j) {
        B[j] = a[base + j];
        N[j] = n[j];
    }

    uint t[MONT_OPT2_FIXED_LIMBS + 2u];
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS + 2u; ++i) {
        t[i] = 0u;
    }

    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        uint ai = a[base + i];
        ulong carry = 0ul;
        for (uint j = 0u; j < MONT_OPT2_FIXED_LIMBS; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong top = (ulong)t[MONT_OPT2_FIXED_LIMBS] + carry;
        t[MONT_OPT2_FIXED_LIMBS] = (uint)top;
        t[MONT_OPT2_FIXED_LIMBS + 1u] = (uint)(top >> 32);

        uint m = (uint)((ulong)t[0] * (ulong)np0);
        carry = 0ul;
        for (uint j = 0u; j < MONT_OPT2_FIXED_LIMBS; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
            if (j > 0u) {
                t[j - 1u] = (uint)uv;
            }
            carry = uv >> 32;
        }
        top = (ulong)t[MONT_OPT2_FIXED_LIMBS] + carry;
        t[MONT_OPT2_FIXED_LIMBS - 1u] = (uint)top;
        top = (ulong)t[MONT_OPT2_FIXED_LIMBS + 1u] + (top >> 32);
        t[MONT_OPT2_FIXED_LIMBS] = (uint)top;
        t[MONT_OPT2_FIXED_LIMBS + 1u] = (uint)(top >> 32);
    }

    ulong borrow = 0ul;
    uint D[MONT_OPT2_FIXED_LIMBS];
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        ulong tv = (ulong)t[i];
        ulong nv = (ulong)N[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }

    uint need_sub = (t[MONT_OPT2_FIXED_LIMBS] != 0u || t[MONT_OPT2_FIXED_LIMBS + 1u] != 0u) ? 1u : 0u;
    need_sub = (borrow == 0u) ? 1u : need_sub;
    uint mask = 0u - need_sub;
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        out[base + i] = (D[i] & mask) | (t[i] & ~mask);
    }
}

// Fixed 4096-bit (128 limbs) local-only path:
// keep loops dynamic to avoid huge codegen / VGPR blow-up from full unroll.
static inline void mont_mul_priv_local_only_4096_body(
    __global uint *out,
    __global const uint *a,
    __global const uint *b,
    __constant uint *n,
    uint base,
    uint np0,
    __local uint *local_mem,
    uint lid,
    uint lsize)
{
    __local uint *B_cache = local_mem;
    __local uint *N_cache = B_cache + lsize * MONT_FIXED_4096_LIMBS;
    __local uint *B = B_cache + lid * MONT_FIXED_4096_LIMBS;
    __local uint *N = N_cache + lid * MONT_FIXED_4096_LIMBS;

    for (uint j = 0u; j < MONT_FIXED_4096_LIMBS; ++j) {
        B[j] = b[base + j];
        N[j] = n[j];
    }

    uint t[MONT_FIXED_4096_LIMBS + 2u];
    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS + 2u; ++i) {
        t[i] = 0u;
    }

    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        uint ai = a[base + i];
        ulong carry = 0ul;
        for (uint j = 0u; j < MONT_FIXED_4096_LIMBS; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong top = (ulong)t[MONT_FIXED_4096_LIMBS] + carry;
        t[MONT_FIXED_4096_LIMBS] = (uint)top;
        t[MONT_FIXED_4096_LIMBS + 1u] = (uint)(top >> 32);

        uint m = (uint)((ulong)t[0] * (ulong)np0);
        carry = 0ul;
        for (uint j = 0u; j < MONT_FIXED_4096_LIMBS; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
            if (j > 0u) {
                t[j - 1u] = (uint)uv;
            }
            carry = uv >> 32;
        }
        top = (ulong)t[MONT_FIXED_4096_LIMBS] + carry;
        t[MONT_FIXED_4096_LIMBS - 1u] = (uint)top;
        top = (ulong)t[MONT_FIXED_4096_LIMBS + 1u] + (top >> 32);
        t[MONT_FIXED_4096_LIMBS] = (uint)top;
        t[MONT_FIXED_4096_LIMBS + 1u] = (uint)(top >> 32);
    }

    ulong borrow = 0ul;
    uint D[MONT_FIXED_4096_LIMBS];
    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        ulong tv = (ulong)t[i];
        ulong nv = (ulong)N[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }

    uint need_sub = (t[MONT_FIXED_4096_LIMBS] != 0u || t[MONT_FIXED_4096_LIMBS + 1u] != 0u) ? 1u : 0u;
    need_sub = (borrow == 0u) ? 1u : need_sub;
    uint mask = 0u - need_sub;
    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        out[base + i] = (D[i] & mask) | (t[i] & ~mask);
    }
}

static inline void mont_sqr_priv_local_only_4096_body(
    __global uint *out,
    __global const uint *a,
    __constant uint *n,
    uint base,
    uint np0,
    __local uint *local_mem,
    uint lid,
    uint lsize)
{
    __local uint *B_cache = local_mem;
    __local uint *N_cache = B_cache + lsize * MONT_FIXED_4096_LIMBS;
    __local uint *B = B_cache + lid * MONT_FIXED_4096_LIMBS;
    __local uint *N = N_cache + lid * MONT_FIXED_4096_LIMBS;

    for (uint j = 0u; j < MONT_FIXED_4096_LIMBS; ++j) {
        B[j] = a[base + j];
        N[j] = n[j];
    }

    uint t[MONT_FIXED_4096_LIMBS + 2u];
    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS + 2u; ++i) {
        t[i] = 0u;
    }

    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        uint ai = a[base + i];
        ulong carry = 0ul;
        for (uint j = 0u; j < MONT_FIXED_4096_LIMBS; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong top = (ulong)t[MONT_FIXED_4096_LIMBS] + carry;
        t[MONT_FIXED_4096_LIMBS] = (uint)top;
        t[MONT_FIXED_4096_LIMBS + 1u] = (uint)(top >> 32);

        uint m = (uint)((ulong)t[0] * (ulong)np0);
        carry = 0ul;
        for (uint j = 0u; j < MONT_FIXED_4096_LIMBS; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
            if (j > 0u) {
                t[j - 1u] = (uint)uv;
            }
            carry = uv >> 32;
        }
        top = (ulong)t[MONT_FIXED_4096_LIMBS] + carry;
        t[MONT_FIXED_4096_LIMBS - 1u] = (uint)top;
        top = (ulong)t[MONT_FIXED_4096_LIMBS + 1u] + (top >> 32);
        t[MONT_FIXED_4096_LIMBS] = (uint)top;
        t[MONT_FIXED_4096_LIMBS + 1u] = (uint)(top >> 32);
    }

    ulong borrow = 0ul;
    uint D[MONT_FIXED_4096_LIMBS];
    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        ulong tv = (ulong)t[i];
        ulong nv = (ulong)N[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }

    uint need_sub = (t[MONT_FIXED_4096_LIMBS] != 0u || t[MONT_FIXED_4096_LIMBS + 1u] != 0u) ? 1u : 0u;
    need_sub = (borrow == 0u) ? 1u : need_sub;
    uint mask = 0u - need_sub;
    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        out[base + i] = (D[i] & mask) | (t[i] & ~mask);
    }
}

#define FIPS4096_T_WORDS (2u * MONT_FIXED_4096_LIMBS + 2u)
#define FIPS4096_P_WORDS (MONT_FIXED_4096_LIMBS + 1u)
#define FIPS4096_MT_PROD_WORDS (2u * MONT_FIXED_4096_LIMBS)

static inline void fips4096_finalize_p_priv(__global uint *out, const uint P[FIPS4096_P_WORDS],
                                              __constant uint *n, uint base) {
    uint ps = P[MONT_FIXED_4096_LIMBS];
    ulong borrow = 0ul;
    uint D[MONT_FIXED_4096_LIMBS];
    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        ulong tv = (ulong)P[i];
        ulong nv = (ulong)n[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }
    uint any_high = (ps != 0u) ? 1u : 0u;
    uint need_sub = any_high | ((borrow == 0u) ? 1u : 0u);
    uint mask = 0u - need_sub;
    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        out[base + i] = (D[i] & mask) | (P[i] & ~mask);
    }
}

static inline void fips4096_finalize_p_local(__global uint *out, __local uint *P, __constant uint *n,
                                             uint base) {
    uint ps = P[MONT_FIXED_4096_LIMBS];
    ulong borrow = 0ul;
    uint D[MONT_FIXED_4096_LIMBS];
    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        ulong tv = (ulong)P[i];
        ulong nv = (ulong)n[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }
    uint any_high = (ps != 0u) ? 1u : 0u;
    uint need_sub = any_high | ((borrow == 0u) ? 1u : 0u);
    uint mask = 0u - need_sub;
    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        out[base + i] = (D[i] & mask) | (P[i] & ~mask);
    }
}

static inline void fips4096_finalize_t_local(__global uint *out, __local uint *t, __constant uint *n,
                                             uint base) {
    ulong borrow = 0ul;
    uint D[MONT_FIXED_4096_LIMBS];
    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        ulong tv = (ulong)t[i];
        ulong nv = (ulong)n[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }
    uint need_sub = (t[MONT_FIXED_4096_LIMBS] != 0u || t[MONT_FIXED_4096_LIMBS + 1u] != 0u) ? 1u : 0u;
    need_sub = (borrow == 0u) ? 1u : need_sub;
    uint mask = 0u - need_sub;
    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        out[base + i] = (D[i] & mask) | (t[i] & ~mask);
    }
}

static inline void mont_redc_cios_4096_local(__local uint *tt, __constant uint *n, uint np0) {
    for (uint iter = 0u; iter < MONT_FIXED_4096_LIMBS; ++iter) {
        uint m = (uint)((ulong)tt[0] * (ulong)np0);
        ulong carry = 0ul;
        for (uint j = 0u; j < FIPS4096_T_WORDS; ++j) {
            ulong add = (j < MONT_FIXED_4096_LIMBS) ? (ulong)m * (ulong)n[j] : 0ul;
            ulong uv = (ulong)tt[j] + add + carry;
            if (j > 0u) {
                tt[j - 1u] = (uint)uv;
            }
            carry = uv >> 32;
        }
        tt[FIPS4096_T_WORDS - 1u] = (uint)carry;
    }
}

// Single-lane FIPS (Koç Algorithm 1) for 4096-bit.
static inline void mont_mul_priv_fips4096_body(__global uint *out, __global const uint *a,
                                               __global const uint *b, __constant uint *n,
                                               uint base, uint np0) {
    uint P[FIPS4096_P_WORDS];
    uint t = 0u;
    uint u = 0u;
    uint v = 0u;

    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        for (uint j = 0u; j < i; ++j) {
            fips512_csa_add_priv(&t, &u, &v, (ulong)a[base + j] * (ulong)b[base + i - j]);
            fips512_csa_add_priv(&t, &u, &v, (ulong)P[j] * (ulong)n[i - j]);
        }
        fips512_csa_add_priv(&t, &u, &v, (ulong)a[base + i] * (ulong)b[base]);
        uint pi = (uint)((ulong)v * (ulong)np0);
        fips512_csa_add_priv(&t, &u, &v, (ulong)pi * (ulong)n[0]);
        P[i] = pi;
        fips512_csa_shift_priv(&t, &u, &v);
    }

    for (uint i = MONT_FIXED_4096_LIMBS; i < 2u * MONT_FIXED_4096_LIMBS; ++i) {
        for (uint j = i - MONT_FIXED_4096_LIMBS + 1u; j < MONT_FIXED_4096_LIMBS; ++j) {
            fips512_csa_add_priv(&t, &u, &v, (ulong)a[base + j] * (ulong)b[base + i - j]);
            fips512_csa_add_priv(&t, &u, &v, (ulong)P[j] * (ulong)n[i - j]);
        }
        P[i - MONT_FIXED_4096_LIMBS] = v;
        fips512_csa_shift_priv(&t, &u, &v);
    }
    P[MONT_FIXED_4096_LIMBS] = v;

    fips4096_finalize_p_priv(out, P, n, base);
}

static inline void mont_sqr_priv_fips4096_body(__global uint *out, __global const uint *a,
                                               __constant uint *n, uint base, uint np0) {
    mont_mul_priv_fips4096_body(out, a, a, n, base, np0);
}

static inline void mont_mul_priv_fips4096_mtn_body(__global uint *out, __global const uint *a,
                                                   __global const uint *b, __constant uint *n,
                                                   uint base, uint np0, __local uint *local_mem,
                                                   uint lid, uint mt) {
    __local uint *A = local_mem;
    __local uint *B = A + MONT_FIXED_4096_LIMBS;
    __local ulong *prods = (__local ulong *)(B + MONT_FIXED_4096_LIMBS);
    __local uint *P = (__local uint *)(prods + FIPS4096_MT_PROD_WORDS);

    for (uint j = lid; j < MONT_FIXED_4096_LIMBS; j += mt) {
        A[j] = a[base + j];
        B[j] = b[base + j];
    }
    if (lid == 0u) {
        for (uint i = 0u; i < FIPS4096_P_WORDS; ++i) {
            P[i] = 0u;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint t = 0u;
    uint u = 0u;
    uint v = 0u;

    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        for (uint j = lid; j < i; j += mt) {
            prods[2u * j] = (ulong)A[j] * (ulong)B[i - j];
            prods[2u * j + 1u] = (ulong)P[j] * (ulong)n[i - j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid == 0u) {
            for (uint j = 0u; j < i; ++j) {
                fips512_csa_add_priv(&t, &u, &v, prods[2u * j]);
                fips512_csa_add_priv(&t, &u, &v, prods[2u * j + 1u]);
            }
            fips512_csa_add_priv(&t, &u, &v, (ulong)A[i] * (ulong)B[0]);
            uint pi = (uint)((ulong)v * (ulong)np0);
            fips512_csa_add_priv(&t, &u, &v, (ulong)pi * (ulong)n[0]);
            P[i] = pi;
            fips512_csa_shift_priv(&t, &u, &v);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint i = MONT_FIXED_4096_LIMBS; i < 2u * MONT_FIXED_4096_LIMBS; ++i) {
        for (uint j = i - MONT_FIXED_4096_LIMBS + 1u + lid; j < MONT_FIXED_4096_LIMBS; j += mt) {
            prods[2u * j] = (ulong)A[j] * (ulong)B[i - j];
            prods[2u * j + 1u] = (ulong)P[j] * (ulong)n[i - j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid == 0u) {
            for (uint j = i - MONT_FIXED_4096_LIMBS + 1u; j < MONT_FIXED_4096_LIMBS; ++j) {
                fips512_csa_add_priv(&t, &u, &v, prods[2u * j]);
                fips512_csa_add_priv(&t, &u, &v, prods[2u * j + 1u]);
            }
            P[i - MONT_FIXED_4096_LIMBS] = v;
            fips512_csa_shift_priv(&t, &u, &v);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0u) {
        P[MONT_FIXED_4096_LIMBS] = v;
        fips4096_finalize_p_local(out, P, n, base);
    }
}

static inline void mont_mul_priv_fips4096_mt4_body(__global uint *out, __global const uint *a,
                                                   __global const uint *b, __constant uint *n,
                                                   uint base, uint np0, __local uint *local_mem,
                                                   uint lid) {
    mont_mul_priv_fips4096_mtn_body(out, a, b, n, base, np0, local_mem, lid, 4u);
}

static inline void mont_mul_priv_fips4096_mt8_body(__global uint *out, __global const uint *a,
                                                   __global const uint *b, __constant uint *n,
                                                   uint base, uint np0, __local uint *local_mem,
                                                   uint lid) {
    mont_mul_priv_fips4096_mtn_body(out, a, b, n, base, np0, local_mem, lid, 8u);
}

static inline void mont_mul_priv_fips4096_mt16_body(__global uint *out, __global const uint *a,
                                                    __global const uint *b, __constant uint *n,
                                                    uint base, uint np0, __local uint *local_mem,
                                                    uint lid) {
    mont_mul_priv_fips4096_mtn_body(out, a, b, n, base, np0, local_mem, lid, 16u);
}

static inline void mont_sqr_priv_fips4096_mt4_body(__global uint *out, __global const uint *a,
                                                   __constant uint *n, uint base, uint np0,
                                                   __local uint *local_mem, uint lid) {
    mont_mul_priv_fips4096_mt4_body(out, a, a, n, base, np0, local_mem, lid);
}

static inline void mont_sqr_priv_fips4096_mt8_body(__global uint *out, __global const uint *a,
                                                   __constant uint *n, uint base, uint np0,
                                                   __local uint *local_mem, uint lid) {
    mont_mul_priv_fips4096_mt8_body(out, a, a, n, base, np0, local_mem, lid);
}

static inline void mont_sqr_priv_fips4096_mt16_body(__global uint *out, __global const uint *a,
                                                    __constant uint *n, uint base, uint np0,
                                                    __local uint *local_mem, uint lid) {
    mont_mul_priv_fips4096_mt16_body(out, a, a, n, base, np0, local_mem, lid);
}

static inline void mont_mul_priv_fips4096_mtn_cs_body(__global uint *out, __global const uint *a,
                                                      __global const uint *b, __constant uint *n,
                                                      uint base, uint np0, __local uint *local_mem,
                                                      uint lid, uint mt) {
    __local uint *A = local_mem;
    __local uint *B = A + MONT_FIXED_4096_LIMBS;
    __local uint *parts = B + MONT_FIXED_4096_LIMBS;
    __local uint *t = parts + mt * FIPS4096_T_WORDS;
    const uint tile = lid * FIPS4096_T_WORDS;

    for (uint j = lid; j < MONT_FIXED_4096_LIMBS; j += mt) {
        A[j] = a[base + j];
        B[j] = b[base + j];
    }
    for (uint idx = 0u; idx < FIPS4096_T_WORDS; ++idx) {
        parts[tile + idx] = 0u;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint t_priv[FIPS4096_T_WORDS];
    for (uint idx = 0u; idx < FIPS4096_T_WORDS; ++idx) {
        t_priv[idx] = 0u;
    }

    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        for (uint j = lid; j < MONT_FIXED_4096_LIMBS; j += mt) {
            uint k = i + j;
            ulong prod = (ulong)A[i] * (ulong)B[j];
            ulong uv = (ulong)t_priv[k] + (prod & 0xFFFFFFFFul);
            t_priv[k] = (uint)uv;
            uv = (ulong)t_priv[k + 1u] + (prod >> 32) + (uv >> 32);
            t_priv[k + 1u] = (uint)uv;
            uint pos = k + 2u;
            ulong carry = uv >> 32;
            while (carry != 0ul && pos < FIPS4096_T_WORDS) {
                uv = (ulong)t_priv[pos] + carry;
                t_priv[pos] = (uint)uv;
                carry = uv >> 32;
                pos++;
            }
        }
    }

    for (uint idx = 0u; idx < FIPS4096_T_WORDS; ++idx) {
        parts[tile + idx] = t_priv[idx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = mt >> 1; stride > 0u; stride >>= 1) {
        if (lid < stride) {
            uint src = (lid + stride) * FIPS4096_T_WORDS;
            uint dst = lid * FIPS4096_T_WORDS;
            ulong carry = 0ul;
            uint pos = 0u;
            for (; pos < FIPS4096_T_WORDS; ++pos) {
                ulong uv = (ulong)parts[dst + pos] + (ulong)parts[src + pos] + carry;
                parts[dst + pos] = (uint)uv;
                carry = uv >> 32;
            }
            while (carry != 0ul && pos < FIPS4096_T_WORDS) {
                ulong uv = (ulong)parts[dst + pos] + carry;
                parts[dst + pos] = (uint)uv;
                carry = uv >> 32;
                pos++;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0u) {
        for (uint idx = 0u; idx < FIPS4096_T_WORDS; ++idx) {
            t[idx] = parts[idx];
        }
        mont_redc_cios_4096_local(t, n, np0);
        fips4096_finalize_t_local(out, t, n, base);
    }
}

static inline void mont_mul_priv_fips4096_mt8_cs_body(__global uint *out, __global const uint *a,
                                                      __global const uint *b, __constant uint *n,
                                                      uint base, uint np0, __local uint *local_mem,
                                                      uint lid) {
    mont_mul_priv_fips4096_mtn_cs_body(out, a, b, n, base, np0, local_mem, lid, 8u);
}

static inline void mont_mul_priv_fips4096_mt16_cs_body(__global uint *out, __global const uint *a,
                                                       __global const uint *b, __constant uint *n,
                                                       uint base, uint np0, __local uint *local_mem,
                                                       uint lid) {
    mont_mul_priv_fips4096_mtn_cs_body(out, a, b, n, base, np0, local_mem, lid, 16u);
}

static inline void mont_sqr_priv_fips4096_mt8_cs_body(__global uint *out, __global const uint *a,
                                                      __constant uint *n, uint base, uint np0,
                                                      __local uint *local_mem, uint lid) {
    mont_mul_priv_fips4096_mt8_cs_body(out, a, a, n, base, np0, local_mem, lid);
}

static inline void mont_sqr_priv_fips4096_mt16_cs_body(__global uint *out, __global const uint *a,
                                                       __constant uint *n, uint base, uint np0,
                                                       __local uint *local_mem, uint lid) {
    mont_mul_priv_fips4096_mt16_cs_body(out, a, a, n, base, np0, local_mem, lid);
}

// Fixed 4096-bit unroll-factor path (no local memory).
// Use mild unroll factor (x64) for better ILP without exploding VGPR/code size.
static inline void mont_mul_priv_unroll64_4096_body(
    __global uint *out,
    __global const uint *a,
    __global const uint *b,
    __constant uint *n,
    uint base,
    uint np0)
{
    uint t[MONT_FIXED_4096_LIMBS + 2u];
    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS + 2u; ++i) {
        t[i] = 0u;
    }
    uint B[MONT_FIXED_4096_LIMBS];
    for (uint j = 0u; j < MONT_FIXED_4096_LIMBS; ++j) {
        B[j] = b[base + j];
    }

    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        uint ai = a[base + i];
        ulong carry = 0ul;
        #pragma unroll 64
        for (uint j = 0u; j < MONT_FIXED_4096_LIMBS; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong top = (ulong)t[MONT_FIXED_4096_LIMBS] + carry;
        t[MONT_FIXED_4096_LIMBS] = (uint)top;
        t[MONT_FIXED_4096_LIMBS + 1u] = (uint)(top >> 32);

        uint m = (uint)((ulong)t[0] * (ulong)np0);

        // Pass 2: 提取 j=0，消除 if(j>0) 分支
        ulong uv0 = (ulong)t[0] + (ulong)m * (ulong)n[0];
        carry = uv0 >> 32;

        #pragma unroll 64
        for (uint j = 1u; j < MONT_FIXED_4096_LIMBS; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)n[j] + carry;
            t[j - 1u] = (uint)uv;
            carry = uv >> 32;
        }
        // carry = 0ul;
        // #pragma unroll 64
        // for (uint j = 0u; j < MONT_FIXED_4096_LIMBS; ++j) {
        //     ulong uv = (ulong)t[j] + (ulong)m * (ulong)n[j] + carry;
        //     if (j > 0u) {
        //         t[j - 1u] = (uint)uv;
        //     }
        //     carry = uv >> 32;
        // }
        top = (ulong)t[MONT_FIXED_4096_LIMBS] + carry;
        t[MONT_FIXED_4096_LIMBS - 1u] = (uint)top;
        top = (ulong)t[MONT_FIXED_4096_LIMBS + 1u] + (top >> 32);
        t[MONT_FIXED_4096_LIMBS] = (uint)top;
        t[MONT_FIXED_4096_LIMBS + 1u] = (uint)(top >> 32);
    }

    ulong borrow = 0ul;
    uint D[MONT_FIXED_4096_LIMBS];
    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        ulong tv = (ulong)t[i];
        ulong nv = (ulong)n[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }
    uint need_sub = (t[MONT_FIXED_4096_LIMBS] != 0u || t[MONT_FIXED_4096_LIMBS + 1u] != 0u) ? 1u : 0u;
    need_sub = (borrow == 0u) ? 1u : need_sub;
    uint mask = 0u - need_sub;
    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        out[base + i] = (D[i] & mask) | (t[i] & ~mask);
    }
}

static inline void mont_sqr_priv_unroll64_4096_body(
    __global uint *out,
    __global const uint *a,
    __constant uint *n,
    uint base,
    uint np0)
{
    mont_mul_priv_unroll64_4096_body(out, a, a, n, base, np0);
}

// Fixed 4096-bit single-lane variant without D[128] temp.
// Do one borrow-probe pass, then a writeback pass based on need_sub.
static inline void mont_mul_priv_unroll64_4096_nod_body(
    __global uint *out,
    __global const uint *a,
    __global const uint *b,
    __constant uint *n,
    uint base,
    uint np0)
{
    uint t[MONT_FIXED_4096_LIMBS + 2u];
    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS + 2u; ++i) {
        t[i] = 0u;
    }
    uint B[MONT_FIXED_4096_LIMBS];
    for (uint j = 0u; j < MONT_FIXED_4096_LIMBS; ++j) {
        B[j] = b[base + j];
    }

    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        uint ai = a[base + i];
        ulong carry = 0ul;
        #pragma unroll 64
        for (uint j = 0u; j < MONT_FIXED_4096_LIMBS; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong top = (ulong)t[MONT_FIXED_4096_LIMBS] + carry;
        t[MONT_FIXED_4096_LIMBS] = (uint)top;
        t[MONT_FIXED_4096_LIMBS + 1u] = (uint)(top >> 32);

        uint m = (uint)((ulong)t[0] * (ulong)np0);
        ulong uv0 = (ulong)t[0] + (ulong)m * (ulong)n[0];
        carry = uv0 >> 32;

        #pragma unroll 64
        for (uint j = 1u; j < MONT_FIXED_4096_LIMBS; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)n[j] + carry;
            t[j - 1u] = (uint)uv;
            carry = uv >> 32;
        }
        top = (ulong)t[MONT_FIXED_4096_LIMBS] + carry;
        t[MONT_FIXED_4096_LIMBS - 1u] = (uint)top;
        top = (ulong)t[MONT_FIXED_4096_LIMBS + 1u] + (top >> 32);
        t[MONT_FIXED_4096_LIMBS] = (uint)top;
        t[MONT_FIXED_4096_LIMBS + 1u] = (uint)(top >> 32);
    }

    ulong borrow = 0ul;
    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        ulong tv = (ulong)t[i];
        ulong nv = (ulong)n[i];
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }
    uint need_sub = (t[MONT_FIXED_4096_LIMBS] != 0u || t[MONT_FIXED_4096_LIMBS + 1u] != 0u) ? 1u : 0u;
    need_sub = (borrow == 0u) ? 1u : need_sub;

    if (need_sub) {
        borrow = 0ul;
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            ulong tv = (ulong)t[i];
            ulong nv = (ulong)n[i];
            ulong w = tv - nv - borrow;
            out[base + i] = (uint)w;
            borrow = (tv < nv + borrow) ? 1ul : 0ul;
        }
    } else {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            out[base + i] = t[i];
        }
    }
}

static inline void mont_sqr_priv_unroll64_4096_nod_body(
    __global uint *out,
    __global const uint *a,
    __constant uint *n,
    uint base,
    uint np0)
{
    mont_mul_priv_unroll64_4096_nod_body(out, a, a, n, base, np0);
}

// Fixed 4096-bit 2-thread cooperative path:
// one work-group (local_size=2) handles one instance, split as 2x64 limbs.
static inline void mont_mul_priv_unroll64_4096_mt2_body(
    __global uint *out,
    __global const uint *a,
    __global const uint *b,
    __constant uint *n,
    uint base,
    uint np0,
    __local uint *local_mem,
    uint lid)
{
    __local uint *t = local_mem;                                           // 130
    __local uint *B = t + (MONT_FIXED_4096_LIMBS + 2u);                    // 128
    __local uint *D = B + MONT_FIXED_4096_LIMBS;                           // 128
    __local uint *meta = D + MONT_FIXED_4096_LIMBS;                        // >= 3
    const uint half_words = MONT_FIXED_4096_LIMBS / 2u;                    // 64
    const uint j_begin = lid * half_words;
    const uint j_end = j_begin + half_words;

    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS + 2u; ++i) {
            t[i] = 0u;
        }
    }

    #pragma unroll 64
    for (uint j = j_begin; j < j_end; ++j) {
        B[j] = b[base + j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        uint ai = a[base + i];

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
            ulong uv0 = (ulong)t[0] + (ulong)m * (ulong)n[0];
            ulong carry = uv0 >> 32;
            #pragma unroll 64
            for (uint j = 1u; j < half_words; ++j) {
                ulong uv = (ulong)t[j] + (ulong)m * (ulong)n[j] + carry;
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
                ulong uv = (ulong)t[j] + (ulong)m * (ulong)n[j] + carry;
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
            ulong nv = (ulong)n[i];
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
            ulong nv = (ulong)n[i];
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
        out[base + i] = (D[i] & mask) | (t[i] & ~mask);
    }
}

static inline void mont_sqr_priv_unroll64_4096_mt2_body(
    __global uint *out,
    __global const uint *a,
    __constant uint *n,
    uint base,
    uint np0,
    __local uint *local_mem,
    uint lid)
{
    mont_mul_priv_unroll64_4096_mt2_body(out, a, a, n, base, np0, local_mem, lid);
}

// 4096-bit N-lane cooperative CIOS (pass1/pass2/finalize split into N chunks).
// meta layout: meta[0..N-2] carry chain, meta[N-1] m, meta[N] need_sub.
static inline void mont_mul_priv_unroll64_4096_mt4_body(
    __global uint *out,
    __global const uint *a,
    __global const uint *b,
    __constant uint *n,
    uint base,
    uint np0,
    __local uint *local_mem,
    uint lid)
{
    const uint mt = 4u;
    const uint chunk = MONT_FIXED_4096_LIMBS / mt;
    __local uint *t = local_mem;
    __local uint *B = t + (MONT_FIXED_4096_LIMBS + 2u);
    __local uint *D = B + MONT_FIXED_4096_LIMBS;
    __local uint *meta = D + MONT_FIXED_4096_LIMBS;
    const uint j_begin = lid * chunk;
    const uint j_end = j_begin + chunk;

    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS + 2u; ++i) {
            t[i] = 0u;
        }
    }

    #pragma unroll 32
    for (uint j = j_begin; j < j_end; ++j) {
        B[j] = b[base + j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        uint ai = a[base + i];

        for (uint phase = 0u; phase < mt; ++phase) {
            if (lid == phase) {
                ulong carry = (phase == 0u) ? 0ul : (ulong)meta[phase - 1u];
                #pragma unroll 32
                for (uint j = phase * chunk; j < (phase + 1u) * chunk; ++j) {
                    ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
                    t[j] = (uint)uv;
                    carry = uv >> 32;
                }
                if (phase + 1u < mt) {
                    meta[phase] = (uint)carry;
                } else {
                    ulong top = (ulong)t[MONT_FIXED_4096_LIMBS] + carry;
                    t[MONT_FIXED_4096_LIMBS] = (uint)top;
                    t[MONT_FIXED_4096_LIMBS + 1u] = (uint)(top >> 32);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        for (uint phase = 0u; phase < mt; ++phase) {
            if (lid == phase) {
                if (phase == 0u) {
                    uint m = (uint)((ulong)t[0] * (ulong)np0);
                    meta[mt - 1u] = m;
                    ulong uv0 = (ulong)t[0] + (ulong)m * (ulong)n[0];
                    ulong carry = uv0 >> 32;
                    #pragma unroll 32
                    for (uint j = 1u; j < chunk; ++j) {
                        ulong uv = (ulong)t[j] + (ulong)m * (ulong)n[j] + carry;
                        t[j - 1u] = (uint)uv;
                        carry = uv >> 32;
                    }
                    meta[0] = (uint)carry;
                } else {
                    uint m = meta[mt - 1u];
                    ulong carry = (ulong)meta[phase - 1u];
                    #pragma unroll 32
                    for (uint j = phase * chunk; j < (phase + 1u) * chunk; ++j) {
                        ulong uv = (ulong)t[j] + (ulong)m * (ulong)n[j] + carry;
                        t[j - 1u] = (uint)uv;
                        carry = uv >> 32;
                    }
                    if (phase + 1u < mt) {
                        meta[phase] = (uint)carry;
                    } else {
                        ulong top = (ulong)t[MONT_FIXED_4096_LIMBS] + carry;
                        t[MONT_FIXED_4096_LIMBS - 1u] = (uint)top;
                        top = (ulong)t[MONT_FIXED_4096_LIMBS + 1u] + (top >> 32);
                        t[MONT_FIXED_4096_LIMBS] = (uint)top;
                        t[MONT_FIXED_4096_LIMBS + 1u] = (uint)(top >> 32);
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    for (uint phase = 0u; phase < mt; ++phase) {
        if (lid == phase) {
            ulong borrow = (phase == 0u) ? 0ul : (ulong)meta[phase - 1u];
            #pragma unroll 32
            for (uint j = phase * chunk; j < (phase + 1u) * chunk; ++j) {
                ulong tv = (ulong)t[j];
                ulong nv = (ulong)n[j];
                ulong w = tv - nv - borrow;
                D[j] = (uint)w;
                borrow = (tv < nv + borrow) ? 1ul : 0ul;
            }
            if (phase + 1u < mt) {
                meta[phase] = (uint)borrow;
            } else {
                uint need_sub =
                    (t[MONT_FIXED_4096_LIMBS] != 0u || t[MONT_FIXED_4096_LIMBS + 1u] != 0u) ? 1u : 0u;
                need_sub = (borrow == 0u) ? 1u : need_sub;
                meta[mt] = need_sub;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    uint mask = 0u - meta[mt];
    #pragma unroll 32
    for (uint j = j_begin; j < j_end; ++j) {
        out[base + j] = (D[j] & mask) | (t[j] & ~mask);
    }
}

static inline void mont_sqr_priv_unroll64_4096_mt4_body(
    __global uint *out,
    __global const uint *a,
    __constant uint *n,
    uint base,
    uint np0,
    __local uint *local_mem,
    uint lid)
{
    mont_mul_priv_unroll64_4096_mt4_body(out, a, a, n, base, np0, local_mem, lid);
}

static inline void mont_mul_priv_unroll64_4096_mt8_body(
    __global uint *out,
    __global const uint *a,
    __global const uint *b,
    __constant uint *n,
    uint base,
    uint np0,
    __local uint *local_mem,
    uint lid)
{
    const uint mt = 8u;
    const uint chunk = MONT_FIXED_4096_LIMBS / mt;
    __local uint *t = local_mem;
    __local uint *B = t + (MONT_FIXED_4096_LIMBS + 2u);
    __local uint *D = B + MONT_FIXED_4096_LIMBS;
    __local uint *meta = D + MONT_FIXED_4096_LIMBS;
    const uint j_begin = lid * chunk;
    const uint j_end = j_begin + chunk;

    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS + 2u; ++i) {
            t[i] = 0u;
        }
    }

    #pragma unroll 16
    for (uint j = j_begin; j < j_end; ++j) {
        B[j] = b[base + j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        uint ai = a[base + i];

        for (uint phase = 0u; phase < mt; ++phase) {
            if (lid == phase) {
                ulong carry = (phase == 0u) ? 0ul : (ulong)meta[phase - 1u];
                #pragma unroll 16
                for (uint j = phase * chunk; j < (phase + 1u) * chunk; ++j) {
                    ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
                    t[j] = (uint)uv;
                    carry = uv >> 32;
                }
                if (phase + 1u < mt) {
                    meta[phase] = (uint)carry;
                } else {
                    ulong top = (ulong)t[MONT_FIXED_4096_LIMBS] + carry;
                    t[MONT_FIXED_4096_LIMBS] = (uint)top;
                    t[MONT_FIXED_4096_LIMBS + 1u] = (uint)(top >> 32);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        for (uint phase = 0u; phase < mt; ++phase) {
            if (lid == phase) {
                if (phase == 0u) {
                    uint m = (uint)((ulong)t[0] * (ulong)np0);
                    meta[mt - 1u] = m;
                    ulong uv0 = (ulong)t[0] + (ulong)m * (ulong)n[0];
                    ulong carry = uv0 >> 32;
                    #pragma unroll 16
                    for (uint j = 1u; j < chunk; ++j) {
                        ulong uv = (ulong)t[j] + (ulong)m * (ulong)n[j] + carry;
                        t[j - 1u] = (uint)uv;
                        carry = uv >> 32;
                    }
                    meta[0] = (uint)carry;
                } else {
                    uint m = meta[mt - 1u];
                    ulong carry = (ulong)meta[phase - 1u];
                    #pragma unroll 16
                    for (uint j = phase * chunk; j < (phase + 1u) * chunk; ++j) {
                        ulong uv = (ulong)t[j] + (ulong)m * (ulong)n[j] + carry;
                        t[j - 1u] = (uint)uv;
                        carry = uv >> 32;
                    }
                    if (phase + 1u < mt) {
                        meta[phase] = (uint)carry;
                    } else {
                        ulong top = (ulong)t[MONT_FIXED_4096_LIMBS] + carry;
                        t[MONT_FIXED_4096_LIMBS - 1u] = (uint)top;
                        top = (ulong)t[MONT_FIXED_4096_LIMBS + 1u] + (top >> 32);
                        t[MONT_FIXED_4096_LIMBS] = (uint)top;
                        t[MONT_FIXED_4096_LIMBS + 1u] = (uint)(top >> 32);
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    for (uint phase = 0u; phase < mt; ++phase) {
        if (lid == phase) {
            ulong borrow = (phase == 0u) ? 0ul : (ulong)meta[phase - 1u];
            #pragma unroll 16
            for (uint j = phase * chunk; j < (phase + 1u) * chunk; ++j) {
                ulong tv = (ulong)t[j];
                ulong nv = (ulong)n[j];
                ulong w = tv - nv - borrow;
                D[j] = (uint)w;
                borrow = (tv < nv + borrow) ? 1ul : 0ul;
            }
            if (phase + 1u < mt) {
                meta[phase] = (uint)borrow;
            } else {
                uint need_sub =
                    (t[MONT_FIXED_4096_LIMBS] != 0u || t[MONT_FIXED_4096_LIMBS + 1u] != 0u) ? 1u : 0u;
                need_sub = (borrow == 0u) ? 1u : need_sub;
                meta[mt] = need_sub;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    uint mask = 0u - meta[mt];
    #pragma unroll 16
    for (uint j = j_begin; j < j_end; ++j) {
        out[base + j] = (D[j] & mask) | (t[j] & ~mask);
    }
}

static inline void mont_sqr_priv_unroll64_4096_mt8_body(
    __global uint *out,
    __global const uint *a,
    __constant uint *n,
    uint base,
    uint np0,
    __local uint *local_mem,
    uint lid)
{
    mont_mul_priv_unroll64_4096_mt8_body(out, a, a, n, base, np0, local_mem, lid);
}

// Fixed 4096-bit weak-sync 2-thread path:
// lane1 only helps preload B and split final writeback.
static inline void mont_mul_priv_unroll64_4096_mt2_weak_body(
    __global uint *out,
    __global const uint *a,
    __global const uint *b,
    __constant uint *n,
    uint base,
    uint np0,
    __local uint *local_mem,
    uint lid)
{
    __local uint *t = local_mem;                                           // 130
    __local uint *B = t + (MONT_FIXED_4096_LIMBS + 2u);                    // 128
    __local uint *D = B + MONT_FIXED_4096_LIMBS;                           // 128
    __local uint *meta = D + MONT_FIXED_4096_LIMBS;                        // >= 1
    const uint half_words = MONT_FIXED_4096_LIMBS / 2u;                    // 64
    const uint j_begin = lid * half_words;
    const uint j_end = j_begin + half_words;

    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS + 2u; ++i) {
            t[i] = 0u;
        }
    }

    #pragma unroll 64
    for (uint j = j_begin; j < j_end; ++j) {
        B[j] = b[base + j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            uint ai = a[base + i];
            ulong carry = 0ul;
            #pragma unroll 64
            for (uint j = 0u; j < MONT_FIXED_4096_LIMBS; ++j) {
                ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
                t[j] = (uint)uv;
                carry = uv >> 32;
            }
            ulong top = (ulong)t[MONT_FIXED_4096_LIMBS] + carry;
            t[MONT_FIXED_4096_LIMBS] = (uint)top;
            t[MONT_FIXED_4096_LIMBS + 1u] = (uint)(top >> 32);

            uint m = (uint)((ulong)t[0] * (ulong)np0);
            ulong uv0 = (ulong)t[0] + (ulong)m * (ulong)n[0];
            carry = uv0 >> 32;

            #pragma unroll 64
            for (uint j = 1u; j < MONT_FIXED_4096_LIMBS; ++j) {
                ulong uv = (ulong)t[j] + (ulong)m * (ulong)n[j] + carry;
                t[j - 1u] = (uint)uv;
                carry = uv >> 32;
            }
            top = (ulong)t[MONT_FIXED_4096_LIMBS] + carry;
            t[MONT_FIXED_4096_LIMBS - 1u] = (uint)top;
            top = (ulong)t[MONT_FIXED_4096_LIMBS + 1u] + (top >> 32);
            t[MONT_FIXED_4096_LIMBS] = (uint)top;
            t[MONT_FIXED_4096_LIMBS + 1u] = (uint)(top >> 32);
        }

        ulong borrow = 0ul;
        #pragma unroll 64
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            ulong tv = (ulong)t[i];
            ulong nv = (ulong)n[i];
            ulong w = tv - nv - borrow;
            D[i] = (uint)w;
            borrow = (tv < nv + borrow) ? 1ul : 0ul;
        }
        uint need_sub = (t[MONT_FIXED_4096_LIMBS] != 0u || t[MONT_FIXED_4096_LIMBS + 1u] != 0u) ? 1u : 0u;
        need_sub = (borrow == 0u) ? 1u : need_sub;
        meta[0] = need_sub;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint mask = 0u - meta[0];
    #pragma unroll 64
    for (uint i = j_begin; i < j_end; ++i) {
        out[base + i] = (D[i] & mask) | (t[i] & ~mask);
    }
}

static inline void mont_sqr_priv_unroll64_4096_mt2_weak_body(
    __global uint *out,
    __global const uint *a,
    __constant uint *n,
    uint base,
    uint np0,
    __local uint *local_mem,
    uint lid)
{
    mont_mul_priv_unroll64_4096_mt2_weak_body(out, a, a, n, base, np0, local_mem, lid);
}


// Raw kernels for RGA analysis (no benchmark loop wrapper).
__kernel void cgbn_mont_mul_opt2_512_local(__global const uint *a, __global const uint *b,
                                            __constant uint *n, __global uint *out,
                                            __constant uint *np0_ptr, uint limbs,
                                            __local uint *local_mem) {
    if (limbs != MONT_OPT2_FIXED_LIMBS) {
        return;
    }
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint lsize = get_local_size(0);
    uint base = gid * limbs;
    uint np0 = np0_ptr[0];
    mont_mul_priv_opt2_512_local_body(out, a, b, n, base, np0, local_mem, lid, lsize);
}

__kernel void cgbn_mont_sqr_opt2_512_local(__global const uint *a, __constant uint *n,
                                            __global uint *out, __constant uint *np0_ptr,
                                            uint limbs, __local uint *local_mem) {
    if (limbs != MONT_OPT2_FIXED_LIMBS) {
        return;
    }
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint lsize = get_local_size(0);
    uint base = gid * limbs;
    uint np0 = np0_ptr[0];
    mont_sqr_priv_opt2_512_local_body(out, a, n, base, np0, local_mem, lid, lsize);
}

__kernel void cgbn_mont_mul_unroll_only_512(__global const uint *a, __global const uint *b,
                                             __constant uint *n, __global uint *out,
                                             __constant uint *np0_ptr, uint limbs) {
    if (limbs != MONT_OPT2_FIXED_LIMBS) {
        return;
    }
    uint gid = get_global_id(0);
    uint base = gid * limbs;
    uint np0 = np0_ptr[0];
    mont_mul_priv_unroll_only_512_body(out, a, b, n, base, np0);
}

__kernel void cgbn_mont_sqr_unroll_only_512(__global const uint *a, __constant uint *n,
                                             __global uint *out, __constant uint *np0_ptr,
                                             uint limbs) {
    if (limbs != MONT_OPT2_FIXED_LIMBS) {
        return;
    }
    uint gid = get_global_id(0);
    uint base = gid * limbs;
    uint np0 = np0_ptr[0];
    mont_sqr_priv_unroll_only_512_body(out, a, n, base, np0);
}

__kernel void cgbn_mont_mul_local_only_512(__global const uint *a, __global const uint *b,
                                            __constant uint *n, __global uint *out,
                                            __constant uint *np0_ptr, uint limbs,
                                            __local uint *local_mem) {
    if (limbs != MONT_OPT2_FIXED_LIMBS) {
        return;
    }
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint lsize = get_local_size(0);
    uint base = gid * limbs;
    uint np0 = np0_ptr[0];
    mont_mul_priv_local_only_512_body(out, a, b, n, base, np0, local_mem, lid, lsize);
}

__kernel void cgbn_mont_sqr_local_only_512(__global const uint *a, __constant uint *n,
                                            __global uint *out, __constant uint *np0_ptr,
                                            uint limbs, __local uint *local_mem) {
    if (limbs != MONT_OPT2_FIXED_LIMBS) {
        return;
    }
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint lsize = get_local_size(0);
    uint base = gid * limbs;
    uint np0 = np0_ptr[0];
    mont_sqr_priv_local_only_512_body(out, a, n, base, np0, local_mem, lid, lsize);
}

__kernel void cgbn_mont_mul_local_only_4096(__global const uint *a, __global const uint *b,
                                             __constant uint *n, __global uint *out,
                                             __constant uint *np0_ptr, uint limbs,
                                             __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS) {
        return;
    }
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint lsize = get_local_size(0);
    uint base = gid * limbs;
    uint np0 = np0_ptr[0];
    mont_mul_priv_local_only_4096_body(out, a, b, n, base, np0, local_mem, lid, lsize);
}

__kernel void cgbn_mont_sqr_local_only_4096(__global const uint *a, __constant uint *n,
                                             __global uint *out, __constant uint *np0_ptr,
                                             uint limbs, __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS) {
        return;
    }
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint lsize = get_local_size(0);
    uint base = gid * limbs;
    uint np0 = np0_ptr[0];
    mont_sqr_priv_local_only_4096_body(out, a, n, base, np0, local_mem, lid, lsize);
}

// removed: unroll2/4/8 raw kernels (replaced by unroll32/64)

// Generic unroll32/unroll64 paths for multi-bit benchmarking.
static inline void mont_mul_priv_unroll32_body(
    __global uint *out,
    __global const uint *a,
    __global const uint *b,
    __constant uint *n,
    uint base,
    uint np0,
    uint limbs)
{
    if (limbs == 0u || limbs > MAX_LIMBS) return;
    uint t[MAX_LIMBS + 2u];
    for (uint i = 0u; i < limbs + 2u; ++i) t[i] = 0u;
    uint B[MAX_LIMBS];
    for (uint j = 0u; j < limbs; ++j) B[j] = b[base + j];
    for (uint i = 0u; i < limbs; ++i) {
        uint ai = a[base + i];
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
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)n[j] + carry;
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
        ulong tv = (ulong)t[i], nv = (ulong)n[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }
    uint need_sub = (t[limbs] != 0u || t[limbs + 1u] != 0u) ? 1u : 0u;
    need_sub = (borrow == 0u) ? 1u : need_sub;
    uint mask = 0u - need_sub;
    for (uint i = 0u; i < limbs; ++i) out[base + i] = (D[i] & mask) | (t[i] & ~mask);
}

static inline void mont_sqr_priv_unroll32_body(
    __global uint *out,
    __global const uint *a,
    __constant uint *n,
    uint base,
    uint np0,
    uint limbs)
{
    mont_mul_priv_unroll32_body(out, a, a, n, base, np0, limbs);
}

static inline void mont_mul_priv_unroll64_body(
    __global uint *out,
    __global const uint *a,
    __global const uint *b,
    __constant uint *n,
    uint base,
    uint np0,
    uint limbs)
{
    if (limbs == 0u || limbs > MAX_LIMBS) return;
    uint t[MAX_LIMBS + 2u];
    for (uint i = 0u; i < limbs + 2u; ++i) t[i] = 0u;
    uint B[MAX_LIMBS];
    for (uint j = 0u; j < limbs; ++j) B[j] = b[base + j];
    for (uint i = 0u; i < limbs; ++i) {
        uint ai = a[base + i];
        ulong carry = 0ul;
        #pragma unroll 64
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
        #pragma unroll 64
        for (uint j = 0u; j < limbs; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)n[j] + carry;
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
        ulong tv = (ulong)t[i], nv = (ulong)n[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }
    uint need_sub = (t[limbs] != 0u || t[limbs + 1u] != 0u) ? 1u : 0u;
    need_sub = (borrow == 0u) ? 1u : need_sub;
    uint mask = 0u - need_sub;
    for (uint i = 0u; i < limbs; ++i) out[base + i] = (D[i] & mask) | (t[i] & ~mask);
}

static inline void mont_sqr_priv_unroll64_body(
    __global uint *out,
    __global const uint *a,
    __constant uint *n,
    uint base,
    uint np0,
    uint limbs)
{
    mont_mul_priv_unroll64_body(out, a, a, n, base, np0, limbs);
}

__kernel void cgbn_mont_mul_unroll32(__global const uint *a, __global const uint *b,
                                      __constant uint *n, __global uint *out,
                                      __constant uint *np0_ptr, uint limbs) {
    uint gid = get_global_id(0), base = gid * limbs, np0 = np0_ptr[0];
    mont_mul_priv_unroll32_body(out, a, b, n, base, np0, limbs);
}

__kernel void cgbn_mont_sqr_unroll32(__global const uint *a, __constant uint *n,
                                      __global uint *out, __constant uint *np0_ptr, uint limbs) {
    uint gid = get_global_id(0), base = gid * limbs, np0 = np0_ptr[0];
    mont_sqr_priv_unroll32_body(out, a, n, base, np0, limbs);
}

__kernel void cgbn_mont_mul_unroll64(__global const uint *a, __global const uint *b,
                                      __constant uint *n, __global uint *out,
                                      __constant uint *np0_ptr, uint limbs) {
    uint gid = get_global_id(0), base = gid * limbs, np0 = np0_ptr[0];
    mont_mul_priv_unroll64_body(out, a, b, n, base, np0, limbs);
}

__kernel void cgbn_mont_sqr_unroll64(__global const uint *a, __constant uint *n,
                                      __global uint *out, __constant uint *np0_ptr, uint limbs) {
    uint gid = get_global_id(0), base = gid * limbs, np0 = np0_ptr[0];
    mont_sqr_priv_unroll64_body(out, a, n, base, np0, limbs);
}

// Dedicated fixed-4096 kernels (for ISA/resource comparison vs generic unroll64).
__kernel void cgbn_mont_mul_unroll64_4096(__global const uint *a, __global const uint *b,
                                           __constant uint *n, __global uint *out,
                                           __constant uint *np0_ptr, uint limbs) {
    if (limbs != MONT_FIXED_4096_LIMBS) return;
    uint gid = get_global_id(0), base = gid * limbs, np0 = np0_ptr[0];
    mont_mul_priv_unroll64_4096_body(out, a, b, n, base, np0);
}

__kernel void cgbn_mont_sqr_unroll64_4096(__global const uint *a, __constant uint *n,
                                           __global uint *out, __constant uint *np0_ptr, uint limbs) {
    if (limbs != MONT_FIXED_4096_LIMBS) return;
    uint gid = get_global_id(0), base = gid * limbs, np0 = np0_ptr[0];
    mont_sqr_priv_unroll64_4096_body(out, a, n, base, np0);
}

__kernel void cgbn_mont_mul_unroll64_4096_nod(__global const uint *a, __global const uint *b,
                                               __constant uint *n, __global uint *out,
                                               __constant uint *np0_ptr, uint limbs) {
    if (limbs != MONT_FIXED_4096_LIMBS) return;
    uint gid = get_global_id(0), base = gid * limbs, np0 = np0_ptr[0];
    mont_mul_priv_unroll64_4096_nod_body(out, a, b, n, base, np0);
}

__kernel void cgbn_mont_sqr_unroll64_4096_nod(__global const uint *a, __constant uint *n,
                                               __global uint *out, __constant uint *np0_ptr, uint limbs) {
    if (limbs != MONT_FIXED_4096_LIMBS) return;
    uint gid = get_global_id(0), base = gid * limbs, np0 = np0_ptr[0];
    mont_sqr_priv_unroll64_4096_nod_body(out, a, n, base, np0);
}

__kernel void cgbn_mont_mul_unroll64_4096_mt2(__global const uint *a, __global const uint *b,
                                               __constant uint *n, __global uint *out,
                                               __constant uint *np0_ptr, uint limbs,
                                               __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS || get_local_size(0) != 2u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0];
    uint lid = get_local_id(0);
    mont_mul_priv_unroll64_4096_mt2_body(out, a, b, n, base, np0, local_mem, lid);
}

__kernel void cgbn_mont_sqr_unroll64_4096_mt2(__global const uint *a, __constant uint *n,
                                               __global uint *out, __constant uint *np0_ptr,
                                               uint limbs, __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS || get_local_size(0) != 2u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0];
    uint lid = get_local_id(0);
    mont_sqr_priv_unroll64_4096_mt2_body(out, a, n, base, np0, local_mem, lid);
}

__kernel void cgbn_mont_mul_unroll64_4096_mt2_weak(__global const uint *a, __global const uint *b,
                                                    __constant uint *n, __global uint *out,
                                                    __constant uint *np0_ptr, uint limbs,
                                                    __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS || get_local_size(0) != 2u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0];
    uint lid = get_local_id(0);
    mont_mul_priv_unroll64_4096_mt2_weak_body(out, a, b, n, base, np0, local_mem, lid);
}

__kernel void cgbn_mont_sqr_unroll64_4096_mt2_weak(__global const uint *a, __constant uint *n,
                                                    __global uint *out, __constant uint *np0_ptr,
                                                    uint limbs, __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS || get_local_size(0) != 2u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0];
    uint lid = get_local_id(0);
    mont_sqr_priv_unroll64_4096_mt2_weak_body(out, a, n, base, np0, local_mem, lid);
}

__kernel void cgbn_mont_mul_unroll64_4096_mt4(__global const uint *a, __global const uint *b,
                                              __constant uint *n, __global uint *out,
                                              __constant uint *np0_ptr, uint limbs,
                                              __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS || get_local_size(0) != 4u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0];
    uint lid = get_local_id(0);
    mont_mul_priv_unroll64_4096_mt4_body(out, a, b, n, base, np0, local_mem, lid);
}

__kernel void cgbn_mont_sqr_unroll64_4096_mt4(__global const uint *a, __constant uint *n,
                                               __global uint *out, __constant uint *np0_ptr,
                                               uint limbs, __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS || get_local_size(0) != 4u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0];
    uint lid = get_local_id(0);
    mont_sqr_priv_unroll64_4096_mt4_body(out, a, n, base, np0, local_mem, lid);
}

__kernel void cgbn_mont_mul_unroll64_4096_mt8(__global const uint *a, __global const uint *b,
                                              __constant uint *n, __global uint *out,
                                              __constant uint *np0_ptr, uint limbs,
                                              __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS || get_local_size(0) != 8u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0];
    uint lid = get_local_id(0);
    mont_mul_priv_unroll64_4096_mt8_body(out, a, b, n, base, np0, local_mem, lid);
}

__kernel void cgbn_mont_sqr_unroll64_4096_mt8(__global const uint *a, __constant uint *n,
                                               __global uint *out, __constant uint *np0_ptr,
                                               uint limbs, __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS || get_local_size(0) != 8u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0];
    uint lid = get_local_id(0);
    mont_sqr_priv_unroll64_4096_mt8_body(out, a, n, base, np0, local_mem, lid);
}
