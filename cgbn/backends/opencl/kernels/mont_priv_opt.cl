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

// A/B split helpers for 512-bit analysis:
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

    uint need_sub = (t[MONT_OPT2_FIXED_LIMBS] != 0u || t[MONT_OPT2_FIXED_LIMBS + 1u] != 0u) ? 1u : 0u;
    need_sub = (borrow == 0u) ? 1u : need_sub;
    uint mask = 0u - need_sub;
    #pragma unroll
    for (uint i = 0u; i < MONT_OPT2_FIXED_LIMBS; ++i) {
        out[base + i] = (D[i] & mask) | (t[i] & ~mask);
    }
}

static inline void mont_sqr_priv_unroll_only_512_body(
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
    uint t[MONT_FIXED_4096_LIMBS + 2u];
    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS + 2u; ++i) {
        t[i] = 0u;
    }
    uint B[MONT_FIXED_4096_LIMBS];
    for (uint j = 0u; j < MONT_FIXED_4096_LIMBS; ++j) {
        B[j] = a[base + j];
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
        carry = 0ul;
        #pragma unroll 64
        for (uint j = 0u; j < MONT_FIXED_4096_LIMBS; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)n[j] + carry;
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

// Fixed 4096-bit local=2 dual-lane independent path:
// each lane handles one instance (no inter-lane carry dependency).
__kernel void cgbn_mont_mul_unroll64_4096_l2(__global const uint *a, __global const uint *b,
                                              __constant uint *n, __global uint *out,
                                              __constant uint *np0_ptr, uint limbs,
                                              uint total_instances) {
    if (limbs != MONT_FIXED_4096_LIMBS || get_local_size(0) != 2u) return;
    uint gid2 = get_group_id(0) * 2u + get_local_id(0);
    if (gid2 >= total_instances) return;
    uint base = gid2 * limbs;
    uint np0 = np0_ptr[0];
    mont_mul_priv_unroll64_4096_body(out, a, b, n, base, np0);
}

__kernel void cgbn_mont_sqr_unroll64_4096_l2(__global const uint *a, __constant uint *n,
                                              __global uint *out, __constant uint *np0_ptr,
                                              uint limbs, uint total_instances) {
    if (limbs != MONT_FIXED_4096_LIMBS || get_local_size(0) != 2u) return;
    uint gid2 = get_group_id(0) * 2u + get_local_id(0);
    if (gid2 >= total_instances) return;
    uint base = gid2 * limbs;
    uint np0 = np0_ptr[0];
    mont_sqr_priv_unroll64_4096_body(out, a, n, base, np0);
}
