// Optimized private Montgomery mul: N/np0 in __constant, B cached, speculative final subtract.

#pragma once

#ifndef MAX_LIMBS
#define MAX_LIMBS 128
#endif

#define MONT_OPT2_FIXED_LIMBS 16u

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
