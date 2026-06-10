// Montgomery mul (CIOS) with 24-bit limbs — Adreno mul24/mad24 path.
// Compile with -DMAX_LIMBS=<ceil(bits/24)> and -DMP_LIMB_BITS=24.

#pragma once

#ifndef MP_LIMB_BITS
#define MP_LIMB_BITS 32
#endif

#ifndef MAX_LIMBS
#define MAX_LIMBS 22
#endif

#if MP_LIMB_BITS != 24
#error "mont_mul_unroll_i24.cl requires -DMP_LIMB_BITS=24"
#endif

#define MONT_I24_RADIX_BITS 24u
#define MONT_I24_LIMBS MAX_LIMBS

static inline ulong mont_i24_mul_full(uint a, uint b) {
    const uint mask12 = 0xFFFu;
    const uint a0 = a & mask12;
    const uint a1 = a >> 12;
    const uint b0 = b & mask12;
    const uint b1 = b >> 12;

    const uint p00 = mul24(a0, b0);
    const uint mid1 = mad24(a0, b1, p00 >> 12);
    const uint mid2 = mad24(a1, b0, mid1);
    const uint lo48 = (p00 & mask12) | ((mid2 & mask12) << 12);
    const uint hi48 = mad24(a1, b1, mid2 >> 12);
    return ((ulong)hi48 << 24) | lo48;
}

static inline ulong mont_i24_add3(ulong x, ulong y, ulong z) {
    return x + y + z;
}

static inline void mont_mul_unroll_i24_body(
    __global uint *out,
    __global const uint *a,
    __global const uint *b,
    __constant uint *n,
    uint base,
    uint np0) {
    uint t[MONT_I24_LIMBS + 2u];
    #pragma unroll
    for (uint i = 0u; i < MONT_I24_LIMBS + 2u; ++i) {
        t[i] = 0u;
    }

    uint B[MONT_I24_LIMBS];
    uint N[MONT_I24_LIMBS];
    #pragma unroll
    for (uint j = 0u; j < MONT_I24_LIMBS; ++j) {
        B[j] = b[base + j];
        N[j] = n[j];
    }

    #pragma unroll
    for (uint i = 0u; i < MONT_I24_LIMBS; ++i) {
        const uint ai = a[base + i];
        ulong carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < MONT_I24_LIMBS; ++j) {
            const ulong uv = mont_i24_add3((ulong)t[j], mont_i24_mul_full(ai, B[j]), carry);
            t[j] = (uint)uv;
            carry = uv >> MONT_I24_RADIX_BITS;
        }
        ulong top = (ulong)t[MONT_I24_LIMBS] + carry;
        t[MONT_I24_LIMBS] = (uint)top;
        t[MONT_I24_LIMBS + 1u] = (uint)(top >> MONT_I24_RADIX_BITS);

        const uint m = mul24(t[0], np0);
        carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < MONT_I24_LIMBS; ++j) {
            const ulong uv = mont_i24_add3((ulong)t[j], mont_i24_mul_full(m, N[j]), carry);
            if (j > 0u) {
                t[j - 1u] = (uint)uv;
            }
            carry = uv >> MONT_I24_RADIX_BITS;
        }
        top = (ulong)t[MONT_I24_LIMBS] + carry;
        t[MONT_I24_LIMBS - 1u] = (uint)top;
        top = (ulong)t[MONT_I24_LIMBS + 1u] + (top >> MONT_I24_RADIX_BITS);
        t[MONT_I24_LIMBS] = (uint)top;
        t[MONT_I24_LIMBS + 1u] = (uint)(top >> MONT_I24_RADIX_BITS);
    }

    ulong borrow = 0ul;
    uint D[MONT_I24_LIMBS];
    #pragma unroll
    for (uint i = 0u; i < MONT_I24_LIMBS; ++i) {
        const ulong tv = (ulong)t[i];
        const ulong nv = (ulong)N[i];
        const ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }

    const uint any_high = (t[MONT_I24_LIMBS] | t[MONT_I24_LIMBS + 1u]) != 0u;
    const uint no_borrow = (borrow == 0u);
    const uint need_sub = any_high | no_borrow;
    const uint mask = 0u - need_sub;

    #pragma unroll
    for (uint i = 0u; i < MONT_I24_LIMBS; ++i) {
        out[base + i] = (D[i] & mask) | (t[i] & ~mask);
    }
}

static inline void mont_sqr_unroll_i24_body(
    __global uint *out,
    __global const uint *a,
    __constant uint *n,
    uint base,
    uint np0) {
    mont_mul_unroll_i24_body(out, a, a, n, base, np0);
}
