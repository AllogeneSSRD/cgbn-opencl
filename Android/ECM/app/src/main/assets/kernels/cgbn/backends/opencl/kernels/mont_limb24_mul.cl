// Montgomery mul (CIOS) with 24-bit limbs in uint32 — Adreno mul24/mad24 path.
// Fixed unroll for 512-bit moduli: 22 limbs × 24 bits = 528-bit capacity.

#pragma once

#ifndef MP_LIMB_BITS
#define MP_LIMB_BITS 32
#endif

#ifndef MAX_LIMBS
#define MAX_LIMBS 22
#endif

#if MP_LIMB_BITS != 24
#error "mont_limb24_mul.cl requires -DMP_LIMB_BITS=24"
#endif

#define MONT_LIMB24_RADIX_BITS 24u
#define MONT_LIMB24_MASK 0xFFFFFFu
// unroll_only_512 @ limb24: 512-bit mod => 22 limbs; 504-bit => 21 limbs, etc.
#define MONT_LIMB24_UNROLL512_LIMBS MAX_LIMBS

// 24×24 -> 48-bit product via 12-bit splits (maps to mul24 on Adreno).
static inline ulong u24_mul_full(uint a, uint b) {
    const uint mask12 = 0xFFFu;
    const uint a0 = a & mask12;
    const uint a1 = a >> 12;
    const uint b0 = b & mask12;
    const uint b1 = b >> 12;

    const uint p00 = mul24(a0, b0);
    const uint p01 = mul24(a0, b1);
    const uint p10 = mul24(a1, b0);
    const uint p11 = mul24(a1, b1);

    ulong mid = (ulong)p01 + (ulong)p10 + (ulong)(p00 >> 12);
    const ulong lo48 = ((ulong)(p00 & mask12)) | ((mid & mask12) << 12);
    const ulong hi48 = (ulong)p11 + (mid >> 12);
    return (hi48 << 24) | lo48;
}

static inline ulong u24_add3(ulong x, ulong y, ulong z) {
    return x + y + z;
}

static inline void mont_mul_priv_unroll_only_512_limb24_body(
    __global uint *out,
    __global const uint *a,
    __global const uint *b,
    __constant uint *n,
    uint base,
    uint np0) {
    uint t[MONT_LIMB24_UNROLL512_LIMBS + 2u];
    #pragma unroll
    for (uint i = 0u; i < MONT_LIMB24_UNROLL512_LIMBS + 2u; ++i) {
        t[i] = 0u;
    }

    uint B[MONT_LIMB24_UNROLL512_LIMBS];
    uint N[MONT_LIMB24_UNROLL512_LIMBS];
    #pragma unroll
    for (uint j = 0u; j < MONT_LIMB24_UNROLL512_LIMBS; ++j) {
        B[j] = b[base + j];
        N[j] = n[j];
    }

    #pragma unroll
    for (uint i = 0u; i < MONT_LIMB24_UNROLL512_LIMBS; ++i) {
        const uint ai = a[base + i];
        ulong carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < MONT_LIMB24_UNROLL512_LIMBS; ++j) {
            const ulong uv = u24_add3((ulong)t[j], u24_mul_full(ai, B[j]), carry);
            t[j] = (uint)uv;
            carry = uv >> MONT_LIMB24_RADIX_BITS;
        }
        ulong top = (ulong)t[MONT_LIMB24_UNROLL512_LIMBS] + carry;
        t[MONT_LIMB24_UNROLL512_LIMBS] = (uint)top;
        t[MONT_LIMB24_UNROLL512_LIMBS + 1u] = (uint)(top >> MONT_LIMB24_RADIX_BITS);

        const uint m = mul24(t[0], np0);
        carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < MONT_LIMB24_UNROLL512_LIMBS; ++j) {
            const ulong uv = u24_add3((ulong)t[j], u24_mul_full(m, N[j]), carry);
            if (j > 0u) {
                t[j - 1u] = (uint)uv;
            }
            carry = uv >> MONT_LIMB24_RADIX_BITS;
        }
        top = (ulong)t[MONT_LIMB24_UNROLL512_LIMBS] + carry;
        t[MONT_LIMB24_UNROLL512_LIMBS - 1u] = (uint)top;
        top = (ulong)t[MONT_LIMB24_UNROLL512_LIMBS + 1u] + (top >> MONT_LIMB24_RADIX_BITS);
        t[MONT_LIMB24_UNROLL512_LIMBS] = (uint)top;
        t[MONT_LIMB24_UNROLL512_LIMBS + 1u] = (uint)(top >> MONT_LIMB24_RADIX_BITS);
    }

    ulong borrow = 0ul;
    uint D[MONT_LIMB24_UNROLL512_LIMBS];
    #pragma unroll
    for (uint i = 0u; i < MONT_LIMB24_UNROLL512_LIMBS; ++i) {
        const ulong tv = (ulong)t[i];
        const ulong nv = (ulong)N[i];
        const ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }

    const uint any_high =
        (t[MONT_LIMB24_UNROLL512_LIMBS] | t[MONT_LIMB24_UNROLL512_LIMBS + 1u]) != 0u;
    const uint no_borrow = (borrow == 0u);
    const uint need_sub = any_high | no_borrow;
    const uint mask = 0u - need_sub;

    #pragma unroll
    for (uint i = 0u; i < MONT_LIMB24_UNROLL512_LIMBS; ++i) {
        out[base + i] = (D[i] & mask) | (t[i] & ~mask);
    }
}

static inline void mont_sqr_priv_unroll_only_512_limb24_body(
    __global uint *out,
    __global const uint *a,
    __constant uint *n,
    uint base,
    uint np0) {
    mont_mul_priv_unroll_only_512_limb24_body(out, a, a, n, base, np0);
}
