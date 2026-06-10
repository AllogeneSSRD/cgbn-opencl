// Montgomery mul (CIOS) with 24-bit limbs — Adreno mul24/mad24 path.
// Compile with -DMAX_LIMBS=<ceil(bits/24)> and -DMP_LIMB_BITS=24.
//
// Bodies for A/B bench:
//   mont_mul_unroll_i24_body           — Level 1: ulong CIOS + private B/N
//   mont_mul_unroll_i24_u32_body       — Level 2: 32-bit CIOS MAC + private B/N
//   mont_mul_unroll_i24_nocopy_body    — Level 3: ulong CIOS, b/n from global/__constant
//   mont_mul_unroll_i24_u32_nocopy_body — Level 2+3: u32 MAC, no private B/N

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
#define MONT_I24_LIMB_MASK 0xFFFFFFu

// --- Level 1: mad24 24×24→48 (returns packed ulong) ---

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

// --- Level 2: same mul semantics, split (lo24, hi24) for 32-bit CIOS MAC ---

static inline uint2 mont_i24_mul_full_split(uint a, uint b) {
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
    return (uint2)(lo48, hi48);
}

// t[j] += ai*bj + carry; returns new carry (fits uint32 for MONT_I24_LIMBS <= ~128).
static inline uint mont_i24_cios_mac_u32(uint *t_j, uint ai, uint bj, uint carry) {
    const uint2 prod = mont_i24_mul_full_split(ai, bj);
    const uint lo_sum = *t_j + prod.x + (carry & MONT_I24_LIMB_MASK);
    *t_j = lo_sum & MONT_I24_LIMB_MASK;
    const uint hi_carry = (lo_sum >> MONT_I24_RADIX_BITS) + (carry >> MONT_I24_RADIX_BITS);
    return prod.y + hi_carry;
}

// REDC inner: returns limb_val for caller to shift into t[j-1] when j > 0.
static inline uint mont_i24_cios_mac_shift_u32(uint tj, uint mi, uint nj, uint carry, uint *limb_out) {
    const uint2 prod = mont_i24_mul_full_split(mi, nj);
    const uint lo_sum = tj + prod.x + (carry & MONT_I24_LIMB_MASK);
    *limb_out = lo_sum & MONT_I24_LIMB_MASK;
    const uint hi_carry = (lo_sum >> MONT_I24_RADIX_BITS) + (carry >> MONT_I24_RADIX_BITS);
    return prod.y + hi_carry;
}

// --- Level 1 body: ulong CIOS (baseline for bench) ---

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
            t[j] = (uint)(uv & MONT_I24_LIMB_MASK);
            carry = uv >> MONT_I24_RADIX_BITS;
        }
        ulong top = (ulong)t[MONT_I24_LIMBS] + carry;
        t[MONT_I24_LIMBS] = (uint)(top & MONT_I24_LIMB_MASK);
        t[MONT_I24_LIMBS + 1u] = (uint)(top >> MONT_I24_RADIX_BITS);

        const uint m = mul24(t[0], np0);
        carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < MONT_I24_LIMBS; ++j) {
            const ulong uv = mont_i24_add3((ulong)t[j], mont_i24_mul_full(m, N[j]), carry);
            if (j > 0u) {
                t[j - 1u] = (uint)(uv & MONT_I24_LIMB_MASK);
            }
            carry = uv >> MONT_I24_RADIX_BITS;
        }
        top = (ulong)t[MONT_I24_LIMBS] + carry;
        t[MONT_I24_LIMBS - 1u] = (uint)(top & MONT_I24_LIMB_MASK);
        top = (ulong)t[MONT_I24_LIMBS + 1u] + (top >> MONT_I24_RADIX_BITS);
        t[MONT_I24_LIMBS] = (uint)(top & MONT_I24_LIMB_MASK);
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

// --- Level 2 body: 32-bit CIOS MAC (no ulong in inner loops) ---

static inline void mont_mul_unroll_i24_u32_body(
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
        uint carry = 0u;
        #pragma unroll
        for (uint j = 0u; j < MONT_I24_LIMBS; ++j) {
            carry = mont_i24_cios_mac_u32(&t[j], ai, B[j], carry);
        }
        ulong top = (ulong)t[MONT_I24_LIMBS] + (ulong)carry;
        t[MONT_I24_LIMBS] = (uint)(top & MONT_I24_LIMB_MASK);
        t[MONT_I24_LIMBS + 1u] = (uint)(top >> MONT_I24_RADIX_BITS);

        const uint m = mul24(t[0], np0);
        carry = 0u;
        #pragma unroll
        for (uint j = 0u; j < MONT_I24_LIMBS; ++j) {
            uint limb_val = 0u;
            carry = mont_i24_cios_mac_shift_u32(t[j], m, N[j], carry, &limb_val);
            if (j > 0u) {
                t[j - 1u] = limb_val;
            }
        }
        top = (ulong)t[MONT_I24_LIMBS] + (ulong)carry;
        t[MONT_I24_LIMBS - 1u] = (uint)(top & MONT_I24_LIMB_MASK);
        top = (ulong)t[MONT_I24_LIMBS + 1u] + (top >> MONT_I24_RADIX_BITS);
        t[MONT_I24_LIMBS] = (uint)(top & MONT_I24_LIMB_MASK);
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

// --- Level 3: no private B/N (VGPR); stream b[base+j] and n[j] from cache ---

static inline void mont_mul_unroll_i24_nocopy_body(
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

    #pragma unroll
    for (uint i = 0u; i < MONT_I24_LIMBS; ++i) {
        const uint ai = a[base + i];
        ulong carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < MONT_I24_LIMBS; ++j) {
            const ulong uv =
                mont_i24_add3((ulong)t[j], mont_i24_mul_full(ai, b[base + j]), carry);
            t[j] = (uint)(uv & MONT_I24_LIMB_MASK);
            carry = uv >> MONT_I24_RADIX_BITS;
        }
        ulong top = (ulong)t[MONT_I24_LIMBS] + carry;
        t[MONT_I24_LIMBS] = (uint)(top & MONT_I24_LIMB_MASK);
        t[MONT_I24_LIMBS + 1u] = (uint)(top >> MONT_I24_RADIX_BITS);

        const uint m = mul24(t[0], np0);
        carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < MONT_I24_LIMBS; ++j) {
            const ulong uv = mont_i24_add3((ulong)t[j], mont_i24_mul_full(m, n[j]), carry);
            if (j > 0u) {
                t[j - 1u] = (uint)(uv & MONT_I24_LIMB_MASK);
            }
            carry = uv >> MONT_I24_RADIX_BITS;
        }
        top = (ulong)t[MONT_I24_LIMBS] + carry;
        t[MONT_I24_LIMBS - 1u] = (uint)(top & MONT_I24_LIMB_MASK);
        top = (ulong)t[MONT_I24_LIMBS + 1u] + (top >> MONT_I24_RADIX_BITS);
        t[MONT_I24_LIMBS] = (uint)(top & MONT_I24_LIMB_MASK);
        t[MONT_I24_LIMBS + 1u] = (uint)(top >> MONT_I24_RADIX_BITS);
    }

    ulong borrow = 0ul;
    uint D[MONT_I24_LIMBS];
    #pragma unroll
    for (uint i = 0u; i < MONT_I24_LIMBS; ++i) {
        const ulong tv = (ulong)t[i];
        const ulong nv = (ulong)n[i];
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

static inline void mont_mul_unroll_i24_u32_nocopy_body(
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

    #pragma unroll
    for (uint i = 0u; i < MONT_I24_LIMBS; ++i) {
        const uint ai = a[base + i];
        uint carry = 0u;
        #pragma unroll
        for (uint j = 0u; j < MONT_I24_LIMBS; ++j) {
            carry = mont_i24_cios_mac_u32(&t[j], ai, b[base + j], carry);
        }
        ulong top = (ulong)t[MONT_I24_LIMBS] + (ulong)carry;
        t[MONT_I24_LIMBS] = (uint)(top & MONT_I24_LIMB_MASK);
        t[MONT_I24_LIMBS + 1u] = (uint)(top >> MONT_I24_RADIX_BITS);

        const uint m = mul24(t[0], np0);
        carry = 0u;
        #pragma unroll
        for (uint j = 0u; j < MONT_I24_LIMBS; ++j) {
            uint limb_val = 0u;
            carry = mont_i24_cios_mac_shift_u32(t[j], m, n[j], carry, &limb_val);
            if (j > 0u) {
                t[j - 1u] = limb_val;
            }
        }
        top = (ulong)t[MONT_I24_LIMBS] + (ulong)carry;
        t[MONT_I24_LIMBS - 1u] = (uint)(top & MONT_I24_LIMB_MASK);
        top = (ulong)t[MONT_I24_LIMBS + 1u] + (top >> MONT_I24_RADIX_BITS);
        t[MONT_I24_LIMBS] = (uint)(top & MONT_I24_LIMB_MASK);
        t[MONT_I24_LIMBS + 1u] = (uint)(top >> MONT_I24_RADIX_BITS);
    }

    ulong borrow = 0ul;
    uint D[MONT_I24_LIMBS];
    #pragma unroll
    for (uint i = 0u; i < MONT_I24_LIMBS; ++i) {
        const ulong tv = (ulong)t[i];
        const ulong nv = (ulong)n[i];
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

static inline void mont_sqr_unroll_i24_u32_body(
    __global uint *out,
    __global const uint *a,
    __constant uint *n,
    uint base,
    uint np0) {
    mont_mul_unroll_i24_u32_body(out, a, a, n, base, np0);
}

static inline void mont_sqr_unroll_i24_nocopy_body(
    __global uint *out,
    __global const uint *a,
    __constant uint *n,
    uint base,
    uint np0) {
    mont_mul_unroll_i24_nocopy_body(out, a, a, n, base, np0);
}

static inline void mont_sqr_unroll_i24_u32_nocopy_body(
    __global uint *out,
    __global const uint *a,
    __constant uint *n,
    uint base,
    uint np0) {
    mont_mul_unroll_i24_u32_nocopy_body(out, a, a, n, base, np0);
}
