// ECM stage1 ladder — operator-free; implementations bound via ECM_STAGE1_*_IMPL macros.

// ---------------------------------------------------------------------------
// Single-limb Montgomery multiplication helpers (R = 2^32).
// mul_ui32_limbs / shift_right_32_limbs are shared with ecm_stage1_coop.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// special_mult_ui32_generic — fallback single-limb Montgomery multiply.
// Force noinline: isolate register frame from double_add_v2.
// CIOS pattern — single outer iteration because m is one 32-bit limb.
// ---------------------------------------------------------------------------
#ifdef __GNUC__
__attribute__((noinline))
#endif
static void special_mult_ui32_generic(uint *r, uint m, const uint *N, uint np0, uint limbs) {
    uint t[MAX_LIMBS + 1u];
    for (uint i = 0u; i <= limbs; ++i) t[i] = 0u;

    // 1) product: t = r * m
    {
        ulong carry = 0ul;
        for (uint j = 0u; j < limbs; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)r[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        t[limbs] = (uint)carry;
    }

    // 2) reduction: t = (t + q*N) >> 32  where q = t[0]*np0
    {
        uint mp = (uint)((ulong)t[0] * (ulong)np0);
        ulong carry = 0ul;
        for (uint j = 0u; j < limbs; ++j) {
            ulong uv = (ulong)t[j] + (ulong)mp * (ulong)N[j] + carry;
            if (j > 0u) t[j - 1u] = (uint)uv;
            carry = uv >> 32;
        }
        ulong top = (ulong)t[limbs] + carry;
        t[limbs - 1u] = (uint)top;
        t[limbs] = (uint)(top >> 32);
    }

    // 3) branchless conditional subtraction (same pattern as mont_mul_unroll_512b)
    {
        ulong borrow = 0ul;
        uint D[MAX_LIMBS];
        for (uint i = 0u; i < limbs; ++i) {
            ulong tv = (ulong)t[i];
            ulong nv = (ulong)N[i];
            ulong w = tv - nv - borrow;
            D[i] = (uint)w;
            borrow = (tv < nv + borrow) ? 1ul : 0ul;
        }
        uint need_sub = (t[limbs] != 0u) ? 1u : (borrow == 0ul ? 1u : 0u);
        uint mask = 0u - need_sub;
        for (uint i = 0u; i < limbs; ++i) {
            r[i] = (D[i] & mask) | (t[i] & ~mask);
        }
    }
}

#if MAX_LIMBS <= 16
// Unrolled 512b variant — host injects special_mult_unroll_512b.cl before this file.
static inline void special_mult_ui32(uint *r, uint m, const uint *N, uint np0, uint limbs) {
    special_mult_ui32_unroll_512b(r, m, N, np0, limbs);
}
#else
static void special_mult_ui32(uint *r, uint m, const uint *N, uint np0, uint limbs) {
    special_mult_ui32_generic(r, m, N, np0, limbs);
}
#endif

static inline void special_mult_stage1(uint *r, uint m, const uint *N, uint np0, uint limbs) {
    special_mult_ui32(r, m, N, np0, limbs);
}

static inline void double_add_v2(uint *q, uint *u, uint *w, uint *v, uint d, const uint *N,
                                 uint np0, uint limbs) {
    uint t[MAX_LIMBS], CB[MAX_LIMBS], DA[MAX_LIMBS], AA[MAX_LIMBS], BB[MAX_LIMBS];
    uint K[MAX_LIMBS], dK[MAX_LIMBS];

    add_mod(t, v, w, N, limbs);
    (void)sub_mod(v, v, w, N, limbs);

    add_mod(w, u, q, N, limbs);
    (void)sub_mod(u, u, q, N, limbs);

    mont_mul(CB, t, u, N, np0, limbs);
    maybe_mont_normalize(CB, N, limbs);
    mont_mul(DA, v, w, N, np0, limbs);
    maybe_mont_normalize(DA, N, limbs);

    mont_sqr(AA, w, N, np0, limbs);
    mont_sqr(BB, u, N, np0, limbs);
    maybe_mont_normalize(AA, N, limbs);
    maybe_mont_normalize(BB, N, limbs);

    mont_mul(q, AA, BB, N, np0, limbs);
    maybe_mont_normalize(q, N, limbs);

    (void)sub_mod(K, AA, BB, N, limbs);

    mp_copy(dK, K, limbs);
    special_mult_stage1(dK, d, N, np0, limbs);

    add_mod(u, BB, dK, N, limbs);
    mont_mul(u, K, u, N, np0, limbs);
    maybe_mont_normalize(u, N, limbs);

    add_mod(w, DA, CB, N, limbs);
    (void)sub_mod(v, DA, CB, N, limbs);

    mont_sqr(w, w, N, np0, limbs);
    maybe_mont_normalize(w, N, limbs);
    mont_sqr(v, v, N, np0, limbs);
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

static inline void run_double_add_instance(uint instance_i, __global const uint *s_bits,
                                           ulong s_num_bits, ulong s_bits_start,
                                           ulong s_bits_interval, __global uint *data,
                                           uint sigma_0, uint np0, uint limbs) {
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

#ifndef ECM_STAGE1_USE_COOP_WG
#define ECM_STAGE1_USE_COOP_WG 0
#endif

#if !ECM_STAGE1_USE_COOP_WG
__kernel void kernel_double_add(__global const uint *s_bits, ulong s_num_bits, ulong s_bits_start,
                                ulong s_bits_interval, __global uint *data, uint count,
                                uint sigma_0, uint np0, uint limbs) {
    uint instance_i = get_global_id(0);
    if (instance_i >= count) {
        return;
    }
    if (limbs == 0u || limbs > MAX_LIMBS) {
        return;
    }
    run_double_add_instance(instance_i, s_bits, s_num_bits, s_bits_start, s_bits_interval, data,
                            sigma_0, np0, limbs);
}
#endif
