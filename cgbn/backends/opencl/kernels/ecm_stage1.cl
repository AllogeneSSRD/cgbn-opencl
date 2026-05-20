// OpenCL ECM Stage 1 — Montgomery ladder (double_add_v2), ported from test/cgbn_stage1.cu

#ifndef MAX_LIMBS
#define MAX_LIMBS 64
#endif

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

inline void mp_copy(uint *dst, const uint *src, uint limbs) {
    for (uint i = 0u; i < limbs; ++i) {
        dst[i] = src[i];
    }
}

inline void mp_zero(uint *dst, uint limbs) {
    for (uint i = 0u; i < limbs; ++i) {
        dst[i] = 0u;
    }
}

inline int mp_ge(const uint *a, const uint *N, uint limbs) {
    for (int i = (int)limbs - 1; i >= 0; --i) {
        if (a[(uint)i] > N[(uint)i]) return 1;
        if (a[(uint)i] < N[(uint)i]) return 0;
    }
    return 1;
}

inline void mp_sub_n(uint *r, const uint *a, const uint *N, uint limbs) {
    ulong borrow = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong av = (ulong)a[i];
        ulong nv = (ulong)N[i];
        ulong w = av - nv - borrow;
        r[i] = (uint)w;
        borrow = (av < nv + borrow) ? 1ul : 0ul;
    }
}

inline uint mp_add_n(uint *r, const uint *a, const uint *b, uint limbs) {
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
inline void mp_add_mod(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {
    ulong carry = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong sum = (ulong)a[i] + (ulong)b[i] + carry;
        r[i] = (uint)sum;
        carry = sum >> 32;
    }
    if (carry != 0ul || mp_ge(r, N, limbs)) {
        mp_sub_n(r, r, N, limbs);
    }
}

// Returns 1 if borrow (a < b)
inline int mp_sub_mod(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {
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

inline void mp_shift_left_1_mod(uint *r, const uint *a, const uint *N, uint limbs) {
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

inline void mont_normalize(uint *r, const uint *N, uint limbs) {
    if (mp_ge(r, N, limbs)) {
        mp_sub_n(r, r, N, limbs);
    }
}

// r <- low limbs of (r * m); returns overflow limb above r
inline uint mul_ui32_limbs(uint *r, uint m, uint limbs) {
    ulong carry = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong prod = (ulong)r[i] * (ulong)m + carry;
        r[i] = (uint)prod;
        carry = prod >> 32;
    }
    return (uint)carry;
}

inline void shift_right_32_limbs(uint *r, uint limbs) {
    for (uint i = 0u; i + 1u < limbs; ++i) {
        r[i] = r[i + 1u];
    }
    r[limbs - 1u] = 0u;
}

// (r * m) / 2^32 mod N — ported from CUDA curve_t::special_mult_ui32
void special_mult_ui32(uint *r, uint m, const uint *N, uint np0, uint limbs) {
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
void double_add_v2(
    uint *q, uint *u, uint *w, uint *v,
    uint d, const uint *N, uint np0, uint limbs)
{
    uint t[MAX_LIMBS], CB[MAX_LIMBS], DA[MAX_LIMBS], AA[MAX_LIMBS], BB[MAX_LIMBS];
    uint K[MAX_LIMBS], dK[MAX_LIMBS];

    mp_add_mod(t, v, w, N, limbs);
    (void)mp_sub_mod(v, v, w, N, limbs);

    mp_add_mod(w, u, q, N, limbs);
    (void)mp_sub_mod(u, u, q, N, limbs);

    mont_mul_priv(CB, t, u, N, np0, limbs);
    mont_normalize(CB, N, limbs);
    mont_mul_priv(DA, v, w, N, np0, limbs);
    mont_normalize(DA, N, limbs);

    mont_sqr_priv(AA, w, N, np0, limbs);
    mont_sqr_priv(BB, u, N, np0, limbs);
    mont_normalize(AA, N, limbs);
    mont_normalize(BB, N, limbs);

    mont_mul_priv(q, AA, BB, N, np0, limbs);
    mont_normalize(q, N, limbs);

    (void)mp_sub_mod(K, AA, BB, N, limbs);

    mp_copy(dK, K, limbs);
    special_mult_ui32(dK, d, N, np0, limbs);

    mp_add_mod(u, BB, dK, N, limbs);
    mont_mul_priv(u, K, u, N, np0, limbs);
    mont_normalize(u, N, limbs);

    mp_add_mod(w, DA, CB, N, limbs);
    (void)mp_sub_mod(v, DA, CB, N, limbs);

    mont_sqr_priv(w, w, N, np0, limbs);
    mont_normalize(w, N, limbs);
    mont_sqr_priv(v, v, N, np0, limbs);
    mont_normalize(v, N, limbs);
    mp_shift_left_1_mod(v, v, N, limbs);
}

inline void swap_limbs(uint *a, uint *b, uint limbs) {
    for (uint i = 0u; i < limbs; ++i) {
        uint tmp = a[i];
        a[i] = b[i];
        b[i] = tmp;
    }
}

// ---------------------------------------------------------------------------
// Main kernel — mirrors CUDA kernel_double_add
// data layout per curve (5 * limbs uint32): N, aX, aZ, bX, bZ
// ---------------------------------------------------------------------------

__kernel void kernel_double_add(
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
    uint instance_i = get_global_id(0);
    if (instance_i >= count) {
        return;
    }
    if (limbs == 0u || limbs > MAX_LIMBS) {
        return;
    }

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

    // Curve coordinates are stored in Montgomery form (host set_p_2p).
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
