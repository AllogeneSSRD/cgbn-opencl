// OpenCL ECM Stage 1 — Montgomery ladder (double_add_v2), ported from test/cgbn_stage1.cu

#ifndef MAX_LIMBS
#define MAX_LIMBS 64
#endif

// ---------------------------------------------------------------------------
// Private multi-limb helpers (one curve per work-item)
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
        mp_add_mod(r, r, N, N, limbs);
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

// Montgomery multiplication: out = a*b*R^{-1} mod N  (CIOS, private arrays)
void mont_mul(uint *out, const uint *a, const uint *b, const uint *N, uint np0, uint limbs) {
    if (limbs == 0u || limbs > MAX_LIMBS) {
        return;
    }

    uint t[MAX_LIMBS + 1];
    uint B[MAX_LIMBS];
    for (uint i = 0u; i <= limbs; ++i) {
        t[i] = 0u;
    }
    uint t_hi = 0u;
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
        ulong uvh = (ulong)t[limbs] + carry;
        t[limbs] = (uint)uvh;
        t_hi += (uint)(uvh >> 32);

        uint m = (uint)((ulong)t[0] * (ulong)np0);
        carry = 0ul;
        for (uint j = 0u; j < limbs; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
            if (j > 0u) {
                t[j - 1u] = (uint)uv;
            }
            carry = uv >> 32;
        }
        ulong top = (ulong)t[limbs] + carry;
        t[limbs - 1u] = (uint)top;
        ulong top2 = (ulong)t_hi + (top >> 32);
        t[limbs] = (uint)top2;
        t_hi = (uint)(top2 >> 32);
    }

    int ge = (t_hi != 0u || t[limbs] != 0u) ? 1 : 0;
    if (!ge) {
        for (int i = (int)limbs - 1; i >= 0; --i) {
            if (t[(uint)i] > N[(uint)i]) {
                ge = 1;
                break;
            }
            if (t[(uint)i] < N[(uint)i]) {
                ge = 0;
                break;
            }
        }
    }
    if (ge) {
        ulong borrow = 0ul;
        for (uint i = 0u; i < limbs; ++i) {
            ulong tv = (ulong)t[i];
            ulong nv = (ulong)N[i];
            ulong w = tv - nv - borrow;
            t[i] = (uint)w;
            borrow = (tv < nv + borrow) ? 1ul : 0ul;
        }
    }
    for (uint i = 0u; i < limbs; ++i) {
        out[i] = t[i];
    }
}

inline void mont_sqr(uint *out, const uint *a, const uint *N, uint np0, uint limbs) {
    mont_mul(out, a, a, N, np0, limbs);
}

inline void mont_normalize(uint *r, const uint *N, uint limbs) {
    if (mp_ge(r, N, limbs)) {
        mp_sub_n(r, r, N, limbs);
    }
}

// (r * m) / 2^32 mod N — d = (sigma/2^32) mod N handled on host via uint32 d
void special_mult_ui32(uint *r, uint m, const uint *N, uint np0, uint limbs) {
    ulong carry = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong prod = (ulong)r[i] * (ulong)m + carry;
        r[i] = (uint)prod;
        carry = prod >> 32;
    }
    uint t1_0 = r[0];
    uint q = (uint)((ulong)t1_0 * (ulong)np0);

    uint temp[MAX_LIMBS];
    carry = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong prod = (ulong)N[i] * (ulong)q + carry;
        temp[i] = (uint)prod;
        carry = prod >> 32;
    }
    uint carry_t2 = (uint)carry;

    // shift r and temp right by 32 bits
    for (uint i = 0u; i + 1u < limbs; ++i) {
        r[i] = r[i + 1u];
        temp[i] = temp[i + 1u];
    }
    r[limbs - 1u] = 0u;
    temp[limbs - 1u] = 0u;

    uint carry_t1 = (uint)(carry != 0ul);
    mp_add_mod(r, r, temp, N, limbs);
    if (carry_t1) {
        uint one[MAX_LIMBS];
        mp_zero(one, limbs);
        one[0] = 1u;
        mp_add_mod(r, r, one, N, limbs);
    }
    if (carry_t2) {
        uint one[MAX_LIMBS];
        mp_zero(one, limbs);
        one[0] = 1u;
        mp_add_mod(r, r, one, N, limbs);
    }
    if (t1_0 != 0u) {
        uint one[MAX_LIMBS];
        mp_zero(one, limbs);
        one[0] = 1u;
        mp_add_mod(r, r, one, N, limbs);
    }
}

// Simultaneous double-and-add (CUDA curve_t::double_add_v2)
void double_add_v2(
    uint *q, uint *u, uint *w, uint *v,
    uint d, const uint *N, uint np0, uint limbs)
{
    uint t[MAX_LIMBS], CB[MAX_LIMBS], DA[MAX_LIMBS], AA[MAX_LIMBS], BB[MAX_LIMBS];
    uint K[MAX_LIMBS], dK[MAX_LIMBS];

    mp_add_mod(t, v, w, N, limbs);
    if (mp_sub_mod(v, v, w, N, limbs)) {
        mp_add_mod(v, v, N, N, limbs);
    }

    mp_add_mod(w, u, q, N, limbs);
    if (mp_sub_mod(u, u, q, N, limbs)) {
        mp_add_mod(u, u, N, N, limbs);
    }

    mont_mul(CB, t, u, N, np0, limbs);
    mont_normalize(CB, N, limbs);
    mont_mul(DA, v, w, N, np0, limbs);
    mont_normalize(DA, N, limbs);

    mont_sqr(AA, w, N, np0, limbs);
    mont_sqr(BB, u, N, np0, limbs);
    mont_normalize(AA, N, limbs);
    mont_normalize(BB, N, limbs);

    mont_mul(q, AA, BB, N, np0, limbs);
    mont_normalize(q, N, limbs);

    if (mp_sub_mod(K, AA, BB, N, limbs)) {
        mp_add_mod(K, K, N, N, limbs);
    }

    mp_copy(dK, K, limbs);
    special_mult_ui32(dK, d, N, np0, limbs);

    mp_add_mod(u, BB, dK, N, limbs);
    mont_mul(u, K, u, N, np0, limbs);
    mont_normalize(u, N, limbs);

    mp_add_mod(w, DA, CB, N, limbs);
    if (mp_sub_mod(v, DA, CB, N, limbs)) {
        mp_add_mod(v, v, N, N, limbs);
    }

    mont_sqr(w, w, N, np0, limbs);
    mont_normalize(w, N, limbs);
    mont_sqr(v, v, N, np0, limbs);
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

    // Values are uploaded in Montgomery form from host; ladder runs in mont domain.
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
