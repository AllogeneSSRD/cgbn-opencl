// Pure OpenCL kernels for ECM-style add/sub/mod operators.
// Each __kernel performs exactly one arithmetic operation (no in-kernel timing loop).

#ifndef MAX_LIMBS
#define MAX_LIMBS 64
#endif

inline void mp_copy(uint *dst, const uint *src, uint limbs) {
    for (uint i = 0u; i < limbs; ++i) dst[i] = src[i];
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

// Legacy: add + mp_ge compare loop + conditional sub (3 passes worst case).
inline void mp_add_mod_legacy(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {
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

inline uint mp_sub_n_borrow(uint *r, const uint *a, const uint *b, uint limbs) {
    ulong borrow = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong av = (ulong)a[i];
        ulong bv = (ulong)b[i];
        ulong w = av - bv - borrow;
        r[i] = (uint)w;
        borrow = (av < bv + borrow) ? 1ul : 0ul;
    }
    return (uint)borrow;
}

// Mask select: S=a+b, D=S-N, pick D or S without mp_ge (extra stack for S).
inline void mp_add_mod_mask(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {
    uint S[MAX_LIMBS];
    ulong carry = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong sum = (ulong)a[i] + (ulong)b[i] + carry;
        S[i] = (uint)sum;
        carry = sum >> 32;
    }
    uint borrow = mp_sub_n_borrow(r, S, N, limbs);
    uint need_sub = (uint)(carry | (borrow == 0u));
    uint mask = 0u - need_sub;
    for (uint i = 0u; i < limbs; ++i) {
        r[i] = (r[i] & mask) | (S[i] & ~mask);
    }
}

#ifndef MP_ADD_MOD_FUSED_UNROLL
#define MP_ADD_MOD_FUSED_UNROLL 2
#endif

// Fused speculative subtract (v2): no mp_ge, branchless fix, optional 2-limb unroll.
inline void mp_add_mod(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {
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
        (void)mp_add_n(r, r, N, limbs);
        return 1;
    }
    return 0;
}

// r = a + b
__kernel void ecm_mp_add_n(
    __global const uint *a,
    __global const uint *b,
    __global uint *out,
    uint limbs)
{
    uint gid = get_global_id(0);
    uint base = gid * limbs;

    uint x[MAX_LIMBS], y[MAX_LIMBS], r[MAX_LIMBS];
    for (uint i = 0u; i < limbs; ++i) {
        x[i] = a[base + i];
        y[i] = b[base + i];
    }
    (void)mp_add_n(r, x, y, limbs);
    for (uint i = 0u; i < limbs; ++i) {
        out[base + i] = r[i];
    }
}

// r = a - N
__kernel void ecm_mp_sub_n(
    __global const uint *a,
    __global const uint *n,
    __global uint *out,
    uint limbs)
{
    uint gid = get_global_id(0);
    uint base = gid * limbs;

    uint x[MAX_LIMBS], m[MAX_LIMBS], r[MAX_LIMBS];
    for (uint i = 0u; i < limbs; ++i) {
        x[i] = a[base + i];
        m[i] = n[base + i];
    }
    mp_sub_n(r, x, m, limbs);
    for (uint i = 0u; i < limbs; ++i) {
        out[base + i] = r[i];
    }
}

// r = (a + b) mod N (legacy path, for A/B vs fused mp_add_mod)
__kernel void ecm_mp_add_mod_legacy(
    __global const uint *a,
    __global const uint *b,
    __global const uint *n,
    __global uint *out,
    uint limbs)
{
    uint gid = get_global_id(0);
    uint base = gid * limbs;

    uint x[MAX_LIMBS], y[MAX_LIMBS], m[MAX_LIMBS], r[MAX_LIMBS];
    for (uint i = 0u; i < limbs; ++i) {
        x[i] = a[base + i];
        y[i] = b[base + i];
        m[i] = n[base + i];
    }
    mp_add_mod_legacy(r, x, y, m, limbs);
    for (uint i = 0u; i < limbs; ++i) {
        out[base + i] = r[i];
    }
}

// r = (a + b) mod N (mask select, no mp_ge)
__kernel void ecm_mp_add_mod_mask(
    __global const uint *a,
    __global const uint *b,
    __global const uint *n,
    __global uint *out,
    uint limbs)
{
    uint gid = get_global_id(0);
    uint base = gid * limbs;

    uint x[MAX_LIMBS], y[MAX_LIMBS], m[MAX_LIMBS], r[MAX_LIMBS];
    for (uint i = 0u; i < limbs; ++i) {
        x[i] = a[base + i];
        y[i] = b[base + i];
        m[i] = n[base + i];
    }
    mp_add_mod_mask(r, x, y, m, limbs);
    for (uint i = 0u; i < limbs; ++i) {
        out[base + i] = r[i];
    }
}

// r = (a + b) mod N (fused speculative subtract)
__kernel void ecm_mp_add_mod_fused(
    __global const uint *a,
    __global const uint *b,
    __global const uint *n,
    __global uint *out,
    uint limbs)
{
    uint gid = get_global_id(0);
    uint base = gid * limbs;

    uint x[MAX_LIMBS], y[MAX_LIMBS], m[MAX_LIMBS], r[MAX_LIMBS];
    for (uint i = 0u; i < limbs; ++i) {
        x[i] = a[base + i];
        y[i] = b[base + i];
        m[i] = n[base + i];
    }
    mp_add_mod(r, x, y, m, limbs);
    for (uint i = 0u; i < limbs; ++i) {
        out[base + i] = r[i];
    }
}

// r = (a - b) mod N
__kernel void ecm_mp_sub_mod(
    __global const uint *a,
    __global const uint *b,
    __global const uint *n,
    __global uint *out,
    uint limbs)
{
    uint gid = get_global_id(0);
    uint base = gid * limbs;

    uint x[MAX_LIMBS], y[MAX_LIMBS], m[MAX_LIMBS], r[MAX_LIMBS];
    for (uint i = 0u; i < limbs; ++i) {
        x[i] = a[base + i];
        y[i] = b[base + i];
        m[i] = n[base + i];
    }
    (void)mp_sub_mod(r, x, y, m, limbs);
    for (uint i = 0u; i < limbs; ++i) {
        out[base + i] = r[i];
    }
}
