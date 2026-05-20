// Microbench kernels for ECM-style operators used in ecm_stage1.cl.
// Host should prepend mont_priv.cl when building this source so
// mont_mul_priv/mont_sqr_priv are available.

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

__kernel void ecm_mp_add_n_bench(
    __global const uint *a,
    __global const uint *b,
    __global uint *out,
    uint limbs,
    uint iterations)
{
    uint gid = get_global_id(0);
    uint base = gid * limbs;

    uint x[MAX_LIMBS], y[MAX_LIMBS], r[MAX_LIMBS];
    for (uint i = 0u; i < limbs; ++i) {
        x[i] = a[base + i];
        y[i] = b[base + i];
    }

    for (uint it = 0u; it < iterations; ++it) {
        (void)mp_add_n(r, x, y, limbs);
        mp_copy(x, r, limbs);
    }

    for (uint i = 0u; i < limbs; ++i) out[base + i] = x[i];
}

__kernel void ecm_mp_add_mod_bench(
    __global const uint *a,
    __global const uint *b,
    __global const uint *n,
    __global uint *out,
    uint limbs,
    uint iterations)
{
    uint gid = get_global_id(0);
    uint base = gid * limbs;

    uint x[MAX_LIMBS], y[MAX_LIMBS], m[MAX_LIMBS], r[MAX_LIMBS];
    for (uint i = 0u; i < limbs; ++i) {
        x[i] = a[base + i];
        y[i] = b[base + i];
        m[i] = n[base + i];
    }

    for (uint it = 0u; it < iterations; ++it) {
        mp_add_mod(r, x, y, m, limbs);
        mp_copy(x, r, limbs);
    }

    for (uint i = 0u; i < limbs; ++i) out[base + i] = x[i];
}

__kernel void ecm_mp_sub_mod_bench(
    __global const uint *a,
    __global const uint *b,
    __global const uint *n,
    __global uint *out,
    uint limbs,
    uint iterations)
{
    uint gid = get_global_id(0);
    uint base = gid * limbs;

    uint x[MAX_LIMBS], y[MAX_LIMBS], m[MAX_LIMBS], r[MAX_LIMBS];
    for (uint i = 0u; i < limbs; ++i) {
        x[i] = a[base + i];
        y[i] = b[base + i];
        m[i] = n[base + i];
    }

    for (uint it = 0u; it < iterations; ++it) {
        (void)mp_sub_mod(r, x, y, m, limbs);
        mp_copy(x, r, limbs);
    }

    for (uint i = 0u; i < limbs; ++i) out[base + i] = x[i];
}

__kernel void ecm_mont_mul_priv_bench(
    __global const uint *a,
    __global const uint *b,
    __global const uint *n,
    __global uint *out,
    uint np0,
    uint limbs,
    uint iterations)
{
    uint gid = get_global_id(0);
    uint base = gid * limbs;

    uint x[MAX_LIMBS], y[MAX_LIMBS], m[MAX_LIMBS], r[MAX_LIMBS];
    for (uint i = 0u; i < limbs; ++i) {
        x[i] = a[base + i];
        y[i] = b[base + i];
        m[i] = n[base + i];
    }

    for (uint it = 0u; it < iterations; ++it) {
        mont_mul_priv(r, x, y, m, np0, limbs);
        mp_copy(x, r, limbs);
    }

    for (uint i = 0u; i < limbs; ++i) out[base + i] = x[i];
}

__kernel void ecm_mont_sqr_priv_bench(
    __global const uint *a,
    __global const uint *n,
    __global uint *out,
    uint np0,
    uint limbs,
    uint iterations)
{
    uint gid = get_global_id(0);
    uint base = gid * limbs;

    uint x[MAX_LIMBS], m[MAX_LIMBS], r[MAX_LIMBS];
    for (uint i = 0u; i < limbs; ++i) {
        x[i] = a[base + i];
        m[i] = n[base + i];
    }

    for (uint it = 0u; it < iterations; ++it) {
        mont_sqr_priv(r, x, m, np0, limbs);
        mp_copy(x, r, limbs);
    }

    for (uint i = 0u; i < limbs; ++i) out[base + i] = x[i];
}
