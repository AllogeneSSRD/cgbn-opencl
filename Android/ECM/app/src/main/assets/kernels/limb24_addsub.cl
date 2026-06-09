// ECM add/sub microbench: 24-bit limbs stored in uint32 (Adreno 24-bit int ALU path).
// Each limb uses MP_LIMB_BITS=24; values are masked to 0xFFFFFF per word.

#ifndef MAX_LIMBS
#define MAX_LIMBS 16
#endif

#ifndef MP_LIMB_BITS
#define MP_LIMB_BITS 24
#endif

#define MP_LIMB_MASK ((1u << MP_LIMB_BITS) - 1u)

static inline uint limb24_load(uint v) {
    return v & MP_LIMB_MASK;
}

// Fused speculative subtract (same structure as mp_add_mod in ecm_addsub_bench.cl).
static inline void mp_add_mod_limb24(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {
    ulong carry_add = 0ul;
    ulong carry_sub = 1ul;
    for (uint i = 0u; i < limbs; ++i) {
        const ulong av = (ulong)limb24_load(a[i]);
        const ulong bv = (ulong)limb24_load(b[i]);
        const ulong nv = (ulong)limb24_load(N[i]);
        const ulong sum = av + bv + carry_add;
        carry_add = sum >> MP_LIMB_BITS;
        const ulong limb_sum = sum & (ulong)MP_LIMB_MASK;
        const ulong temp = limb_sum + (ulong)((~nv) & (ulong)MP_LIMB_MASK) + carry_sub;
        carry_sub = temp >> MP_LIMB_BITS;
        r[i] = (uint)(temp & (ulong)MP_LIMB_MASK);
    }
    if ((carry_add | carry_sub) != 0ul) {
        return;
    }
    ulong c = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        const ulong nv = (ulong)limb24_load(N[i]);
        const ulong rv = (ulong)limb24_load(r[i]);
        const ulong s = rv + nv + c;
        r[i] = (uint)(s & (ulong)MP_LIMB_MASK);
        c = s >> MP_LIMB_BITS;
    }
}

#if MAX_LIMBS <= 16
#define ECM_LIMB24_UNROLL_HINT 16
#elif MAX_LIMBS <= 32
#define ECM_LIMB24_UNROLL_HINT 32
#else
#define ECM_LIMB24_UNROLL_HINT 32
#endif

static inline void mp_add_mod_fused_unroll_limb24(uint *r, const uint *a, const uint *b, const uint *N) {
    ulong carry_add = 0ul;
    ulong carry_sub = 1ul;
#if MAX_LIMBS <= 16
    #pragma unroll 16
#else
    #pragma unroll ECM_LIMB24_UNROLL_HINT
#endif
    for (uint i = 0u; i < MAX_LIMBS; ++i) {
        const ulong av = (ulong)limb24_load(a[i]);
        const ulong bv = (ulong)limb24_load(b[i]);
        const ulong nv = (ulong)limb24_load(N[i]);
        const ulong sum = av + bv + carry_add;
        carry_add = sum >> MP_LIMB_BITS;
        const ulong limb_sum = sum & (ulong)MP_LIMB_MASK;
        const ulong temp = limb_sum + (ulong)((~nv) & (ulong)MP_LIMB_MASK) + carry_sub;
        carry_sub = temp >> MP_LIMB_BITS;
        r[i] = (uint)(temp & (ulong)MP_LIMB_MASK);
    }
    if ((carry_add | carry_sub) != 0ul) {
        return;
    }
    ulong c = 0ul;
#if MAX_LIMBS <= 16
    #pragma unroll 16
#else
    #pragma unroll ECM_LIMB24_UNROLL_HINT
#endif
    for (uint i = 0u; i < MAX_LIMBS; ++i) {
        const ulong nv = (ulong)limb24_load(N[i]);
        const ulong rv = (ulong)limb24_load(r[i]);
        const ulong s = rv + nv + c;
        r[i] = (uint)(s & (ulong)MP_LIMB_MASK);
        c = s >> MP_LIMB_BITS;
    }
}

__kernel void ecm_mp_add_mod_fused(
    __global const uint *a,
    __global const uint *b,
    __global const uint *n,
    __global uint *out,
    uint limbs)
{
    if (limbs != MAX_LIMBS) {
        return;
    }
    const uint gid = get_global_id(0);
    const uint base = gid * MAX_LIMBS;

    uint x[MAX_LIMBS], y[MAX_LIMBS], m[MAX_LIMBS], r[MAX_LIMBS];
    for (uint i = 0u; i < MAX_LIMBS; ++i) {
        x[i] = a[base + i];
        y[i] = b[base + i];
        m[i] = n[base + i];
    }
    mp_add_mod_limb24(r, x, y, m, MAX_LIMBS);
    for (uint i = 0u; i < MAX_LIMBS; ++i) {
        out[base + i] = r[i];
    }
}

__kernel void ecm_mp_add_mod_fused_unroll(
    __global const uint *a,
    __global const uint *b,
    __global const uint *n,
    __global uint *out,
    uint limbs)
{
    if (limbs != MAX_LIMBS) {
        return;
    }
    const uint gid = get_global_id(0);
    const uint base = gid * MAX_LIMBS;

    uint x[MAX_LIMBS], y[MAX_LIMBS], m[MAX_LIMBS], r[MAX_LIMBS];
    for (uint i = 0u; i < MAX_LIMBS; ++i) {
        x[i] = a[base + i];
        y[i] = b[base + i];
        m[i] = n[base + i];
    }
    mp_add_mod_fused_unroll_limb24(r, x, y, m);
    for (uint i = 0u; i < MAX_LIMBS; ++i) {
        out[base + i] = r[i];
    }
}
