// ECM add/sub microbench: 24-bit limbs in uint32 (low 24 bits only).
// Host guarantees inputs are pre-masked; kernels avoid per-limb &mask on the hot path.

#ifndef MAX_LIMBS
#define MAX_LIMBS 16
#endif

#ifndef MP_LIMB_BITS
#define MP_LIMB_BITS 24
#endif

#define MP_LIMB_MASK ((1u << MP_LIMB_BITS) - 1u)

#if MAX_LIMBS <= 16
#define ECM_LIMB24_UNROLL_HINT 16
#elif MAX_LIMBS <= 32
#define ECM_LIMB24_UNROLL_HINT 32
#else
#define ECM_LIMB24_UNROLL_HINT 32
#endif

// Mirror 32-bit fused: (ulong)(uint)sum truncates; carry via >> 24. No &mask in loop.
static inline void mp_add_mod_limb24_fast(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {
    ulong carry_add = 0ul;
    ulong carry_sub = 1ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong sum = (ulong)a[i] + (ulong)b[i] + carry_add;
        carry_add = sum >> MP_LIMB_BITS;
        ulong temp = (ulong)(uint)sum + (ulong)(~N[i]) + carry_sub;
        carry_sub = temp >> MP_LIMB_BITS;
        r[i] = (uint)temp;
    }
    if ((carry_add | carry_sub) != 0ul) {
        return;
    }
    ulong c = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong s = (ulong)r[i] + (ulong)N[i] + c;
        r[i] = (uint)s;
        c = s >> MP_LIMB_BITS;
    }
}

// 2-limb unroll (same pattern as MP_ADD_MOD_FUSED_UNROLL=2 on 32-bit path).
static inline void mp_add_mod_limb24_fast_u2(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {
    ulong carry_add = 0ul;
    ulong carry_sub = 1ul;
    uint j = 0u;
    for (; j + 1u < limbs; j += 2u) {
        ulong sum0 = (ulong)a[j] + (ulong)b[j] + carry_add;
        carry_add = sum0 >> MP_LIMB_BITS;
        ulong temp0 = (ulong)(uint)sum0 + (ulong)(~N[j]) + carry_sub;
        carry_sub = temp0 >> MP_LIMB_BITS;
        r[j] = (uint)temp0;

        ulong sum1 = (ulong)a[j + 1u] + (ulong)b[j + 1u] + carry_add;
        carry_add = sum1 >> MP_LIMB_BITS;
        ulong temp1 = (ulong)(uint)sum1 + (ulong)(~N[j + 1u]) + carry_sub;
        carry_sub = temp1 >> MP_LIMB_BITS;
        r[j + 1u] = (uint)temp1;
    }
    if (limbs & 1u) {
        ulong sum = (ulong)a[j] + (ulong)b[j] + carry_add;
        carry_add = sum >> MP_LIMB_BITS;
        ulong temp = (ulong)(uint)sum + (ulong)(~N[j]) + carry_sub;
        carry_sub = temp >> MP_LIMB_BITS;
        r[j] = (uint)temp;
    }
    if ((carry_add | carry_sub) != 0ul) {
        return;
    }
    ulong c = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong s = (ulong)r[i] + (ulong)N[i] + c;
        r[i] = (uint)s;
        c = s >> MP_LIMB_BITS;
    }
}

static inline void mp_add_mod_limb24_fast_unroll(uint *r, const uint *a, const uint *b, const uint *N) {
    ulong carry_add = 0ul;
    ulong carry_sub = 1ul;
#if MAX_LIMBS <= 16
    #pragma unroll 16
#else
    #pragma unroll ECM_LIMB24_UNROLL_HINT
#endif
    for (uint i = 0u; i < MAX_LIMBS; ++i) {
        ulong sum = (ulong)a[i] + (ulong)b[i] + carry_add;
        carry_add = sum >> MP_LIMB_BITS;
        ulong temp = (ulong)(uint)sum + (ulong)(~N[i]) + carry_sub;
        carry_sub = temp >> MP_LIMB_BITS;
        r[i] = (uint)temp;
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
        ulong s = (ulong)r[i] + (ulong)N[i] + c;
        r[i] = (uint)s;
        c = s >> MP_LIMB_BITS;
    }
}

static inline void mp_load_limbs(uint *dst, __global const uint *src, uint base, uint count) {
    for (uint i = 0u; i < count; ++i) {
        dst[i] = src[base + i];
    }
}

static inline void mp_store_limbs(__global uint *dst, const uint *src, uint base, uint count) {
    for (uint i = 0u; i < count; ++i) {
        dst[base + i] = src[i];
    }
}

#define ECM_LIMB24_DISPATCH(KERNEL, MP_FN) \
__kernel void KERNEL( \
    __global const uint *a, \
    __global const uint *b, \
    __global const uint *n, \
    __global uint *out, \
    uint limbs) \
{ \
    if (limbs != MAX_LIMBS) return; \
    const uint gid = get_global_id(0); \
    const uint base = gid * MAX_LIMBS; \
    uint x[MAX_LIMBS], y[MAX_LIMBS], m[MAX_LIMBS], r[MAX_LIMBS]; \
    mp_load_limbs(x, a, base, MAX_LIMBS); \
    mp_load_limbs(y, b, base, MAX_LIMBS); \
    mp_load_limbs(m, n, base, MAX_LIMBS); \
    MP_FN(r, x, y, m, MAX_LIMBS); \
    mp_store_limbs(out, r, base, MAX_LIMBS); \
}

ECM_LIMB24_DISPATCH(ecm_mp_add_mod_fused, mp_add_mod_limb24_fast)
ECM_LIMB24_DISPATCH(ecm_mp_add_mod_fused_u2, mp_add_mod_limb24_fast_u2)

__kernel void ecm_mp_add_mod_fused_unroll(
    __global const uint *a,
    __global const uint *b,
    __global const uint *n,
    __global uint *out,
    uint limbs)
{
    if (limbs != MAX_LIMBS) return;
    const uint gid = get_global_id(0);
    const uint base = gid * MAX_LIMBS;
    uint x[MAX_LIMBS], y[MAX_LIMBS], m[MAX_LIMBS], r[MAX_LIMBS];
    mp_load_limbs(x, a, base, MAX_LIMBS);
    mp_load_limbs(y, b, base, MAX_LIMBS);
    mp_load_limbs(m, n, base, MAX_LIMBS);
    mp_add_mod_limb24_fast_unroll(r, x, y, m);
    mp_store_limbs(out, r, base, MAX_LIMBS);
}

// Hot loop: load once, many add_mod in registers, store once (isolates ALU vs CLPeak).
__kernel void ecm_mp_add_mod_fused_hot(
    __global const uint *a,
    __global const uint *b,
    __global const uint *n,
    __global uint *out,
    uint limbs,
    uint inner_iters)
{
    if (limbs != MAX_LIMBS || inner_iters == 0u) return;
    const uint gid = get_global_id(0);
    const uint base = gid * MAX_LIMBS;

    uint x[MAX_LIMBS], y[MAX_LIMBS], m[MAX_LIMBS], r[MAX_LIMBS];
    mp_load_limbs(x, a, base, MAX_LIMBS);
    mp_load_limbs(y, b, base, MAX_LIMBS);
    mp_load_limbs(m, n, base, MAX_LIMBS);

    for (uint k = 0u; k < inner_iters; ++k) {
        mp_add_mod_limb24_fast_unroll(r, x, y, m);
        for (uint i = 0u; i < MAX_LIMBS; ++i) {
            x[i] = r[i];
        }
        y[0] = (y[0] + 1u) & MP_LIMB_MASK;
    }
    mp_store_limbs(out, r, base, MAX_LIMBS);
}

__kernel void ecm_mp_add_mod_fused_unroll_hot(
    __global const uint *a,
    __global const uint *b,
    __global const uint *n,
    __global uint *out,
    uint limbs,
    uint inner_iters)
{
    if (limbs != MAX_LIMBS || inner_iters == 0u) return;
    const uint gid = get_global_id(0);
    const uint base = gid * MAX_LIMBS;

    uint x[MAX_LIMBS], y[MAX_LIMBS], m[MAX_LIMBS], r[MAX_LIMBS];
    mp_load_limbs(x, a, base, MAX_LIMBS);
    mp_load_limbs(y, b, base, MAX_LIMBS);
    mp_load_limbs(m, n, base, MAX_LIMBS);

    for (uint k = 0u; k < inner_iters; ++k) {
        mp_add_mod_limb24_fast_unroll(r, x, y, m);
        for (uint i = 0u; i < MAX_LIMBS; ++i) {
            x[i] = r[i];
        }
        y[0] = (y[0] + 1u) & MP_LIMB_MASK;
    }
    mp_store_limbs(out, r, base, MAX_LIMBS);
}
