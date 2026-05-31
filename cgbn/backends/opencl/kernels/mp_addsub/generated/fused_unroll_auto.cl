// ECM/bench: fused add/sub-mod with #pragma unroll (path: fused_unroll_auto).

#ifndef MAX_LIMBS
#define MAX_LIMBS 16
#endif

#if MAX_LIMBS <= 16
#define ECM_ADDSUB_UNROLL_HINT 16
#elif MAX_LIMBS <= 32
#define ECM_ADDSUB_UNROLL_HINT 32
#elif MAX_LIMBS <= 64
#define ECM_ADDSUB_UNROLL_HINT 64
#else
#define ECM_ADDSUB_UNROLL_HINT 32
#endif

static inline void mp_add_mod_fused_unroll_auto(uint *r, const uint *a, const uint *b, const uint *N) {
    ulong carry_add = 0ul;
    ulong carry_sub = 1ul;
#if MAX_LIMBS <= 16
    #pragma unroll 16
#else
    #pragma unroll ECM_ADDSUB_UNROLL_HINT
#endif
    for (uint i = 0u; i < MAX_LIMBS; ++i) {
        ulong sum = (ulong)a[i] + (ulong)b[i] + carry_add;
        carry_add = sum >> 32;
        ulong temp = (ulong)(uint)sum + (ulong)(~N[i]) + carry_sub;
        carry_sub = temp >> 32;
        r[i] = (uint)temp;
    }
    if ((carry_add | carry_sub) != 0ul) return;
    ulong c = 0ul;
#if MAX_LIMBS <= 16
    #pragma unroll 16
#else
    #pragma unroll ECM_ADDSUB_UNROLL_HINT
#endif
    for (uint i = 0u; i < MAX_LIMBS; ++i) {
        ulong s = (ulong)r[i] + (ulong)N[i] + c;
        r[i] = (uint)s;
        c = s >> 32;
    }
}

static inline int mp_sub_mod_fused_unroll_auto(uint *r, const uint *a, const uint *b, const uint *N) {
    ulong br = 0ul;
#if MAX_LIMBS <= 16
    #pragma unroll 16
#else
    #pragma unroll ECM_ADDSUB_UNROLL_HINT
#endif
    for (uint i = 0u; i < MAX_LIMBS; ++i) {
        ulong av = (ulong)a[i];
        ulong bv = (ulong)b[i];
        ulong w = av - bv - br;
        r[i] = (uint)w;
        br = (av < bv + br) ? 1ul : 0ul;
    }
    if (br != 0ul) {
        ulong c = 0ul;
#if MAX_LIMBS <= 16
        #pragma unroll 16
#else
        #pragma unroll ECM_ADDSUB_UNROLL_HINT
#endif
        for (uint i = 0u; i < MAX_LIMBS; ++i) {
            ulong s = (ulong)r[i] + (ulong)N[i] + c;
            r[i] = (uint)s;
            c = s >> 32;
        }
        return 1;
    }
    return 0;
}

__kernel void ecm_mp_add_mod_fused_unroll_auto(__global const uint *a, __global const uint *b,
                                                __global const uint *n, __global uint *out,
                                                uint limbs) {
    if (limbs != MAX_LIMBS) return;
    uint gid = get_global_id(0);
    uint base = gid * MAX_LIMBS;
    uint x[MAX_LIMBS], y[MAX_LIMBS], m[MAX_LIMBS], r[MAX_LIMBS];
    for (uint i = 0u; i < MAX_LIMBS; ++i) {
        x[i] = a[base + i];
        y[i] = b[base + i];
        m[i] = n[base + i];
    }
    mp_add_mod_fused_unroll_auto(r, x, y, m);
    for (uint i = 0u; i < MAX_LIMBS; ++i) out[base + i] = r[i];
}

__kernel void ecm_mp_sub_mod_fused_unroll_auto(__global const uint *a, __global const uint *b,
                                               __global const uint *n, __global uint *out,
                                               uint limbs) {
    if (limbs != MAX_LIMBS) return;
    uint gid = get_global_id(0);
    uint base = gid * MAX_LIMBS;
    uint x[MAX_LIMBS], y[MAX_LIMBS], m[MAX_LIMBS], r[MAX_LIMBS];
    for (uint i = 0u; i < MAX_LIMBS; ++i) {
        x[i] = a[base + i];
        y[i] = b[base + i];
        m[i] = n[base + i];
    }
    (void)mp_sub_mod_fused_unroll_auto(r, x, y, m);
    for (uint i = 0u; i < MAX_LIMBS; ++i) out[base + i] = r[i];
}
