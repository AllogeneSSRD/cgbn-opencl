// Fused mp_add_mod — AMDGCN v_add_co / v_add_co_ci carry chain (bench only).

#if defined(MP_ADDMOD_ASM_ENABLE) && defined(__AMDGCN__)

// Fix r += N when S < N: ca==0 && cs==0 (pure C ulong chain, no v_add_co asm).
static inline void c_fix_add_n8(uint *r0, uint *r1, uint *r2, uint *r3, uint *r4, uint *r5, uint *r6,
                                uint *r7, uint n0, uint n1, uint n2, uint n3, uint n4, uint n5,
                                uint n6, uint n7) {
    ulong c = 0ul;
    ulong s;
    s = (ulong)*r0 + (ulong)n0 + c;
    c = s >> 32;
    *r0 = (uint)s;
    s = (ulong)*r1 + (ulong)n1 + c;
    c = s >> 32;
    *r1 = (uint)s;
    s = (ulong)*r2 + (ulong)n2 + c;
    c = s >> 32;
    *r2 = (uint)s;
    s = (ulong)*r3 + (ulong)n3 + c;
    c = s >> 32;
    *r3 = (uint)s;
    s = (ulong)*r4 + (ulong)n4 + c;
    c = s >> 32;
    *r4 = (uint)s;
    s = (ulong)*r5 + (ulong)n5 + c;
    c = s >> 32;
    *r5 = (uint)s;
    s = (ulong)*r6 + (ulong)n6 + c;
    c = s >> 32;
    *r6 = (uint)s;
    s = (ulong)*r7 + (ulong)n7 + c;
    c = s >> 32;
    *r7 = (uint)s;
}

// Fix r += N via v_add_co asm (bench compare vs c_fix_add_n8).
#define MP_ADDMOD_FIX_LIMB(RI, NI)                                                                  \
    do {                                                                                            \
        uint co = 0u;                                                                               \
        if (c) {                                                                                    \
            __asm volatile("v_add_co_u32    %[ri], vcc_lo, %[ri], %[ni]\n\t"                       \
                           "v_add_co_ci_u32 %[ri], vcc_lo, %[c], %[ri], vcc_lo\n\t"                 \
                           "v_cndmask_b32   %[co], %[z], %[o], vcc_lo"                               \
                           : [ri] "+v"(RI), [co] "=&v"(co)                                         \
                           : [ni] "v"(NI), [c] "v"(c), [z] "v"(z), [o] "v"(o)                       \
                           : "vcc_lo");                                                             \
        } else {                                                                                    \
            __asm volatile("v_add_co_u32  %[ri], vcc_lo, %[ri], %[ni]\n\t"                         \
                           "v_cndmask_b32 %[co], %[z], %[o], vcc_lo"                                 \
                           : [ri] "+v"(RI), [co] "=&v"(co)                                         \
                           : [ni] "v"(NI), [z] "v"(z), [o] "v"(o)                                   \
                           : "vcc_lo");                                                             \
        }                                                                                           \
        c = co;                                                                                     \
    } while (0)

static inline void asm_fix_add_n8(uint *r0, uint *r1, uint *r2, uint *r3, uint *r4, uint *r5,
                                  uint *r6, uint *r7, uint n0, uint n1, uint n2, uint n3, uint n4,
                                  uint n5, uint n6, uint n7) {
    const uint z = 0u, o = 1u;
    uint c = 0u;
    uint x0 = *r0, x1 = *r1, x2 = *r2, x3 = *r3, x4 = *r4, x5 = *r5, x6 = *r6, x7 = *r7;
    MP_ADDMOD_FIX_LIMB(x0, n0);
    MP_ADDMOD_FIX_LIMB(x1, n1);
    MP_ADDMOD_FIX_LIMB(x2, n2);
    MP_ADDMOD_FIX_LIMB(x3, n3);
    MP_ADDMOD_FIX_LIMB(x4, n4);
    MP_ADDMOD_FIX_LIMB(x5, n5);
    MP_ADDMOD_FIX_LIMB(x6, n6);
    MP_ADDMOD_FIX_LIMB(x7, n7);
    *r0 = x0;
    *r1 = x1;
    *r2 = x2;
    *r3 = x3;
    *r4 = x4;
    *r5 = x5;
    *r6 = x6;
    *r7 = x7;
}
#undef MP_ADDMOD_FIX_LIMB

// 8-limb fused block; ca_in/cs_in are 0/1 carry into this chunk (VCC preset via cmp).
// a/b/n/r are __global: no private[MAX_LIMBS] arrays in callers (avoids scratch spill).
static inline void asm_fused_block8(__global const uint *a, __global const uint *b,
                                    __global const uint *n, __global uint *r, uint ca_in,
                                    uint cs_in, uint *ca_out, uint *cs_out) {
    uint a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
    uint b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3], b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];
    uint n0 = n[0], n1 = n[1], n2 = n[2], n3 = n[3], n4 = n[4], n5 = n[5], n6 = n[6], n7 = n[7];
    uint s0, s1, s2, s3, s4, s5, s6, s7;
    uint r0, r1, r2, r3, r4, r5, r6, r7;
    uint n0n, n1n, n2n, n3n, n4n, n5n, n6n, n7n;
    uint ca = 0u, cs = 0u;
    const uint z = 0u, o = 1u;
    uint ca_bit = ca_in ? o : z;
    uint cs_bit = cs_in ? o : z;

    __asm volatile(
        "v_cmp_eq_u32    vcc_lo, %[ca_bit], %[o]\n\t"
        "v_add_co_ci_u32 %[s0], vcc_lo, %[a0], %[b0], vcc_lo\n\t"
        "v_add_co_ci_u32 %[s1], vcc_lo, %[a1], %[b1], vcc_lo\n\t"
        "v_add_co_ci_u32 %[s2], vcc_lo, %[a2], %[b2], vcc_lo\n\t"
        "v_add_co_ci_u32 %[s3], vcc_lo, %[a3], %[b3], vcc_lo\n\t"
        "v_add_co_ci_u32 %[s4], vcc_lo, %[a4], %[b4], vcc_lo\n\t"
        "v_add_co_ci_u32 %[s5], vcc_lo, %[a5], %[b5], vcc_lo\n\t"
        "v_add_co_ci_u32 %[s6], vcc_lo, %[a6], %[b6], vcc_lo\n\t"
        "v_add_co_ci_u32 %[s7], vcc_lo, %[a7], %[b7], vcc_lo\n\t"
        "v_cndmask_b32   %[ca], %[z], %[o], vcc_lo\n\t"
        "v_not_b32       %[n0n], %[n0]\n\t"
        "v_not_b32       %[n1n], %[n1]\n\t"
        "v_not_b32       %[n2n], %[n2]\n\t"
        "v_not_b32       %[n3n], %[n3]\n\t"
        "v_not_b32       %[n4n], %[n4]\n\t"
        "v_not_b32       %[n5n], %[n5]\n\t"
        "v_not_b32       %[n6n], %[n6]\n\t"
        "v_not_b32       %[n7n], %[n7]\n\t"
        "v_cmp_eq_u32    vcc_lo, %[cs_bit], %[o]\n\t"
        "v_add_co_ci_u32 %[r0], vcc_lo, %[s0], %[n0n], vcc_lo\n\t"
        "v_add_co_ci_u32 %[r1], vcc_lo, %[s1], %[n1n], vcc_lo\n\t"
        "v_add_co_ci_u32 %[r2], vcc_lo, %[s2], %[n2n], vcc_lo\n\t"
        "v_add_co_ci_u32 %[r3], vcc_lo, %[s3], %[n3n], vcc_lo\n\t"
        "v_add_co_ci_u32 %[r4], vcc_lo, %[s4], %[n4n], vcc_lo\n\t"
        "v_add_co_ci_u32 %[r5], vcc_lo, %[s5], %[n5n], vcc_lo\n\t"
        "v_add_co_ci_u32 %[r6], vcc_lo, %[s6], %[n6n], vcc_lo\n\t"
        "v_add_co_ci_u32 %[r7], vcc_lo, %[s7], %[n7n], vcc_lo\n\t"
        "v_cndmask_b32   %[cs], %[z], %[o], vcc_lo"
        : [s0] "=&v"(s0), [s1] "=&v"(s1), [s2] "=&v"(s2), [s3] "=&v"(s3),
          [s4] "=&v"(s4), [s5] "=&v"(s5), [s6] "=&v"(s6), [s7] "=&v"(s7),
          [r0] "=&v"(r0), [r1] "=&v"(r1), [r2] "=&v"(r2), [r3] "=&v"(r3),
          [r4] "=&v"(r4), [r5] "=&v"(r5), [r6] "=&v"(r6), [r7] "=&v"(r7),
          [n0n] "=&v"(n0n), [n1n] "=&v"(n1n), [n2n] "=&v"(n2n), [n3n] "=&v"(n3n),
          [n4n] "=&v"(n4n), [n5n] "=&v"(n5n), [n6n] "=&v"(n6n), [n7n] "=&v"(n7n),
          [ca] "=&v"(ca), [cs] "=&v"(cs)
        : [a0] "v"(a0), [a1] "v"(a1), [a2] "v"(a2), [a3] "v"(a3),
          [a4] "v"(a4), [a5] "v"(a5), [a6] "v"(a6), [a7] "v"(a7),
          [b0] "v"(b0), [b1] "v"(b1), [b2] "v"(b2), [b3] "v"(b3),
          [b4] "v"(b4), [b5] "v"(b5), [b6] "v"(b6), [b7] "v"(b7),
          [n0] "v"(n0), [n1] "v"(n1), [n2] "v"(n2), [n3] "v"(n3),
          [n4] "v"(n4), [n5] "v"(n5), [n6] "v"(n6), [n7] "v"(n7),
          [ca_bit] "v"(ca_bit), [cs_bit] "v"(cs_bit), [z] "v"(z), [o] "v"(o)
        : "vcc_lo");

    if ((ca | cs) == 0u) {
        c_fix_add_n8(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7, n0, n1, n2, n3, n4, n5, n6, n7);
    }

    r[0] = r0;
    r[1] = r1;
    r[2] = r2;
    r[3] = r3;
    r[4] = r4;
    r[5] = r5;
    r[6] = r6;
    r[7] = r7;

    *ca_out = ca;
    *cs_out = cs;
}

// Same as asm_fused_block8 but fix uses v_add_co asm (A/B vs c_fix_add_n8).
static inline void asm_fused_block8_asmfix(__global const uint *a, __global const uint *b,
                                           __global const uint *n, __global uint *r, uint ca_in,
                                           uint cs_in, uint *ca_out, uint *cs_out) {
    uint a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
    uint b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3], b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];
    uint n0 = n[0], n1 = n[1], n2 = n[2], n3 = n[3], n4 = n[4], n5 = n[5], n6 = n[6], n7 = n[7];
    uint s0, s1, s2, s3, s4, s5, s6, s7;
    uint r0, r1, r2, r3, r4, r5, r6, r7;
    uint n0n, n1n, n2n, n3n, n4n, n5n, n6n, n7n;
    uint ca = 0u, cs = 0u;
    const uint z = 0u, o = 1u;
    uint ca_bit = ca_in ? o : z;
    uint cs_bit = cs_in ? o : z;

    __asm volatile(
        "v_cmp_eq_u32    vcc_lo, %[ca_bit], %[o]\n\t"
        "v_add_co_ci_u32 %[s0], vcc_lo, %[a0], %[b0], vcc_lo\n\t"
        "v_add_co_ci_u32 %[s1], vcc_lo, %[a1], %[b1], vcc_lo\n\t"
        "v_add_co_ci_u32 %[s2], vcc_lo, %[a2], %[b2], vcc_lo\n\t"
        "v_add_co_ci_u32 %[s3], vcc_lo, %[a3], %[b3], vcc_lo\n\t"
        "v_add_co_ci_u32 %[s4], vcc_lo, %[a4], %[b4], vcc_lo\n\t"
        "v_add_co_ci_u32 %[s5], vcc_lo, %[a5], %[b5], vcc_lo\n\t"
        "v_add_co_ci_u32 %[s6], vcc_lo, %[a6], %[b6], vcc_lo\n\t"
        "v_add_co_ci_u32 %[s7], vcc_lo, %[a7], %[b7], vcc_lo\n\t"
        "v_cndmask_b32   %[ca], %[z], %[o], vcc_lo\n\t"
        "v_not_b32       %[n0n], %[n0]\n\t"
        "v_not_b32       %[n1n], %[n1]\n\t"
        "v_not_b32       %[n2n], %[n2]\n\t"
        "v_not_b32       %[n3n], %[n3]\n\t"
        "v_not_b32       %[n4n], %[n4]\n\t"
        "v_not_b32       %[n5n], %[n5]\n\t"
        "v_not_b32       %[n6n], %[n6]\n\t"
        "v_not_b32       %[n7n], %[n7]\n\t"
        "v_cmp_eq_u32    vcc_lo, %[cs_bit], %[o]\n\t"
        "v_add_co_ci_u32 %[r0], vcc_lo, %[s0], %[n0n], vcc_lo\n\t"
        "v_add_co_ci_u32 %[r1], vcc_lo, %[s1], %[n1n], vcc_lo\n\t"
        "v_add_co_ci_u32 %[r2], vcc_lo, %[s2], %[n2n], vcc_lo\n\t"
        "v_add_co_ci_u32 %[r3], vcc_lo, %[s3], %[n3n], vcc_lo\n\t"
        "v_add_co_ci_u32 %[r4], vcc_lo, %[s4], %[n4n], vcc_lo\n\t"
        "v_add_co_ci_u32 %[r5], vcc_lo, %[s5], %[n5n], vcc_lo\n\t"
        "v_add_co_ci_u32 %[r6], vcc_lo, %[s6], %[n6n], vcc_lo\n\t"
        "v_add_co_ci_u32 %[r7], vcc_lo, %[s7], %[n7n], vcc_lo\n\t"
        "v_cndmask_b32   %[cs], %[z], %[o], vcc_lo"
        : [s0] "=&v"(s0), [s1] "=&v"(s1), [s2] "=&v"(s2), [s3] "=&v"(s3),
          [s4] "=&v"(s4), [s5] "=&v"(s5), [s6] "=&v"(s6), [s7] "=&v"(s7),
          [r0] "=&v"(r0), [r1] "=&v"(r1), [r2] "=&v"(r2), [r3] "=&v"(r3),
          [r4] "=&v"(r4), [r5] "=&v"(r5), [r6] "=&v"(r6), [r7] "=&v"(r7),
          [n0n] "=&v"(n0n), [n1n] "=&v"(n1n), [n2n] "=&v"(n2n), [n3n] "=&v"(n3n),
          [n4n] "=&v"(n4n), [n5n] "=&v"(n5n), [n6n] "=&v"(n6n), [n7n] "=&v"(n7n),
          [ca] "=&v"(ca), [cs] "=&v"(cs)
        : [a0] "v"(a0), [a1] "v"(a1), [a2] "v"(a2), [a3] "v"(a3),
          [a4] "v"(a4), [a5] "v"(a5), [a6] "v"(a6), [a7] "v"(a7),
          [b0] "v"(b0), [b1] "v"(b1), [b2] "v"(b2), [b3] "v"(b3),
          [b4] "v"(b4), [b5] "v"(b5), [b6] "v"(b6), [b7] "v"(b7),
          [n0] "v"(n0), [n1] "v"(n1), [n2] "v"(n2), [n3] "v"(n3),
          [n4] "v"(n4), [n5] "v"(n5), [n6] "v"(n6), [n7] "v"(n7),
          [ca_bit] "v"(ca_bit), [cs_bit] "v"(cs_bit), [z] "v"(z), [o] "v"(o)
        : "vcc_lo");

    if ((ca | cs) == 0u) {
        asm_fix_add_n8(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7, n0, n1, n2, n3, n4, n5, n6, n7);
    }

    r[0] = r0;
    r[1] = r1;
    r[2] = r2;
    r[3] = r3;
    r[4] = r4;
    r[5] = r5;
    r[6] = r6;
    r[7] = r7;

    *ca_out = ca;
    *cs_out = cs;
}

// Reference interleaved 8-limb (same semantics as fused mp_add_mod, for asm validation).
__attribute__((always_inline)) static inline void mp_add_mod_8_interleaved_c(__global uint *r,
                                                                             __global const uint *a,
                                                                             __global const uint *b,
                                                                             __global const uint *N) {
    ulong ca = 0ul, cs = 1ul;
    for (uint i = 0u; i < 8u; ++i) {
        ulong sum = (ulong)a[i] + (ulong)b[i] + ca;
        ca = sum >> 32;
        ulong temp = (ulong)(uint)sum + (ulong)(~N[i]) + cs;
        cs = temp >> 32;
        r[i] = (uint)temp;
    }
    if ((ca | cs) != 0ul) {
        return;
    }
    ulong c = 0ul;
    for (uint i = 0u; i < 8u; ++i) {
        ulong s = (ulong)r[i] + (ulong)N[i] + c;
        r[i] = (uint)s;
        c = s >> 32;
    }
}

// 8-limb fused: VCC soft-switch (per-limb ca/cs reload via cmp + cndmask).
// Scalar a/b/N loads keep operands in VGPR; one asm region, shared s_tmp/nn_tmp.
__attribute__((always_inline)) static inline void mp_add_mod_8_asm_vccsoft(__global uint *r,
                                                                           __global const uint *a,
                                                                           __global const uint *b,
                                                                           __global const uint *N) {
    uint a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
    uint b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3], b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];
    uint n0 = N[0], n1 = N[1], n2 = N[2], n3 = N[3], n4 = N[4], n5 = N[5], n6 = N[6], n7 = N[7];
    uint r0, r1, r2, r3, r4, r5, r6, r7;
    uint v_ca = 0u, v_cs = 0u;
    uint s_tmp, nn_tmp;
    const uint z = 0u, one = 1u;

    __asm__ volatile(
        "v_add_co_u32    %[s], vcc_lo, %[a0], %[b0]\n\t"
        "v_cndmask_b32   %[ca], %[z], %[one], vcc_lo\n\t"
        "v_not_b32       %[nn], %[n0]\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[one]\n\t"
        "v_add_co_ci_u32 %[r0], vcc_lo, %[s], %[nn], vcc_lo\n\t"
        "v_cndmask_b32   %[cs], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[ca]\n\t"
        "v_add_co_ci_u32 %[s], vcc_lo, %[a1], %[b1], vcc_lo\n\t"
        "v_cndmask_b32   %[ca], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[cs]\n\t"
        "v_not_b32       %[nn], %[n1]\n\t"
        "v_add_co_ci_u32 %[r1], vcc_lo, %[s], %[nn], vcc_lo\n\t"
        "v_cndmask_b32   %[cs], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[ca]\n\t"
        "v_add_co_ci_u32 %[s], vcc_lo, %[a2], %[b2], vcc_lo\n\t"
        "v_cndmask_b32   %[ca], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[cs]\n\t"
        "v_not_b32       %[nn], %[n2]\n\t"
        "v_add_co_ci_u32 %[r2], vcc_lo, %[s], %[nn], vcc_lo\n\t"
        "v_cndmask_b32   %[cs], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[ca]\n\t"
        "v_add_co_ci_u32 %[s], vcc_lo, %[a3], %[b3], vcc_lo\n\t"
        "v_cndmask_b32   %[ca], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[cs]\n\t"
        "v_not_b32       %[nn], %[n3]\n\t"
        "v_add_co_ci_u32 %[r3], vcc_lo, %[s], %[nn], vcc_lo\n\t"
        "v_cndmask_b32   %[cs], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[ca]\n\t"
        "v_add_co_ci_u32 %[s], vcc_lo, %[a4], %[b4], vcc_lo\n\t"
        "v_cndmask_b32   %[ca], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[cs]\n\t"
        "v_not_b32       %[nn], %[n4]\n\t"
        "v_add_co_ci_u32 %[r4], vcc_lo, %[s], %[nn], vcc_lo\n\t"
        "v_cndmask_b32   %[cs], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[ca]\n\t"
        "v_add_co_ci_u32 %[s], vcc_lo, %[a5], %[b5], vcc_lo\n\t"
        "v_cndmask_b32   %[ca], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[cs]\n\t"
        "v_not_b32       %[nn], %[n5]\n\t"
        "v_add_co_ci_u32 %[r5], vcc_lo, %[s], %[nn], vcc_lo\n\t"
        "v_cndmask_b32   %[cs], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[ca]\n\t"
        "v_add_co_ci_u32 %[s], vcc_lo, %[a6], %[b6], vcc_lo\n\t"
        "v_cndmask_b32   %[ca], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[cs]\n\t"
        "v_not_b32       %[nn], %[n6]\n\t"
        "v_add_co_ci_u32 %[r6], vcc_lo, %[s], %[nn], vcc_lo\n\t"
        "v_cndmask_b32   %[cs], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[ca]\n\t"
        "v_add_co_ci_u32 %[s], vcc_lo, %[a7], %[b7], vcc_lo\n\t"
        "v_cndmask_b32   %[ca], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[cs]\n\t"
        "v_not_b32       %[nn], %[n7]\n\t"
        "v_add_co_ci_u32 %[r7], vcc_lo, %[s], %[nn], vcc_lo\n\t"
        "v_cndmask_b32   %[cs], %[z], %[one], vcc_lo"
        : [r0] "=&v"(r0), [r1] "=&v"(r1), [r2] "=&v"(r2), [r3] "=&v"(r3), [r4] "=&v"(r4),
          [r5] "=&v"(r5), [r6] "=&v"(r6), [r7] "=&v"(r7), [ca] "=&v"(v_ca), [cs] "=&v"(v_cs),
          [s] "=&v"(s_tmp), [nn] "=&v"(nn_tmp)
        : [a0] "v"(a0), [a1] "v"(a1), [a2] "v"(a2), [a3] "v"(a3), [a4] "v"(a4), [a5] "v"(a5),
          [a6] "v"(a6), [a7] "v"(a7), [b0] "v"(b0), [b1] "v"(b1), [b2] "v"(b2), [b3] "v"(b3),
          [b4] "v"(b4), [b5] "v"(b5), [b6] "v"(b6), [b7] "v"(b7), [n0] "v"(n0), [n1] "v"(n1),
          [n2] "v"(n2), [n3] "v"(n3), [n4] "v"(n4), [n5] "v"(n5), [n6] "v"(n6), [n7] "v"(n7),
          [z] "v"(z), [one] "v"(one)
        : "vcc_lo");

    if ((v_ca | v_cs) == 0u) {
        c_fix_add_n8(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7, n0, n1, n2, n3, n4, n5, n6, n7);
    }

    r[0] = r0;
    r[1] = r1;
    r[2] = r2;
    r[3] = r3;
    r[4] = r4;
    r[5] = r5;
    r[6] = r6;
    r[7] = r7;
}

// Chunk wrapper: soft-switch with arbitrary ca_in/cs_in (for 4096 = 16 chunks).
__attribute__((always_inline)) static inline void asm_fused_block8_vccsoft(__global const uint *a,
                                                                           __global const uint *b,
                                                                           __global const uint *n,
                                                                           __global uint *r,
                                                                           uint ca_in,
                                                                           uint cs_in,
                                                                           uint *ca_out,
                                                                           uint *cs_out) {
    uint a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
    uint b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3], b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];
    uint n0 = n[0], n1 = n[1], n2 = n[2], n3 = n[3], n4 = n[4], n5 = n[5], n6 = n[6], n7 = n[7];
    uint r0, r1, r2, r3, r4, r5, r6, r7;
    uint v_ca = ca_in;
    uint v_cs = cs_in;
    uint s_tmp, nn_tmp;
    const uint z = 0u, one = 1u;

    __asm__ volatile(
        "v_cmp_eq_u32    vcc_lo, %[one], %[ca]\n\t"
        "v_add_co_ci_u32 %[s], vcc_lo, %[a0], %[b0], vcc_lo\n\t"
        "v_cndmask_b32   %[ca], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[cs]\n\t"
        "v_not_b32       %[nn], %[n0]\n\t"
        "v_add_co_ci_u32 %[r0], vcc_lo, %[s], %[nn], vcc_lo\n\t"
        "v_cndmask_b32   %[cs], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[ca]\n\t"
        "v_add_co_ci_u32 %[s], vcc_lo, %[a1], %[b1], vcc_lo\n\t"
        "v_cndmask_b32   %[ca], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[cs]\n\t"
        "v_not_b32       %[nn], %[n1]\n\t"
        "v_add_co_ci_u32 %[r1], vcc_lo, %[s], %[nn], vcc_lo\n\t"
        "v_cndmask_b32   %[cs], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[ca]\n\t"
        "v_add_co_ci_u32 %[s], vcc_lo, %[a2], %[b2], vcc_lo\n\t"
        "v_cndmask_b32   %[ca], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[cs]\n\t"
        "v_not_b32       %[nn], %[n2]\n\t"
        "v_add_co_ci_u32 %[r2], vcc_lo, %[s], %[nn], vcc_lo\n\t"
        "v_cndmask_b32   %[cs], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[ca]\n\t"
        "v_add_co_ci_u32 %[s], vcc_lo, %[a3], %[b3], vcc_lo\n\t"
        "v_cndmask_b32   %[ca], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[cs]\n\t"
        "v_not_b32       %[nn], %[n3]\n\t"
        "v_add_co_ci_u32 %[r3], vcc_lo, %[s], %[nn], vcc_lo\n\t"
        "v_cndmask_b32   %[cs], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[ca]\n\t"
        "v_add_co_ci_u32 %[s], vcc_lo, %[a4], %[b4], vcc_lo\n\t"
        "v_cndmask_b32   %[ca], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[cs]\n\t"
        "v_not_b32       %[nn], %[n4]\n\t"
        "v_add_co_ci_u32 %[r4], vcc_lo, %[s], %[nn], vcc_lo\n\t"
        "v_cndmask_b32   %[cs], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[ca]\n\t"
        "v_add_co_ci_u32 %[s], vcc_lo, %[a5], %[b5], vcc_lo\n\t"
        "v_cndmask_b32   %[ca], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[cs]\n\t"
        "v_not_b32       %[nn], %[n5]\n\t"
        "v_add_co_ci_u32 %[r5], vcc_lo, %[s], %[nn], vcc_lo\n\t"
        "v_cndmask_b32   %[cs], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[ca]\n\t"
        "v_add_co_ci_u32 %[s], vcc_lo, %[a6], %[b6], vcc_lo\n\t"
        "v_cndmask_b32   %[ca], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[cs]\n\t"
        "v_not_b32       %[nn], %[n6]\n\t"
        "v_add_co_ci_u32 %[r6], vcc_lo, %[s], %[nn], vcc_lo\n\t"
        "v_cndmask_b32   %[cs], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[ca]\n\t"
        "v_add_co_ci_u32 %[s], vcc_lo, %[a7], %[b7], vcc_lo\n\t"
        "v_cndmask_b32   %[ca], %[z], %[one], vcc_lo\n\t"
        "v_cmp_eq_u32    vcc_lo, %[one], %[cs]\n\t"
        "v_not_b32       %[nn], %[n7]\n\t"
        "v_add_co_ci_u32 %[r7], vcc_lo, %[s], %[nn], vcc_lo\n\t"
        "v_cndmask_b32   %[cs], %[z], %[one], vcc_lo"
        : [r0] "=&v"(r0), [r1] "=&v"(r1), [r2] "=&v"(r2), [r3] "=&v"(r3), [r4] "=&v"(r4),
          [r5] "=&v"(r5), [r6] "=&v"(r6), [r7] "=&v"(r7), [ca] "+v"(v_ca), [cs] "+v"(v_cs),
          [s] "=&v"(s_tmp), [nn] "=&v"(nn_tmp)
        : [a0] "v"(a0), [a1] "v"(a1), [a2] "v"(a2), [a3] "v"(a3), [a4] "v"(a4), [a5] "v"(a5),
          [a6] "v"(a6), [a7] "v"(a7), [b0] "v"(b0), [b1] "v"(b1), [b2] "v"(b2), [b3] "v"(b3),
          [b4] "v"(b4), [b5] "v"(b5), [b6] "v"(b6), [b7] "v"(b7), [n0] "v"(n0), [n1] "v"(n1),
          [n2] "v"(n2), [n3] "v"(n3), [n4] "v"(n4), [n5] "v"(n5), [n6] "v"(n6), [n7] "v"(n7),
          [z] "v"(z), [one] "v"(one)
        : "vcc_lo");

    if ((v_ca | v_cs) == 0u) {
        c_fix_add_n8(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7, n0, n1, n2, n3, n4, n5, n6, n7);
    }

    r[0] = r0;
    r[1] = r1;
    r[2] = r2;
    r[3] = r3;
    r[4] = r4;
    r[5] = r5;
    r[6] = r6;
    r[7] = r7;

    *ca_out = v_ca;
    *cs_out = v_cs;
}

#endif
