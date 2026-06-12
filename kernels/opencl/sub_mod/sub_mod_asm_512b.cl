// AUTO-GENERATED -- asm sub_mod 512b (n16).
#if defined(__AMDGCN__)
static inline int sub_mod_asm_512b_body(uint *r, const uint *a, const uint *b, const uint *N) {
    uint br = 0u;
    asm_sub_fused_block16_priv(a, b, N, r, br, &br);
    return br != 0u ? 1 : 0;
}
#else
static inline int sub_mod_asm_512b_body(uint *r, const uint *a, const uint *b, const uint *N) {
    ulong br = 0ul;
    for (uint i = 0u; i < MAX_LIMBS; ++i) {
        ulong av = (ulong)a[i], bv = (ulong)b[i];
        ulong w = av - bv - br;
        r[i] = (uint)w;
        br = (av < bv + br) ? 1ul : 0ul;
    }
    if (br != 0ul) {
        ulong c = 0ul;
        for (uint i = 0u; i < MAX_LIMBS; ++i) {
            ulong s = (ulong)r[i] + (ulong)N[i] + c;
            r[i] = (uint)s;
            c = s >> 32;
        }
        return 1;
    }
    return 0;
}
#endif

static inline int sub_mod_asm_512b(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {
    if (limbs == 16u) { return sub_mod_asm_512b_body(r, a, b, N); }
    return 0;
}
