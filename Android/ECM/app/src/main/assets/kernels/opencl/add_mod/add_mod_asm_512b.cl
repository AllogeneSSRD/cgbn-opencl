// AUTO-GENERATED -- asm add_mod 512b (n16).
#if defined(__AMDGCN__)
static inline void add_mod_asm_512b_body(uint *r, const uint *a, const uint *b, const uint *N) {
    uint ca = 0u, cs = 1u;
    asm_fused_block16_priv(a, b, N, r, ca, cs, &ca, &cs);
}
#else
static inline void add_mod_asm_512b_body(uint *r, const uint *a, const uint *b, const uint *N) {
    // non-AMD: use unrolled fused add-mod
    ulong carry_add = 0ul, carry_sub = 1ul;
    #pragma unroll 16
    for (uint i = 0u; i < MAX_LIMBS; ++i) {
        ulong sum = (ulong)a[i] + (ulong)b[i] + carry_add;
        carry_add = sum >> 32;
        ulong temp = (ulong)(uint)sum + (ulong)(~N[i]) + carry_sub;
        carry_sub = temp >> 32;
        r[i] = (uint)temp;
    }
    if ((carry_add | carry_sub) != 0ul) { return; }
    ulong c = 0ul;
    #pragma unroll 16
    for (uint i = 0u; i < MAX_LIMBS; ++i) {
        ulong s = (ulong)r[i] + (ulong)N[i] + c;
        r[i] = (uint)s;
        c = s >> 32;
    }
}
#endif

static inline void add_mod_asm_512b(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {
    if (limbs == 16u) { add_mod_asm_512b_body(r, a, b, N); }
    (void)limbs;
}
