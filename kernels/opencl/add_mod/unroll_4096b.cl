// Extracted stage1 add_mod/unroll_4096b.cl
static inline void mp_add_mod_fused_unroll_b32_4096(uint *r, const uint *a, const uint *b, const uint *N) {
    ulong carry_add = 0ul;
    ulong carry_sub = 1ul;

    #pragma unroll
    for (uint blk = 0u; blk < 4u; ++blk) {
        uint off = blk * 32u;
        #pragma unroll 32
        for (uint j = 0u; j < 32u; ++j) {
            uint i = off + j;
            ulong sum = (ulong)a[i] + (ulong)b[i] + carry_add;
            carry_add = sum >> 32;
            ulong temp = (ulong)(uint)sum + (ulong)(~N[i]) + carry_sub;
            carry_sub = temp >> 32;
            r[i] = (uint)temp;
        }
    }

    if ((carry_add | carry_sub) != 0ul) {
        return;
    }
    ulong c = 0ul;
    #pragma unroll 32
    for (uint i = 0u; i < 128u; ++i) {
        ulong s = (ulong)r[i] + (ulong)N[i] + c;
        r[i] = (uint)s;
        c = s >> 32;
    }
}

static inline void ecm_stage1_add_mod(uint *r, const uint *a, const uint *b,
                                        const uint *N, uint limbs) {
    if (limbs == 128u) { mp_add_mod_fused_unroll_b32_4096(r, a, b, N); return; }
    (void)limbs;
}
