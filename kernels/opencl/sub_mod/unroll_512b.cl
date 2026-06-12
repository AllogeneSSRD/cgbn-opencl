// Extracted stage1 sub_mod/unroll_512b.cl
static inline int mp_sub_mod_fused_unroll_b16_512(uint *r, const uint *a, const uint *b,
                                                  const uint *N) {
    return mp_sub_mod_fused_unroll(r, a, b, N);
}

static inline int ecm_stage1_sub_mod(uint *r, const uint *a, const uint *b,
                                       const uint *N, uint limbs) {
    if (limbs == 16u) { return mp_sub_mod_fused_unroll_b16_512(r, a, b, N); }
    return 0;
}
