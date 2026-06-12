// Extracted stage1 add_mod/unroll_512b.cl
// 512-bit alias (16 limbs): same as mp_add_mod_fused_unroll when MAX_LIMBS==16.
static inline void mp_add_mod_fused_unroll_b16_512(uint *r, const uint *a, const uint *b,
                                                     const uint *N) {
    mp_add_mod_fused_unroll(r, a, b, N);
}

static inline void ecm_stage1_add_mod(uint *r, const uint *a, const uint *b,
                                        const uint *N, uint limbs) {
    if (limbs == 16u) { mp_add_mod_fused_unroll_b16_512(r, a, b, N); return; }
    (void)limbs;
}
