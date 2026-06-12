// Extracted stage1 add_mod/unroll_512b.cl
// 512-bit alias (16 limbs): same as add_mod_fused_unroll when MAX_LIMBS==16.
static inline static inline void add_mod_unroll_512b_body(uint *r, const uint *a, const uint *b,
                                                     const uint *N) {
    add_mod_fused_unroll(r, a, b, N);
}

static inline void add_mod_unroll_512b(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {
    add_mod_unroll_512b_body(r, a, b, N, limbs);
    (void)limbs;
}
