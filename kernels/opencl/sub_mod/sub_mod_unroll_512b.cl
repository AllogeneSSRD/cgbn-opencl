// Extracted stage1 sub_mod/unroll_512b.cl
static inline static inline int sub_mod_unroll_512b_body(uint *r, const uint *a, const uint *b,
                                                  const uint *N) {
    return sub_mod_fused_unroll(r, a, b, N);
}

static inline int sub_mod_unroll_512b(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {
    return sub_mod_unroll_512b_body(r, a, b, N, limbs);
}
