static inline int mp_sub_mod(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {
    return ecm_stage1_sub_mod(r, a, b, N, limbs);
}
