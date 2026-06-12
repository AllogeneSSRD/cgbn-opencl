static inline void mp_add_mod(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {
    ecm_stage1_add_mod(r, a, b, N, limbs);
}
