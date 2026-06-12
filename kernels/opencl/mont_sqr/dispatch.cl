// Montgomery sqr dispatch shell (implementation from loaded operator file).
static inline void mont_sqr_stage1(uint *out, const uint *a, const uint *N, uint np0,
                                   uint limbs) {
    ecm_stage1_mont_sqr(out, a, N, np0, limbs);
}
