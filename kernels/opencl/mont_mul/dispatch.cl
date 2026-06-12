// Montgomery mul dispatch shell (implementation from loaded operator file).
static inline void mont_mul_stage1(uint *out, const uint *a, const uint *b,
                                   const uint *N, uint np0, uint limbs) {
    ecm_stage1_mont_mul(out, a, b, N, np0, limbs);
}
