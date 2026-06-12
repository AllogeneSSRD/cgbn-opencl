// Generic modular sub (scalar fallback).
static inline int ecm_stage1_sub_mod(uint *r, const uint *a, const uint *b, const uint *N,
                                     uint limbs) {
    ulong borrow = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong av = (ulong)a[i];
        ulong bv = (ulong)b[i];
        ulong w = av - bv - borrow;
        r[i] = (uint)w;
        borrow = (av < bv + borrow) ? 1ul : 0ul;
    }
    if (borrow) {
        (void)mp_add_n(r, r, N, limbs);
        return 1;
    }
    return 0;
}
