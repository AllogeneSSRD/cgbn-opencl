// Curve ladder helpers (independent of operator selection).
// 依赖 common/mp_priv.h.cl 中的 mp_ge / mp_sub_n 等基础 limb 原语。

static inline void mp_shift_left_1_mod(uint *r, const uint *a, const uint *N, uint limbs) {
    uint carry = 0u;
    for (uint i = 0u; i < limbs; ++i) {
        uint old = a[i];
        r[i] = (old << 1) | carry;
        carry = old >> 31;
    }
    if (carry || mp_ge(r, N, limbs)) {
        mp_sub_n(r, r, N, limbs);
    }
}

static inline void mont_normalize(uint *r, const uint *N, uint limbs) {
    if (mp_ge(r, N, limbs)) {
        mp_sub_n(r, r, N, limbs);
    }
}

static inline void maybe_mont_normalize(uint *r, const uint *N, uint limbs) {
#if ECM_STAGE1_FORCE_NORMALIZE
    mont_normalize(r, N, limbs);
#else
    (void)r;
    (void)N;
    (void)limbs;
#endif
}
