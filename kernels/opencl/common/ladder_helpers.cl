// Curve ladder helpers (independent of operator selection).

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

static inline uint mul_ui32_limbs(uint *r, uint m, uint limbs) {
    ulong carry = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong prod = (ulong)r[i] * (ulong)m + carry;
        r[i] = (uint)prod;
        carry = prod >> 32;
    }
    return (uint)carry;
}

static inline void shift_right_32_limbs(uint *r, uint limbs) {
    for (uint i = 0u; i + 1u < limbs; ++i) {
        r[i] = r[i + 1u];
    }
    r[limbs - 1u] = 0u;
}

static inline void special_mult_ui32(uint *r, uint m, const uint *N, uint np0, uint limbs) {
    uint carry_t1 = mul_ui32_limbs(r, m, limbs);
    uint t1_0 = r[0];
    uint q = (uint)((ulong)t1_0 * (ulong)np0);

    uint temp[MAX_LIMBS];
    mp_copy(temp, N, limbs);
    uint carry_t2 = mul_ui32_limbs(temp, q, limbs);

    shift_right_32_limbs(r, limbs);
    shift_right_32_limbs(temp, limbs);
    r[limbs - 1u] = carry_t1;
    temp[limbs - 1u] = carry_t2;

    {
        int carry_q = (int)mp_add_n(r, r, temp, limbs);
        if (t1_0 != 0u) {
            uint carry1 = 1u;
            for (uint i = 0u; i < limbs && carry1 != 0u; ++i) {
                ulong sum = (ulong)r[i] + (ulong)carry1;
                r[i] = (uint)sum;
                carry1 = (uint)(sum >> 32);
            }
            carry_q += (int)carry1;
        }
        if (carry_q > 0) {
            mp_sub_n(r, r, N, limbs);
        }
        if (mp_ge(r, N, limbs)) {
            mp_sub_n(r, r, N, limbs);
        }
    }
    (void)np0;
}

static inline void special_mult_stage1(uint *r, uint m, const uint *N, uint np0, uint limbs) {
    special_mult_ui32(r, m, N, np0, limbs);
}
