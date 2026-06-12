// Stage1 Montgomery mul — 384-bit unroll-only.
// 384-bit CIOS: 12 active 32-bit limbs in a 16-limb private layout.
// Valid only when N + CARRY_BITS < 384 (host: opencl_ecm_stage1_n_fits_unroll384).
static inline void mont_mul_unroll_384b(uint *out, const uint *a, const uint *b,
                                                   const uint *N, uint np0, uint limbs) {
    uint t[ECM_STAGE1_384_LIMBS + 2u];
    #pragma unroll
    for (uint i = 0u; i < ECM_STAGE1_384_LIMBS + 2u; ++i) {
        t[i] = 0u;
    }
    uint B[ECM_STAGE1_512_CONTAINER_LIMBS];
    #pragma unroll
    for (uint j = 0u; j < ECM_STAGE1_512_CONTAINER_LIMBS; ++j) {
        B[j] = b[j];
    }

    #pragma unroll
    for (uint i = 0u; i < ECM_STAGE1_384_LIMBS; ++i) {
        uint ai = a[i];
        ulong carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < ECM_STAGE1_384_LIMBS; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong top = (ulong)t[ECM_STAGE1_384_LIMBS] + carry;
        t[ECM_STAGE1_384_LIMBS] = (uint)top;
        t[ECM_STAGE1_384_LIMBS + 1u] = (uint)(top >> 32);

        uint m = (uint)((ulong)t[0] * (ulong)np0);
        carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < ECM_STAGE1_384_LIMBS; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
            if (j > 0u) {
                t[j - 1u] = (uint)uv;
            }
            carry = uv >> 32;
        }
        top = (ulong)t[ECM_STAGE1_384_LIMBS] + carry;
        t[ECM_STAGE1_384_LIMBS - 1u] = (uint)top;
        top = (ulong)t[ECM_STAGE1_384_LIMBS + 1u] + (top >> 32);
        t[ECM_STAGE1_384_LIMBS] = (uint)top;
        t[ECM_STAGE1_384_LIMBS + 1u] = (uint)(top >> 32);
    }

    ulong borrow = 0ul;
    uint D[ECM_STAGE1_384_LIMBS];
    #pragma unroll
    for (uint i = 0u; i < ECM_STAGE1_384_LIMBS; ++i) {
        ulong tv = (ulong)t[i];
        ulong nv = (ulong)N[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }

    uint any_high =
        (t[ECM_STAGE1_384_LIMBS] | t[ECM_STAGE1_384_LIMBS + 1u]) != 0u;
    uint need_sub = any_high | (borrow == 0u);
    uint mask = 0u - need_sub;
    #pragma unroll
    for (uint i = 0u; i < ECM_STAGE1_384_LIMBS; ++i) {
        out[i] = (D[i] & mask) | (t[i] & ~mask);
    }
    #pragma unroll
    for (uint i = ECM_STAGE1_384_LIMBS; i < ECM_STAGE1_512_CONTAINER_LIMBS; ++i) {
        out[i] = 0u;
    }
}

static inline void mont_sqr_unroll_384b(uint *out, const uint *a, const uint *N, uint np0, uint limbs) {
    mont_mul_unroll_384b(out, a, a, N, np0, limbs);
}
