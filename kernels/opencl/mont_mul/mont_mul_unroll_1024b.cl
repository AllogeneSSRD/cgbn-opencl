// Stage1 Montgomery mul — 1024-bit unroll-only.
static inline void mont_mul_unroll_1024b(uint *out, const uint *a, const uint *b,
                                                   const uint *N, uint np0, uint limbs) {
    uint t[32u + 2u];
    #pragma unroll
    for (uint i = 0u; i < 34u; ++i) t[i] = 0u;
    uint B[32u];
    #pragma unroll
    for (uint j = 0u; j < 32u; ++j) B[j] = b[j];
    #pragma unroll
    for (uint i = 0u; i < 32u; ++i) {
        uint ai = a[i];
        ulong carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < 32u; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong top = (ulong)t[32u] + carry;
        t[32u] = (uint)top;
        t[33u] = (uint)(top >> 32);
        uint m = (uint)((ulong)t[0] * (ulong)np0);
        carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < 32u; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
            if (j > 0u) t[j - 1u] = (uint)uv;
            carry = uv >> 32;
        }
        top = (ulong)t[32u] + carry;
        t[31u] = (uint)top;
        top = (ulong)t[33u] + (top >> 32);
        t[32u] = (uint)top;
        t[33u] = (uint)(top >> 32);
    }
    ulong borrow = 0ul;
    uint D[32u];
    #pragma unroll
    for (uint i = 0u; i < 32u; ++i) {
        ulong tv = (ulong)t[i], nv = (ulong)N[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }
    uint need_sub = (t[32u] != 0u || t[33u] != 0u) ? 1u : 0u;
    need_sub = (borrow == 0u) ? 1u : need_sub;
    uint mask = 0u - need_sub;
    #pragma unroll
    for (uint i = 0u; i < 32u; ++i) out[i] = (D[i] & mask) | (t[i] & ~mask);
}

static inline void mont_sqr_unroll_1024b(uint *out, const uint *a, const uint *N, uint np0, uint limbs) {
    mont_mul_unroll_1024b(out, a, a, N, np0, limbs);
}
