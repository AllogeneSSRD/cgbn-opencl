// Stage1 Montgomery mul — 768-bit unroll-only.
static inline void mont_mul_unroll_768b(uint *out, const uint *a, const uint *b,
                                                   const uint *N, uint np0, uint limbs) {
    uint t[24u + 2u];
    #pragma unroll
    for (uint i = 0u; i < 26u; ++i) t[i] = 0u;
    uint B[24u];
    #pragma unroll
    for (uint j = 0u; j < 24u; ++j) B[j] = b[j];
    #pragma unroll
    for (uint i = 0u; i < 24u; ++i) {
        uint ai = a[i];
        ulong carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < 24u; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong top = (ulong)t[24u] + carry;
        t[24u] = (uint)top;
        t[25u] = (uint)(top >> 32);
        uint m = (uint)((ulong)t[0] * (ulong)np0);
        carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < 24u; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
            if (j > 0u) t[j - 1u] = (uint)uv;
            carry = uv >> 32;
        }
        top = (ulong)t[24u] + carry;
        t[23u] = (uint)top;
        top = (ulong)t[25u] + (top >> 32);
        t[24u] = (uint)top;
        t[25u] = (uint)(top >> 32);
    }
    ulong borrow = 0ul;
    uint D[24u];
    #pragma unroll
    for (uint i = 0u; i < 24u; ++i) {
        ulong tv = (ulong)t[i], nv = (ulong)N[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }
    uint need_sub = (t[24u] != 0u || t[25u] != 0u) ? 1u : 0u;
    need_sub = (borrow == 0u) ? 1u : need_sub;
    uint mask = 0u - need_sub;
    #pragma unroll
    for (uint i = 0u; i < 24u; ++i) out[i] = (D[i] & mask) | (t[i] & ~mask);
}

static inline void mont_sqr_unroll_768b(uint *out, const uint *a, const uint *N, uint np0, uint limbs) {
    mont_mul_unroll_768b(out, a, a, N, np0, limbs);
}
