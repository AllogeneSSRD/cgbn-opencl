// Stage1 Montgomery mul — 512-bit unroll-only.
// Stage1-private Montgomery variants (private pointer ABI).
static inline void mont_mul_unroll_512b(uint *out, const uint *a, const uint *b,
                                                   const uint *N, uint np0, uint limbs) {
    uint t[16u + 2u];
    #pragma unroll
    for (uint i = 0u; i < 18u; ++i) t[i] = 0u;
    uint B[16u];
    #pragma unroll
    for (uint j = 0u; j < 16u; ++j) B[j] = b[j];
    #pragma unroll
    for (uint i = 0u; i < 16u; ++i) {
        uint ai = a[i];
        ulong carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < 16u; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong top = (ulong)t[16u] + carry;
        t[16u] = (uint)top;
        t[17u] = (uint)(top >> 32);
        uint m = (uint)((ulong)t[0] * (ulong)np0);
        carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < 16u; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
            if (j > 0u) t[j - 1u] = (uint)uv;
            carry = uv >> 32;
        }
        top = (ulong)t[16u] + carry;
        t[15u] = (uint)top;
        top = (ulong)t[17u] + (top >> 32);
        t[16u] = (uint)top;
        t[17u] = (uint)(top >> 32);
    }
    ulong borrow = 0ul;
    uint D[16u];
    #pragma unroll
    for (uint i = 0u; i < 16u; ++i) {
        ulong tv = (ulong)t[i], nv = (ulong)N[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }
    uint need_sub = (t[16u] != 0u || t[17u] != 0u) ? 1u : 0u;
    need_sub = (borrow == 0u) ? 1u : need_sub;
    uint mask = 0u - need_sub;
    #pragma unroll
    for (uint i = 0u; i < 16u; ++i) out[i] = (D[i] & mask) | (t[i] & ~mask);
}

static inline void mont_sqr_unroll_512b(uint *out, const uint *a, const uint *N, uint np0, uint limbs) {
    mont_mul_unroll_512b(out, a, a, N, np0, limbs);
}
