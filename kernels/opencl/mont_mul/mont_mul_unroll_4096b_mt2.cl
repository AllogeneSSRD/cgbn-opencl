// Stage1 Montgomery mul — 4096-bit unroll64 MT2 local.
static inline void mont_mul_unroll_4096b_mt2(
    __local uint *out,
    __local const uint *a,
    __local const uint *b,
    __local const uint *N,
    uint np0,
    __local uint *local_mem,
    uint lid)
{
    __local uint *t = local_mem;
    __local uint *B = t + (MONT_FIXED_4096_LIMBS + 2u);
    __local uint *D = B + MONT_FIXED_4096_LIMBS;
    __local uint *meta = D + MONT_FIXED_4096_LIMBS;
    const uint half_words = MONT_FIXED_4096_LIMBS / 2u;
    const uint j_begin = lid * half_words;
    const uint j_end = j_begin + half_words;

    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS + 2u; ++i) {
            t[i] = 0u;
        }
    }

    #pragma unroll 64
    for (uint j = j_begin; j < j_end; ++j) {
        B[j] = b[j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        uint ai = a[i];

        if (lid == 0u) {
            ulong carry = 0ul;
            #pragma unroll 64
            for (uint j = 0u; j < half_words; ++j) {
                ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
                t[j] = (uint)uv;
                carry = uv >> 32;
            }
            meta[0] = (uint)carry;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid == 1u) {
            ulong carry = (ulong)meta[0];
            #pragma unroll 64
            for (uint j = half_words; j < MONT_FIXED_4096_LIMBS; ++j) {
                ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
                t[j] = (uint)uv;
                carry = uv >> 32;
            }
            ulong top = (ulong)t[MONT_FIXED_4096_LIMBS] + carry;
            t[MONT_FIXED_4096_LIMBS] = (uint)top;
            t[MONT_FIXED_4096_LIMBS + 1u] = (uint)(top >> 32);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid == 0u) {
            uint m = (uint)((ulong)t[0] * (ulong)np0);
            ulong uv0 = (ulong)t[0] + (ulong)m * (ulong)N[0];
            ulong carry = uv0 >> 32;
            #pragma unroll 64
            for (uint j = 1u; j < half_words; ++j) {
                ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
                t[j - 1u] = (uint)uv;
                carry = uv >> 32;
            }
            meta[0] = (uint)carry;
            meta[1] = m;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid == 1u) {
            uint m = meta[1];
            ulong carry = (ulong)meta[0];
            #pragma unroll 64
            for (uint j = half_words; j < MONT_FIXED_4096_LIMBS; ++j) {
                ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
                t[j - 1u] = (uint)uv;
                carry = uv >> 32;
            }
            ulong top = (ulong)t[MONT_FIXED_4096_LIMBS] + carry;
            t[MONT_FIXED_4096_LIMBS - 1u] = (uint)top;
            top = (ulong)t[MONT_FIXED_4096_LIMBS + 1u] + (top >> 32);
            t[MONT_FIXED_4096_LIMBS] = (uint)top;
            t[MONT_FIXED_4096_LIMBS + 1u] = (uint)(top >> 32);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0u) {
        ulong borrow = 0ul;
        #pragma unroll 64
        for (uint i = 0u; i < half_words; ++i) {
            ulong tv = (ulong)t[i];
            ulong nv = (ulong)N[i];
            ulong w = tv - nv - borrow;
            D[i] = (uint)w;
            borrow = (tv < nv + borrow) ? 1ul : 0ul;
        }
        meta[0] = (uint)borrow;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 1u) {
        ulong borrow = (ulong)meta[0];
        #pragma unroll 64
        for (uint i = half_words; i < MONT_FIXED_4096_LIMBS; ++i) {
            ulong tv = (ulong)t[i];
            ulong nv = (ulong)N[i];
            ulong w = tv - nv - borrow;
            D[i] = (uint)w;
            borrow = (tv < nv + borrow) ? 1ul : 0ul;
        }
        uint need_sub = (t[MONT_FIXED_4096_LIMBS] != 0u || t[MONT_FIXED_4096_LIMBS + 1u] != 0u) ? 1u : 0u;
        need_sub = (borrow == 0u) ? 1u : need_sub;
        meta[2] = need_sub;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint mask = 0u - meta[2];
    #pragma unroll 64
    for (uint i = j_begin; i < j_end; ++i) {
        out[i] = (D[i] & mask) | (t[i] & ~mask);
    }
}

static inline void mont_sqr_unroll_4096b_mt2(uint *out, const uint *a, const uint *N, uint np0, uint limbs) {
    mont_mul_unroll_4096b_mt2(out, a, a, N, np0, limbs);
}
