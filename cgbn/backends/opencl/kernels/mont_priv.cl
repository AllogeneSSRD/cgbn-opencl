// Private Montgomery mul/sqr (CIOS), extracted from mont.cl for ECM stage 1.

#ifndef MAX_LIMBS
#define MAX_LIMBS 128
#endif

void mont_mul_priv(uint *out, const uint *a, const uint *b, const uint *N, uint np0, uint limbs) {
    if (limbs == 0u || limbs > MAX_LIMBS) {
        return;
    }

    uint t[MAX_LIMBS + 1];
    for (uint i = 0u; i <= limbs; ++i) {
        t[i] = 0u;
    }
    uint t_hi = 0u;

    uint B[MAX_LIMBS];
    uint Nloc[MAX_LIMBS];
    for (uint j = 0u; j < limbs; ++j) {
        B[j] = b[j];
        Nloc[j] = N[j];
    }

    for (uint i = 0u; i < limbs; ++i) {
        uint ai = a[i];
        ulong carry = 0ul;
        for (uint j = 0u; j < limbs; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong uvh = (ulong)t[limbs] + carry;
        t[limbs] = (uint)uvh;
        t_hi += (uint)(uvh >> 32);

        uint m = (uint)((ulong)t[0] * (ulong)np0);
        carry = 0ul;
        for (uint j = 0u; j < limbs; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)Nloc[j] + carry;
            if (j > 0u) {
                t[j - 1u] = (uint)uv;
            }
            carry = uv >> 32;
        }
        ulong top = (ulong)t[limbs] + carry;
        t[limbs - 1u] = (uint)top;
        ulong top2 = (ulong)t_hi + (top >> 32);
        t[limbs] = (uint)top2;
        t_hi = (uint)(top2 >> 32);
    }

    int ge = (t_hi != 0u || t[limbs] != 0u) ? 1 : 0;
    if (!ge) {
        for (int i = (int)limbs - 1; i >= 0; --i) {
            uint tv = t[(uint)i];
            uint nv = Nloc[(uint)i];
            if (tv > nv) {
                ge = 1;
                break;
            }
            if (tv < nv) {
                ge = 0;
                break;
            }
        }
    }
    if (ge) {
        ulong borrow = 0ul;
        for (uint i = 0u; i < limbs; ++i) {
            ulong tv = (ulong)t[i];
            ulong nv = (ulong)Nloc[i];
            ulong w = tv - nv - borrow;
            t[i] = (uint)w;
            borrow = (tv < nv + borrow) ? 1ul : 0ul;
        }
    }
    for (uint i = 0u; i < limbs; ++i) {
        out[i] = t[i];
    }
}

void mont_sqr_priv(uint *out, const uint *a, const uint *N, uint np0, uint limbs) {
    mont_mul_priv(out, a, a, N, np0, limbs);
}
