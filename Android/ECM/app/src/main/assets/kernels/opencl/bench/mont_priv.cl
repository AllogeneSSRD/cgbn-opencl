// Private Montgomery mul/sqr (CIOS). Core for ECM stage 1 (local/private pointers)
// and global __kernel entry points (benchmark / host enqueue).


#ifndef MAX_LIMBS
#define MAX_LIMBS 128
#endif

// CIOS on private/local operand pointers (ecm_stage1 local arrays).
void mont_mul_priv(uint *out, const uint *a, const uint *b, const uint *N, uint np0, uint limbs) {
    if (limbs == 0u || limbs > MAX_LIMBS) {
        return;
    }

    uint t[MAX_LIMBS + 1];
    for (uint i = 0u; i <= limbs; ++i) {
        t[i] = 0u;
    }
    uint t_hi = 0u;

    for (uint i = 0u; i < limbs; ++i) {
        uint ai = a[i];
        ulong carry = 0ul;
        for (uint j = 0u; j < limbs; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)b[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong uvh = (ulong)t[limbs] + carry;
        t[limbs] = (uint)uvh;
        t_hi += (uint)(uvh >> 32);

        uint m = (uint)((ulong)t[0] * (ulong)np0);
        carry = 0ul;
        for (uint j = 0u; j < limbs; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
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
            uint nv = N[(uint)i];
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
            ulong nv = (ulong)N[i];
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

// Streaming CIOS from __global operands; only t[MAX_LIMBS+1] in private memory.
static inline void mont_mul_priv_global_core(__global uint *out, __global const uint *a,
                                             __global const uint *b, __global const uint *N,
                                             uint base, uint np0, uint limbs) {
    if (limbs == 0u || limbs > MAX_LIMBS) {
        return;
    }

    uint t[MAX_LIMBS + 1];
    for (uint i = 0u; i <= limbs; ++i) {
        t[i] = 0u;
    }
    uint t_hi = 0u;

    for (uint i = 0u; i < limbs; ++i) {
        uint ai = a[base + i];
        ulong carry = 0ul;
        for (uint j = 0u; j < limbs; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)b[base + j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong uvh = (ulong)t[limbs] + carry;
        t[limbs] = (uint)uvh;
        t_hi += (uint)(uvh >> 32);

        uint m = (uint)((ulong)t[0] * (ulong)np0);
        carry = 0ul;
        for (uint j = 0u; j < limbs; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[base + j] + carry;
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
            uint nv = N[base + (uint)i];
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
            ulong nv = (ulong)N[base + i];
            ulong w = tv - nv - borrow;
            t[i] = (uint)w;
            borrow = (tv < nv + borrow) ? 1ul : 0ul;
        }
    }
    for (uint i = 0u; i < limbs; ++i) {
        out[base + i] = t[i];
    }
}

static inline void mont_sqr_priv_global_core(__global uint *out, __global const uint *a,
                                             __global const uint *N, uint base, uint np0,
                                             uint limbs) {
    mont_mul_priv_global_core(out, a, a, N, base, np0, limbs);
}

// Global __kernel entry points (one work-item per instance), mirroring cgbn_mont_*_wg.
__kernel void cgbn_mont_mul_priv(__global const uint *a, __global const uint *b,
                                 __global const uint *n, __global uint *out, uint np0,
                                 uint limbs) {
    uint gid = get_global_id(0);
    uint base = gid * limbs;
    mont_mul_priv_global_core(out, a, b, n, base, np0, limbs);
}

__kernel void cgbn_mont_sqr_priv(__global const uint *a, __global const uint *n,
                                 __global uint *out, uint np0, uint limbs) {
    uint gid = get_global_id(0);
    uint base = gid * limbs;
    mont_sqr_priv_global_core(out, a, n, base, np0, limbs);
}
