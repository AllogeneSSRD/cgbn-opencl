// Sliced CIOS Montgomery Multiplication — 32-lane cooperative.
// Verified against GMP for 1-10000 iterations on (2^991-1).
//
// Reliable instruction set (gfx1150 OpenCL 2.0):
//   ds_bpermute(i*4, val)         — constant-source broadcast (lane i → all)
//   ds_bpermute((lid-1)*4, v)     — carry chain left→right (reads left neighbor)
//   readfirstlane(val)            — m broadcast (lane 0 → all)
//   LDS + barrier                 — shift-right (lane j ← lane j+1)
//
// Architecture:
//   32-lane × 1-limb = 1024-bit Montgomery multiplication.
//   CIOS interleaving: product+reduce per outer iteration,
//   overflow carry consumed within same iteration via shift-right.
//   32 barriers total (1 per CIOS outer iteration).
//
// VGPR per lane: ~8 (T, A, B, N, carry, t32, t33, scratch).
// LDS: 34 uints (136 bytes).
//
// CRITICAL: my_T starts at 0u (standard CIOS accumulator convention).
// Initializing my_T = A[lid] adds A to product, causing double-count.

__kernel __attribute__((reqd_work_group_size(32u, 1u, 1u)))
void sliced_cios_mul(
    __global const uint *A, __global const uint *B,
    __global const uint *N, __global       uint *R,
    uint np0)
{
    const uint lid = get_local_id(0);
    __local uint L_T[34];  // L_T[0..31]=T[j], L_T[32]=t32, L_T[33]=t33

    uint my_T = 0u;         // CIOS accumulator — starts at ZERO
    uint my_A = A[lid];     // a[lane] — read-only, for broadcast
    uint my_B = B[lid];     // b[lane] — read-only
    uint my_N = N[lid];     // N[lane] — read-only
    uint t32  = 0u;         // overflow word (high half)
    uint t33  = 0u;         // carry from t32

    for (uint i = 0u; i < 32u; ++i) {
        // ═════  Phase 1: T += A[i] * B  ═════
        uint A_i = __builtin_amdgcn_ds_bpermute(i * 4u, my_A);
        uint carry = 0u, C_prod = 0u;
        for (uint k = 0u; k < 32u; ++k) {
            if (lid == k) {
                ulong uv = (ulong)my_T + (ulong)A_i * (ulong)my_B + carry;
                my_T  = (uint)uv;
                carry = (uint)(uv >> 32);
            }
            if (lid == 31u && k == 31u) C_prod = carry;
            carry = __builtin_amdgcn_ds_bpermute(((lid - 1u) & 31u) * 4u, carry);
        }
        { ulong t = (ulong)t32 + (ulong)C_prod; t32 = (uint)t; t33 += (uint)(t >> 32); }

        // ═════  Phase 2: T += m * N  ═════
        uint m = 0u;
        if (lid == 0u) m = (uint)((ulong)my_T * (ulong)np0);
        m = __builtin_amdgcn_readfirstlane(m);

        carry = 0u; uint C_red = 0u;
        for (uint k = 0u; k < 32u; ++k) {
            if (lid == k) {
                ulong uv = (ulong)my_T + (ulong)m * (ulong)my_N + carry;
                my_T  = (uint)uv;
                carry = (uint)(uv >> 32);
            }
            if (lid == 31u && k == 31u) C_red = carry;
            carry = __builtin_amdgcn_ds_bpermute(((lid - 1u) & 31u) * 4u, carry);
        }
        { ulong t = (ulong)t32 + (ulong)C_red; t32 = (uint)t; t33 += (uint)(t >> 32); }

        // ═════  Phase 3: T >>= 32 (shift-right via LDS)  ═════
        L_T[lid] = my_T;
        if (lid == 31u) { L_T[32u] = t32; L_T[33u] = t33; }
        barrier(CLK_LOCAL_MEM_FENCE);
        my_T = L_T[lid + 1u];
        t32  = L_T[33u];
        t33  = 0u;
    }

    // ═════  Conditional subtraction: if result >= N, subtract N  ═════
    {
        uint D = 0u, borrow = 0u;
        for (uint k = 0u; k < 32u; ++k) {
            if (lid == k) {
                ulong tv = (ulong)my_T, nv = (ulong)my_N;
                ulong w  = tv - nv - borrow;
                D      = (uint)w;
                borrow = (tv < nv + borrow) ? 1u : 0u;
            }
            borrow = __builtin_amdgcn_ds_bpermute(((lid - 1u) & 31u) * 4u, borrow);
        }
        uint need_sub = (borrow == 0u) ? 1u : 0u;
        need_sub = __builtin_amdgcn_readfirstlane(need_sub);
        uint mask = 0u - need_sub;
        R[lid] = (D & mask) | (my_T & ~mask);
    }
}
