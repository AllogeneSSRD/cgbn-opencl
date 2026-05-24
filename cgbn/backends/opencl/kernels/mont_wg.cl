// ============================================================================
// Work-group parallel versions: multiple work-items cooperatively handle one instance
// TPI = Threads Per Instance (must divide limbs evenly, typically 4 or 8)
// ============================================================================
#pragma once

#ifndef TPI
#define TPI 4  // Default: 4 threads per instance
#endif

#ifndef MAX_LIMBS
#define MAX_LIMBS 128
#endif

#ifndef MONT_WG_IMPL
// 0: legacy serial (tid0-only)
// 1: parallel base terms + serial full merge
// 4: parallel base terms + serial full merge (2-limb unroll)
#define MONT_WG_IMPL 1
#endif

#ifndef MONT_WG_IMPL4_UNROLL
// impl4 merge unroll factor:
// 1 = scalar merge (same order as impl1)
// 2 = 2-limb unrolled merge
#define MONT_WG_IMPL4_UNROLL 2
#endif

#if MONT_WG_IMPL == 0
#define MONT_WG_SCRATCH_WORDS (MAX_LIMBS + 1u)
#else
#define MONT_WG_SCRATCH_WORDS (3u * MAX_LIMBS + 1u)
#endif

// Shared local-array core for WG Montgomery.
// This is intended to be reused by ECM stage1 WG path so arithmetic code stays unified.
static inline void cgbn_mont_mul_wg_local_core(
    __local uint *out,
    __local const uint *a,
    __local const uint *b,
    __local const uint *N,
    uint np0,
    uint limbs,
    uint tid,
    __local uint *scratch)
{
    if (limbs == 0u || limbs > MAX_LIMBS || (limbs % TPI) != 0u) {
        return;
    }

    __local uint *t = scratch;
#if MONT_WG_IMPL != 0
    __local uint *sum_lo = t + (MAX_LIMBS + 1);
    __local uint *sum_hi = sum_lo + MAX_LIMBS;
#endif

    for (uint i = tid; i <= limbs; i += TPI) {
        if (i <= MAX_LIMBS) t[i] = 0u;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#if MONT_WG_IMPL == 0
    if (tid == 0u) {
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

        if (tid == 0u) {
            int ge = (t_hi != 0u || t[limbs] != 0u) ? 1 : 0;
            if (!ge) {
                for (int i = (int)limbs - 1; i >= 0; --i) {
                    if (t[(uint)i] > N[(uint)i]) {
                        ge = 1;
                        break;
                    }
                    if (t[(uint)i] < N[(uint)i]) {
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
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#else
    uint t_hi = 0u;
    for (uint i = 0u; i < limbs; ++i) {
        uint ai = a[i];
        // Phase A (parallelizable): base[j] = t[j] + ai*b[j]
        for (uint j = tid; j < limbs; j += TPI) {
            ulong base = (ulong)t[j] + (ulong)ai * (ulong)b[j];
            sum_lo[j] = (uint)base;
            sum_hi[j] = (uint)(base >> 32);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (tid == 0u) {
            ulong carry = 0ul;
#if MONT_WG_IMPL == 4
#if MONT_WG_IMPL4_UNROLL == 1
            for (uint j = 0u; j < limbs; ++j) {
                ulong uv = (ulong)sum_lo[j] + carry;
                t[j] = (uint)uv;
                carry = (uv >> 32) + (ulong)sum_hi[j];
            }
#else
            for (uint j = 0u; j + 1u < limbs; j += 2u) {
                ulong uv0 = (ulong)sum_lo[j] + carry;
                t[j] = (uint)uv0;
                carry = (uv0 >> 32) + (ulong)sum_hi[j];
                ulong uv1 = (ulong)sum_lo[j + 1u] + carry;
                t[j + 1u] = (uint)uv1;
                carry = (uv1 >> 32) + (ulong)sum_hi[j + 1u];
            }
            if (limbs & 1u) {
                uint j = limbs - 1u;
                ulong uv = (ulong)sum_lo[j] + carry;
                t[j] = (uint)uv;
                carry = (uv >> 32) + (ulong)sum_hi[j];
            }
#endif
#else
            for (uint j = 0u; j < limbs; ++j) {
                ulong uv = (ulong)sum_lo[j] + carry;
                t[j] = (uint)uv;
                carry = (uv >> 32) + (ulong)sum_hi[j];
            }
#endif
            {
                ulong uvh = (ulong)t[limbs] + carry;
                t[limbs] = (uint)uvh;
                t_hi += (uint)(uvh >> 32);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        uint m = (uint)((ulong)t[0] * (ulong)np0);
        for (uint j = tid; j < limbs; j += TPI) {
            ulong base = (ulong)t[j] + (ulong)m * (ulong)N[j];
            sum_lo[j] = (uint)base;
            sum_hi[j] = (uint)(base >> 32);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (tid == 0u) {
            ulong carry = 0ul;
#if MONT_WG_IMPL == 4
#if MONT_WG_IMPL4_UNROLL == 1
            for (uint j = 0u; j < limbs; ++j) {
                ulong uv = (ulong)sum_lo[j] + carry;
                if (j > 0u) {
                    t[j - 1u] = (uint)uv;
                }
                carry = (uv >> 32) + (ulong)sum_hi[j];
            }
#else
            for (uint j = 0u; j + 1u < limbs; j += 2u) {
                ulong uv0 = (ulong)sum_lo[j] + carry;
                if (j > 0u) {
                    t[j - 1u] = (uint)uv0;
                }
                carry = (uv0 >> 32) + (ulong)sum_hi[j];
                ulong uv1 = (ulong)sum_lo[j + 1u] + carry;
                t[j] = (uint)uv1;
                carry = (uv1 >> 32) + (ulong)sum_hi[j + 1u];
            }
            if (limbs & 1u) {
                uint j = limbs - 1u;
                ulong uv = (ulong)sum_lo[j] + carry;
                if (j > 0u) {
                    t[j - 1u] = (uint)uv;
                }
                carry = (uv >> 32) + (ulong)sum_hi[j];
            }
#endif
#else
            for (uint j = 0u; j < limbs; ++j) {
                ulong uv = (ulong)sum_lo[j] + carry;
                if (j > 0u) {
                    t[j - 1u] = (uint)uv;
                }
                carry = (uv >> 32) + (ulong)sum_hi[j];
            }
#endif
            {
                ulong top = (ulong)t[limbs] + carry;
                t[limbs - 1u] = (uint)top;
                ulong top2 = (ulong)t_hi + (top >> 32);
                t[limbs] = (uint)top2;
                t_hi = (uint)(top2 >> 32);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0u) {
        int ge = (t_hi != 0u || t[limbs] != 0u) ? 1 : 0;
        if (!ge) {
            for (int i = (int)limbs - 1; i >= 0; --i) {
                if (t[(uint)i] > N[(uint)i]) {
                    ge = 1;
                    break;
                }
                if (t[(uint)i] < N[(uint)i]) {
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
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    for (uint i = tid; i < limbs; i += TPI) {
        out[i] = t[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

static inline void cgbn_mont_sqr_wg_local_core(
    __local uint *out,
    __local const uint *a,
    __local const uint *N,
    uint np0,
    uint limbs,
    uint tid,
    __local uint *t)
{
    cgbn_mont_mul_wg_local_core(out, a, a, N, np0, limbs, tid, t);
}


// Work-group parallel Montgomery multiplication
// Launch with global size = (num_instances * TPI), local size divisible by TPI
__kernel void cgbn_mont_mul_wg(
    __global const uint *a,
    __global const uint *b,
    __global const uint *n,
    __global uint *out,
    uint np0,
    uint limbs,
    __local uint *local_mem)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    
    uint instance = gid / TPI;
    uint tid = gid % TPI;
    uint limbs_per_thread = limbs / TPI;
    uint base = instance * limbs;
    
    __local uint *t = local_mem;
    __local uint *B = t + MONT_WG_SCRATCH_WORDS;
    __local uint *N = B + MAX_LIMBS;
    __local uint *A = N + MAX_LIMBS;
    
    if (limbs == 0u || limbs > MAX_LIMBS || (limbs % TPI) != 0u) {
        return;
    }
    
    // Initialize t (all threads participate)
    for (uint i = tid; i <= limbs; i += TPI) {
        if (i <= MAX_LIMBS) t[i] = 0u;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Load B and N (distributed)
    for (uint i = tid; i < limbs; i += TPI) {
        B[i] = b[base + i];
        N[i] = n[base + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (uint i = tid; i < limbs; i += TPI) {
        A[i] = a[base + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    cgbn_mont_mul_wg_local_core(t, A, B, N, np0, limbs, tid, t);

    for (uint i = tid * limbs_per_thread; i < (tid + 1u) * limbs_per_thread; ++i) {
        out[base + i] = t[i];
    }
}



// Work-group parallel Montgomery squaring (same local layout as mul; core is mul(a,a)).
__kernel void cgbn_mont_sqr_wg(
    __global const uint *a,
    __global const uint *n,
    __global uint *out,
    uint np0,
    uint limbs,
    __local uint *local_mem)
{
    uint gid = get_global_id(0);
    uint instance = gid / TPI;
    uint tid = gid % TPI;
    uint limbs_per_thread = limbs / TPI;
    uint base = instance * limbs;

    __local uint *t = local_mem;
    __local uint *B = t + MONT_WG_SCRATCH_WORDS;
    __local uint *N = B + MAX_LIMBS;
    __local uint *A = N + MAX_LIMBS;

    if (limbs == 0u || limbs > MAX_LIMBS || (limbs % TPI) != 0u) {
        return;
    }

    for (uint i = tid; i <= limbs; i += TPI) {
        if (i <= MAX_LIMBS) t[i] = 0u;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = tid; i < limbs; i += TPI) {
        A[i] = a[base + i];
        B[i] = A[i];
        N[i] = n[base + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    cgbn_mont_mul_wg_local_core(t, A, B, N, np0, limbs, tid, t);

    for (uint i = tid * limbs_per_thread; i < (tid + 1u) * limbs_per_thread; ++i) {
        out[base + i] = t[i];
    }
}
