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
// 2: parallel base terms + serial chunked merge
#define MONT_WG_IMPL 1
#endif

#ifndef MONT_WG_MERGE_CHUNK
#define MONT_WG_MERGE_CHUNK 32
#endif

#if MONT_WG_IMPL == 0
#define MONT_WG_SCRATCH_WORDS (MAX_LIMBS + 1u)
#elif MONT_WG_IMPL == 3
#define MONT_WG_SCRATCH_WORDS (3u * MAX_LIMBS + 1u + 4u * TPI)
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
#if MONT_WG_IMPL == 3
    __local uint *carry_in_lo = sum_hi + MAX_LIMBS;
    __local uint *carry_in_hi = carry_in_lo + TPI;
    __local uint *carry_out_lo = carry_in_hi + TPI;
    __local uint *carry_out_hi = carry_out_lo + TPI;
#endif
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
#if MONT_WG_IMPL == 2
            for (uint j0 = 0u; j0 < limbs; j0 += MONT_WG_MERGE_CHUNK) {
                uint j_end = min(limbs, j0 + (uint)MONT_WG_MERGE_CHUNK);
                for (uint j = j0; j < j_end; ++j) {
                    ulong uv = (ulong)sum_lo[j] + carry;
                    t[j] = (uint)uv;
                    carry = (uv >> 32) + (ulong)sum_hi[j];
                }
            }
#elif MONT_WG_IMPL == 3
            for (uint lane = 0u; lane < TPI; ++lane) {
                carry_in_lo[lane] = 0u;
                carry_in_hi[lane] = 0u;
                carry_out_lo[lane] = 0u;
                carry_out_hi[lane] = 0u;
            }
#else
            for (uint j = 0u; j < limbs; ++j) {
                ulong uv = (ulong)sum_lo[j] + carry;
                t[j] = (uint)uv;
                carry = (uv >> 32) + (ulong)sum_hi[j];
            }
#endif
            if (MONT_WG_IMPL != 3) {
                ulong uvh = (ulong)t[limbs] + carry;
                t[limbs] = (uint)uvh;
                t_hi += (uint)(uvh >> 32);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

#if MONT_WG_IMPL == 3
        {
            const uint chunk = limbs / TPI;
            const uint j_begin = tid * chunk;
            const uint j_end = j_begin + chunk;
            for (uint iter = 0u; iter < TPI; ++iter) {
                ulong carry = ((ulong)carry_in_hi[tid] << 32) | (ulong)carry_in_lo[tid];
                for (uint j = j_begin; j < j_end; ++j) {
                    ulong uv = (ulong)sum_lo[j] + carry;
                    t[j] = (uint)uv;
                    carry = (uv >> 32) + (ulong)sum_hi[j];
                }
                carry_out_lo[tid] = (uint)carry;
                carry_out_hi[tid] = (uint)(carry >> 32);
                barrier(CLK_LOCAL_MEM_FENCE);
                if (tid == 0u) {
                    carry_in_lo[0] = 0u;
                    carry_in_hi[0] = 0u;
                    for (uint lane = 1u; lane < TPI; ++lane) {
                        carry_in_lo[lane] = carry_out_lo[lane - 1u];
                        carry_in_hi[lane] = carry_out_hi[lane - 1u];
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (tid == 0u) {
                ulong carry = ((ulong)carry_out_hi[TPI - 1u] << 32) |
                              (ulong)carry_out_lo[TPI - 1u];
                ulong uvh = (ulong)t[limbs] + carry;
                t[limbs] = (uint)uvh;
                t_hi += (uint)(uvh >> 32);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#endif

        uint m = (uint)((ulong)t[0] * (ulong)np0);
        for (uint j = tid; j < limbs; j += TPI) {
            ulong base = (ulong)t[j] + (ulong)m * (ulong)N[j];
            sum_lo[j] = (uint)base;
            sum_hi[j] = (uint)(base >> 32);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (tid == 0u) {
            ulong carry = 0ul;
#if MONT_WG_IMPL == 2
            for (uint j0 = 0u; j0 < limbs; j0 += MONT_WG_MERGE_CHUNK) {
                uint j_end = min(limbs, j0 + (uint)MONT_WG_MERGE_CHUNK);
                for (uint j = j0; j < j_end; ++j) {
                    ulong uv = (ulong)sum_lo[j] + carry;
                    if (j > 0u) {
                        t[j - 1u] = (uint)uv;
                    }
                    carry = (uv >> 32) + (ulong)sum_hi[j];
                }
            }
#elif MONT_WG_IMPL == 3
            for (uint lane = 0u; lane < TPI; ++lane) {
                carry_in_lo[lane] = 0u;
                carry_in_hi[lane] = 0u;
                carry_out_lo[lane] = 0u;
                carry_out_hi[lane] = 0u;
            }
#else
            for (uint j = 0u; j < limbs; ++j) {
                ulong uv = (ulong)sum_lo[j] + carry;
                if (j > 0u) {
                    t[j - 1u] = (uint)uv;
                }
                carry = (uv >> 32) + (ulong)sum_hi[j];
            }
#endif
            if (MONT_WG_IMPL != 3) {
                ulong top = (ulong)t[limbs] + carry;
                t[limbs - 1u] = (uint)top;
                ulong top2 = (ulong)t_hi + (top >> 32);
                t[limbs] = (uint)top2;
                t_hi = (uint)(top2 >> 32);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

#if MONT_WG_IMPL == 3
        {
            const uint chunk = limbs / TPI;
            const uint j_begin = tid * chunk;
            const uint j_end = j_begin + chunk;
            for (uint iter = 0u; iter < TPI; ++iter) {
                ulong carry = ((ulong)carry_in_hi[tid] << 32) | (ulong)carry_in_lo[tid];
                for (uint j = j_begin; j < j_end; ++j) {
                    ulong uv = (ulong)sum_lo[j] + carry;
                    if (j > 0u) {
                        t[j - 1u] = (uint)uv;
                    }
                    carry = (uv >> 32) + (ulong)sum_hi[j];
                }
                carry_out_lo[tid] = (uint)carry;
                carry_out_hi[tid] = (uint)(carry >> 32);
                barrier(CLK_LOCAL_MEM_FENCE);
                if (tid == 0u) {
                    carry_in_lo[0] = 0u;
                    carry_in_hi[0] = 0u;
                    for (uint lane = 1u; lane < TPI; ++lane) {
                        carry_in_lo[lane] = carry_out_lo[lane - 1u];
                        carry_in_hi[lane] = carry_out_hi[lane - 1u];
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (tid == 0u) {
                ulong carry = ((ulong)carry_out_hi[TPI - 1u] << 32) |
                              (ulong)carry_out_lo[TPI - 1u];
                ulong top = (ulong)t[limbs] + carry;
                t[limbs - 1u] = (uint)top;
                ulong top2 = (ulong)t_hi + (top >> 32);
                t[limbs] = (uint)top2;
                t_hi = (uint)(top2 >> 32);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
    }
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



// Work-group parallel Montgomery squaring
__kernel void cgbn_mont_sqr_wg(
    __global const uint *a,
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
    __local uint *A = t + MONT_WG_SCRATCH_WORDS;
    __local uint *N = A + MAX_LIMBS;
    
    if (limbs == 0u || limbs > MAX_LIMBS || (limbs % TPI) != 0u) {
        return;
    }
    
    // Initialize t (all threads participate)
    for (uint i = tid; i <= limbs; i += TPI) {
        if (i <= MAX_LIMBS) t[i] = 0u;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Load A and N (distributed)
    for (uint i = tid; i < limbs; i += TPI) {
        A[i] = a[base + i];
        N[i] = n[base + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    cgbn_mont_sqr_wg_local_core(t, A, N, np0, limbs, tid, t);

    // Write results (distributed)
    for (uint i = tid * limbs_per_thread; i < (tid + 1u) * limbs_per_thread; ++i) {
        out[base + i] = t[i];
    }
}
