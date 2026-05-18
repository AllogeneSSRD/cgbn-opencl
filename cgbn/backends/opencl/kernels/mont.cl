// OpenCL kernels for Montgomery multiplication/squaring (CIOS style).
// One work-item handles one instance.

#ifndef MAX_LIMBS
#define MAX_LIMBS 128
#endif

__kernel void cgbn_mont_mul(
    __global const uint *a,
    __global const uint *b,
    __global const uint *n,
    __global uint *out,
    uint np0,
    uint limbs)
{
    uint idx = get_global_id(0);
    uint base = idx * limbs;

    if (limbs == 0u || limbs > MAX_LIMBS) {
        return;
    }

    uint t[MAX_LIMBS + 1];
    for (uint i = 0u; i <= limbs; ++i) {
        t[i] = 0u;
    }
    uint t_hi = 0u;

    // Cache b and n into private arrays to reduce global memory traffic
    uint B[MAX_LIMBS];
    uint N[MAX_LIMBS];
    for (uint j = 0u; j < limbs; ++j) {
        B[j] = b[base + j];
        N[j] = n[base + j];
    }

    // Classic CIOS (Column Operand Scanning) Montgomery multiplication
    for (uint i = 0u; i < limbs; ++i) {
        uint ai = a[base + i];

        // t += ai * B (use cached B)
        ulong carry = 0ul;
        #pragma unroll 4
        for (uint j = 0u; j < limbs; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong uvh = (ulong)t[limbs] + carry;
        t[limbs] = (uint)uvh;
        t_hi += (uint)(uvh >> 32);

        uint m = (uint)((ulong)t[0] * (ulong)np0);

        // t = (t + m*N) / 2^32 (use cached N)
        carry = 0ul;
        #pragma unroll 4
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

    // Conditional subtraction: if t >= n then t -= n
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
        out[base + i] = t[i];
    }
}

// ============================================================================
// Work-group parallel versions: multiple work-items cooperatively handle one instance
// TPI = Threads Per Instance (must divide limbs evenly, typically 4 or 8)
// ============================================================================

#ifndef TPI
#define TPI 4  // Default: 4 threads per instance
#endif

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
    __local uint *B = t + (MAX_LIMBS + 1);
    __local uint *N = B + MAX_LIMBS;
    
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
    
    // Thread 0 executes the full CIOS, other threads help with subtraction
    if (tid == 0u) {
        uint t_hi = 0u;
        
        // Main CIOS loop
        for (uint i = 0u; i < limbs; ++i) {
            uint ai = a[base + i];
            
            // Multiplication: t += ai * B
            ulong carry = 0ul;
            for (uint j = 0u; j < limbs; ++j) {
                ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
                t[j] = (uint)uv;
                carry = uv >> 32;
            }
            ulong uvh = (ulong)t[limbs] + carry;
            t[limbs] = (uint)uvh;
            t_hi += (uint)(uvh >> 32);
            
            // Reduction: m = t[0] * np0
            uint m = (uint)((ulong)t[0] * (ulong)np0);
            
            // t = (t + m*N) / 2^32
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
        
        // Store t_hi for conditional subtraction
        local_mem[(MAX_LIMBS + 1) * 3 + MAX_LIMBS * 2] = t_hi;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Conditional subtraction (all threads participate)
    uint t_hi = local_mem[(MAX_LIMBS + 1) * 3 + MAX_LIMBS * 2];
    int ge = (t_hi != 0u || t[limbs] != 0u) ? 1 : 0;
    if (!ge) {
        if (tid == 0u) {
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
            local_mem[(MAX_LIMBS + 1) * 3 + MAX_LIMBS * 2 + 1u] = (uint)ge;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        ge = (int)local_mem[(MAX_LIMBS + 1) * 3 + MAX_LIMBS * 2 + 1u];
    }
    
    if (ge) {
        ulong borrow = 0ul;
        for (uint i = tid * limbs_per_thread; i < (tid + 1u) * limbs_per_thread; ++i) {
            ulong tv = (ulong)t[i];
            ulong nv = (ulong)N[i];
            ulong w = tv - nv - borrow;
            t[i] = (uint)w;
            borrow = (tv < nv + borrow) ? 1ul : 0ul;
        }
        local_mem[(MAX_LIMBS + 1) * 3 + MAX_LIMBS * 2 + 2u + tid] = (uint)borrow;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Propagate borrow
        if (tid == 0u) {
            for (uint k = 1u; k < TPI; ++k) {
                if (local_mem[(MAX_LIMBS + 1) * 3 + MAX_LIMBS * 2 + 2u + k] != 0u) {
                    ulong w = (ulong)t[k * limbs_per_thread] - 1ul;
                    t[k * limbs_per_thread] = (uint)w;
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Write results (distributed)
    for (uint i = tid * limbs_per_thread; i < (tid + 1u) * limbs_per_thread; ++i) {
        out[base + i] = t[i];
    }
}

__kernel void cgbn_mont_sqr(
    __global const uint *a,
    __global const uint *n,
    __global uint *out,
    uint np0,
    uint limbs)
{
    uint idx = get_global_id(0);
    uint base = idx * limbs;

    if (limbs == 0u || limbs > MAX_LIMBS) {
        return;
    }

    uint t[MAX_LIMBS + 1];
    for (uint i = 0u; i <= limbs; ++i) {
        t[i] = 0u;
    }
    uint t_hi = 0u;

    // Cache a and n into private arrays to reduce global memory traffic (sqr uses a twice)
    uint A[MAX_LIMBS];
    uint N[MAX_LIMBS];
    for (uint j = 0u; j < limbs; ++j) {
        A[j] = a[base + j];
        N[j] = n[base + j];
    }

    for (uint i = 0u; i < limbs; ++i) {
        uint ai = A[i];

        // t += ai * a (use cached A)
        ulong carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < limbs; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)A[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong uvh = (ulong)t[limbs] + carry;
        t[limbs] = (uint)uvh;
        t_hi += (uint)(uvh >> 32);

        uint m = (uint)((ulong)t[0] * (ulong)np0);

        // t = (t + m*n) / 2^32 (use cached N)
        carry = 0ul;
        #pragma unroll
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
            uint nv = n[base + (uint)i];
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
            ulong nv = (ulong)n[base + i];
            ulong w = tv - nv - borrow;
            t[i] = (uint)w;
            borrow = (tv < nv + borrow) ? 1ul : 0ul;
        }
    }

    for (uint i = 0u; i < limbs; ++i) {
        out[base + i] = t[i];
    }
}
