// OpenCL kernels for multi-limb addition and subtraction

// Layout: numbers are stored as consecutive uint limbs in little-endian limb order.
// Each work-item processes one instance (one multi-limb integer pair).

__kernel void cgbn_add(
    __global const uint *a,
    __global const uint *b,
    __global uint *out,
    uint limbs)
{
    uint idx = get_global_id(0);
    uint base = idx * limbs;
    // Try a lightweight uint4 vectorized path when possible to reduce
    // global memory traffic. This keeps the arithmetic lane-serial so
    // correctness is preserved; vectorization only batches loads/stores.
    uint chunks = limbs / 4u;
    uint rem = limbs % 4u;

    if (chunks > 0u) {
        __global const uint4 *a4 = (__global const uint4 *)(a + base);
        __global const uint4 *b4 = (__global const uint4 *)(b + base);
        __global uint4 *out4 = (__global uint4 *)(out + base);

        ulong carry = 0UL;
        for (uint i = 0u; i < chunks; ++i) {
            uint4 av = a4[i];
            uint4 bv = b4[i];
            uint r0 = (uint)(((ulong)av.x + (ulong)bv.x + carry) & 0xFFFFFFFFUL);
            carry = (((ulong)av.x + (ulong)bv.x + carry) >> 32);
            uint r1 = (uint)(((ulong)av.y + (ulong)bv.y + carry) & 0xFFFFFFFFUL);
            carry = (((ulong)av.y + (ulong)bv.y + carry) >> 32);
            uint r2 = (uint)(((ulong)av.z + (ulong)bv.z + carry) & 0xFFFFFFFFUL);
            carry = (((ulong)av.z + (ulong)bv.z + carry) >> 32);
            uint r3 = (uint)(((ulong)av.w + (ulong)bv.w + carry) & 0xFFFFFFFFUL);
            carry = (((ulong)av.w + (ulong)bv.w + carry) >> 32);
            out4[i] = (uint4)(r0, r1, r2, r3);
        }

        // remainder scalar words
        for (uint i = chunks * 4u; i < limbs; ++i) {
            ulong av = (ulong)a[base + i];
            ulong bv = (ulong)b[base + i];
            ulong sum = av + bv + carry;
            out[base + i] = (uint)sum;
            carry = sum >> 32;
        }

        // r = r + a  (prevent optimization; propagate carry)
        carry = 0UL;
        for (uint i = 0u; i < chunks; ++i) {
            uint4 rv = out4[i];
            uint4 av = a4[i];
            uint s0 = (uint)(((ulong)rv.x + (ulong)av.x + carry) & 0xFFFFFFFFUL);
            carry = (((ulong)rv.x + (ulong)av.x + carry) >> 32);
            uint s1 = (uint)(((ulong)rv.y + (ulong)av.y + carry) & 0xFFFFFFFFUL);
            carry = (((ulong)rv.y + (ulong)av.y + carry) >> 32);
            uint s2 = (uint)(((ulong)rv.z + (ulong)av.z + carry) & 0xFFFFFFFFUL);
            carry = (((ulong)rv.z + (ulong)av.z + carry) >> 32);
            uint s3 = (uint)(((ulong)rv.w + (ulong)av.w + carry) & 0xFFFFFFFFUL);
            carry = (((ulong)rv.w + (ulong)av.w + carry) >> 32);
            out4[i] = (uint4)(s0, s1, s2, s3);
        }
        for (uint i = chunks * 4u; i < limbs; ++i) {
            ulong rv = (ulong)out[base + i];
            ulong av = (ulong)a[base + i];
            ulong sum2 = rv + av + carry;
            out[base + i] = (uint)sum2;
            carry = sum2 >> 32;
        }

    } else {
        // fallback scalar path
        ulong carry = 0UL;
        // r = a + b
        for (uint i = 0; i < limbs; ++i) {
            ulong av = (ulong)a[base + i];
            ulong bv = (ulong)b[base + i];
            ulong sum = av + bv + carry;
            out[base + i] = (uint)sum;
            carry = sum >> 32;
        }

        // r = r + a  (prevent optimization; propagate carry)
        carry = 0UL;
        for (uint i = 0; i < limbs; ++i) {
            ulong rv = (ulong)out[base + i];
            ulong av = (ulong)a[base + i];
            ulong sum2 = rv + av + carry;
            out[base + i] = (uint)sum2;
            carry = sum2 >> 32;
        }
    }
}

// Work-group-parallel version: each work-group handles one multi-limb instance.
// Each work-item handles a contiguous block of limbs; partial sums computed
// per-lane (with internal carry) are stored in local memory. Then a prefix
// scan on per-lane carry-outs computes incoming carries, which are applied
// to each lane and propagated within the lane. This reduces global-memory
// traffic and allows parallelism across blocks.
__kernel void cgbn_add_wg(
    __global const uint *a,
    __global const uint *b,
    __global uint *out,
    uint limbs,
    __local uint *local_r,
    __local uint *carry_flags)
{
    uint group = get_group_id(0);
    uint lid = get_local_id(0);
    uint lsize = get_local_size(0);
    uint base = group * limbs;

    __local uint *g_arr = carry_flags;
    __local uint *p_arr = carry_flags + lsize;

    // Fast path inspired by CUDA resolver: one lane per limb and (G,P)
    // prefix propagation across lanes. This keeps correctness while
    // enabling parallel carry handling.
    if (lsize >= limbs && limbs > 0u) {
        if (lid < limbs) {
            ulong av = (ulong)a[base + lid];
            ulong bv = (ulong)b[base + lid];
            ulong s = av + bv;
            local_r[lid] = (uint)s;
            g_arr[lid] = (uint)(s >> 32);
            p_arr[lid] = (local_r[lid] == 0xFFFFFFFFu) ? 1u : 0u;
        }
        if (lid >= limbs) {
            g_arr[lid] = 0u;
            p_arr[lid] = 0u;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Inclusive prefix on (G,P):
        // G' = G | (P & G_left), P' = P & P_left
        for (uint offset = 1u; offset < limbs; offset <<= 1u) {
            uint g_old = g_arr[lid];
            uint p_old = p_arr[lid];
            uint g_new = g_old;
            uint p_new = p_old;
            if (lid < limbs && lid >= offset) {
                uint gl = g_arr[lid - offset];
                uint pl = p_arr[lid - offset];
                g_new = (g_old | (p_old & gl)) & 1u;
                p_new = (p_old & pl) & 1u;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lid < limbs) {
                g_arr[lid] = g_new;
                p_arr[lid] = p_new;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid < limbs) {
            uint carry_in = (lid == 0u) ? 0u : g_arr[lid - 1u];
            local_r[lid] = local_r[lid] + carry_in;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Second pass: r = r + a, then same carry propagation.
        if (lid < limbs) {
            ulong rv = (ulong)local_r[lid];
            ulong av = (ulong)a[base + lid];
            ulong s2 = rv + av;
            local_r[lid] = (uint)s2;
            g_arr[lid] = (uint)(s2 >> 32);
            p_arr[lid] = (local_r[lid] == 0xFFFFFFFFu) ? 1u : 0u;
        }
        if (lid >= limbs) {
            g_arr[lid] = 0u;
            p_arr[lid] = 0u;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint offset = 1u; offset < limbs; offset <<= 1u) {
            uint g_old = g_arr[lid];
            uint p_old = p_arr[lid];
            uint g_new = g_old;
            uint p_new = p_old;
            if (lid < limbs && lid >= offset) {
                uint gl = g_arr[lid - offset];
                uint pl = p_arr[lid - offset];
                g_new = (g_old | (p_old & gl)) & 1u;
                p_new = (p_old & pl) & 1u;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lid < limbs) {
                g_arr[lid] = g_new;
                p_arr[lid] = p_new;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid < limbs) {
            uint carry_in = (lid == 0u) ? 0u : g_arr[lid - 1u];
            local_r[lid] = local_r[lid] + carry_in;
            out[base + lid] = local_r[lid];
        }
        return;
    }

    // Each lane copies a contiguous chunk of the instance into local memory
    uint block = (limbs + lsize - 1u) / lsize;
    uint start = lid * block;
    uint end = start + block;
    if (end > limbs) end = limbs;

    for (uint i = start; i < end; ++i) {
        local_r[i] = a[base + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Single serial thread applies full scalar add/sub to local_r to guarantee correctness
    if (lid == 0u) {
        // r = a + b
        ulong carry = 0UL;
        for (uint i = 0u; i < (uint)limbs; ++i) {
            ulong av = (ulong)local_r[i];
            ulong bv = (ulong)b[base + i];
            ulong sum = av + bv + carry;
            local_r[i] = (uint)sum;
            carry = sum >> 32;
        }
        // r = r + a  (prevent optimization; propagate carry)
        carry = 0UL;
        for (uint i = 0u; i < (uint)limbs; ++i) {
            ulong rv = (ulong)local_r[i];
            ulong av = (ulong)a[base + i];
            ulong sum2 = rv + av + carry;
            local_r[i] = (uint)sum2;
            carry = sum2 >> 32;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Write back
    for (uint i = start; i < end; ++i) {
        out[base + i] = local_r[i];
    }
}

__kernel void cgbn_sub_wg(
    __global const uint *a,
    __global const uint *b,
    __global uint *out,
    uint limbs,
    __local uint *local_r,
    __local uint *carry_flags)
{
    uint group = get_group_id(0);
    uint lid = get_local_id(0);
    uint lsize = get_local_size(0);
    uint base = group * limbs;

    uint block = (limbs + lsize - 1u) / lsize;
    uint start = lid * block;
    uint end = start + block;
    if (end > limbs) end = limbs;

    for (uint i = start; i < end; ++i) {
        local_r[i] = a[base + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0u) {
        // r = a - b
        ulong borrow = 0UL;
        for (uint i = 0u; i < (uint)limbs; ++i) {
            ulong av = (ulong)local_r[i];
            ulong bv = (ulong)b[base + i];
            ulong w = av - bv - borrow;
            local_r[i] = (uint)w;
            borrow = ((av < bv + borrow) ? 1UL : 0UL);
        }
        // r = r - a  (prevent optimization; propagate borrow)
        borrow = 0UL;
        for (uint i = 0u; i < (uint)limbs; ++i) {
            ulong rv = (ulong)local_r[i];
            ulong av = (ulong)a[base + i];
            ulong w2 = rv - av - borrow;
            local_r[i] = (uint)w2;
            borrow = ((rv < av + borrow) ? 1UL : 0UL);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = start; i < end; ++i) {
        out[base + i] = local_r[i];
    }
}

__kernel void cgbn_sub(
    __global const uint *a,
    __global const uint *b,
    __global uint *out,
    uint limbs)
{
    uint idx = get_global_id(0);
    uint base = idx * limbs;
    uint chunks = limbs / 4u;
    uint rem = limbs % 4u;

    if (chunks > 0u) {
        __global const uint4 *a4 = (__global const uint4 *)(a + base);
        __global const uint4 *b4 = (__global const uint4 *)(b + base);
        __global uint4 *out4 = (__global uint4 *)(out + base);

        ulong borrow = 0UL;
        // r = a - b
        for (uint i = 0u; i < chunks; ++i) {
            uint4 av = a4[i];
            uint4 bv = b4[i];
            ulong w0 = (ulong)av.x - (ulong)bv.x - borrow;
            uint r0 = (uint)w0;
            borrow = (((ulong)av.x < (ulong)bv.x + borrow) ? 1UL : 0UL);
            ulong w1 = (ulong)av.y - (ulong)bv.y - borrow;
            uint r1 = (uint)w1;
            borrow = (((ulong)av.y < (ulong)bv.y + borrow) ? 1UL : 0UL);
            ulong w2 = (ulong)av.z - (ulong)bv.z - borrow;
            uint r2 = (uint)w2;
            borrow = (((ulong)av.z < (ulong)bv.z + borrow) ? 1UL : 0UL);
            ulong w3 = (ulong)av.w - (ulong)bv.w - borrow;
            uint r3 = (uint)w3;
            borrow = (((ulong)av.w < (ulong)bv.w + borrow) ? 1UL : 0UL);
            out4[i] = (uint4)(r0, r1, r2, r3);
        }

        for (uint i = chunks * 4u; i < limbs; ++i) {
            ulong av = (ulong)a[base + i];
            ulong bv = (ulong)b[base + i];
            ulong wide = av - bv - borrow;
            out[base + i] = (uint)wide;
            borrow = ((av < bv + borrow) ? 1UL : 0UL);
        }

        // r = r - a  (prevent optimization; propagate borrow)
        borrow = 0UL;
        for (uint i = 0u; i < chunks; ++i) {
            uint4 rv = out4[i];
            uint4 av = a4[i];
            ulong w0 = (ulong)rv.x - (ulong)av.x - borrow;
            uint s0 = (uint)w0;
            borrow = (((ulong)rv.x < (ulong)av.x + borrow) ? 1UL : 0UL);
            ulong w1 = (ulong)rv.y - (ulong)av.y - borrow;
            uint s1 = (uint)w1;
            borrow = (((ulong)rv.y < (ulong)av.y + borrow) ? 1UL : 0UL);
            ulong w2 = (ulong)rv.z - (ulong)av.z - borrow;
            uint s2 = (uint)w2;
            borrow = (((ulong)rv.z < (ulong)av.z + borrow) ? 1UL : 0UL);
            ulong w3 = (ulong)rv.w - (ulong)av.w - borrow;
            uint s3 = (uint)w3;
            borrow = (((ulong)rv.w < (ulong)av.w + borrow) ? 1UL : 0UL);
            out4[i] = (uint4)(s0, s1, s2, s3);
        }
        for (uint i = chunks * 4u; i < limbs; ++i) {
            ulong rv = (ulong)out[base + i];
            ulong av = (ulong)a[base + i];
            ulong wide2 = rv - av - borrow;
            out[base + i] = (uint)wide2;
            borrow = ((rv < av + borrow) ? 1UL : 0UL);
        }
    } else {
        ulong borrow = 0UL;
        // r = a - b
        for (uint i = 0; i < limbs; ++i) {
            ulong av = (ulong)a[base + i];
            ulong bv = (ulong)b[base + i];
            ulong wide = av - bv - borrow;
            out[base + i] = (uint)wide;
            borrow = ((av < bv + borrow) ? 1UL : 0UL);
        }

        // r = r - a  (prevent optimization; propagate borrow)
        borrow = 0UL;
        for (uint i = 0; i < limbs; ++i) {
            ulong rv = (ulong)out[base + i];
            ulong av = (ulong)a[base + i];
            ulong wide2 = rv - av - borrow;
            out[base + i] = (uint)wide2;
            borrow = ((rv < av + borrow) ? 1UL : 0UL);
        }
    }
}
