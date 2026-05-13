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
