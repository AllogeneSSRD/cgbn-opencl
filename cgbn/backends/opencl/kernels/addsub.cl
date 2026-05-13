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

    ulong carry = 0UL;
    for (uint i = 0; i < limbs; ++i) {
        ulong av = (ulong)a[base + i];
        ulong bv = (ulong)b[base + i];
        ulong sum = av + bv + carry;
        out[base + i] = (uint)sum;
        carry = sum >> 32;
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

    ulong borrow = 0UL;
    for (uint i = 0; i < limbs; ++i) {
        ulong av = (ulong)a[base + i];
        ulong bv = (ulong)b[base + i];
        ulong wide = av - bv - borrow;
        out[base + i] = (uint)wide;
        borrow = ((av < bv + borrow) ? 1UL : 0UL);
    }
}
