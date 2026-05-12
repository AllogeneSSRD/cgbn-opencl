// Minimal OpenCL kernels for cgbn backend testing.

__kernel void cgbn_set_ui32(__global uint *out, uint value, uint limbs) {
    for (uint i = 0; i < limbs; ++i) {
        out[i] = 0u;
    }

    if (limbs > 0u) {
        out[0] = value;
    }
}

__kernel void cgbn_bitwise_and(__global const uint *a, __global const uint *b, __global uint *out, uint limbs) {
    for (uint i = 0; i < limbs; ++i) {
        out[i] = a[i] & b[i];
    }
}

__kernel void cgbn_bitwise_ior(__global const uint *a, __global const uint *b, __global uint *out, uint limbs) {
    for (uint i = 0; i < limbs; ++i) {
        out[i] = a[i] | b[i];
    }
}

__kernel void cgbn_bitwise_xor(__global const uint *a, __global const uint *b, __global uint *out, uint limbs) {
    for (uint i = 0; i < limbs; ++i) {
        out[i] = a[i] ^ b[i];
    }
}

__kernel void cgbn_add_ui32(__global const uint *a, __global uint *out, uint value, uint limbs) {
    ulong carry = (ulong)value;

    for (uint i = 0; i < limbs; ++i) {
        ulong sum = (ulong)a[i] + carry;
        out[i] = (uint)sum;
        carry = sum >> 32;
    }
}

__kernel void cgbn_sub_ui32(__global const uint *a, __global uint *out, uint value, uint limbs) {
    ulong borrow = (ulong)value;

    for (uint i = 0; i < limbs; ++i) {
        ulong minuend = (ulong)a[i];
        ulong subtrahend = borrow;
        ulong wide = minuend - subtrahend;
        out[i] = (uint)wide;
        borrow = (minuend < subtrahend) ? 1u : 0u;
    }
}
