// Minimal OpenCL kernels for cgbn backend testing.

__kernel void cgbn_set_ui32(__global uint *out, uint value, uint limbs) {
    for (uint i = 0; i < limbs; ++i) {
        out[i] = 0u;
    }

    if (limbs > 0u) {
        out[0] = value;
    }
}

__kernel void cgbn_set_one(__global uint *out, uint limbs) {
    for (uint i = 0; i < limbs; ++i) {
        out[i] = 0u;
    }
    if (limbs > 0u) {
        out[0] = 1u;
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

__kernel void cgbn_bitwise_complement(__global const uint *a, __global uint *out, uint limbs) {
    for (uint i = 0; i < limbs; ++i) {
        out[i] = ~a[i];
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

__kernel void cgbn_is_zero(__global const uint *a, __global int *out, uint limbs) {
    int allZero = 1;
    for (uint i = 0; i < limbs; ++i) {
        if (a[i] != 0u) {
            allZero = 0;
            break;
        }
    }
    out[0] = allZero;
}

__kernel void cgbn_equals(__global const uint *a, __global const uint *b, __global int *out, uint limbs) {
    int eq = 1;
    for (uint i = 0; i < limbs; ++i) {
        if (a[i] != b[i]) {
            eq = 0;
            break;
        }
    }
    out[0] = eq;
}

__kernel void cgbn_compare(__global const uint *a, __global const uint *b, __global int *out, uint limbs) {
    int cmp = 0;
    for (int i = (int)limbs - 1; i >= 0; --i) {
        uint av = a[(uint)i];
        uint bv = b[(uint)i];
        if (av > bv) {
            cmp = 1;
            break;
        }
        if (av < bv) {
            cmp = -1;
            break;
        }
    }
    out[0] = cmp;
}

__kernel void cgbn_shift_left(__global const uint *a, __global uint *out, uint shift, uint limbs) {
    uint limbShift = shift / 32u;
    uint bitShift = shift % 32u;

    for (uint i = 0; i < limbs; ++i) {
        if (i < limbShift) {
            out[i] = 0u;
            continue;
        }

        uint src = i - limbShift;
        uint lo = a[src] << bitShift;
        uint hi = 0u;
        if (bitShift != 0u && src > 0u) {
            hi = a[src - 1u] >> (32u - bitShift);
        }
        out[i] = lo | hi;
    }
}

__kernel void cgbn_shift_right(__global const uint *a, __global uint *out, uint shift, uint limbs) {
    uint limbShift = shift / 32u;
    uint bitShift = shift % 32u;

    for (uint i = 0; i < limbs; ++i) {
        uint src = i + limbShift;
        if (src >= limbs) {
            out[i] = 0u;
            continue;
        }

        uint lo = a[src] >> bitShift;
        uint hi = 0u;
        if (bitShift != 0u && (src + 1u) < limbs) {
            hi = a[src + 1u] << (32u - bitShift);
        }
        out[i] = lo | hi;
    }
}
