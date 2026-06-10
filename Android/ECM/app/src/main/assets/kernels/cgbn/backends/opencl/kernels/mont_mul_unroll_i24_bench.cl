// ECM mont mul/sqr microbench — 24-bit limb unroll_i24 (mul24).

#include "mont_mul_unroll_i24.cl"

__kernel void ecm_mont_mul_unroll_i24_bench(__global const uint *a,
                                            __global const uint *b,
                                            __constant uint *n,
                                            __global uint *out,
                                            __constant uint *np0_ptr,
                                            uint limbs, uint iterations) {
    if (limbs != MONT_I24_LIMBS) {
        return;
    }
    const uint gid = get_global_id(0);
    const uint base = gid * limbs;
    const uint np0 = np0_ptr[0];
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_unroll_i24_body(out, a, b, n, base, np0);
        } else {
            mont_mul_unroll_i24_body(out, out, b, n, base, np0);
        }
    }
}

__kernel void ecm_mont_sqr_unroll_i24_bench(__global const uint *a,
                                              __constant uint *n,
                                              __global uint *out,
                                              __constant uint *np0_ptr,
                                              uint limbs, uint iterations) {
    if (limbs != MONT_I24_LIMBS) {
        return;
    }
    const uint gid = get_global_id(0);
    const uint base = gid * limbs;
    const uint np0 = np0_ptr[0];
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_sqr_unroll_i24_body(out, a, n, base, np0);
        } else {
            mont_sqr_unroll_i24_body(out, out, n, base, np0);
        }
    }
}
