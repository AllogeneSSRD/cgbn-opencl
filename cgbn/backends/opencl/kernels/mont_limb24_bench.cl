// ECM mont mul/sqr microbench — 24-bit limb unroll_only_512 (mul24).

#include "mont_limb24_mul.cl"

__kernel void ecm_mont_mul_priv_unroll_only_512_limb24_bench(__global const uint *a,
                                                             __global const uint *b,
                                                             __constant uint *n,
                                                             __global uint *out,
                                                             __constant uint *np0_ptr,
                                                             uint limbs, uint iterations) {
    if (limbs != MONT_LIMB24_UNROLL512_LIMBS) {
        return;
    }
    const uint gid = get_global_id(0);
    const uint base = gid * limbs;
    const uint np0 = np0_ptr[0];
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_priv_unroll_only_512_limb24_body(out, a, b, n, base, np0);
        } else {
            mont_mul_priv_unroll_only_512_limb24_body(out, out, b, n, base, np0);
        }
    }
}

__kernel void ecm_mont_sqr_priv_unroll_only_512_limb24_bench(__global const uint *a,
                                                               __constant uint *n,
                                                               __global uint *out,
                                                               __constant uint *np0_ptr,
                                                               uint limbs, uint iterations) {
    if (limbs != MONT_LIMB24_UNROLL512_LIMBS) {
        return;
    }
    const uint gid = get_global_id(0);
    const uint base = gid * limbs;
    const uint np0 = np0_ptr[0];
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_sqr_priv_unroll_only_512_limb24_body(out, a, n, base, np0);
        } else {
            mont_sqr_priv_unroll_only_512_limb24_body(out, out, n, base, np0);
        }
    }
}
