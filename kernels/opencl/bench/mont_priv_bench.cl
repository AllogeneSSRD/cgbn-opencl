// Benchmark wrappers for private Montgomery kernels.
// Core + global kernels live in mont_priv.cl (used by ECM stage1 via mont_mul_priv only).

#include "mont_priv.cl"

#ifndef MAX_LIMBS
#define MAX_LIMBS 128
#endif

__kernel void ecm_mont_mul_priv_bench(__global const uint *a, __global const uint *b,
                                      __global const uint *n, __global uint *out, uint np0,
                                      uint limbs, uint iterations) {
    uint gid = get_global_id(0);
    uint base = gid * limbs;

    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_priv_global_core(out, a, b, n, base, np0, limbs);
        } else {
            mont_mul_priv_global_core(out, out, b, n, base, np0, limbs);
        }
    }
}

__kernel void ecm_mont_sqr_priv_bench(__global const uint *a, __global const uint *n,
                                      __global uint *out, uint np0, uint limbs,
                                      uint iterations) {
    uint gid = get_global_id(0);
    uint base = gid * limbs;

    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_sqr_priv_global_core(out, a, n, base, np0, limbs);
        } else {
            mont_mul_priv_global_core(out, out, out, n, base, np0, limbs);
        }
    }
}
