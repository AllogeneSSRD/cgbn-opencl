// Benchmark wrappers for Montgomery work-group kernels.
// Core implementation lives in mont_wg.cl (used by ECM stage1).

#include "mont_wg.cl"

#ifndef TPI
#define TPI 4
#endif

#ifndef MAX_LIMBS
#define MAX_LIMBS 128
#endif

__kernel void cgbn_mont_mul_wg_bench(
    __global const uint *a,
    __global const uint *b,
    __global const uint *n,
    __global uint *out,
    uint np0,
    uint limbs,
    uint iterations,
    __local uint *local_mem)
{
    for (uint iter = 0u; iter < iterations; ++iter) {
        cgbn_mont_mul_wg(a, b, n, out, np0, limbs, local_mem);
    }
}

__kernel void cgbn_mont_sqr_wg_bench(
    __global const uint *a,
    __global const uint *n,
    __global uint *out,
    uint np0,
    uint limbs,
    uint iterations,
    __local uint *local_mem)
{
    for (uint iter = 0u; iter < iterations; ++iter) {
        cgbn_mont_sqr_wg(a, n, out, np0, limbs, local_mem);
    }
}
