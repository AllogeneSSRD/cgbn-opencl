// Benchmark wrappers for Montgomery work-group kernels.
// The heavy implementation lives in mont_wg.cl and is reused here.

#include "mont_wg.cl"

#ifndef TPI
#define TPI 4
#endif

#ifndef MAX_LIMBS
#define MAX_LIMBS 128
#endif

__kernel void ecm_mont_mul_priv_bench(
    __global const uint *a,
    __global const uint *b,
    __global const uint *n,
    __global uint *out,
    uint np0,
    uint limbs,
    uint iterations)
{
    uint gid = get_global_id(0);
    uint base = gid * limbs;

    uint x[MAX_LIMBS], y[MAX_LIMBS], m[MAX_LIMBS], r[MAX_LIMBS];
    for (uint i = 0u; i < limbs; ++i) {
        x[i] = a[base + i];
        y[i] = b[base + i];
        m[i] = n[base + i];
    }

    for (uint it = 0u; it < iterations; ++it) {
        mont_mul_priv(r, x, y, m, np0, limbs);
        mp_copy(x, r, limbs);
    }

    for (uint i = 0u; i < limbs; ++i) out[base + i] = x[i];
}

__kernel void ecm_mont_sqr_priv_bench(
    __global const uint *a,
    __global const uint *n,
    __global uint *out,
    uint np0,
    uint limbs,
    uint iterations)
{
    uint gid = get_global_id(0);
    uint base = gid * limbs;

    uint x[MAX_LIMBS], m[MAX_LIMBS], r[MAX_LIMBS];
    for (uint i = 0u; i < limbs; ++i) {
        x[i] = a[base + i];
        m[i] = n[base + i];
    }

    for (uint it = 0u; it < iterations; ++it) {
        mont_sqr_priv(r, x, m, np0, limbs);
        mp_copy(x, r, limbs);
    }

    for (uint i = 0u; i < limbs; ++i) out[base + i] = x[i];
}

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
