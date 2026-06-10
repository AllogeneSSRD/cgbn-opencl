// ECM mont mul/sqr microbench — 24-bit limb unroll_i24 (mul24).
// L1 ulong | L2 u32 MAC | L4 branchless final sub (blsub)

#include "mont_mul_unroll_i24.cl"

#define ECM_MONT_I24_BENCH_KERNEL(MUL_NAME, BODY) \
    __kernel void MUL_NAME(__global const uint *a, __global const uint *b, \
                           __constant uint *n, __global uint *out, \
                           __constant uint *np0_ptr, uint limbs, uint iterations) { \
        if (limbs != MONT_I24_LIMBS) { \
            return; \
        } \
        const uint gid = get_global_id(0); \
        const uint base = gid * limbs; \
        const uint np0 = np0_ptr[0]; \
        for (uint it = 0u; it < iterations; ++it) { \
            if (it == 0u) { \
                BODY(out, a, b, n, base, np0); \
            } else { \
                BODY(out, out, b, n, base, np0); \
            } \
        } \
    }

#define ECM_MONT_I24_SQR_BENCH_KERNEL(SQR_NAME, BODY) \
    __kernel void SQR_NAME(__global const uint *a, __constant uint *n, __global uint *out, \
                           __constant uint *np0_ptr, uint limbs, uint iterations) { \
        if (limbs != MONT_I24_LIMBS) { \
            return; \
        } \
        const uint gid = get_global_id(0); \
        const uint base = gid * limbs; \
        const uint np0 = np0_ptr[0]; \
        for (uint it = 0u; it < iterations; ++it) { \
            if (it == 0u) { \
                BODY(out, a, n, base, np0); \
            } else { \
                BODY(out, out, n, base, np0); \
            } \
        } \
    }

ECM_MONT_I24_BENCH_KERNEL(ecm_mont_mul_unroll_i24_bench, mont_mul_unroll_i24_body)
ECM_MONT_I24_BENCH_KERNEL(ecm_mont_mul_unroll_i24_u32_bench, mont_mul_unroll_i24_u32_body)
ECM_MONT_I24_BENCH_KERNEL(ecm_mont_mul_unroll_i24_blsub_bench, mont_mul_unroll_i24_blsub_body)
ECM_MONT_I24_BENCH_KERNEL(ecm_mont_mul_unroll_i24_u32_blsub_bench, mont_mul_unroll_i24_u32_blsub_body)

ECM_MONT_I24_SQR_BENCH_KERNEL(ecm_mont_sqr_unroll_i24_bench, mont_sqr_unroll_i24_body)
ECM_MONT_I24_SQR_BENCH_KERNEL(ecm_mont_sqr_unroll_i24_u32_bench, mont_sqr_unroll_i24_u32_body)
ECM_MONT_I24_SQR_BENCH_KERNEL(ecm_mont_sqr_unroll_i24_blsub_bench, mont_sqr_unroll_i24_blsub_body)
ECM_MONT_I24_SQR_BENCH_KERNEL(ecm_mont_sqr_unroll_i24_u32_blsub_bench, mont_sqr_unroll_i24_u32_blsub_body)
