// Benchmark wrappers for mont_priv_opt (constant-cache N/np0 variant).

#include "mont_priv_opt.cl"

#ifndef MAX_LIMBS
#define MAX_LIMBS 128
#endif

__kernel void ecm_mont_mul_priv_opt_bench(__global const uint *a, __global const uint *b,
                                          __constant uint *n, __global uint *out,
                                          __constant uint *np0_ptr, uint limbs,
                                          uint iterations) {
    uint gid = get_global_id(0);
    uint base = gid * limbs;
    uint np0 = np0_ptr[0];

    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_priv_opt_core(out, a, b, n, base, np0, limbs);
        } else {
            mont_mul_priv_opt_core(out, out, b, n, base, np0, limbs);
        }
    }
}

__kernel void ecm_mont_sqr_priv_opt_bench(__global const uint *a, __constant uint *n,
                                          __global uint *out, __constant uint *np0_ptr,
                                          uint limbs, uint iterations) {
    uint gid = get_global_id(0);
    uint base = gid * limbs;
    uint np0 = np0_ptr[0];

    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_sqr_priv_opt_core(out, a, n, base, np0, limbs);
        } else {
            mont_sqr_priv_opt_core(out, out, n, base, np0, limbs);
        }
    }
}

#if MAX_LIMBS == 16

__kernel void ecm_mont_mul_priv_opt2_512_local_bench(__global const uint *a, __global const uint *b,
                                                      __constant uint *n, __global uint *out,
                                                      __constant uint *np0_ptr, uint limbs,
                                                      uint iterations, __local uint *local_mem) {
    if (limbs != MONT_OPT2_FIXED_LIMBS) {
        return;
    }
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint lsize = get_local_size(0);
    uint base = gid * limbs;
    uint np0 = np0_ptr[0];

    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_priv_opt2_512_local_body(out, a, b, n, base, np0, local_mem, lid, lsize);
        } else {
            mont_mul_priv_opt2_512_local_body(out, out, b, n, base, np0, local_mem, lid, lsize);
        }
    }
}

__kernel void ecm_mont_sqr_priv_opt2_512_local_bench(__global const uint *a, __constant uint *n,
                                                      __global uint *out, __constant uint *np0_ptr,
                                                      uint limbs, uint iterations, __local uint *local_mem) {
    if (limbs != MONT_OPT2_FIXED_LIMBS) {
        return;
    }
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint lsize = get_local_size(0);
    uint base = gid * limbs;
    uint np0 = np0_ptr[0];

    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_sqr_priv_opt2_512_local_body(out, a, n, base, np0, local_mem, lid, lsize);
        } else {
            mont_sqr_priv_opt2_512_local_body(out, out, n, base, np0, local_mem, lid, lsize);
        }
    }
}

__kernel void ecm_mont_mul_priv_unroll_only_512_bench(__global const uint *a, __global const uint *b,
                                                       __constant uint *n, __global uint *out,
                                                       __constant uint *np0_ptr, uint limbs,
                                                       uint iterations) {
    if (limbs != MONT_OPT2_FIXED_LIMBS) {
        return;
    }
    uint gid = get_global_id(0);
    uint base = gid * limbs;
    uint np0 = np0_ptr[0];
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_priv_unroll_only_512_body(out, a, b, n, base, np0);
        } else {
            mont_mul_priv_unroll_only_512_body(out, out, b, n, base, np0);
        }
    }
}

__kernel void ecm_mont_mul_priv_unroll_only_512_manual_bench(__global const uint *a,
                                                             __global const uint *b,
                                                             __constant uint *n,
                                                             __global uint *out,
                                                             __constant uint *np0_ptr,
                                                             uint limbs, uint iterations) {
    if (limbs != MONT_OPT2_FIXED_LIMBS) {
        return;
    }
    uint gid = get_global_id(0);
    uint base = gid * limbs;
    uint np0 = np0_ptr[0];
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_priv_unroll_only_512_manual_body(out, a, b, n, base, np0);
        } else {
            mont_mul_priv_unroll_only_512_manual_body(out, out, b, n, base, np0);
        }
    }
}

__kernel void ecm_mont_sqr_priv_unroll_only_512_bench(__global const uint *a, __constant uint *n,
                                                       __global uint *out, __constant uint *np0_ptr,
                                                       uint limbs, uint iterations) {
    if (limbs != MONT_OPT2_FIXED_LIMBS) {
        return;
    }
    uint gid = get_global_id(0);
    uint base = gid * limbs;
    uint np0 = np0_ptr[0];
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_sqr_priv_unroll_only_512_body(out, a, n, base, np0);
        } else {
            mont_sqr_priv_unroll_only_512_body(out, out, n, base, np0);
        }
    }
}

__kernel void ecm_mont_mul_priv_fips512_bench(__global const uint *a, __global const uint *b,
                                               __constant uint *n, __global uint *out,
                                               __constant uint *np0_ptr, uint limbs,
                                               uint iterations) {
    if (limbs != MONT_OPT2_FIXED_LIMBS) return;
    uint gid = get_global_id(0), base = gid * limbs, np0 = np0_ptr[0];
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_priv_fips512_body(out, a, b, n, base, np0);
        } else {
            mont_mul_priv_fips512_body(out, out, b, n, base, np0);
        }
    }
}

__kernel void ecm_mont_sqr_priv_fips512_bench(__global const uint *a, __constant uint *n,
                                              __global uint *out, __constant uint *np0_ptr,
                                              uint limbs, uint iterations) {
    if (limbs != MONT_OPT2_FIXED_LIMBS) return;
    uint gid = get_global_id(0), base = gid * limbs, np0 = np0_ptr[0];
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_sqr_priv_fips512_body(out, a, n, base, np0);
        } else {
            mont_sqr_priv_fips512_body(out, out, n, base, np0);
        }
    }
}

#define FIPS512_MT_LOCAL_U32 (MONT_OPT2_FIXED_LIMBS + MONT_OPT2_FIXED_LIMBS + FIPS512_MT_PROD_WORDS * 2u + FIPS512_P_WORDS)

__kernel void ecm_mont_mul_priv_fips512_mt4_bench(__global const uint *a, __global const uint *b,
                                                   __constant uint *n, __global uint *out,
                                                   __constant uint *np0_ptr, uint limbs,
                                                   uint iterations, __local uint *local_mem) {
    if (limbs != MONT_OPT2_FIXED_LIMBS || get_local_size(0) != 4u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0], lid = get_local_id(0);
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_priv_fips512_mt4_body(out, a, b, n, base, np0, local_mem, lid);
        } else {
            mont_mul_priv_fips512_mt4_body(out, out, b, n, base, np0, local_mem, lid);
        }
    }
}

__kernel void ecm_mont_sqr_priv_fips512_mt4_bench(__global const uint *a, __constant uint *n,
                                                   __global uint *out, __constant uint *np0_ptr,
                                                   uint limbs, uint iterations, __local uint *local_mem) {
    if (limbs != MONT_OPT2_FIXED_LIMBS || get_local_size(0) != 4u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0], lid = get_local_id(0);
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_sqr_priv_fips512_mt4_body(out, a, n, base, np0, local_mem, lid);
        } else {
            mont_sqr_priv_fips512_mt4_body(out, out, n, base, np0, local_mem, lid);
        }
    }
}

__kernel void ecm_mont_mul_priv_fips512_mt8_bench(__global const uint *a, __global const uint *b,
                                                   __constant uint *n, __global uint *out,
                                                   __constant uint *np0_ptr, uint limbs,
                                                   uint iterations, __local uint *local_mem) {
    if (limbs != MONT_OPT2_FIXED_LIMBS || get_local_size(0) != 8u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0], lid = get_local_id(0);
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_priv_fips512_mt8_body(out, a, b, n, base, np0, local_mem, lid);
        } else {
            mont_mul_priv_fips512_mt8_body(out, out, b, n, base, np0, local_mem, lid);
        }
    }
}

__kernel void ecm_mont_sqr_priv_fips512_mt8_bench(__global const uint *a, __constant uint *n,
                                                   __global uint *out, __constant uint *np0_ptr,
                                                   uint limbs, uint iterations, __local uint *local_mem) {
    if (limbs != MONT_OPT2_FIXED_LIMBS || get_local_size(0) != 8u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0], lid = get_local_id(0);
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_sqr_priv_fips512_mt8_body(out, a, n, base, np0, local_mem, lid);
        } else {
            mont_sqr_priv_fips512_mt8_body(out, out, n, base, np0, local_mem, lid);
        }
    }
}

__kernel void ecm_mont_mul_priv_fips512_mt16_bench(__global const uint *a, __global const uint *b,
                                                    __constant uint *n, __global uint *out,
                                                    __constant uint *np0_ptr, uint limbs,
                                                    uint iterations, __local uint *local_mem) {
    if (limbs != MONT_OPT2_FIXED_LIMBS || get_local_size(0) != 16u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0], lid = get_local_id(0);
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_priv_fips512_mt16_body(out, a, b, n, base, np0, local_mem, lid);
        } else {
            mont_mul_priv_fips512_mt16_body(out, out, b, n, base, np0, local_mem, lid);
        }
    }
}

__kernel void ecm_mont_sqr_priv_fips512_mt16_bench(__global const uint *a, __constant uint *n,
                                                    __global uint *out, __constant uint *np0_ptr,
                                                    uint limbs, uint iterations, __local uint *local_mem) {
    if (limbs != MONT_OPT2_FIXED_LIMBS || get_local_size(0) != 16u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0], lid = get_local_id(0);
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_sqr_priv_fips512_mt16_body(out, a, n, base, np0, local_mem, lid);
        } else {
            mont_sqr_priv_fips512_mt16_body(out, out, n, base, np0, local_mem, lid);
        }
    }
}

#define FIPS512_CS_LOCAL_U32 (MONT_OPT2_FIXED_LIMBS + MONT_OPT2_FIXED_LIMBS + 8u * FIPS512_T_WORDS + FIPS512_T_WORDS)
#define FIPS512_CS16_LOCAL_U32 (MONT_OPT2_FIXED_LIMBS + MONT_OPT2_FIXED_LIMBS + 16u * FIPS512_T_WORDS + FIPS512_T_WORDS)

__kernel void ecm_mont_mul_priv_fips512_mt8_cs_bench(__global const uint *a, __global const uint *b,
                                                      __constant uint *n, __global uint *out,
                                                      __constant uint *np0_ptr, uint limbs,
                                                      uint iterations, __local uint *local_mem) {
    if (limbs != MONT_OPT2_FIXED_LIMBS || get_local_size(0) != 8u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0], lid = get_local_id(0);
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_priv_fips512_mt8_cs_body(out, a, b, n, base, np0, local_mem, lid);
        } else {
            mont_mul_priv_fips512_mt8_cs_body(out, out, b, n, base, np0, local_mem, lid);
        }
    }
}

__kernel void ecm_mont_sqr_priv_fips512_mt8_cs_bench(__global const uint *a, __constant uint *n,
                                                      __global uint *out, __constant uint *np0_ptr,
                                                      uint limbs, uint iterations, __local uint *local_mem) {
    if (limbs != MONT_OPT2_FIXED_LIMBS || get_local_size(0) != 8u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0], lid = get_local_id(0);
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_sqr_priv_fips512_mt8_cs_body(out, a, n, base, np0, local_mem, lid);
        } else {
            mont_sqr_priv_fips512_mt8_cs_body(out, out, n, base, np0, local_mem, lid);
        }
    }
}

__kernel void ecm_mont_mul_priv_fips512_mt16_cs_bench(__global const uint *a, __global const uint *b,
                                                       __constant uint *n, __global uint *out,
                                                       __constant uint *np0_ptr, uint limbs,
                                                       uint iterations, __local uint *local_mem) {
    if (limbs != MONT_OPT2_FIXED_LIMBS || get_local_size(0) != 16u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0], lid = get_local_id(0);
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_priv_fips512_mt16_cs_body(out, a, b, n, base, np0, local_mem, lid);
        } else {
            mont_mul_priv_fips512_mt16_cs_body(out, out, b, n, base, np0, local_mem, lid);
        }
    }
}

__kernel void ecm_mont_sqr_priv_fips512_mt16_cs_bench(__global const uint *a, __constant uint *n,
                                                       __global uint *out, __constant uint *np0_ptr,
                                                       uint limbs, uint iterations, __local uint *local_mem) {
    if (limbs != MONT_OPT2_FIXED_LIMBS || get_local_size(0) != 16u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0], lid = get_local_id(0);
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_sqr_priv_fips512_mt16_cs_body(out, a, n, base, np0, local_mem, lid);
        } else {
            mont_sqr_priv_fips512_mt16_cs_body(out, out, n, base, np0, local_mem, lid);
        }
    }
}

__kernel void ecm_mont_mul_priv_local_only_512_bench(__global const uint *a, __global const uint *b,
                                                      __constant uint *n, __global uint *out,
                                                      __constant uint *np0_ptr, uint limbs,
                                                      uint iterations, __local uint *local_mem) {
    if (limbs != MONT_OPT2_FIXED_LIMBS) {
        return;
    }
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint lsize = get_local_size(0);
    uint base = gid * limbs;
    uint np0 = np0_ptr[0];
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_priv_local_only_512_body(out, a, b, n, base, np0, local_mem, lid, lsize);
        } else {
            mont_mul_priv_local_only_512_body(out, out, b, n, base, np0, local_mem, lid, lsize);
        }
    }
}

__kernel void ecm_mont_sqr_priv_local_only_512_bench(__global const uint *a, __constant uint *n,
                                                      __global uint *out, __constant uint *np0_ptr,
                                                      uint limbs, uint iterations, __local uint *local_mem) {
    if (limbs != MONT_OPT2_FIXED_LIMBS) {
        return;
    }
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint lsize = get_local_size(0);
    uint base = gid * limbs;
    uint np0 = np0_ptr[0];
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_sqr_priv_local_only_512_body(out, a, n, base, np0, local_mem, lid, lsize);
        } else {
            mont_sqr_priv_local_only_512_body(out, out, n, base, np0, local_mem, lid, lsize);
        }
    }
}

#endif // MAX_LIMBS == 16

#if MAX_LIMBS == 128

__kernel void ecm_mont_mul_priv_local_only_4096_bench(__global const uint *a, __global const uint *b,
                                                       __constant uint *n, __global uint *out,
                                                       __constant uint *np0_ptr, uint limbs,
                                                       uint iterations, __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS) {
        return;
    }
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint lsize = get_local_size(0);
    uint base = gid * limbs;
    uint np0 = np0_ptr[0];
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_priv_local_only_4096_body(out, a, b, n, base, np0, local_mem, lid, lsize);
        } else {
            mont_mul_priv_local_only_4096_body(out, out, b, n, base, np0, local_mem, lid, lsize);
        }
    }
}

__kernel void ecm_mont_sqr_priv_local_only_4096_bench(__global const uint *a, __constant uint *n,
                                                       __global uint *out, __constant uint *np0_ptr,
                                                       uint limbs, uint iterations, __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS) {
        return;
    }
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint lsize = get_local_size(0);
    uint base = gid * limbs;
    uint np0 = np0_ptr[0];
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_sqr_priv_local_only_4096_body(out, a, n, base, np0, local_mem, lid, lsize);
        } else {
            mont_sqr_priv_local_only_4096_body(out, out, n, base, np0, local_mem, lid, lsize);
        }
    }
}

#endif // MAX_LIMBS == 128

// removed: unroll2/4/8 benches (replaced by unroll32/64)

__kernel void ecm_mont_mul_priv_unroll32_bench(__global const uint *a, __global const uint *b,
                                                __constant uint *n, __global uint *out,
                                                __constant uint *np0_ptr, uint limbs,
                                                uint iterations) {
    uint gid = get_global_id(0);
    uint base = gid * limbs;
    uint np0 = np0_ptr[0];
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_priv_unroll32_body(out, a, b, n, base, np0, limbs);
        } else {
            mont_mul_priv_unroll32_body(out, out, b, n, base, np0, limbs);
        }
    }
}

__kernel void ecm_mont_sqr_priv_unroll32_bench(__global const uint *a, __constant uint *n,
                                                __global uint *out, __constant uint *np0_ptr,
                                                uint limbs, uint iterations) {
    uint gid = get_global_id(0);
    uint base = gid * limbs;
    uint np0 = np0_ptr[0];
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_sqr_priv_unroll32_body(out, a, n, base, np0, limbs);
        } else {
            mont_sqr_priv_unroll32_body(out, out, n, base, np0, limbs);
        }
    }
}

__kernel void ecm_mont_mul_priv_unroll64_bench(__global const uint *a, __global const uint *b,
                                                __constant uint *n, __global uint *out,
                                                __constant uint *np0_ptr, uint limbs,
                                                uint iterations) {
    uint gid = get_global_id(0);
    uint base = gid * limbs;
    uint np0 = np0_ptr[0];
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_priv_unroll64_body(out, a, b, n, base, np0, limbs);
        } else {
            mont_mul_priv_unroll64_body(out, out, b, n, base, np0, limbs);
        }
    }
}

__kernel void ecm_mont_sqr_priv_unroll64_bench(__global const uint *a, __constant uint *n,
                                                __global uint *out, __constant uint *np0_ptr,
                                                uint limbs, uint iterations) {
    uint gid = get_global_id(0);
    uint base = gid * limbs;
    uint np0 = np0_ptr[0];
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_sqr_priv_unroll64_body(out, a, n, base, np0, limbs);
        } else {
            mont_sqr_priv_unroll64_body(out, out, n, base, np0, limbs);
        }
    }
}

#if MAX_LIMBS == 128

__kernel void ecm_mont_mul_priv_unroll64_4096_bench(__global const uint *a, __global const uint *b,
                                                     __constant uint *n, __global uint *out,
                                                     __constant uint *np0_ptr, uint limbs,
                                                     uint iterations) {
    if (limbs != MONT_FIXED_4096_LIMBS) return;
    uint gid = get_global_id(0), base = gid * limbs, np0 = np0_ptr[0];
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_priv_unroll64_4096_body(out, a, b, n, base, np0);
        } else {
            mont_mul_priv_unroll64_4096_body(out, out, b, n, base, np0);
        }
    }
}

__kernel void ecm_mont_sqr_priv_unroll64_4096_bench(__global const uint *a, __constant uint *n,
                                                     __global uint *out, __constant uint *np0_ptr,
                                                     uint limbs, uint iterations) {
    if (limbs != MONT_FIXED_4096_LIMBS) return;
    uint gid = get_global_id(0), base = gid * limbs, np0 = np0_ptr[0];
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_sqr_priv_unroll64_4096_body(out, a, n, base, np0);
        } else {
            mont_sqr_priv_unroll64_4096_body(out, out, n, base, np0);
        }
    }
}

__kernel void ecm_mont_mul_priv_unroll64_4096_nod_bench(__global const uint *a, __global const uint *b,
                                                         __constant uint *n, __global uint *out,
                                                         __constant uint *np0_ptr, uint limbs,
                                                         uint iterations) {
    if (limbs != MONT_FIXED_4096_LIMBS) return;
    uint gid = get_global_id(0), base = gid * limbs, np0 = np0_ptr[0];
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_priv_unroll64_4096_nod_body(out, a, b, n, base, np0);
        } else {
            mont_mul_priv_unroll64_4096_nod_body(out, out, b, n, base, np0);
        }
    }
}

__kernel void ecm_mont_sqr_priv_unroll64_4096_nod_bench(__global const uint *a, __constant uint *n,
                                                         __global uint *out, __constant uint *np0_ptr,
                                                         uint limbs, uint iterations) {
    if (limbs != MONT_FIXED_4096_LIMBS) return;
    uint gid = get_global_id(0), base = gid * limbs, np0 = np0_ptr[0];
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_sqr_priv_unroll64_4096_nod_body(out, a, n, base, np0);
        } else {
            mont_sqr_priv_unroll64_4096_nod_body(out, out, n, base, np0);
        }
    }
}

__kernel void ecm_mont_mul_priv_unroll64_4096_mt2_bench(__global const uint *a, __global const uint *b,
                                                         __constant uint *n, __global uint *out,
                                                         __constant uint *np0_ptr, uint limbs,
                                                         uint iterations, __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS || get_local_size(0) != 2u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0];
    uint lid = get_local_id(0);
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_priv_unroll64_4096_mt2_body(out, a, b, n, base, np0, local_mem, lid);
        } else {
            mont_mul_priv_unroll64_4096_mt2_body(out, out, b, n, base, np0, local_mem, lid);
        }
    }
}

__kernel void ecm_mont_sqr_priv_unroll64_4096_mt2_bench(__global const uint *a, __constant uint *n,
                                                         __global uint *out, __constant uint *np0_ptr,
                                                         uint limbs, uint iterations, __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS || get_local_size(0) != 2u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0];
    uint lid = get_local_id(0);
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_sqr_priv_unroll64_4096_mt2_body(out, a, n, base, np0, local_mem, lid);
        } else {
            mont_sqr_priv_unroll64_4096_mt2_body(out, out, n, base, np0, local_mem, lid);
        }
    }
}

__kernel void ecm_mont_mul_priv_unroll64_4096_mt2_weak_bench(__global const uint *a, __global const uint *b,
                                                              __constant uint *n, __global uint *out,
                                                              __constant uint *np0_ptr, uint limbs,
                                                              uint iterations, __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS || get_local_size(0) != 2u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0];
    uint lid = get_local_id(0);
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_priv_unroll64_4096_mt2_weak_body(out, a, b, n, base, np0, local_mem, lid);
        } else {
            mont_mul_priv_unroll64_4096_mt2_weak_body(out, out, b, n, base, np0, local_mem, lid);
        }
    }
}

__kernel void ecm_mont_sqr_priv_unroll64_4096_mt2_weak_bench(__global const uint *a, __constant uint *n,
                                                              __global uint *out, __constant uint *np0_ptr,
                                                              uint limbs, uint iterations, __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS || get_local_size(0) != 2u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0];
    uint lid = get_local_id(0);
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_sqr_priv_unroll64_4096_mt2_weak_body(out, a, n, base, np0, local_mem, lid);
        } else {
            mont_sqr_priv_unroll64_4096_mt2_weak_body(out, out, n, base, np0, local_mem, lid);
        }
    }
}

__kernel void ecm_mont_mul_priv_unroll64_4096_mt4_bench(__global const uint *a, __global const uint *b,
                                                         __constant uint *n, __global uint *out,
                                                         __constant uint *np0_ptr, uint limbs,
                                                         uint iterations, __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS || get_local_size(0) != 4u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0];
    uint lid = get_local_id(0);
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_priv_unroll64_4096_mt4_body(out, a, b, n, base, np0, local_mem, lid);
        } else {
            mont_mul_priv_unroll64_4096_mt4_body(out, out, b, n, base, np0, local_mem, lid);
        }
    }
}

__kernel void ecm_mont_sqr_priv_unroll64_4096_mt4_bench(__global const uint *a, __constant uint *n,
                                                         __global uint *out, __constant uint *np0_ptr,
                                                         uint limbs, uint iterations, __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS || get_local_size(0) != 4u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0];
    uint lid = get_local_id(0);
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_sqr_priv_unroll64_4096_mt4_body(out, a, n, base, np0, local_mem, lid);
        } else {
            mont_sqr_priv_unroll64_4096_mt4_body(out, out, n, base, np0, local_mem, lid);
        }
    }
}

__kernel void ecm_mont_mul_priv_unroll64_4096_mt8_bench(__global const uint *a, __global const uint *b,
                                                         __constant uint *n, __global uint *out,
                                                         __constant uint *np0_ptr, uint limbs,
                                                         uint iterations, __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS || get_local_size(0) != 8u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0];
    uint lid = get_local_id(0);
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_priv_unroll64_4096_mt8_body(out, a, b, n, base, np0, local_mem, lid);
        } else {
            mont_mul_priv_unroll64_4096_mt8_body(out, out, b, n, base, np0, local_mem, lid);
        }
    }
}

__kernel void ecm_mont_sqr_priv_unroll64_4096_mt8_bench(__global const uint *a, __constant uint *n,
                                                         __global uint *out, __constant uint *np0_ptr,
                                                         uint limbs, uint iterations, __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS || get_local_size(0) != 8u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0];
    uint lid = get_local_id(0);
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_sqr_priv_unroll64_4096_mt8_body(out, a, n, base, np0, local_mem, lid);
        } else {
            mont_sqr_priv_unroll64_4096_mt8_body(out, out, n, base, np0, local_mem, lid);
        }
    }
}

#define FIPS4096_MT_LOCAL_U32 (MONT_FIXED_4096_LIMBS + MONT_FIXED_4096_LIMBS + FIPS4096_MT_PROD_WORDS * 2u + FIPS4096_P_WORDS)
#define FIPS4096_CS_LOCAL_U32 (MONT_FIXED_4096_LIMBS + MONT_FIXED_4096_LIMBS + 8u * FIPS4096_T_WORDS + FIPS4096_T_WORDS)
#define FIPS4096_CS16_LOCAL_U32 (MONT_FIXED_4096_LIMBS + MONT_FIXED_4096_LIMBS + 16u * FIPS4096_T_WORDS + FIPS4096_T_WORDS)

__kernel void ecm_mont_mul_priv_fips4096_bench(__global const uint *a, __global const uint *b,
                                                  __constant uint *n, __global uint *out,
                                                  __constant uint *np0_ptr, uint limbs, uint iterations) {
    if (limbs != MONT_FIXED_4096_LIMBS) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0];
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_priv_fips4096_body(out, a, b, n, base, np0);
        } else {
            mont_mul_priv_fips4096_body(out, out, b, n, base, np0);
        }
    }
}

__kernel void ecm_mont_sqr_priv_fips4096_bench(__global const uint *a, __constant uint *n,
                                               __global uint *out, __constant uint *np0_ptr,
                                               uint limbs, uint iterations) {
    if (limbs != MONT_FIXED_4096_LIMBS) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0];
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_sqr_priv_fips4096_body(out, a, n, base, np0);
        } else {
            mont_sqr_priv_fips4096_body(out, out, n, base, np0);
        }
    }
}

__kernel void ecm_mont_mul_priv_fips4096_mt4_bench(__global const uint *a, __global const uint *b,
                                                   __constant uint *n, __global uint *out,
                                                   __constant uint *np0_ptr, uint limbs,
                                                   uint iterations, __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS || get_local_size(0) != 4u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0], lid = get_local_id(0);
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_priv_fips4096_mt4_body(out, a, b, n, base, np0, local_mem, lid);
        } else {
            mont_mul_priv_fips4096_mt4_body(out, out, b, n, base, np0, local_mem, lid);
        }
    }
}

__kernel void ecm_mont_sqr_priv_fips4096_mt4_bench(__global const uint *a, __constant uint *n,
                                                    __global uint *out, __constant uint *np0_ptr,
                                                    uint limbs, uint iterations, __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS || get_local_size(0) != 4u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0], lid = get_local_id(0);
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_sqr_priv_fips4096_mt4_body(out, a, n, base, np0, local_mem, lid);
        } else {
            mont_sqr_priv_fips4096_mt4_body(out, out, n, base, np0, local_mem, lid);
        }
    }
}

__kernel void ecm_mont_mul_priv_fips4096_mt8_bench(__global const uint *a, __global const uint *b,
                                                   __constant uint *n, __global uint *out,
                                                   __constant uint *np0_ptr, uint limbs,
                                                   uint iterations, __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS || get_local_size(0) != 8u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0], lid = get_local_id(0);
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_priv_fips4096_mt8_body(out, a, b, n, base, np0, local_mem, lid);
        } else {
            mont_mul_priv_fips4096_mt8_body(out, out, b, n, base, np0, local_mem, lid);
        }
    }
}

__kernel void ecm_mont_sqr_priv_fips4096_mt8_bench(__global const uint *a, __constant uint *n,
                                                    __global uint *out, __constant uint *np0_ptr,
                                                    uint limbs, uint iterations, __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS || get_local_size(0) != 8u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0], lid = get_local_id(0);
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_sqr_priv_fips4096_mt8_body(out, a, n, base, np0, local_mem, lid);
        } else {
            mont_sqr_priv_fips4096_mt8_body(out, out, n, base, np0, local_mem, lid);
        }
    }
}

__kernel void ecm_mont_mul_priv_fips4096_mt16_bench(__global const uint *a, __global const uint *b,
                                                     __constant uint *n, __global uint *out,
                                                     __constant uint *np0_ptr, uint limbs,
                                                     uint iterations, __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS || get_local_size(0) != 16u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0], lid = get_local_id(0);
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_priv_fips4096_mt16_body(out, a, b, n, base, np0, local_mem, lid);
        } else {
            mont_mul_priv_fips4096_mt16_body(out, out, b, n, base, np0, local_mem, lid);
        }
    }
}

__kernel void ecm_mont_sqr_priv_fips4096_mt16_bench(__global const uint *a, __constant uint *n,
                                                     __global uint *out, __constant uint *np0_ptr,
                                                     uint limbs, uint iterations, __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS || get_local_size(0) != 16u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0], lid = get_local_id(0);
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_sqr_priv_fips4096_mt16_body(out, a, n, base, np0, local_mem, lid);
        } else {
            mont_sqr_priv_fips4096_mt16_body(out, out, n, base, np0, local_mem, lid);
        }
    }
}

__kernel void ecm_mont_mul_priv_fips4096_mt8_cs_bench(__global const uint *a, __global const uint *b,
                                                       __constant uint *n, __global uint *out,
                                                       __constant uint *np0_ptr, uint limbs,
                                                       uint iterations, __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS || get_local_size(0) != 8u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0], lid = get_local_id(0);
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_priv_fips4096_mt8_cs_body(out, a, b, n, base, np0, local_mem, lid);
        } else {
            mont_mul_priv_fips4096_mt8_cs_body(out, out, b, n, base, np0, local_mem, lid);
        }
    }
}

__kernel void ecm_mont_sqr_priv_fips4096_mt8_cs_bench(__global const uint *a, __constant uint *n,
                                                        __global uint *out, __constant uint *np0_ptr,
                                                        uint limbs, uint iterations, __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS || get_local_size(0) != 8u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0], lid = get_local_id(0);
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_sqr_priv_fips4096_mt8_cs_body(out, a, n, base, np0, local_mem, lid);
        } else {
            mont_sqr_priv_fips4096_mt8_cs_body(out, out, n, base, np0, local_mem, lid);
        }
    }
}

__kernel void ecm_mont_mul_priv_fips4096_mt16_cs_bench(__global const uint *a, __global const uint *b,
                                                        __constant uint *n, __global uint *out,
                                                        __constant uint *np0_ptr, uint limbs,
                                                        uint iterations, __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS || get_local_size(0) != 16u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0], lid = get_local_id(0);
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_mul_priv_fips4096_mt16_cs_body(out, a, b, n, base, np0, local_mem, lid);
        } else {
            mont_mul_priv_fips4096_mt16_cs_body(out, out, b, n, base, np0, local_mem, lid);
        }
    }
}

__kernel void ecm_mont_sqr_priv_fips4096_mt16_cs_bench(__global const uint *a, __constant uint *n,
                                                        __global uint *out, __constant uint *np0_ptr,
                                                        uint limbs, uint iterations, __local uint *local_mem) {
    if (limbs != MONT_FIXED_4096_LIMBS || get_local_size(0) != 16u) return;
    uint gid = get_group_id(0), base = gid * limbs, np0 = np0_ptr[0], lid = get_local_id(0);
    for (uint it = 0u; it < iterations; ++it) {
        if (it == 0u) {
            mont_sqr_priv_fips4096_mt16_cs_body(out, a, n, base, np0, local_mem, lid);
        } else {
            mont_sqr_priv_fips4096_mt16_cs_body(out, out, n, base, np0, local_mem, lid);
        }
    }
}

#endif // MAX_LIMBS == 128
