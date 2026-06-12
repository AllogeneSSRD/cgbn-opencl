// 4096-bit cooperative work-group supplement.
#if ECM_STAGE1_USE_COOP_WG

#ifndef ECM_STAGE1_MUL_PATH
#define ECM_STAGE1_MUL_PATH 0
#endif
#ifndef ECM_STAGE1_SQR_PATH
#define ECM_STAGE1_SQR_PATH 0
#endif
#define MONT_FIXED_4096_LIMBS MAX_LIMBS
#define ECM_STAGE1_MT2_LOCAL_U32 (MONT_FIXED_4096_LIMBS + 2u + MONT_FIXED_4096_LIMBS + MONT_FIXED_4096_LIMBS + 3u)

#if ECM_STAGE1_USE_COOP_WG
static inline void mont_mul_stage1_unroll64_4096_local(__local uint *out, __local const uint *a,
                                                       __local const uint *b, __local const uint *N,
                                                       uint np0) {
    uint t[128u + 2u];
    for (uint i = 0u; i < 130u; ++i) t[i] = 0u;
    uint B[128u];
    for (uint j = 0u; j < 128u; ++j) B[j] = b[j];
    for (uint i = 0u; i < 128u; ++i) {
        uint ai = a[i];
        ulong carry = 0ul;
        #pragma unroll 64
        for (uint j = 0u; j < 128u; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong top = (ulong)t[128u] + carry;
        t[128u] = (uint)top;
        t[129u] = (uint)(top >> 32);
        uint m = (uint)((ulong)t[0] * (ulong)np0);
        ulong uv0 = (ulong)t[0] + (ulong)m * (ulong)N[0];
        carry = uv0 >> 32;
        #pragma unroll 64
        for (uint j = 1u; j < 128u; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
            t[j - 1u] = (uint)uv;
            carry = uv >> 32;
        }
        top = (ulong)t[128u] + carry;
        t[127u] = (uint)top;
        top = (ulong)t[129u] + (top >> 32);
        t[128u] = (uint)top;
        t[129u] = (uint)(top >> 32);
    }
    ulong borrow = 0ul;
    uint D[128u];
    for (uint i = 0u; i < 128u; ++i) {
        ulong tv = (ulong)t[i], nv = (ulong)N[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }
    uint need_sub = (t[128u] != 0u || t[129u] != 0u) ? 1u : 0u;
    need_sub = (borrow == 0u) ? 1u : need_sub;
    uint mask = 0u - need_sub;
    for (uint i = 0u; i < 128u; ++i) out[i] = (D[i] & mask) | (t[i] & ~mask);
}

static inline void mp_add_mod_fused_unroll_b32_4096_local(__local uint *r, __local const uint *a,
                                                          __local const uint *b,
                                                          __local const uint *N) {
    ulong carry_add = 0ul;
    ulong carry_sub = 1ul;
    #pragma unroll
    for (uint blk = 0u; blk < 4u; ++blk) {
        uint off = blk * 32u;
        #pragma unroll 32
        for (uint j = 0u; j < 32u; ++j) {
            uint i = off + j;
            ulong sum = (ulong)a[i] + (ulong)b[i] + carry_add;
            carry_add = sum >> 32;
            ulong temp = (ulong)(uint)sum + (ulong)(~N[i]) + carry_sub;
            carry_sub = temp >> 32;
            r[i] = (uint)temp;
        }
    }
    if ((carry_add | carry_sub) != 0ul) {
        return;
    }
    ulong c = 0ul;
    #pragma unroll 32
    for (uint i = 0u; i < 128u; ++i) {
        ulong s = (ulong)r[i] + (ulong)N[i] + c;
        r[i] = (uint)s;
        c = s >> 32;
    }
}

static inline void mp_add_mod_local(__local uint *r, __local const uint *a, __local const uint *b,
                                    __local const uint *N, uint limbs) {
    mp_add_mod_fused_unroll_b32_4096_local(r, a, b, N);
    (void)limbs;
}

static inline uint mp_add_n_local(__local uint *r, __local const uint *a, __local const uint *b,
                                  uint limbs) {
    ulong carry = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong sum = (ulong)a[i] + (ulong)b[i] + carry;
        r[i] = (uint)sum;
        carry = sum >> 32;
    }
    return (uint)carry;
}

static inline void mp_sub_n_local(__local uint *r, __local const uint *a, __local const uint *N,
                                  uint limbs) {
    ulong borrow = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong av = (ulong)a[i];
        ulong nv = (ulong)N[i];
        ulong w = av - nv - borrow;
        r[i] = (uint)w;
        borrow = (av < nv + borrow) ? 1ul : 0ul;
    }
}

static inline int mp_sub_mod_local(__local uint *r, __local const uint *a, __local const uint *b,
                                   __local const uint *N, uint limbs) {
    ulong borrow = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong av = (ulong)a[i];
        ulong bv = (ulong)b[i];
        ulong w = av - bv - borrow;
        r[i] = (uint)w;
        borrow = (av < bv + borrow) ? 1ul : 0ul;
    }
    if (borrow) {
        ulong c = 0ul;
        for (uint i = 0u; i < limbs; ++i) {
            ulong s = (ulong)r[i] + (ulong)N[i] + c;
            r[i] = (uint)s;
            c = s >> 32;
        }
        return 1;
    }
    return 0;
}

static inline void mp_copy_local(__local uint *dst, __local const uint *src, uint limbs) {
    for (uint i = 0u; i < limbs; ++i) {
        dst[i] = src[i];
    }
}

static inline int mp_ge_local(__local const uint *a, __local const uint *N, uint limbs) {
    for (int i = (int)limbs - 1; i >= 0; --i) {
        if (a[(uint)i] > N[(uint)i]) return 1;
        if (a[(uint)i] < N[(uint)i]) return 0;
    }
    return 1;
}

static inline void mont_normalize_local(__local uint *r, __local const uint *N, uint limbs) {
    if (mp_ge_local(r, N, limbs)) {
        mp_sub_n_local(r, r, N, limbs);
    }
}

static inline void maybe_mont_normalize_local(__local uint *r, __local const uint *N, uint limbs) {
#if ECM_STAGE1_FORCE_NORMALIZE
    mont_normalize_local(r, N, limbs);
#else
    (void)r;
    (void)N;
    (void)limbs;
#endif
}

static inline uint mul_ui32_limbs_local(__local uint *r, uint m, uint limbs) {
    ulong carry = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong prod = (ulong)r[i] * (ulong)m + carry;
        r[i] = (uint)prod;
        carry = prod >> 32;
    }
    return (uint)carry;
}

static inline void shift_right_32_limbs_local(__local uint *r, uint limbs) {
    for (uint i = 0u; i + 1u < limbs; ++i) {
        r[i] = r[i + 1u];
    }
    r[limbs - 1u] = 0u;
}

static inline void special_mult_ui32_local(__local uint *r, uint m, __local const uint *N,
                                           uint np0, uint limbs) {
    uint carry_t1 = mul_ui32_limbs_local(r, m, limbs);
    uint t1_0 = r[0];
    uint q = (uint)((ulong)t1_0 * (ulong)np0);
    uint temp[MAX_LIMBS];
    for (uint i = 0u; i < limbs; ++i) {
        temp[i] = N[i];
    }
    uint carry_t2 = mul_ui32_limbs(temp, q, limbs);
    shift_right_32_limbs_local(r, limbs);
    shift_right_32_limbs(temp, limbs);
    r[limbs - 1u] = carry_t1;
    temp[limbs - 1u] = carry_t2;
    {
        uint carry = 0u;
        for (uint i = 0u; i < limbs; ++i) {
            ulong sum = (ulong)r[i] + (ulong)temp[i] + (ulong)carry;
            r[i] = (uint)sum;
            carry = (uint)(sum >> 32);
        }
        int carry_q = (int)carry;
        if (t1_0 != 0u) {
            carry = 0u;
            for (uint i = 0u; i < limbs; ++i) {
                ulong sum = (ulong)r[i] + (ulong)carry;
                r[i] = (uint)sum;
                carry = (uint)(sum >> 32);
            }
            carry_q += (int)carry;
        }
        if (carry_q > 0) {
            mp_sub_n_local(r, r, N, limbs);
        }
        if (mp_ge_local(r, N, limbs)) {
            mp_sub_n_local(r, r, N, limbs);
        }
    }
}

static inline void mp_shift_left_1_mod_local(__local uint *r, __local const uint *a,
                                             __local const uint *N, uint limbs) {
    uint carry = 0u;
    for (uint i = 0u; i < limbs; ++i) {
        uint old = a[i];
        r[i] = (old << 1) | carry;
        carry = old >> 31;
    }
    if (carry || mp_ge_local(r, N, limbs)) {
        mp_sub_n_local(r, r, N, limbs);
    }
}
#endif

#if ECM_STAGE1_USE_COOP_WG
static inline void mont_mul_stage1_coop(
    uint *out, const uint *a, const uint *b, const uint *N, __local const uint *N_loc,
    uint np0, __local uint *op_a, __local uint *op_b, __local uint *op_out,
    __local uint *mont_scratch, uint lid)
{
#if ECM_STAGE1_MUL_PATH == 1
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            op_a[i] = a[i];
            op_b[i] = b[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mont_mul_unroll_4096b_mt2(op_out, op_a, op_b, N_loc, np0, mont_scratch, lid);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            out[i] = op_out[i];
        }
    }
#elif ECM_STAGE1_MUL_PATH == 2
    if (lid == 0u) {
        mont_mul_fips_4096b(out, a, b, N, np0, MONT_FIXED_4096_LIMBS);
    }
#elif ECM_STAGE1_MUL_PATH == 3
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            op_a[i] = a[i];
            op_b[i] = b[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mont_mul_stage1_fips4096_mtn_local(op_out, op_a, op_b, N_loc, np0, mont_scratch, lid, 8u);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            out[i] = op_out[i];
        }
    }
#elif ECM_STAGE1_MUL_PATH == 4
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            op_a[i] = a[i];
            op_b[i] = b[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mont_mul_stage1_fips4096_mtn_local(op_out, op_a, op_b, N_loc, np0, mont_scratch, lid, 16u);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            out[i] = op_out[i];
        }
    }
#else
    if (lid == 0u) {
        mont_mul_unroll_4096b(out, a, b, N, np0, MONT_FIXED_4096_LIMBS);
    }
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
}

static inline void mont_sqr_stage1_coop(
    uint *out, const uint *a, const uint *N, __local const uint *N_loc, uint np0,
    __local uint *op_a, __local uint *op_b, __local uint *op_out, __local uint *mont_scratch,
    uint lid)
{
#if ECM_STAGE1_SQR_PATH == 1
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            op_a[i] = a[i];
            op_b[i] = a[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mont_mul_unroll_4096b_mt2(op_out, op_a, op_b, N_loc, np0, mont_scratch, lid);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            out[i] = op_out[i];
        }
    }
#elif ECM_STAGE1_SQR_PATH == 2
    if (lid == 0u) {
        mont_mul_fips_4096b(out, a, a, N, np0, MONT_FIXED_4096_LIMBS);
    }
#elif ECM_STAGE1_SQR_PATH == 3
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            op_a[i] = a[i];
            op_b[i] = a[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mont_mul_stage1_fips4096_mtn_local(op_out, op_a, op_b, N_loc, np0, mont_scratch, lid, 8u);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            out[i] = op_out[i];
        }
    }
#elif ECM_STAGE1_SQR_PATH == 4
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            op_a[i] = a[i];
            op_b[i] = a[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mont_mul_stage1_fips4096_mtn_local(op_out, op_a, op_b, N_loc, np0, mont_scratch, lid, 16u);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            out[i] = op_out[i];
        }
    }
#else
    if (lid == 0u) {
        mont_mul_unroll_4096b(out, a, a, N, np0, MONT_FIXED_4096_LIMBS);
    }
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
}
#endif

// Simultaneous double-and-add (CUDA curve_t::double_add_v2)
//
// Note: each call executes a fixed "operator mix" and is the performance hotspot:
//   - mp_add_mod:      4 calls
//   - mp_sub_mod:      4 calls
//   - mont_mul_priv:   4 calls
//   - mont_sqr_priv:   4 calls
//   - mont_normalize:  8 calls
//   - special_mult_ui32: 1 call
//   - mp_shift_left_1_mod: 1 call
#if ECM_STAGE1_USE_COOP_WG
__kernel __attribute__((reqd_work_group_size(ECM_STAGE1_COOP_WG, 1, 1)))
#else
__kernel
#endif
void kernel_double_add(
    __global const uint *s_bits,
    ulong s_num_bits,
    ulong s_bits_start,
    ulong s_bits_interval,
    __global uint *data,
    uint count,
    uint sigma_0,
    uint np0,
    uint limbs)
{
#if ECM_STAGE1_USE_COOP_WG
    __local uint N_local[MONT_FIXED_4096_LIMBS];
    __local uint mont_op_a[MONT_FIXED_4096_LIMBS];
    __local uint mont_op_b[MONT_FIXED_4096_LIMBS];
    __local uint mont_op_out[MONT_FIXED_4096_LIMBS];
    __local uint mont_scratch[ECM_STAGE1_COOP_SCRATCH_U32];
    const uint instance_i = get_group_id(0);
    const uint lid = get_local_id(0);
    if (instance_i >= count) {
        return;
    }
    if (limbs != MONT_FIXED_4096_LIMBS) {
        return;
    }
    run_double_add_instance_mt2_wg(instance_i, s_bits, s_num_bits, s_bits_start, s_bits_interval,
                                   data, sigma_0, np0, limbs, N_local, mont_op_a, mont_op_b,
                                   mont_op_out, mont_scratch, lid);
#else
    uint instance_i = get_global_id(0);
    if (instance_i >= count) {
        return;
    }
    if (limbs == 0u || limbs > MAX_LIMBS) {
        return;
    }
    run_double_add_instance(instance_i, s_bits, s_num_bits, s_bits_start, s_bits_interval, data,
                            sigma_0, np0, limbs);
#endif
}
#endif
