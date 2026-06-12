// OpenCL ECM Stage 1 — Montgomery ladder (double_add_v2), ported from test/cgbn_stage1.cu

#ifndef MAX_LIMBS
#define MAX_LIMBS 64
#endif

#ifndef TPI
#define TPI 8
#endif

#ifndef ECM_STAGE1_FORCE_NORMALIZE
#define ECM_STAGE1_FORCE_NORMALIZE 0
#endif

#ifndef ECM_STAGE1_MUL_PATH
#define ECM_STAGE1_MUL_PATH 0
#endif

#ifndef ECM_STAGE1_SQR_PATH
#define ECM_STAGE1_SQR_PATH 0
#endif

#ifndef ECM_STAGE1_COOP_WG
#define ECM_STAGE1_COOP_WG 0
#endif

#ifndef ECM_STAGE1_COOP_SCRATCH_U32
#define ECM_STAGE1_COOP_SCRATCH_U32 0
#endif

#ifndef ECM_STAGE1_MUL_FORCE_UNROLL32
#define ECM_STAGE1_MUL_FORCE_UNROLL32 0
#endif

#ifndef ECM_STAGE1_MUL_FORCE_UNROLL384
#define ECM_STAGE1_MUL_FORCE_UNROLL384 0
#endif

#ifndef ECM_STAGE1_MUL_FORCE_PRIV_OPT
#define ECM_STAGE1_MUL_FORCE_PRIV_OPT 0
#endif

#ifndef ECM_STAGE1_SQR_FORCE_UNROLL32
#define ECM_STAGE1_SQR_FORCE_UNROLL32 0
#endif

#ifndef ECM_STAGE1_SQR_FORCE_UNROLL384
#define ECM_STAGE1_SQR_FORCE_UNROLL384 0
#endif

#ifndef ECM_STAGE1_SQR_FORCE_PRIV_OPT
#define ECM_STAGE1_SQR_FORCE_PRIV_OPT 0
#endif

#ifndef ECM_STAGE1_384_LIMBS
#define ECM_STAGE1_384_LIMBS 12u
#endif

#ifndef ECM_STAGE1_512_CONTAINER_LIMBS
#define ECM_STAGE1_512_CONTAINER_LIMBS 16u
#endif

// Path ids: 0=unroll64_4096, 1=unroll64_4096_mt2, 2=fips4096, 3=fips4096_mt8, 4=fips4096_mt16
#if ECM_STAGE1_COOP_WG > 1
#define ECM_STAGE1_USE_COOP_WG 1
#else
#define ECM_STAGE1_USE_COOP_WG 0
#endif

#define MONT_FIXED_4096_LIMBS 128u
#define ECM_STAGE1_MT2_LOCAL_U32 (MONT_FIXED_4096_LIMBS + 2u + MONT_FIXED_4096_LIMBS + MONT_FIXED_4096_LIMBS + 3u)



static inline uint mul_ui32_limbs(uint *r, uint m, uint limbs);
static inline void shift_right_32_limbs(uint *r, uint limbs);
static inline void special_mult_ui32(uint *r, uint m, const uint *N, uint np0, uint limbs);

static inline void special_mult_stage1(uint *r, uint m, const uint *N, uint np0, uint limbs) {
    special_mult_ui32(r, m, N, np0, limbs);
}

static inline void mp_shift_left_1_mod(uint *r, const uint *a, const uint *N, uint limbs) {
    uint carry = 0u;
    for (uint i = 0u; i < limbs; ++i) {
        uint old = a[i];
        r[i] = (old << 1) | carry;
        carry = old >> 31;
    }
    if (carry || mp_ge(r, N, limbs)) {
        mp_sub_n(r, r, N, limbs);
    }
}

static inline void mont_normalize(uint *r, const uint *N, uint limbs) {
    if (mp_ge(r, N, limbs)) {
        mp_sub_n(r, r, N, limbs);
    }
}

static inline void maybe_mont_normalize(uint *r, const uint *N, uint limbs) {
#if ECM_STAGE1_FORCE_NORMALIZE
    mont_normalize(r, N, limbs);
#else
    (void)r;
    (void)N;
    (void)limbs;
#endif
}

// r <- low limbs of (r * m); returns overflow limb above r
static inline uint mul_ui32_limbs(uint *r, uint m, uint limbs) {
    ulong carry = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong prod = (ulong)r[i] * (ulong)m + carry;
        r[i] = (uint)prod;
        carry = prod >> 32;
    }
    return (uint)carry;
}

static inline void shift_right_32_limbs(uint *r, uint limbs) {
    for (uint i = 0u; i + 1u < limbs; ++i) {
        r[i] = r[i + 1u];
    }
    r[limbs - 1u] = 0u;
}

// (r * m) / 2^32 mod N — ported from CUDA curve_t::special_mult_ui32
static inline void special_mult_ui32(uint *r, uint m, const uint *N, uint np0, uint limbs) {
    uint carry_t1 = mul_ui32_limbs(r, m, limbs);
    uint t1_0 = r[0];
    uint q = (uint)((ulong)t1_0 * (ulong)np0);

    uint temp[MAX_LIMBS];
    mp_copy(temp, N, limbs);
    uint carry_t2 = mul_ui32_limbs(temp, q, limbs);

    shift_right_32_limbs(r, limbs);
    shift_right_32_limbs(temp, limbs);
    r[limbs - 1u] = carry_t1;
    temp[limbs - 1u] = carry_t2;

    {
        int carry_q = (int)mp_add_n(r, r, temp, limbs);
        if (t1_0 != 0u) {
            uint carry1 = 1u;
            for (uint i = 0u; i < limbs && carry1 != 0u; ++i) {
                ulong sum = (ulong)r[i] + (ulong)carry1;
                r[i] = (uint)sum;
                carry1 = (uint)(sum >> 32);
            }
            carry_q += (int)carry1;
        }
        if (carry_q > 0) {
            mp_sub_n(r, r, N, limbs);
        }
        if (mp_ge(r, N, limbs)) {
            mp_sub_n(r, r, N, limbs);
        }
    }
}

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
    mp_add_mod(r, a, b, N, limbs);
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
    return mp_sub_mod(r, a, b, N, limbs);
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
    mont_mul_stage1_unroll64_4096_mt2_local(op_out, op_a, op_b, N_loc, np0, mont_scratch, lid);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            out[i] = op_out[i];
        }
    }
#elif ECM_STAGE1_MUL_PATH == 2
    if (lid == 0u) {
        mont_mul_stage1_fips4096(out, a, b, N, np0);
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
        mont_mul_stage1_unroll64_4096(out, a, b, N, np0);
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
    mont_mul_stage1_unroll64_4096_mt2_local(op_out, op_a, op_b, N_loc, np0, mont_scratch, lid);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            out[i] = op_out[i];
        }
    }
#elif ECM_STAGE1_SQR_PATH == 2
    if (lid == 0u) {
        mont_mul_stage1_fips4096(out, a, a, N, np0);
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
        mont_mul_stage1_unroll64_4096(out, a, a, N, np0);
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
static inline void double_add_v2(
    uint *q, uint *u, uint *w, uint *v,
    uint d, const uint *N, uint np0, uint limbs)
{
    uint t[MAX_LIMBS], CB[MAX_LIMBS], DA[MAX_LIMBS], AA[MAX_LIMBS], BB[MAX_LIMBS];
    uint K[MAX_LIMBS], dK[MAX_LIMBS];

    mp_add_mod(t, v, w, N, limbs);
    (void)mp_sub_mod(v, v, w, N, limbs);

    mp_add_mod(w, u, q, N, limbs);
    (void)mp_sub_mod(u, u, q, N, limbs);

    mont_mul_stage1(CB, t, u, N, np0, limbs);
    maybe_mont_normalize(CB, N, limbs);
    mont_mul_stage1(DA, v, w, N, np0, limbs);
    maybe_mont_normalize(DA, N, limbs);

    mont_sqr_stage1(AA, w, N, np0, limbs);
    mont_sqr_stage1(BB, u, N, np0, limbs);
    maybe_mont_normalize(AA, N, limbs);
    maybe_mont_normalize(BB, N, limbs);

    mont_mul_stage1(q, AA, BB, N, np0, limbs);
    maybe_mont_normalize(q, N, limbs);

    (void)mp_sub_mod(K, AA, BB, N, limbs);

    mp_copy(dK, K, limbs);
    special_mult_stage1(dK, d, N, np0, limbs);

    mp_add_mod(u, BB, dK, N, limbs);
    mont_mul_stage1(u, K, u, N, np0, limbs);
    maybe_mont_normalize(u, N, limbs);

    mp_add_mod(w, DA, CB, N, limbs);
    (void)mp_sub_mod(v, DA, CB, N, limbs);

    mont_sqr_stage1(w, w, N, np0, limbs);
    maybe_mont_normalize(w, N, limbs);
    mont_sqr_stage1(v, v, N, np0, limbs);
    maybe_mont_normalize(v, N, limbs);
    mp_shift_left_1_mod(v, v, N, limbs);
}

static inline void swap_limbs(uint *a, uint *b, uint limbs) {
    for (uint i = 0u; i < limbs; ++i) {
        uint tmp = a[i];
        a[i] = b[i];
        b[i] = tmp;
    }
}

#if ECM_STAGE1_USE_COOP_WG
static inline void double_add_v2_coop(
    uint *q, uint *u, uint *w, uint *v,
    uint d, const uint *N, uint np0, uint limbs,
    __local const uint *N_loc,
    __local uint *op_a, __local uint *op_b, __local uint *op_out,
    __local uint *mont_scratch, uint lid)
{
    uint t[MAX_LIMBS], CB[MAX_LIMBS], DA[MAX_LIMBS], AA[MAX_LIMBS], BB[MAX_LIMBS];
    uint K[MAX_LIMBS], dK[MAX_LIMBS];

    if (lid == 0u) {
        mp_add_mod(t, v, w, N, limbs);
        (void)mp_sub_mod(v, v, w, N, limbs);
        mp_add_mod(w, u, q, N, limbs);
        (void)mp_sub_mod(u, u, q, N, limbs);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    mont_mul_stage1_coop(CB, t, u, N, N_loc, np0, op_a, op_b, op_out, mont_scratch, lid);
    if (lid == 0u) {
        maybe_mont_normalize(CB, N, limbs);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    mont_mul_stage1_coop(DA, v, w, N, N_loc, np0, op_a, op_b, op_out, mont_scratch, lid);
    if (lid == 0u) {
        maybe_mont_normalize(DA, N, limbs);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    mont_sqr_stage1_coop(AA, w, N, N_loc, np0, op_a, op_b, op_out, mont_scratch, lid);
    mont_sqr_stage1_coop(BB, u, N, N_loc, np0, op_a, op_b, op_out, mont_scratch, lid);
    if (lid == 0u) {
        maybe_mont_normalize(AA, N, limbs);
        maybe_mont_normalize(BB, N, limbs);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    mont_mul_stage1_coop(q, AA, BB, N, N_loc, np0, op_a, op_b, op_out, mont_scratch, lid);
    if (lid == 0u) {
        maybe_mont_normalize(q, N, limbs);
        (void)mp_sub_mod(K, AA, BB, N, limbs);
        mp_copy(dK, K, limbs);
        special_mult_stage1(dK, d, N, np0, limbs);
        mp_add_mod(u, BB, dK, N, limbs);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    mont_mul_stage1_coop(u, K, u, N, N_loc, np0, op_a, op_b, op_out, mont_scratch, lid);
    if (lid == 0u) {
        maybe_mont_normalize(u, N, limbs);
        mp_add_mod(w, DA, CB, N, limbs);
        (void)mp_sub_mod(v, DA, CB, N, limbs);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    mont_sqr_stage1_coop(w, w, N, N_loc, np0, op_a, op_b, op_out, mont_scratch, lid);
    mont_sqr_stage1_coop(v, v, N, N_loc, np0, op_a, op_b, op_out, mont_scratch, lid);
    if (lid == 0u) {
        maybe_mont_normalize(w, N, limbs);
        maybe_mont_normalize(v, N, limbs);
        mp_shift_left_1_mod(v, v, N, limbs);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

static inline void run_double_add_instance_mt2_wg(
    uint instance_i,
    __global const uint *s_bits,
    ulong s_num_bits,
    ulong s_bits_start,
    ulong s_bits_interval,
    __global uint *data,
    uint sigma_0,
    uint np0,
    uint limbs,
    __local uint *N_loc,
    __local uint *op_a,
    __local uint *op_b,
    __local uint *op_out,
    __local uint *mont_scratch,
    uint lid)
{
    const uint base = instance_i * 5u * limbs;
    uint N[MAX_LIMBS];
    uint aX[MAX_LIMBS], aZ[MAX_LIMBS], bX[MAX_LIMBS], bZ[MAX_LIMBS];
    const uint half_limbs = limbs / 2u;
    const uint j_begin = lid * half_limbs;
    const uint j_end = j_begin + half_limbs;

    #pragma unroll 64
    for (uint i = j_begin; i < j_end; ++i) {
        N_loc[i] = data[base + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0u) {
        for (uint i = 0u; i < limbs; ++i) {
            N[i] = data[base + i];
            aX[i] = data[base + limbs + i];
            aZ[i] = data[base + 2u * limbs + i];
            bX[i] = data[base + 3u * limbs + i];
            bZ[i] = data[base + 4u * limbs + i];
        }
    }

    const uint d = sigma_0 + instance_i;
    int swapped = 0;
    ulong s_end = s_bits_start + s_bits_interval;
    if (s_end > s_num_bits) {
        s_end = s_num_bits;
    }

    for (ulong b = s_bits_start; b < s_end; ++b) {
        ulong nth = s_num_bits - 1ul - b;
        uint limb_idx = (uint)(nth >> 5);
        uint bit_idx = (uint)(nth & 31ul);
        int bit = (int)((s_bits[limb_idx] >> bit_idx) & 1u);

        if (lid == 0u && bit != swapped) {
            swapped = !swapped;
            swap_limbs(aX, bX, limbs);
            swap_limbs(aZ, bZ, limbs);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        double_add_v2_coop(aX, aZ, bX, bZ, d, N, np0, limbs, N_loc, op_a, op_b, op_out,
                           mont_scratch, lid);
    }

    if (lid == 0u && swapped) {
        swap_limbs(aX, bX, limbs);
        swap_limbs(aZ, bZ, limbs);
    }

    if (lid == 0u) {
        for (uint i = 0u; i < limbs; ++i) {
            data[base + limbs + i] = aX[i];
            data[base + 2u * limbs + i] = aZ[i];
            data[base + 3u * limbs + i] = bX[i];
            data[base + 4u * limbs + i] = bZ[i];
        }
    }
}
#endif

// ---------------------------------------------------------------------------
// Main kernel — mirrors CUDA kernel_double_add
// data layout per curve (5 * limbs uint32): N, aX, aZ, bX, bZ
// ---------------------------------------------------------------------------

static inline void run_double_add_instance(
    uint instance_i,
    __global const uint *s_bits,
    ulong s_num_bits,
    ulong s_bits_start,
    ulong s_bits_interval,
    __global uint *data,
    uint sigma_0,
    uint np0,
    uint limbs)
{
    uint base = instance_i * 5u * limbs;
    uint N[MAX_LIMBS];
    uint aX[MAX_LIMBS], aZ[MAX_LIMBS], bX[MAX_LIMBS], bZ[MAX_LIMBS];

    for (uint i = 0u; i < limbs; ++i) {
        N[i] = data[base + i];
        aX[i] = data[base + limbs + i];
        aZ[i] = data[base + 2u * limbs + i];
        bX[i] = data[base + 3u * limbs + i];
        bZ[i] = data[base + 4u * limbs + i];
    }

    uint d = sigma_0 + instance_i;
    int swapped = 0;

    ulong s_end = s_bits_start + s_bits_interval;
    if (s_end > s_num_bits) {
        s_end = s_num_bits;
    }

    for (ulong b = s_bits_start; b < s_end; ++b) {
        ulong nth = s_num_bits - 1ul - b;
        uint limb_idx = (uint)(nth >> 5);
        uint bit_idx = (uint)(nth & 31ul);
        int bit = (int)((s_bits[limb_idx] >> bit_idx) & 1u);

        if (bit != swapped) {
            swapped = !swapped;
            swap_limbs(aX, bX, limbs);
            swap_limbs(aZ, bZ, limbs);
        }
        double_add_v2(aX, aZ, bX, bZ, d, N, np0, limbs);
    }

    if (swapped) {
        swap_limbs(aX, bX, limbs);
        swap_limbs(aZ, bZ, limbs);
    }

    for (uint i = 0u; i < limbs; ++i) {
        data[base + limbs + i] = aX[i];
        data[base + 2u * limbs + i] = aZ[i];
        data[base + 3u * limbs + i] = bX[i];
        data[base + 4u * limbs + i] = bZ[i];
    }
}

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
