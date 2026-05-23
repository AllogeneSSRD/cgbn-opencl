// OpenCL ECM Stage 1 — Montgomery ladder (double_add_v2), ported from test/cgbn_stage1.cu

#ifndef MAX_LIMBS
#define MAX_LIMBS 64
#endif

#ifndef TPI
#define TPI 8
#endif

#ifndef ECM_STAGE1_FORCE_NORMALIZE
#define ECM_STAGE1_FORCE_NORMALIZE 1
#endif

#include "mont_wg.cl"

// ---------------------------------------------------------------------------
// Private multi-limb helpers (one curve per work-item)
//
// Naming convention:
//   `mp_*` means "multi-precision integer primitive". These are the low-level
//   bignum building blocks used by higher-level curve operations.
//
// Why keep the `mp_` prefix:
// 1) Distinguish bignum operators from point/curve operators (double_add_v2, etc.).
// 2) Make call-sites read like arithmetic formulas over Z and Z/NZ.
// 3) Avoid ambiguity with OpenCL scalar/vector add/sub intrinsics.
// ---------------------------------------------------------------------------

inline void mp_copy(uint *dst, const uint *src, uint limbs) {
    for (uint i = 0u; i < limbs; ++i) {
        dst[i] = src[i];
    }
}

inline void mp_zero(uint *dst, uint limbs) {
    for (uint i = 0u; i < limbs; ++i) {
        dst[i] = 0u;
    }
}

inline int mp_ge(const uint *a, const uint *N, uint limbs) {
    for (int i = (int)limbs - 1; i >= 0; --i) {
        if (a[(uint)i] > N[(uint)i]) return 1;
        if (a[(uint)i] < N[(uint)i]) return 0;
    }
    return 1;
}

inline void mp_sub_n(uint *r, const uint *a, const uint *N, uint limbs) {
    ulong borrow = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong av = (ulong)a[i];
        ulong nv = (ulong)N[i];
        ulong w = av - nv - borrow;
        r[i] = (uint)w;
        borrow = (av < nv + borrow) ? 1ul : 0ul;
    }
}

inline uint mp_add_n(uint *r, const uint *a, const uint *b, uint limbs) {
    ulong carry = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong sum = (ulong)a[i] + (ulong)b[i] + carry;
        r[i] = (uint)sum;
        carry = sum >> 32;
    }
    return (uint)carry;
}

// Modular add over Z/NZ:
//   r = a + b (mod N)
//
// Implementation idea:
//   1) Compute raw limb sum S = a + b in base 2^32.
//   2) If S overflowed (carry != 0) OR S >= N, subtract N once.
// This is valid because a,b are already reduced, so S is in [0, 2N-2].
#ifndef MP_ADD_MOD_FUSED_UNROLL
#define MP_ADD_MOD_FUSED_UNROLL 2
#endif

inline void mp_add_mod(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {
    ulong carry_add = 0ul;
    ulong carry_sub = 1ul;
#if MP_ADD_MOD_FUSED_UNROLL == 2
    uint j = 0u;
    for (; j + 1u < limbs; j += 2u) {
        ulong sum0 = (ulong)a[j] + (ulong)b[j] + carry_add;
        carry_add = sum0 >> 32;
        ulong temp0 = (ulong)(uint)sum0 + (ulong)(~N[j]) + carry_sub;
        carry_sub = temp0 >> 32;
        r[j] = (uint)temp0;

        ulong sum1 = (ulong)a[j + 1u] + (ulong)b[j + 1u] + carry_add;
        carry_add = sum1 >> 32;
        ulong temp1 = (ulong)(uint)sum1 + (ulong)(~N[j + 1u]) + carry_sub;
        carry_sub = temp1 >> 32;
        r[j + 1u] = (uint)temp1;
    }
    if (limbs & 1u) {
        ulong sum = (ulong)a[j] + (ulong)b[j] + carry_add;
        carry_add = sum >> 32;
        ulong temp = (ulong)(uint)sum + (ulong)(~N[j]) + carry_sub;
        carry_sub = temp >> 32;
        r[j] = (uint)temp;
    }
#else
    for (uint i = 0u; i < limbs; ++i) {
        ulong sum = (ulong)a[i] + (ulong)b[i] + carry_add;
        carry_add = sum >> 32;
        ulong temp = (ulong)(uint)sum + (ulong)(~N[i]) + carry_sub;
        carry_sub = temp >> 32;
        r[i] = (uint)temp;
    }
#endif
    if ((carry_add | carry_sub) != 0ul) {
        return;
    }
    ulong c = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong s = (ulong)r[i] + (ulong)N[i] + c;
        r[i] = (uint)s;
        c = s >> 32;
    }
}

// Returns 1 if borrow (a < b)
inline int mp_sub_mod(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {
    ulong borrow = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong av = (ulong)a[i];
        ulong bv = (ulong)b[i];
        ulong w = av - bv - borrow;
        r[i] = (uint)w;
        borrow = (av < bv + borrow) ? 1ul : 0ul;
    }
    if (borrow) {
        // For subtraction underflow, add modulus without modular reduction.
        (void)mp_add_n(r, r, N, limbs);
        return 1;
    }
    return 0;
}

inline void mp_shift_left_1_mod(uint *r, const uint *a, const uint *N, uint limbs) {
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

inline void mont_normalize(uint *r, const uint *N, uint limbs) {
    if (mp_ge(r, N, limbs)) {
        mp_sub_n(r, r, N, limbs);
    }
}

inline void maybe_mont_normalize(uint *r, const uint *N, uint limbs) {
#if ECM_STAGE1_FORCE_NORMALIZE
    mont_normalize(r, N, limbs);
#else
    (void)r;
    (void)N;
    (void)limbs;
#endif
}

// r <- low limbs of (r * m); returns overflow limb above r
inline uint mul_ui32_limbs(uint *r, uint m, uint limbs) {
    ulong carry = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong prod = (ulong)r[i] * (ulong)m + carry;
        r[i] = (uint)prod;
        carry = prod >> 32;
    }
    return (uint)carry;
}

inline void shift_right_32_limbs(uint *r, uint limbs) {
    for (uint i = 0u; i + 1u < limbs; ++i) {
        r[i] = r[i + 1u];
    }
    r[limbs - 1u] = 0u;
}

// (r * m) / 2^32 mod N — ported from CUDA curve_t::special_mult_ui32
void special_mult_ui32(uint *r, uint m, const uint *N, uint np0, uint limbs) {
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
void double_add_v2(
    uint *q, uint *u, uint *w, uint *v,
    uint d, const uint *N, uint np0, uint limbs)
{
    uint t[MAX_LIMBS], CB[MAX_LIMBS], DA[MAX_LIMBS], AA[MAX_LIMBS], BB[MAX_LIMBS];
    uint K[MAX_LIMBS], dK[MAX_LIMBS];

    mp_add_mod(t, v, w, N, limbs);
    (void)mp_sub_mod(v, v, w, N, limbs);

    mp_add_mod(w, u, q, N, limbs);
    (void)mp_sub_mod(u, u, q, N, limbs);

    mont_mul_priv(CB, t, u, N, np0, limbs);
    maybe_mont_normalize(CB, N, limbs);
    mont_mul_priv(DA, v, w, N, np0, limbs);
    maybe_mont_normalize(DA, N, limbs);

    mont_sqr_priv(AA, w, N, np0, limbs);
    mont_sqr_priv(BB, u, N, np0, limbs);
    maybe_mont_normalize(AA, N, limbs);
    maybe_mont_normalize(BB, N, limbs);

    mont_mul_priv(q, AA, BB, N, np0, limbs);
    maybe_mont_normalize(q, N, limbs);

    (void)mp_sub_mod(K, AA, BB, N, limbs);

    mp_copy(dK, K, limbs);
    special_mult_ui32(dK, d, N, np0, limbs);

    mp_add_mod(u, BB, dK, N, limbs);
    mont_mul_priv(u, K, u, N, np0, limbs);
    maybe_mont_normalize(u, N, limbs);

    mp_add_mod(w, DA, CB, N, limbs);
    (void)mp_sub_mod(v, DA, CB, N, limbs);

    mont_sqr_priv(w, w, N, np0, limbs);
    maybe_mont_normalize(w, N, limbs);
    mont_sqr_priv(v, v, N, np0, limbs);
    maybe_mont_normalize(v, N, limbs);
    mp_shift_left_1_mod(v, v, N, limbs);
}

inline void swap_limbs(uint *a, uint *b, uint limbs) {
    for (uint i = 0u; i < limbs; ++i) {
        uint tmp = a[i];
        a[i] = b[i];
        b[i] = tmp;
    }
}

// ---------------------------------------------------------------------------
// Main kernel — mirrors CUDA kernel_double_add
// data layout per curve (5 * limbs uint32): N, aX, aZ, bX, bZ
// ---------------------------------------------------------------------------

inline void run_double_add_instance(
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

__kernel void kernel_double_add(
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
    uint instance_i = get_global_id(0);
    if (instance_i >= count) {
        return;
    }
    if (limbs == 0u || limbs > MAX_LIMBS) {
        return;
    }
    run_double_add_instance(instance_i, s_bits, s_num_bits, s_bits_start, s_bits_interval, data,
                            sigma_0, np0, limbs);
}

static inline int mp_ge_l(__local const uint *a, __local const uint *N, uint limbs) {
    for (int i = (int)limbs - 1; i >= 0; --i) {
        if (a[(uint)i] > N[(uint)i]) return 1;
        if (a[(uint)i] < N[(uint)i]) return 0;
    }
    return 1;
}

static inline void mp_sub_n_l(__local uint *r, __local const uint *a, __local const uint *N, uint limbs) {
    ulong borrow = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong av = (ulong)a[i];
        ulong nv = (ulong)N[i];
        ulong w = av - nv - borrow;
        r[i] = (uint)w;
        borrow = (av < nv + borrow) ? 1ul : 0ul;
    }
}

static inline uint mp_add_n_l(__local uint *r, __local const uint *a, __local const uint *b, uint limbs) {
    ulong carry = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong sum = (ulong)a[i] + (ulong)b[i] + carry;
        r[i] = (uint)sum;
        carry = sum >> 32;
    }
    return (uint)carry;
}

static inline void mp_copy_l(__local uint *dst, __local const uint *src, uint limbs) {
    for (uint i = 0u; i < limbs; ++i) dst[i] = src[i];
}

static inline void mp_add_mod_l(__local uint *r, __local const uint *a, __local const uint *b,
                         __local const uint *N, uint limbs) {
    ulong carry_add = 0ul;
    ulong carry_sub = 1ul;
#if MP_ADD_MOD_FUSED_UNROLL == 2
    uint j = 0u;
    for (; j + 1u < limbs; j += 2u) {
        ulong sum0 = (ulong)a[j] + (ulong)b[j] + carry_add;
        carry_add = sum0 >> 32;
        ulong temp0 = (ulong)(uint)sum0 + (ulong)(~N[j]) + carry_sub;
        carry_sub = temp0 >> 32;
        r[j] = (uint)temp0;

        ulong sum1 = (ulong)a[j + 1u] + (ulong)b[j + 1u] + carry_add;
        carry_add = sum1 >> 32;
        ulong temp1 = (ulong)(uint)sum1 + (ulong)(~N[j + 1u]) + carry_sub;
        carry_sub = temp1 >> 32;
        r[j + 1u] = (uint)temp1;
    }
    if (limbs & 1u) {
        ulong sum = (ulong)a[j] + (ulong)b[j] + carry_add;
        carry_add = sum >> 32;
        ulong temp = (ulong)(uint)sum + (ulong)(~N[j]) + carry_sub;
        carry_sub = temp >> 32;
        r[j] = (uint)temp;
    }
#else
    for (uint i = 0u; i < limbs; ++i) {
        ulong sum = (ulong)a[i] + (ulong)b[i] + carry_add;
        carry_add = sum >> 32;
        ulong temp = (ulong)(uint)sum + (ulong)(~N[i]) + carry_sub;
        carry_sub = temp >> 32;
        r[i] = (uint)temp;
    }
#endif
    if ((carry_add | carry_sub) != 0ul) {
        return;
    }
    ulong c = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong s = (ulong)r[i] + (ulong)N[i] + c;
        r[i] = (uint)s;
        c = s >> 32;
    }
}

static inline int mp_sub_mod_l(__local uint *r, __local const uint *a, __local const uint *b,
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
        (void)mp_add_n_l(r, r, N, limbs);
        return 1;
    }
    return 0;
}

static inline void mp_shift_left_1_mod_l(__local uint *r, __local const uint *a, __local const uint *N,
                                  uint limbs) {
    uint carry = 0u;
    for (uint i = 0u; i < limbs; ++i) {
        uint old = a[i];
        r[i] = (old << 1) | carry;
        carry = old >> 31;
    }
    if (carry || mp_ge_l(r, N, limbs)) mp_sub_n_l(r, r, N, limbs);
}

static inline void mont_normalize_l(__local uint *r, __local const uint *N, uint limbs) {
    if (mp_ge_l(r, N, limbs)) mp_sub_n_l(r, r, N, limbs);
}

static inline void maybe_mont_normalize_l(__local uint *r, __local const uint *N, uint limbs) {
#if ECM_STAGE1_FORCE_NORMALIZE
    mont_normalize_l(r, N, limbs);
#else
    (void)r;
    (void)N;
    (void)limbs;
#endif
}

static inline uint mul_ui32_limbs_l(__local uint *r, uint m, uint limbs) {
    ulong carry = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong prod = (ulong)r[i] * (ulong)m + carry;
        r[i] = (uint)prod;
        carry = prod >> 32;
    }
    return (uint)carry;
}

static inline void shift_right_32_limbs_l(__local uint *r, uint limbs) {
    for (uint i = 0u; i + 1u < limbs; ++i) r[i] = r[i + 1u];
    r[limbs - 1u] = 0u;
}

static inline void special_mult_ui32_l(__local uint *r, uint m, __local const uint *N, uint np0, uint limbs,
                                __local uint *tmp0) {
    uint carry_t1 = mul_ui32_limbs_l(r, m, limbs);
    uint t1_0 = r[0];
    uint q = (uint)((ulong)t1_0 * (ulong)np0);

    mp_copy_l(tmp0, N, limbs);
    uint carry_t2 = mul_ui32_limbs_l(tmp0, q, limbs);
    shift_right_32_limbs_l(r, limbs);
    shift_right_32_limbs_l(tmp0, limbs);
    r[limbs - 1u] = carry_t1;
    tmp0[limbs - 1u] = carry_t2;

    int carry_q = (int)mp_add_n_l(r, r, tmp0, limbs);
    if (t1_0 != 0u) {
        uint carry1 = 1u;
        for (uint i = 0u; i < limbs && carry1 != 0u; ++i) {
            ulong sum = (ulong)r[i] + (ulong)carry1;
            r[i] = (uint)sum;
            carry1 = (uint)(sum >> 32);
        }
        carry_q += (int)carry1;
    }
    if (carry_q > 0) mp_sub_n_l(r, r, N, limbs);
    if (mp_ge_l(r, N, limbs)) mp_sub_n_l(r, r, N, limbs);
}

static inline void mont_mul_wg_local(__local uint *out, __local const uint *a, __local const uint *b,
                              __local const uint *n, uint np0, uint limbs, uint tid,
                              __local uint *scratch) {
    cgbn_mont_mul_wg_local_core(out, a, b, n, np0, limbs, tid, scratch);
}

static inline void mont_sqr_wg_local(__local uint *out, __local const uint *a, __local const uint *n,
                              uint np0, uint limbs, uint tid, __local uint *scratch) {
    cgbn_mont_sqr_wg_local_core(out, a, n, np0, limbs, tid, scratch);
}

static inline void double_add_v2_wg_local(__local uint *q, __local uint *u, __local uint *w, __local uint *v,
                                   uint d, __local uint *N, uint np0, uint limbs, uint tid,
                                   __local uint *t, __local uint *CB, __local uint *DA,
                                   __local uint *AA, __local uint *BB, __local uint *K,
                                   __local uint *dK, __local uint *mont_scratch) {
    if (tid == 0u) {
        mp_add_mod_l(t, v, w, N, limbs);
        (void)mp_sub_mod_l(v, v, w, N, limbs);
        mp_add_mod_l(w, u, q, N, limbs);
        (void)mp_sub_mod_l(u, u, q, N, limbs);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    mont_mul_wg_local(CB, t, u, N, np0, limbs, tid, mont_scratch);
    if (tid == 0u) maybe_mont_normalize_l(CB, N, limbs);
    barrier(CLK_LOCAL_MEM_FENCE);

    mont_mul_wg_local(DA, v, w, N, np0, limbs, tid, mont_scratch);
    if (tid == 0u) maybe_mont_normalize_l(DA, N, limbs);
    barrier(CLK_LOCAL_MEM_FENCE);

    mont_sqr_wg_local(AA, w, N, np0, limbs, tid, mont_scratch);
    mont_sqr_wg_local(BB, u, N, np0, limbs, tid, mont_scratch);
    if (tid == 0u) {
        maybe_mont_normalize_l(AA, N, limbs);
        maybe_mont_normalize_l(BB, N, limbs);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    mont_mul_wg_local(q, AA, BB, N, np0, limbs, tid, mont_scratch);
    if (tid == 0u) {
        maybe_mont_normalize_l(q, N, limbs);
        (void)mp_sub_mod_l(K, AA, BB, N, limbs);
        mp_copy_l(dK, K, limbs);
        special_mult_ui32_l(dK, d, N, np0, limbs, t);
        mp_add_mod_l(u, BB, dK, N, limbs);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    mont_mul_wg_local(u, K, u, N, np0, limbs, tid, mont_scratch);
    if (tid == 0u) {
        maybe_mont_normalize_l(u, N, limbs);
        mp_add_mod_l(w, DA, CB, N, limbs);
        (void)mp_sub_mod_l(v, DA, CB, N, limbs);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    mont_sqr_wg_local(w, w, N, np0, limbs, tid, mont_scratch);
    mont_sqr_wg_local(v, v, N, np0, limbs, tid, mont_scratch);
    if (tid == 0u) {
        maybe_mont_normalize_l(w, N, limbs);
        maybe_mont_normalize_l(v, N, limbs);
        mp_shift_left_1_mod_l(v, v, N, limbs);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void kernel_double_add_wg(
    __global const uint *s_bits,
    ulong s_num_bits,
    ulong s_bits_start,
    ulong s_bits_interval,
    __global uint *data,
    uint count,
    uint sigma_0,
    uint np0,
    uint limbs,
    __local uint *local_mem)
{
    uint instance_i = get_group_id(0);
    uint lane = get_local_id(0);
    if (instance_i >= count) {
        return;
    }
    if (limbs == 0u || limbs > MAX_LIMBS) {
        return;
    }

    uint local_words_needed = 12u * limbs + MONT_WG_SCRATCH_WORDS + 1u;
    (void)local_words_needed;
    __local uint *ptr = local_mem;
    __local uint *N = ptr; ptr += limbs;
    __local uint *q = ptr; ptr += limbs;
    __local uint *u = ptr; ptr += limbs;
    __local uint *w = ptr; ptr += limbs;
    __local uint *v = ptr; ptr += limbs;
    __local uint *t = ptr; ptr += limbs;
    __local uint *CB = ptr; ptr += limbs;
    __local uint *DA = ptr; ptr += limbs;
    __local uint *AA = ptr; ptr += limbs;
    __local uint *BB = ptr; ptr += limbs;
    __local uint *K = ptr; ptr += limbs;
    __local uint *dK = ptr; ptr += limbs;
    __local uint *mont_scratch = ptr; // MONT_WG_SCRATCH_WORDS
    __local int *swapped_l = (__local int *)(mont_scratch + MONT_WG_SCRATCH_WORDS);

    uint base = instance_i * 5u * limbs;
    for (uint i = lane; i < limbs; i += TPI) {
        N[i] = data[base + i];
        q[i] = data[base + limbs + i];
        u[i] = data[base + 2u * limbs + i];
        w[i] = data[base + 3u * limbs + i];
        v[i] = data[base + 4u * limbs + i];
    }
    if (lane == 0u) {
        *swapped_l = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint d = sigma_0 + instance_i;
    ulong s_end = s_bits_start + s_bits_interval;
    if (s_end > s_num_bits) s_end = s_num_bits;
    for (ulong b = s_bits_start; b < s_end; ++b) {
        if (lane == 0u) {
            ulong nth = s_num_bits - 1ul - b;
            uint limb_idx = (uint)(nth >> 5);
            uint bit_idx = (uint)(nth & 31ul);
            int bit = (int)((s_bits[limb_idx] >> bit_idx) & 1u);
            if (bit != *swapped_l) {
                *swapped_l = !(*swapped_l);
                for (uint i = 0u; i < limbs; ++i) {
                    uint tmp = q[i];
                    q[i] = w[i];
                    w[i] = tmp;
                    tmp = u[i];
                    u[i] = v[i];
                    v[i] = tmp;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        double_add_v2_wg_local(q, u, w, v, d, N, np0, limbs, lane, t, CB, DA, AA, BB, K, dK,
                               mont_scratch);
    }

    if (lane == 0u && *swapped_l) {
        for (uint i = 0u; i < limbs; ++i) {
            uint tmp = q[i];
            q[i] = w[i];
            w[i] = tmp;
            tmp = u[i];
            u[i] = v[i];
            v[i] = tmp;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = lane; i < limbs; i += TPI) {
        data[base + limbs + i] = q[i];
        data[base + 2u * limbs + i] = u[i];
        data[base + 3u * limbs + i] = w[i];
        data[base + 4u * limbs + i] = v[i];
    }
}
