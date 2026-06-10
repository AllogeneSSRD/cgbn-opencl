#include "opencl_ecm_limb24.h"

#include "opencl_ecm_log.h"

#include <cstring>

uint32_t ecm_inv24_odd(uint32_t x) {
    uint64_t y = 1;
    for (int i = 0; i < 4; ++i) {
        y = y * (2ull - static_cast<uint64_t>(x & ECM_LIMB24_MASK) * y);
        y &= 0xFFFFFFull;
    }
    return static_cast<uint32_t>(y) & ECM_LIMB24_MASK;
}

void ecm_mpz_to_limb24(uint32_t *out, uint32_t limbs, const mpz_t z) {
    std::memset(out, 0, sizeof(uint32_t) * limbs);
    mpz_t t;
    mpz_init_set(t, z);
    for (uint32_t i = 0; i < limbs; ++i) {
        out[i] = static_cast<uint32_t>(mpz_get_ui(t)) & ECM_LIMB24_MASK;
        mpz_fdiv_q_2exp(t, t, 24);
        if (mpz_sgn(t) == 0) {
            break;
        }
    }
    mpz_clear(t);
}

void ecm_limb24_from_mpz(uint32_t *out, uint32_t limbs, const mpz_t s) {
    if (mpz_sizeinbase(s, 2) > static_cast<size_t>(limbs) * 24u) {
        ecm_ts_fprintf(stderr, "limb24: value does not fit in %u limbs\n", limbs);
        std::memset(out, 0, sizeof(uint32_t) * limbs);
        return;
    }
    ecm_mpz_to_limb24(out, limbs, s);
}

void ecm_limb24_to_mpz(mpz_t r, const uint32_t *x, uint32_t limbs) {
    mpz_set_ui(r, 0);
    for (uint32_t i = 0; i < limbs; ++i) {
        mpz_mul_2exp(r, r, 24);
        mpz_add_ui(r, r, x[i] & ECM_LIMB24_MASK);
    }
}

uint32_t ecm_find_np0_limb24(const uint32_t *n_limbs) {
    return 0u - ecm_inv24_odd(n_limbs[0] | 1u);
}

void ecm_to_montgomery_limb24(uint32_t *out, const mpz_t bn, const mpz_t N, uint32_t mont_bits,
                              uint32_t limbs) {
    mpz_t t;
    mpz_init(t);
    mpz_mul_2exp(t, bn, mont_bits);
    mpz_fdiv_r(t, t, N);
    ecm_mpz_to_limb24(out, limbs, t);
    mpz_clear(t);
}

void ecm_from_montgomery_limb24(mpz_t out, const mpz_t mont, const mpz_t N, uint32_t np0,
                                uint32_t limbs) {
    mpz_t prod, add;
    mpz_init(prod);
    mpz_init(add);
    mpz_set(prod, mont);
    for (uint32_t index = 0; index < limbs; ++index) {
        const uint32_t low = np0 * (static_cast<uint32_t>(mpz_get_ui(prod)) & ECM_LIMB24_MASK);
        mpz_mul_ui(add, N, low);
        mpz_add(prod, prod, add);
        mpz_fdiv_q_2exp(prod, prod, 24);
    }
    if (mpz_cmp(prod, N) < 0) {
        mpz_set(out, prod);
    } else {
        mpz_sub(out, prod, N);
    }
    mpz_clear(prod);
    mpz_clear(add);
}
