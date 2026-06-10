#pragma once

#include <gmp.h>

#include <cstdint>

constexpr uint32_t ECM_LIMB24_MASK = 0xFFFFFFu;
constexpr uint32_t ECM_I24_384_LIMBS = 16u;
constexpr uint32_t ECM_I24_384_BITS = ECM_I24_384_LIMBS * 24u;

uint32_t ecm_inv24_odd(uint32_t x);
void ecm_mpz_to_limb24(uint32_t *out, uint32_t limbs, const mpz_t z);
void ecm_limb24_from_mpz(uint32_t *out, uint32_t limbs, const mpz_t s);
void ecm_limb24_to_mpz(mpz_t r, const uint32_t *x, uint32_t limbs);
uint32_t ecm_find_np0_limb24(const uint32_t *n_limbs);
void ecm_to_montgomery_limb24(uint32_t *out, const mpz_t bn, const mpz_t N, uint32_t mont_bits,
                              uint32_t limbs);
void ecm_from_montgomery_limb24(mpz_t out, const mpz_t mont, const mpz_t N, uint32_t np0,
                                uint32_t limbs);
