#ifndef OPENCL_ECM_ENTRY_H
#define OPENCL_ECM_ENTRY_H

#include <cstdint>

#include <gmp.h>

#include "cgbn_stage1.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Thin host-side entry point that mirrors the gmp-ecm GPU call chain.
 * The caller owns the factor buffers and initializes them for `curves` items.
 */
int opencl_ecm_stage1(mpz_t *factors,
                      int *array_found,
                      const mpz_t n,
                      const mpz_t s,
                      uint32_t curves,
                      uint32_t *sigma,
                      unsigned long checkpoint_interval_ms,
                      float *gputime,
                      int verbose);

#ifdef __cplusplus
}
#endif

#endif /* OPENCL_ECM_ENTRY_H */