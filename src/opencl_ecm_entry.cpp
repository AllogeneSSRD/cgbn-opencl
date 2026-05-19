#include "opencl_ecm_entry.h"

int opencl_ecm_stage1(mpz_t *factors,
                      int *array_found,
                      const mpz_t n,
                      const mpz_t s,
                      uint32_t curves,
                      uint32_t *sigma,
                      unsigned long checkpoint_interval_ms,
                      float *gputime,
                      int verbose)
{
    if (factors == nullptr || array_found == nullptr || sigma == nullptr || gputime == nullptr) {
        return -1;
    }

    if (mpz_cmp_ui(n, 1) <= 0 || mpz_divisible_2exp_p(n, 1)) {
        return -2;
    }

    return cgbn_ecm_stage1(factors,
                           array_found,
                           n,
                           s,
                           curves,
                           sigma,
                           checkpoint_interval_ms,
                           gputime,
                           verbose);
}
