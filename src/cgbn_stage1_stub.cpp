#include <gmp.h>
#include <cstdint>
#include "ecm.h"
#include <iostream>

extern "C" int cgbn_ecm_stage1(mpz_t *factors, int *array_found,
                               const mpz_t N, const mpz_t s,
                               uint32_t curves, uint32_t *sigma,
                               mpz_t *stage1_x_residues,
                               unsigned long checkpoint_interval_ms,
                               float *gputime, int verbose)
{
    // Minimal stub: mark no factors found, set gputime to 0
    std::cout << "Warning: cgbn_ecm_stage1 is a stub that does not perform any computation.\n";
    if(gputime) *gputime = 0.0f;
    for(uint32_t i=0;i<curves;i++){
        array_found[i] = ECM_NO_FACTOR_FOUND;
        if(factors) mpz_set_ui(factors[i], 0);
        if(stage1_x_residues) mpz_set_ui(stage1_x_residues[i], 0);
    }
    return ECM_NO_FACTOR_FOUND;
}