/* opencl_backend_glue.cpp — OpenCL implementation of the ECM backend seam.

   Linked only into the `ecm` executable. Each hook forwards to the pre-existing
   OpenCL host functions, so the shared driver behaves exactly as before.
*/

#include "ecm_backend.h"

#include "cgbn_stage1.h"            /* gpu_prepare_opencl */
#include "cl_probe.h"              /* configureOpenclDeviceIndex */
#include "opencl_ecm_entry.h"       /* opencl_ecm_stage1 */
#include "opencl_ecm_path_registry.h" /* opencl_ecm_print_available_kernels */

extern "C" void ecm_backend_print_kernels(FILE *out) {
    opencl_ecm_print_available_kernels(out);
}

extern "C" int ecm_backend_prepare(size_t n_log2, int verbose, int device_index,
                                   const char *gpu_mul_path, const char *gpu_sqr_path,
                                   const char *gpu_add_path, const char *gpu_sub_path,
                                   const char *gpu_special_mult_path) {
    if (!configureOpenclDeviceIndex(device_index, true)) {
        return 1;
    }
    return gpu_prepare_opencl(n_log2, verbose, gpu_mul_path, gpu_sqr_path,
                              gpu_add_path, gpu_sub_path, gpu_special_mult_path);
}

extern "C" int ecm_backend_stage1(mpz_t *factors, int *array_found,
                                  const mpz_t N, const mpz_t s,
                                  uint32_t curves, uint32_t *sigma,
                                  unsigned long checkpoint_interval_ms,
                                  float *gputime, int verbose,
                                  const char *gpu_mul_path, const char *gpu_sqr_path,
                                  const char *gpu_add_path, const char *gpu_sub_path,
                                  const char *gpu_special_mult_path) {
    return opencl_ecm_stage1(factors, array_found, N, s, curves, sigma,
                             checkpoint_interval_ms, gputime, verbose,
                             gpu_mul_path, gpu_sqr_path, gpu_add_path,
                             gpu_sub_path, gpu_special_mult_path);
}
