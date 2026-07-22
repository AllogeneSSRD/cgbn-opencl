/* cudacommon.h — slim CUDA helper header for the MPA-OpenCl CUDA/CGBN backend.

   This is a trimmed replacement for GMP-ECM's cudacommon.h. It provides only
   what kernels/cuda/cgbn_stage1.cu actually needs:
     - CUDA_CHECK / cuda_check   (error-checking wrapper, used pervasively)
     - kernel_info               (occupancy/reg info; defined in the CUDA glue)

   Device enumeration / selection (GMP-ECM's select_and_init_GPU / get_device_prop)
   is intentionally omitted: the ecm_cuda driver performs device selection through
   the backend prepare hook (src/cuda/ecm_cuda_backend.cu).
*/

#ifndef _CUDACOMMON_SLIM_H
#define _CUDACOMMON_SLIM_H 1

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef __cplusplus
/* C++ / CUDA only */

#define CUDA_CHECK(action) cuda_check(action, #action, __FILE__, __LINE__)

inline void cuda_check(cudaError_t status, const char *action = NULL,
                       const char *file = NULL, int32_t line = 0) {
  if (status != cudaSuccess) {
    fprintf(stderr, "CUDA error (%d) occurred: %s\n", status,
            cudaGetErrorString(status));
    if (action != NULL)
      fprintf(stderr, "While running %s   (file %s, line %d)\n", action, file,
              line);
    exit(EXIT_FAILURE);
  }
}

/* Defined in src/cuda/ecm_cuda_backend.cu. Prints kernel register/occupancy
   info when verbose >= OUTPUT_VERBOSE. */
void kernel_info(const void *func, int verbose);

#endif /* __cplusplus */

#endif /* _CUDACOMMON_SLIM_H */
