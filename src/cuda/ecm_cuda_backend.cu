/* ecm_cuda_backend.cu — CUDA implementation of the ECM backend seam.

   Linked only into the `ecm_cuda` executable. Provides:
     - the GMP-ECM logging shims (outputf / test_verbose) that
       kernels/cuda/cgbn_stage1.cu expects, routed to the project's timestamped
       logger (ecm_ts_vfprintf);
     - kernel_info (occupancy/register report);
     - the three backend hooks (print_kernels / prepare / stage1), with device
       enumeration + selection done here (the .cu never calls cudaSetDevice).
*/

#include "ecm_backend.h"
#include "cgbn_stage1_cuda.h"   /* 9-arg cgbn_ecm_stage1 */
#include "cuda_ecm_shim.h"      /* OUTPUT_*, outputf/test_verbose/ecm_cuda_set_verbose */
#include "cudacommon.h"         /* kernel_info declaration */
#include "opencl_ecm_log.h"     /* ecm_ts_vfprintf / ecm_ts_fprintf */

#include <cuda_runtime.h>

#include <cstdarg>
#include <cstdio>

/* ── Verbose threshold + logging shims (used by cgbn_stage1.cu) ──────────── */

/* Default to OUTPUT_NORMAL so informative "GPU:" lines show; ecm_backend_stage1
   overrides this from the driver's -v flag before the run. */
static int g_cuda_verbose = OUTPUT_NORMAL;

extern "C" void ecm_cuda_set_verbose(int level) { g_cuda_verbose = level; }

extern "C" int test_verbose(int level) { return g_cuda_verbose >= level; }

extern "C" void outputf(int verbosity, const char *format, ...) {
    /* OUTPUT_ERROR is always emitted (to stderr); everything else is gated. */
    if (verbosity != OUTPUT_ERROR && g_cuda_verbose < verbosity)
        return;
    FILE *stream = (verbosity == OUTPUT_ERROR) ? stderr : stdout;
    va_list ap;
    va_start(ap, format);
    ecm_ts_vfprintf(stream, format, ap);
    va_end(ap);
    fflush(stream);
}

/* ── kernel_info (used once by cgbn_stage1.cu) ──────────────────────────── */

void kernel_info(const void *func, int verbose) {
    if (verbose >= OUTPUT_VERBOSE) {
        struct cudaFuncAttributes attr;
        cudaError_t err = cudaFuncGetAttributes(&attr, func);
        if (err == cudaSuccess) {
            outputf(OUTPUT_VERBOSE,
                    "GPU: kernel binaryVersion=%d ptxVersion=%d "
                    "maxThreadsPerBlock=%d numRegs=%d sharedMemPerBlock=%zu bytes\n",
                    attr.binaryVersion, attr.ptxVersion, attr.maxThreadsPerBlock,
                    attr.numRegs, attr.sharedSizeBytes);
        }
    }
}

/* ── Backend hooks ──────────────────────────────────────────────────────── */

extern "C" void ecm_backend_print_kernels(FILE *out) {
    fprintf(out, "CUDA/CGBN backend: compiled kernel sizes (bits):\n");
#ifdef ECM_CUDA_FULL_BUILD
    fprintf(out, "  128 192 256 384 512 768 1024 1280 1536 ... 32768 (full build)\n");
#else
    fprintf(out, "  128 192 256 384 512 768 1024 (dev build)\n");
    fprintf(out, "  define ECM_CUDA_FULL_BUILD to build the full kernel set.\n");
#endif
    fprintf(out, "  note: --mul/--sqr/--add/--sub/--special_mult are OpenCL-only "
                 "and ignored by the CUDA backend.\n");
    fflush(out);
}

extern "C" int ecm_backend_prepare(size_t n_log2, int verbose, int device_index,
                                   const char *gpu_mul_path, const char *gpu_sqr_path,
                                   const char *gpu_add_path, const char *gpu_sub_path,
                                   const char *gpu_special_mult_path) {
    (void)n_log2;
    (void)gpu_mul_path; (void)gpu_sqr_path; (void)gpu_add_path;
    (void)gpu_sub_path; (void)gpu_special_mult_path;

    ecm_cuda_set_verbose(verbose);

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        ecm_ts_fprintf(stderr, "GPU: no CUDA devices available: %s\n",
                       cudaGetErrorString(err));
        return 1;
    }

    ecm_ts_fprintf(stdout, "Available CUDA devices:\n");
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp p;
        if (cudaGetDeviceProperties(&p, i) == cudaSuccess) {
            ecm_ts_fprintf(stdout,
                           "  [%d] %s | CC %d.%d | %d SMs | %.0f MB\n",
                           i, p.name, p.major, p.minor, p.multiProcessorCount,
                           (double)p.totalGlobalMem / (1024.0 * 1024.0));
        }
    }

    int dev = (device_index < 0) ? 0 : device_index;
    if (dev >= count) {
        ecm_ts_fprintf(stderr, "GPU: requested device %d out of range (%d present)\n",
                       dev, count);
        return 1;
    }

    err = cudaSetDevice(dev);
    if (err != cudaSuccess) {
        ecm_ts_fprintf(stderr, "GPU: cudaSetDevice(%d) failed: %s\n", dev,
                       cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, dev) == cudaSuccess) {
        ecm_ts_fprintf(stdout,
                       "GPU: will use device %d: %s, compute capability %d.%d, %d MPs.\n",
                       dev, p.name, p.major, p.minor, p.multiProcessorCount);
        ecm_ts_fprintf(stdout,
                       "GPU: maxSharedPerBlock = %zu maxThreadsPerBlock = %d "
                       "maxRegsPerBlock = %d\n",
                       p.sharedMemPerBlock, p.maxThreadsPerBlock, p.regsPerBlock);
    }

    /* Light context warmup (blocking sync scheduling + establish context). */
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    err = cudaFree(0);
    if (err != cudaSuccess) {
        ecm_ts_fprintf(stderr, "GPU: context init failed: %s\n",
                       cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

extern "C" int ecm_backend_stage1(mpz_t *factors, int *array_found,
                                  const mpz_t N, const mpz_t s,
                                  uint32_t curves, uint32_t *sigma,
                                  unsigned long checkpoint_interval_ms,
                                  float *gputime, int verbose,
                                  const char *gpu_mul_path, const char *gpu_sqr_path,
                                  const char *gpu_add_path, const char *gpu_sub_path,
                                  const char *gpu_special_mult_path) {
    (void)gpu_mul_path; (void)gpu_sqr_path; (void)gpu_add_path;
    (void)gpu_sub_path; (void)gpu_special_mult_path;

    ecm_cuda_set_verbose(verbose);
    return cgbn_ecm_stage1(factors, array_found, N, s, curves, sigma,
                           checkpoint_interval_ms, gputime, verbose);
}
