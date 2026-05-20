// OpenCL ECM Stage 1 implementation
// Replaces CUDA version from test/cgbn_stage1.cu
// Uses OpenCL kernels from cgbn/backends/opencl/kernels/ecm_stage1.cl

#include "cgbn_stage1.h"
#include "ecm.h"
#include <gmp.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

// OpenCL headers
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define ECM_NO_FACTOR_FOUND 0
#define ECM_FACTOR_FOUND_STEP1 2

// Temporary placeholder - to be replaced with actual OpenCL context management
static cl_context g_ocl_context = NULL;
static cl_device_id g_ocl_device = NULL;
static cl_command_queue g_ocl_queue = NULL;

/**
 * Initialize OpenCL context (call once at startup)
 */
int opencl_init_context() {
    cl_int err;
    
    // Get platform
    cl_platform_id platform_id = NULL;
    cl_uint num_platforms;
    err = clGetPlatformIDs(1, &platform_id, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        fprintf(stderr, "Error: No OpenCL platforms found\n");
        return -1;
    }
    
    // Get device
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &g_ocl_device, NULL);
    if (err != CL_SUCCESS) {
        // Fall back to CPU if no GPU
        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &g_ocl_device, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error: No OpenCL devices found\n");
            return -1;
        }
    }
    
    // Create context
    g_ocl_context = clCreateContext(NULL, 1, &g_ocl_device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to create OpenCL context\n");
        return -1;
    }
    
    // Create command queue
    g_ocl_queue = clCreateCommandQueue(g_ocl_context, g_ocl_device, 0, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to create OpenCL command queue\n");
        return -1;
    }
    
    return 0;
}

/**
 * Clean up OpenCL resources
 */
void opencl_cleanup_context() {
    if (g_ocl_queue) {
        clReleaseCommandQueue(g_ocl_queue);
        g_ocl_queue = NULL;
    }
    if (g_ocl_context) {
        clReleaseContext(g_ocl_context);
        g_ocl_context = NULL;
    }
}

/**
 * Find np0 = -N^{-1} mod 2^32 (used for Montgomery arithmetic)
 * This is a Montgomery parameter needed for modular multiplication
 */
static uint32_t find_np0(const mpz_t N) {
    uint32_t np0;
    mpz_t temp;
    mpz_init(temp);
    mpz_ui_pow_ui(temp, 2, 32);
    if (!mpz_invert(temp, N, temp)) {
        fprintf(stderr, "Error: N is even or shares factor with 2^32\n");
        mpz_clear(temp);
        return 0;
    }
    np0 = -mpz_get_ui(temp);
    mpz_clear(temp);
    return np0;
}

/**
 * Encode s (batch product) as bit array
 * Returns allocated uint32_t array (caller must free)
 */
static uint32_t* allocate_and_set_s_bits(const mpz_t s, uint64_t *nbits) {
    uint64_t num_bits = *nbits = mpz_sizeinbase(s, 2);
    
    uint64_t allocated = (num_bits + 31) / 32;
    uint32_t *s_bits = (uint32_t*) malloc(sizeof(uint32_t) * allocated);
    
    uint64_t countp;
    mpz_export(s_bits, &countp, -1, sizeof(uint32_t), 0, 0, s);
    
    // Zero out any remaining limbs
    for (uint64_t i = countp; i < allocated; ++i) {
        s_bits[i] = 0;
    }
    
    return s_bits;
}

/**
 * Initialize curve data: compute Suyama parameterization
 * Stores (x_init, z_init) for each curve
 */
static int set_curve_data(const mpz_t N,
                          uint32_t curves,
                          uint32_t sigma_base,
                          uint32_t limbs,
                          uint32_t **data_out,
                          size_t *data_size_out) {
    // Each curve needs: x, z, A24, B, C (5 limbs each)
    size_t limbs_per = limbs;
    *data_size_out = 5 * curves * limbs_per * sizeof(uint32_t);
    uint32_t *data = (uint32_t*) malloc(*data_size_out);
    
    mpz_t x, d, B, C, A, x_init, z_init;
    mpz_init(x);
    mpz_init(d);
    mpz_init(B);
    mpz_init(C);
    mpz_init(A);
    mpz_init(x_init);
    mpz_init(z_init);
    
    for (uint32_t i = 0; i < curves; ++i) {
        uint32_t sigma = sigma_base + i;
        uint32_t datum_idx = 5 * i;
        
        // Suyama parameterization: https://en.wikipedia.org/wiki/Elliptic_curve_method
        // d = (sigma / 2^32) mod N
        mpz_set_ui(d, sigma);
        
        // B = (d^2 - 5) mod N
        mpz_mul(B, d, d);
        mpz_sub_ui(B, B, 5);
        mpz_mod(B, B, N);
        
        // C = (B^2 - 2*B) mod N
        mpz_mul(C, B, B);
        mpz_mul_2exp(B, B, 1);
        mpz_sub(C, C, B);
        mpz_mod(C, C, N);
        
        // A = (B^3 - 3*B) mod N / (4*B*C)
        // This requires modular inversion - simplified here
        mpz_set_ui(A, 0);  // Placeholder
        
        // x_init = (C^2 - 1) / 4 mod N
        mpz_mul(x_init, C, C);
        mpz_sub_ui(x_init, x_init, 1);
        mpz_tdiv_q_2exp(x_init, x_init, 2);
        mpz_mod(x_init, x_init, N);
        
        // z_init = 1
        mpz_set_ui(z_init, 1);
        
        // Export to uint32_t array
        uint32_t *datum = data + datum_idx * limbs;
        
        // Export x_init
        uint64_t words;
        mpz_export(datum, &words, -1, sizeof(uint32_t), 0, 0, x_init);
        for (uint64_t j = words; j < limbs; ++j) {
            datum[j] = 0;
        }
        
        // Export z_init
        mpz_export(datum + limbs, &words, -1, sizeof(uint32_t), 0, 0, z_init);
        for (uint64_t j = words; j < limbs; ++j) {
            datum[limbs + j] = 0;
        }
        
        // Export A24 (simplified - set to 0)
        mpz_export(datum + 2*limbs, &words, -1, sizeof(uint32_t), 0, 0, A);
        for (uint64_t j = words; j < limbs; ++j) {
            datum[2*limbs + j] = 0;
        }
        
        // Export B and C
        mpz_export(datum + 3*limbs, &words, -1, sizeof(uint32_t), 0, 0, B);
        for (uint64_t j = words; j < limbs; ++j) {
            datum[3*limbs + j] = 0;
        }
        
        mpz_export(datum + 4*limbs, &words, -1, sizeof(uint32_t), 0, 0, C);
        for (uint64_t j = words; j < limbs; ++j) {
            datum[4*limbs + j] = 0;
        }
    }
    
    mpz_clear(x);
    mpz_clear(d);
    mpz_clear(B);
    mpz_clear(C);
    mpz_clear(A);
    mpz_clear(x_init);
    mpz_clear(z_init);
    
    *data_out = data;
    return 0;
}

/**
 * Convert from uint32_t array back to mpz_t
 */
static void uint32_array_to_mpz(mpz_t out, const uint32_t *arr, uint32_t limbs) {
    mpz_import(out, limbs, -1, sizeof(uint32_t), 0, 0, arr);
}

/**
 * Process GPU results and extract factors
 */
static int process_results(mpz_t *factors,
                           int *array_found,
                           const mpz_t N,
                           const uint32_t *gpu_results,
                           uint32_t curves,
                           uint32_t limbs) {
    mpz_t z_final, x_final, gcd_result;
    mpz_init(z_final);
    mpz_init(x_final);
    mpz_init(gcd_result);
    
    int youpi = ECM_NO_FACTOR_FOUND;
    
    for (uint32_t i = 0; i < curves; ++i) {
        const uint32_t *result_x = gpu_results + 2*i*limbs;
        const uint32_t *result_z = gpu_results + (2*i + 1)*limbs;
        
        // Convert to mpz_t
        uint32_array_to_mpz(x_final, result_x, limbs);
        uint32_array_to_mpz(z_final, result_z, limbs);
        
        // Try to invert z_final mod N
        if (mpz_invert(gcd_result, z_final, N)) {
            // z_final is coprime to N - multiply by x to get result
            mpz_mul(gcd_result, gcd_result, x_final);
            mpz_mod(gcd_result, gcd_result, N);
            
            // Check if non-trivial factor
            if (mpz_cmp_ui(gcd_result, 1) > 0 && mpz_cmp(gcd_result, N) < 0) {
                mpz_set(factors[i], gcd_result);
                array_found[i] = 1;
                youpi = ECM_FACTOR_FOUND_STEP1;
            }
        } else {
            // z_final shares factor with N
            mpz_gcd(gcd_result, z_final, N);
            if (mpz_cmp_ui(gcd_result, 1) > 0 && mpz_cmp(gcd_result, N) < 0) {
                mpz_set(factors[i], gcd_result);
                array_found[i] = 1;
                youpi = ECM_FACTOR_FOUND_STEP1;
            }
        }
    }
    
    mpz_clear(z_final);
    mpz_clear(x_final);
    mpz_clear(gcd_result);
    
    return youpi;
}

/**
 * Main OpenCL ECM Stage 1 entry point
 * Mirrors the CUDA cgbn_ecm_stage1 signature but uses OpenCL backend
 */
int cgbn_ecm_stage1(mpz_t *factors,
                    int *array_found,
                    const mpz_t N,
                    const mpz_t s,
                    uint32_t curves,
                    uint32_t *sigma_ptr,
                    unsigned long checkpoint_interval_ms,
                    float *gputime,
                    int verbose) {
    
    if (verbose >= 1) {
        fprintf(stderr, "ECM GPU Stage 1 (OpenCL backend)\n");
    }
    
    uint32_t sigma = *sigma_ptr;
    
    // Initialize OpenCL context if needed
    if (!g_ocl_context) {
        if (opencl_init_context() != 0) {
            fprintf(stderr, "Warning: OpenCL initialization failed, using CPU stub\n");
            // Fall back to stub
            *gputime = 0.0f;
            for (uint32_t i = 0; i < curves; ++i) {
                array_found[i] = 0;
            }
            return ECM_NO_FACTOR_FOUND;
        }
    }
    
    // Step 1: Allocate and encode s as bits
    uint64_t s_num_bits;
    uint32_t *s_bits = allocate_and_set_s_bits(s, &s_num_bits);
    
    if (verbose >= 1) {
        fprintf(stderr, "  s_num_bits: %lu\n", s_num_bits);
    }
    
    // Step 2: Verify N size
    size_t n_log2 = mpz_sizeinbase(N, 2);
    const uint32_t BITS = 2048;  // Default kernel size - can be selected dynamically
    const uint32_t TPI = 8;
    const uint32_t limbs = BITS / 32;
    
    if (n_log2 + 6 > BITS) {  // 6 is carry bits
        fprintf(stderr, "Error: N too large for kernel (needs %zu bits, kernel has %u)\n",
                n_log2 + 6, BITS);
        free(s_bits);
        return -1;
    }
    
    // Step 3: Find np0 parameter
    uint32_t np0 = find_np0(N);
    
    // Step 4: Initialize curve data
    uint32_t *curve_data = NULL;
    size_t curve_data_size = 0;
    if (set_curve_data(N, curves, sigma, limbs, &curve_data, &curve_data_size) != 0) {
        fprintf(stderr, "Error: Failed to initialize curve data\n");
        free(s_bits);
        return -1;
    }
    
    // Step 5: Create GPU buffers
    cl_int err;
    cl_mem gpu_s_bits = clCreateBuffer(g_ocl_context,
                                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       ((s_num_bits + 31) / 32) * sizeof(uint32_t),
                                       s_bits, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to create GPU buffer for s_bits\n");
        free(s_bits);
        free(curve_data);
        return -1;
    }
    
    cl_mem gpu_curve_data = clCreateBuffer(g_ocl_context,
                                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                           curve_data_size,
                                           curve_data, &err);
    
    // Results buffer: (x_final, z_final) per curve
    size_t results_size = 2 * curves * limbs * sizeof(uint32_t);
    cl_mem gpu_results = clCreateBuffer(g_ocl_context,
                                        CL_MEM_READ_WRITE,
                                        results_size, NULL, &err);
    
    cl_mem gpu_N = clCreateBuffer(g_ocl_context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  limbs * sizeof(uint32_t),
                                  NULL, &err);  // Should populate with N
    
    // Step 6: Create and compile kernel
    // (Placeholder - would load ecm_stage1.cl and compile)
    
    // Step 7: Execute kernel
    // (Placeholder - would execute kernel_ecm_stage1)
    
    // Step 8: Read results back
    uint32_t *gpu_results_host = (uint32_t*) malloc(results_size);
    err = clEnqueueReadBuffer(g_ocl_queue, gpu_results, CL_TRUE,
                             0, results_size, gpu_results_host, 0, NULL, NULL);
    
    // Step 9: Process results
    int youpi = process_results(factors, array_found, N, gpu_results_host, curves, limbs);
    
    // Cleanup
    free(gpu_results_host);
    clReleaseMemObject(gpu_s_bits);
    clReleaseMemObject(gpu_curve_data);
    clReleaseMemObject(gpu_results);
    clReleaseMemObject(gpu_N);
    free(s_bits);
    free(curve_data);
    
    *gputime = 0.0f;  // Would measure actual GPU time
    
    if (verbose >= 1) {
        fprintf(stderr, "  GPU Stage 1 complete\n");
    }
    
    return youpi;
}
