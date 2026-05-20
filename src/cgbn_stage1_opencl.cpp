// OpenCL ECM Stage 1 host (mirrors test/cgbn_stage1.cu)

#include "cgbn_stage1.h"
#include "ecm.h"
#include "cgbn_opencl.h"

#include <CL/cl.h>
#include <gmp.h>

#include <algorithm>
#include <cstdarg>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CARRY_BITS 6

static cgbn::opencl::context_t g_ctx;
static bool g_ctx_ready = false;
static cl_program g_ecm_program = nullptr;
static cl_kernel g_ecm_kernel = nullptr;
static uint32_t g_kernel_limbs = 0;

static void ocl_log_verbose(int verbose, const char *fmt, ...) {
    if (verbose < 1) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}

static void from_mpz(const mpz_t s, uint32_t *x, uint32_t count) {
    size_t words;
    if (mpz_sizeinbase(s, 2) > (size_t)count * 32) {
        fprintf(stderr, "from_mpz: value does not fit in %u limbs\n", count);
        exit(EXIT_FAILURE);
    }
    mpz_export(x, &words, -1, sizeof(uint32_t), 0, 0, s);
    while (words < count) {
        x[words++] = 0;
    }
}

static void to_mpz(mpz_t r, const uint32_t *x, uint32_t count) {
    mpz_import(r, count, -1, sizeof(uint32_t), 0, 0, x);
}

static uint32_t find_np0(const mpz_t N) {
    mpz_t temp;
    mpz_init(temp);
    mpz_ui_pow_ui(temp, 2, 32);
    if (!mpz_invert(temp, N, temp)) {
        mpz_clear(temp);
        return 0;
    }
    uint32_t np0 = (uint32_t)(-mpz_get_ui(temp));
    mpz_clear(temp);
    return np0;
}

static void to_montgomery(uint32_t *out, const mpz_t bn, const mpz_t N, uint32_t bits, uint32_t limbs) {
    mpz_t R, t;
    mpz_init(R);
    mpz_init(t);
    mpz_ui_pow_ui(R, 2, bits);
    mpz_mul(t, bn, R);
    mpz_mod(t, t, N);
    from_mpz(t, out, limbs);
    mpz_clear(R);
    mpz_clear(t);
}

static void from_montgomery(mpz_t out, const mpz_t mont, const mpz_t N, uint32_t bits) {
    mpz_t R, t;
    mpz_init(R);
    mpz_init(t);
    mpz_ui_pow_ui(R, 2, bits);
    if (!mpz_invert(R, R, N)) {
        mpz_clear(R);
        mpz_clear(t);
        mpz_set(out, mont);
        return;
    }
    mpz_mul(t, mont, R);
    mpz_mod(out, t, N);
    mpz_clear(R);
    mpz_clear(t);
}

static uint32_t *allocate_and_set_s_bits(const mpz_t s, uint64_t *nbits) {
    uint64_t num_bits = *nbits = mpz_sizeinbase(s, 2);
    uint64_t allocated = (num_bits + 31) / 32;
    uint32_t *s_bits = (uint32_t *)malloc(sizeof(uint32_t) * allocated);
    uint64_t countp;
    mpz_export(s_bits, &countp, -1, sizeof(uint32_t), 0, 0, s);
    for (uint64_t i = countp; i < allocated; ++i) {
        s_bits[i] = 0;
    }
    return s_bits;
}

// CUDA set_p_2p: N, P=(2,1), 2P=(9, 64*d+8) per curve, then Montgomery on GPU.
// Here we convert to Montgomery on host before upload.
static uint32_t *set_p_2p(const mpz_t N, uint32_t curves, uint32_t sigma, uint32_t BITS,
                          size_t *data_size) {
    const uint32_t limbs_per = BITS / 32;
    *data_size = 5 * curves * limbs_per * sizeof(uint32_t);
    uint32_t *data = (uint32_t *)malloc(*data_size);
    uint32_t *datum = data;

    mpz_t x, t;
    mpz_init(x);
    mpz_init(t);

    for (uint32_t index = 0; index < curves; index++) {
        uint32_t d = sigma + index;

        from_mpz(N, datum + 0 * limbs_per, limbs_per);

        mpz_set_ui(x, 2);
        to_montgomery(datum + 1 * limbs_per, x, N, BITS, limbs_per);
        mpz_set_ui(x, 1);
        to_montgomery(datum + 2 * limbs_per, x, N, BITS, limbs_per);

        mpz_set_ui(x, 9);
        to_montgomery(datum + 3 * limbs_per, x, N, BITS, limbs_per);

        mpz_ui_pow_ui(t, 2, 32);
        mpz_invert(t, t, N);
        mpz_mul_ui(t, t, d);
        mpz_mul_ui(t, t, 64);
        mpz_add_ui(t, t, 8);
        mpz_mod(t, t, N);
        to_montgomery(datum + 4 * limbs_per, t, N, BITS, limbs_per);

        datum += 5 * limbs_per;
    }

    mpz_clear(x);
    mpz_clear(t);
    return data;
}

static int findfactor(mpz_t factor, const mpz_t N, const mpz_t x_final, const mpz_t z_final) {
    if (mpz_invert(factor, z_final, N)) {
        mpz_mul(factor, x_final, factor);
        mpz_mod(factor, factor, N);
        return ECM_NO_FACTOR_FOUND;
    }
    mpz_gcd(factor, z_final, N);
    return ECM_FACTOR_FOUND_STEP1;
}

static int process_results(mpz_t *factors, int *array_found, const mpz_t N,
                           const uint32_t *data, uint32_t cgbn_bits, int curves,
                           uint32_t sigma) {
    mpz_t x_final, z_final, modulo, x_std, z_std;
    mpz_init(modulo);
    mpz_init(x_final);
    mpz_init(z_final);
    mpz_init(x_std);
    mpz_init(z_std);

    const uint32_t limbs_per = cgbn_bits / 32;
    int youpi = ECM_NO_FACTOR_FOUND;
    int errors = 0;

    for (int i = 0; i < curves; i++) {
        const uint32_t *datum = data + (5 * i * limbs_per);

        to_mpz(modulo, datum + 0 * limbs_per, limbs_per);
        if (mpz_cmp(modulo, N) != 0) {
            fprintf(stderr, "GPU: curve %d modulus mismatch\n", i);
        }

        to_mpz(x_final, datum + 1 * limbs_per, limbs_per);
        to_mpz(z_final, datum + 2 * limbs_per, limbs_per);
        from_montgomery(x_std, x_final, N, cgbn_bits);
        from_montgomery(z_std, z_final, N, cgbn_bits);

        if (mpz_cmp_ui(x_std, 2) == 0 && mpz_cmp_ui(z_std, 1) == 0) {
            errors++;
            if (errors < 10) {
                fprintf(stderr, "GPU: curve %d may not have computed (still at initial point)\n", i);
            }
        }

        array_found[i] = findfactor(factors[i], N, x_std, z_std);
        if (array_found[i] != ECM_NO_FACTOR_FOUND) {
            youpi = array_found[i];
            fprintf(stderr, "GPU: factor found in Step 1 with curve %d (-sigma %d:%u)\n",
                    i, ECM_PARAM_BATCH_32BITS_D, sigma + (uint32_t)i);
        }
    }

    mpz_clear(modulo);
    mpz_clear(x_final);
    mpz_clear(z_final);
    mpz_clear(x_std);
    mpz_clear(z_std);

    if (errors > 2) {
        return ECM_ERROR;
    }
    return youpi;
}

static uint32_t select_bits(size_t n_log2) {
    static const uint32_t candidates[] = {
        512, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096,
        4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192};
    for (uint32_t b : candidates) {
        if (n_log2 + CARRY_BITS <= b) {
            return b;
        }
    }
    return 0;
}

static int ensure_ecm_kernel(uint32_t limbs, int verbose) {
    if (g_ecm_kernel && g_kernel_limbs == limbs) {
        return 0;
    }
    if (g_ecm_kernel) {
        clReleaseKernel(g_ecm_kernel);
        g_ecm_kernel = nullptr;
    }
    if (g_ecm_program) {
        clReleaseProgram(g_ecm_program);
        g_ecm_program = nullptr;
    }

    if (!g_ctx_ready) {
        cl_int err = cgbn::opencl::create_context(g_ctx);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "OpenCL: failed to create context (%d)\n", err);
            return -1;
        }
        g_ctx_ready = true;
    }

    std::string src = cgbn::opencl::load_text_file("cgbn/backends/opencl/kernels/ecm_stage1.cl");
    if (src.empty()) {
        fprintf(stderr, "OpenCL: failed to load ecm_stage1.cl (run from project root)\n");
        return -1;
    }

    char opts[64];
    snprintf(opts, sizeof(opts), "-DMAX_LIMBS=%u", limbs);

    cl_int buildErr = CL_SUCCESS;
    g_ecm_program = cgbn::opencl::build_program_from_source(g_ctx, src.c_str(), opts, buildErr);
    if (g_ecm_program == nullptr || buildErr != CL_SUCCESS) {
        fprintf(stderr, "OpenCL: failed to build ecm_stage1.cl\n");
        return -1;
    }

    cl_int err;
    g_ecm_kernel = clCreateKernel(g_ecm_program, "kernel_double_add", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL: kernel_double_add not found (%d)\n", err);
        return -1;
    }
    g_kernel_limbs = limbs;
    ocl_log_verbose(verbose, "OpenCL: built kernel MAX_LIMBS=%u\n", limbs);
    return 0;
}

int cgbn_ecm_stage1(mpz_t *factors, int *array_found, const mpz_t N, const mpz_t s,
                    uint32_t curves, uint32_t *sigma_ptr,
                    unsigned long checkpoint_interval_ms, float *gputime, int verbose) {
    (void)checkpoint_interval_ms;

    uint32_t sigma = *sigma_ptr;
    if (sigma == 0 || (uint64_t)sigma + curves > 0xFFFFFFFFull) {
        fprintf(stderr, "Invalid sigma/curves range\n");
        return ECM_ERROR;
    }

    uint64_t s_num_bits;
    uint32_t *s_bits = allocate_and_set_s_bits(s, &s_num_bits);
    if (!s_bits) {
        return ECM_ERROR;
    }

    size_t n_log2 = mpz_sizeinbase(N, 2);
    uint32_t BITS = select_bits(n_log2);
    if (BITS == 0) {
        fprintf(stderr, "No OpenCL kernel large enough for N (%zu bits)\n", n_log2);
        free(s_bits);
        return ECM_ERROR;
    }
    const uint32_t limbs = BITS / 32;

    if (ensure_ecm_kernel(limbs, verbose) != 0) {
        free(s_bits);
        return ECM_ERROR;
    }

    uint32_t np0 = find_np0(N);
    size_t data_size = 0;
    uint32_t *data = set_p_2p(N, curves, sigma, BITS, &data_size);

    uint64_t s_partial = 1; // first bit handled by initial P / 2P setup
    uint64_t batch_size = 200;

    cl_int err;
    uint32_t s_words = (uint32_t)((s_num_bits + 31) / 32);
    cl_mem gpu_s_bits = clCreateBuffer(g_ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(uint32_t) * s_words, s_bits, &err);
    cl_mem gpu_data = clCreateBuffer(g_ctx.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                     data_size, data, &err);

    auto t_global_start = std::chrono::high_resolution_clock::now();

    ocl_log_verbose(verbose, "OpenCL ECM<%u> N=%zu bits, curves=%u, s=%lu bits\n",
                    BITS, n_log2, curves, s_num_bits);

    int batches_complete = 0;
    while (s_partial < s_num_bits) {
        uint64_t this_batch = std::min(batch_size, s_num_bits - s_partial);

        ocl_log_verbose(verbose, "  batch %d: bits %lu..%lu (%.1f%%)\n", batches_complete,
                        s_partial, s_partial + this_batch,
                        100.0 * (double)s_partial / (double)s_num_bits);

        cl_ulong s_num_bits_arg = (cl_ulong)s_num_bits;
        cl_ulong s_start_arg = (cl_ulong)s_partial;
        cl_ulong s_interval_arg = (cl_ulong)this_batch;
        cl_uint count_arg = curves;
        cl_uint sigma_arg = sigma;
        cl_uint np0_arg = np0;
        cl_uint limbs_arg = limbs;

        err = clSetKernelArg(g_ecm_kernel, 0, sizeof(cl_mem), &gpu_s_bits);
        err |= clSetKernelArg(g_ecm_kernel, 1, sizeof(cl_ulong), &s_num_bits_arg);
        err |= clSetKernelArg(g_ecm_kernel, 2, sizeof(cl_ulong), &s_start_arg);
        err |= clSetKernelArg(g_ecm_kernel, 3, sizeof(cl_ulong), &s_interval_arg);
        err |= clSetKernelArg(g_ecm_kernel, 4, sizeof(cl_mem), &gpu_data);
        err |= clSetKernelArg(g_ecm_kernel, 5, sizeof(cl_uint), &count_arg);
        err |= clSetKernelArg(g_ecm_kernel, 6, sizeof(cl_uint), &sigma_arg);
        err |= clSetKernelArg(g_ecm_kernel, 7, sizeof(cl_uint), &np0_arg);
        err |= clSetKernelArg(g_ecm_kernel, 8, sizeof(cl_uint), &limbs_arg);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "clSetKernelArg failed\n");
            break;
        }

        size_t global = curves;
        auto t0 = std::chrono::high_resolution_clock::now();
        err = clEnqueueNDRangeKernel(g_ctx.queue, g_ecm_kernel, 1, nullptr, &global, nullptr,
                                     0, nullptr, nullptr);
        clFinish(g_ctx.queue);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (err != CL_SUCCESS) {
            fprintf(stderr, "kernel enqueue failed (%d)\n", err);
            break;
        }

        double batch_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (batch_ms < 80.0) {
            batch_size = 11 * batch_size / 10;
        } else if (batch_ms > 120.0) {
            batch_size = std::max<uint64_t>(100, 9 * batch_size / 10);
        }

        s_partial += this_batch;
        batches_complete++;
    }

    err = clEnqueueReadBuffer(g_ctx.queue, gpu_data, CL_TRUE, 0, data_size, data, 0, nullptr,
                              nullptr);

    auto t_global_end = std::chrono::high_resolution_clock::now();
    if (gputime) {
        *gputime = (float)std::chrono::duration<double, std::milli>(t_global_end - t_global_start)
                       .count();
    }

    int youpi = ECM_NO_FACTOR_FOUND;
    if (err == CL_SUCCESS) {
        youpi = process_results(factors, array_found, N, data, BITS, (int)curves, sigma);
    } else {
        youpi = ECM_ERROR;
    }

    clReleaseMemObject(gpu_s_bits);
    clReleaseMemObject(gpu_data);
    free(s_bits);
    free(data);

    *sigma_ptr = sigma;
    return youpi;
}
