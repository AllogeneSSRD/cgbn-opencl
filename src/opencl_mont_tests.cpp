#include "cgbn_opencl.h"

#include <CL/cl.h>
#include <gmp.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

void fill_from_gmp(mpz_t v, uint32_t *out_words, size_t words) {
    mpz_t tmp, mod;
    mpz_init(tmp);
    mpz_init(mod);
    mpz_ui_pow_ui(mod, 2, (unsigned long)(words * 32));
    mpz_mod(tmp, v, mod);

    size_t count = 0;
    mpz_export(out_words, &count, -1, sizeof(uint32_t), 0, 0, tmp);
    for (size_t i = count; i < words; ++i) out_words[i] = 0u;

    mpz_clear(tmp);
    mpz_clear(mod);
}

void gmp_from_words(mpz_t out, const uint32_t *words, size_t wordCount) {
    mpz_import(out, wordCount, -1, sizeof(uint32_t), 0, 0, words);
}

uint32_t inv32_odd(uint32_t x) {
    // Newton iteration for inverse modulo 2^32, x must be odd.
    uint64_t y = 1;
    for (int i = 0; i < 5; ++i) {
        y = y * (2ull - (uint64_t)x * y);
        y &= 0xFFFFFFFFull;
    }
    return (uint32_t)y;
}

bool runOpenClMontgomeryBenchmark(int iterations, int instances) {
    constexpr int BITS = 128;
    const size_t WORDS = BITS / 32; // 4

    std::cout << "OpenCL mont benchmark: " << BITS << "-bit, iterations=" << iterations
              << ", instances=" << instances << std::endl;

    // Choose an odd modulus n < 2^128
    mpz_t n_gmp, a_gmp, b_gmp, r_mul_gmp, r_sqr_gmp, R, Rinv, tmp;
    mpz_inits(n_gmp, a_gmp, b_gmp, r_mul_gmp, r_sqr_gmp, R, Rinv, tmp, nullptr);

    mpz_set_str(n_gmp, "340282366920938463463374607431768211283", 10); // 2^128 - 173, odd
    mpz_set_str(a_gmp, "123456789012345678901234567890123456", 10);
    mpz_set_str(b_gmp, "98765432109876543210987654321098765", 10);
    mpz_mod(a_gmp, a_gmp, n_gmp);
    mpz_mod(b_gmp, b_gmp, n_gmp);

    mpz_ui_pow_ui(R, 2, BITS);
    if (mpz_invert(Rinv, R, n_gmp) == 0) {
        std::cerr << "Failed to invert R modulo n" << std::endl;
        mpz_clears(n_gmp, a_gmp, b_gmp, r_mul_gmp, r_sqr_gmp, R, Rinv, tmp, nullptr);
        return false;
    }

    // Reference: mont_mul(a,b) = a*b*R^{-1} mod n
    mpz_mul(tmp, a_gmp, b_gmp);
    mpz_mul(tmp, tmp, Rinv);
    mpz_mod(r_mul_gmp, tmp, n_gmp);

    mpz_mul(tmp, a_gmp, a_gmp);
    mpz_mul(tmp, tmp, Rinv);
    mpz_mod(r_sqr_gmp, tmp, n_gmp);

    std::vector<uint32_t> host_a((size_t)instances * WORDS);
    std::vector<uint32_t> host_b((size_t)instances * WORDS);
    std::vector<uint32_t> host_n((size_t)instances * WORDS);
    std::vector<uint32_t> host_out((size_t)instances * WORDS);

    std::vector<uint32_t> a_words(WORDS), b_words(WORDS), n_words(WORDS);
    fill_from_gmp(a_gmp, a_words.data(), WORDS);
    fill_from_gmp(b_gmp, b_words.data(), WORDS);
    fill_from_gmp(n_gmp, n_words.data(), WORDS);

    for (int i = 0; i < instances; ++i) {
        for (size_t j = 0; j < WORDS; ++j) {
            host_a[(size_t)i * WORDS + j] = a_words[j];
            host_b[(size_t)i * WORDS + j] = b_words[j];
            host_n[(size_t)i * WORDS + j] = n_words[j];
        }
    }

    if ((n_words[0] & 1u) == 0u) {
        std::cerr << "n must be odd for Montgomery" << std::endl;
        mpz_clears(n_gmp, a_gmp, b_gmp, r_mul_gmp, r_sqr_gmp, R, Rinv, tmp, nullptr);
        return false;
    }
    uint32_t inv = inv32_odd(n_words[0]);
    uint32_t np0 = 0u - inv; // np0 * n0 == 0xFFFFFFFF (mod 2^32)

    cgbn::opencl::context_t ctx;
    cl_int err = cgbn::opencl::create_context(ctx);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL context: " << err << std::endl;
        mpz_clears(n_gmp, a_gmp, b_gmp, r_mul_gmp, r_sqr_gmp, R, Rinv, tmp, nullptr);
        return false;
    }

    std::string src = cgbn::opencl::load_text_file("cgbn/backends/opencl/kernels/mont.cl");
    if (src.empty()) {
        std::cerr << "Failed to load mont.cl" << std::endl;
        cgbn::opencl::destroy_context(ctx);
        mpz_clears(n_gmp, a_gmp, b_gmp, r_mul_gmp, r_sqr_gmp, R, Rinv, tmp, nullptr);
        return false;
    }

    cl_int buildErr = CL_SUCCESS;
    cl_program program = cgbn::opencl::build_program_from_source(ctx, src.c_str(), "", buildErr);
    if (program == nullptr || buildErr != CL_SUCCESS) {
        std::cerr << "Failed to build mont program: " << buildErr << std::endl;
        cgbn::opencl::destroy_context(ctx);
        mpz_clears(n_gmp, a_gmp, b_gmp, r_mul_gmp, r_sqr_gmp, R, Rinv, tmp, nullptr);
        return false;
    }

    size_t totalWords = (size_t)instances * WORDS;
    cl_mem bufA = clCreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(uint32_t) * totalWords, host_a.data(), &err);
    if (err != CL_SUCCESS) return false;
    cl_mem bufB = clCreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(uint32_t) * totalWords, host_b.data(), &err);
    if (err != CL_SUCCESS) return false;
    cl_mem bufN = clCreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(uint32_t) * totalWords, host_n.data(), &err);
    if (err != CL_SUCCESS) return false;
    cl_mem bufOut = clCreateBuffer(ctx.ctx, CL_MEM_READ_WRITE,
                                   sizeof(uint32_t) * totalWords, nullptr, &err);
    if (err != CL_SUCCESS) return false;

    size_t global = (size_t)instances;
    cl_uint limbs = (cl_uint)WORDS;

    // mont_mul
    cl_kernel kMul = clCreateKernel(program, "cgbn_mont_mul", &err);
    if (err != CL_SUCCESS) return false;
    clSetKernelArg(kMul, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kMul, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kMul, 2, sizeof(cl_mem), &bufN);
    clSetKernelArg(kMul, 3, sizeof(cl_mem), &bufOut);
    clSetKernelArg(kMul, 4, sizeof(cl_uint), &np0);
    clSetKernelArg(kMul, 5, sizeof(cl_uint), &limbs);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        err = clEnqueueNDRangeKernel(ctx.queue, kMul, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) return false;
    }
    clFinish(ctx.queue);
    auto t1 = std::chrono::high_resolution_clock::now();
    double mul_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    err = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                              host_out.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) return false;

    std::vector<uint32_t> exp_mul(WORDS);
    fill_from_gmp(r_mul_gmp, exp_mul.data(), WORDS);
    bool okMul = true;
    for (size_t i = 0; i < WORDS; ++i) {
        if (host_out[i] != exp_mul[i]) {
            okMul = false;
            break;
        }
    }

    // mont_sqr
    cl_kernel kSqr = clCreateKernel(program, "cgbn_mont_sqr", &err);
    if (err != CL_SUCCESS) return false;
    clSetKernelArg(kSqr, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kSqr, 1, sizeof(cl_mem), &bufN);
    clSetKernelArg(kSqr, 2, sizeof(cl_mem), &bufOut);
    clSetKernelArg(kSqr, 3, sizeof(cl_uint), &np0);
    clSetKernelArg(kSqr, 4, sizeof(cl_uint), &limbs);

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        err = clEnqueueNDRangeKernel(ctx.queue, kSqr, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) return false;
    }
    clFinish(ctx.queue);
    t1 = std::chrono::high_resolution_clock::now();
    double sqr_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    err = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                              host_out.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) return false;

    std::vector<uint32_t> exp_sqr(WORDS);
    fill_from_gmp(r_sqr_gmp, exp_sqr.data(), WORDS);
    bool okSqr = true;
    for (size_t i = 0; i < WORDS; ++i) {
        if (host_out[i] != exp_sqr[i]) {
            okSqr = false;
            break;
        }
    }

    std::cout << "MontMul: OpenCL time (ms)=" << mul_ms << ", equal=" << (okMul ? "YES" : "NO") << std::endl;
    std::cout << "MontSqr: OpenCL time (ms)=" << sqr_ms << ", equal=" << (okSqr ? "YES" : "NO") << std::endl;

    clReleaseKernel(kMul);
    clReleaseKernel(kSqr);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufN);
    clReleaseMemObject(bufOut);
    clReleaseProgram(program);
    cgbn::opencl::destroy_context(ctx);

    mpz_clears(n_gmp, a_gmp, b_gmp, r_mul_gmp, r_sqr_gmp, R, Rinv, tmp, nullptr);
    return okMul && okSqr;
}

} // namespace

#ifdef BUILD_OPENCL_MONT_MAIN
#include <cstdlib>

int main(int argc, char **argv) {
    int iterations = 1000;
    int instances = 256;
    if (argc >= 2) iterations = std::stoi(std::string(argv[1]));
    if (argc >= 3) instances = std::stoi(std::string(argv[2]));

    bool ok = runOpenClMontgomeryBenchmark(iterations, instances);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
#endif
