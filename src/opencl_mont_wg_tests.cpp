#include "cgbn_opencl.h"

#include <CL/cl.h>
#include <gmp.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

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
    uint64_t y = 1;
    for (int i = 0; i < 5; ++i) {
        y = y * (2ull - (uint64_t)x * y);
        y &= 0xFFFFFFFFull;
    }
    return (uint32_t)y;
}

bool runOpenClMontgomeryWGBenchmark(int iterations, int instances, int tpi) {
    constexpr int BITS = 4096;
    const size_t WORDS = BITS / 32; // 128

    std::cout << "\n=== OpenCL Montgomery WG Benchmark (TPI=" << tpi << ") ===" << std::endl;
    std::cout << "Config: " << BITS << "-bit, iterations=" << iterations
              << ", instances=" << instances << ", TPI=" << tpi << std::endl;

    // Verify TPI divides WORDS
    if (WORDS % tpi != 0) {
        std::cerr << "Error: TPI=" << tpi << " does not divide WORDS=" << WORDS << std::endl;
        return false;
    }

    mpz_t n_gmp, a_gmp, b_gmp, r_mul_gmp, r_sqr_gmp, R, Rinv, tmp;
    mpz_inits(n_gmp, a_gmp, b_gmp, r_mul_gmp, r_sqr_gmp, R, Rinv, tmp, nullptr);

    // Generate modulus n = 2^BITS - 189
    mpz_ui_pow_ui(n_gmp, 2, BITS);
    mpz_sub_ui(n_gmp, n_gmp, 189u);

    mpz_set_str(a_gmp, "1234567890123456789012345678901234567890", 10);
    mpz_set_str(b_gmp, "987654321098765432109876543210987654321", 10);
    mpz_mod(a_gmp, a_gmp, n_gmp);
    mpz_mod(b_gmp, b_gmp, n_gmp);

    mpz_ui_pow_ui(R, 2, BITS);
    if (mpz_invert(Rinv, R, n_gmp) == 0) {
        std::cerr << "Failed to invert R modulo n" << std::endl;
        mpz_clears(n_gmp, a_gmp, b_gmp, r_mul_gmp, r_sqr_gmp, R, Rinv, tmp, nullptr);
        return false;
    }

    // Compute reference results
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
    uint32_t np0 = 0u - inv;

    cgbn::opencl::context_t ctx;
    cl_int err = cgbn::opencl::create_context(ctx);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL context: " << err << std::endl;
        mpz_clears(n_gmp, a_gmp, b_gmp, r_mul_gmp, r_sqr_gmp, R, Rinv, tmp, nullptr);
        return false;
    }

    // Load both mont.cl and mont_wg.cl
    std::string src_standard = cgbn::opencl::load_text_file("cgbn/backends/opencl/kernels/mont.cl");
    std::string src_wg = cgbn::opencl::load_text_file("cgbn/backends/opencl/kernels/mont_wg.cl");
    
    if (src_standard.empty() || src_wg.empty()) {
        std::cerr << "Failed to load kernel files" << std::endl;
        cgbn::opencl::destroy_context(ctx);
        mpz_clears(n_gmp, a_gmp, b_gmp, r_mul_gmp, r_sqr_gmp, R, Rinv, tmp, nullptr);
        return false;
    }

    // Combine both sources
    std::string combined_src = src_standard + "\n" + src_wg;

    // Build with TPI definition
    std::string build_opts = std::string("-DTPI=") + std::to_string(tpi) + " -DMAX_LIMBS=128";
    cl_int buildErr = CL_SUCCESS;
    cl_program program = cgbn::opencl::build_program_from_source(ctx, combined_src.c_str(), 
                                                                    build_opts.c_str(), buildErr);
    if (program == nullptr || buildErr != CL_SUCCESS) {
        std::cerr << "Failed to build mont WG program: " << buildErr << std::endl;
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

    cl_uint limbs = (cl_uint)WORDS;

    // Print device info
    {
        char devName[256] = {0};
        clGetDeviceInfo(ctx.device, CL_DEVICE_NAME, sizeof(devName) - 1, devName, nullptr);
        cl_uint computeUnits = 0;
        clGetDeviceInfo(ctx.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, nullptr);
        size_t maxWorkGroup = 0;
        clGetDeviceInfo(ctx.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroup), &maxWorkGroup, nullptr);
        std::cout << "Device: " << devName << ", compute_units=" << computeUnits
                  << ", max_work_group_size=" << maxWorkGroup << std::endl;
    }

    // Work-group kernel: global size = instances * TPI, local size = TPI
    size_t global_size_wg = (size_t)instances * tpi;
    size_t local_size_wg = tpi;
    
    // Local memory size: (MAX_LIMBS+1)*3 + MAX_LIMBS*2 + TPI words
    size_t local_mem_size = ((128u + 1u) * 3u + 128u * 2u + tpi) * sizeof(uint32_t);

    // Test cgbn_mont_mul_wg
    cl_kernel kMulWG = clCreateKernel(program, "cgbn_mont_mul_wg", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create cgbn_mont_mul_wg kernel: " << err << std::endl;
        return false;
    }

    clSetKernelArg(kMulWG, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kMulWG, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kMulWG, 2, sizeof(cl_mem), &bufN);
    clSetKernelArg(kMulWG, 3, sizeof(cl_mem), &bufOut);
    clSetKernelArg(kMulWG, 4, sizeof(cl_uint), &np0);
    clSetKernelArg(kMulWG, 5, sizeof(cl_uint), &limbs);
    clSetKernelArg(kMulWG, 6, local_mem_size, nullptr);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        err = clEnqueueNDRangeKernel(ctx.queue, kMulWG, 1, nullptr, &global_size_wg, &local_size_wg, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to enqueue cgbn_mont_mul_wg: " << err << std::endl;
            return false;
        }
    }
    clFinish(ctx.queue);
    auto t1 = std::chrono::high_resolution_clock::now();
    double mul_wg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    err = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                              host_out.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) return false;

    std::vector<uint32_t> exp_mul(WORDS);
    fill_from_gmp(r_mul_gmp, exp_mul.data(), WORDS);
    bool okMulWG = true;
    for (size_t i = 0; i < WORDS; ++i) {
        if (host_out[i] != exp_mul[i]) {
            okMulWG = false;
            std::cerr << "MontMul WG mismatch at word " << i << ": got 0x" << std::hex 
                      << host_out[i] << " expected 0x" << exp_mul[i] << std::dec << std::endl;
            break;
        }
    }

    // Test cgbn_mont_sqr_wg
    cl_kernel kSqrWG = clCreateKernel(program, "cgbn_mont_sqr_wg", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create cgbn_mont_sqr_wg kernel: " << err << std::endl;
        return false;
    }

    clSetKernelArg(kSqrWG, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kSqrWG, 1, sizeof(cl_mem), &bufN);
    clSetKernelArg(kSqrWG, 2, sizeof(cl_mem), &bufOut);
    clSetKernelArg(kSqrWG, 3, sizeof(cl_uint), &np0);
    clSetKernelArg(kSqrWG, 4, sizeof(cl_uint), &limbs);
    clSetKernelArg(kSqrWG, 5, local_mem_size, nullptr);

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        err = clEnqueueNDRangeKernel(ctx.queue, kSqrWG, 1, nullptr, &global_size_wg, &local_size_wg, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to enqueue cgbn_mont_sqr_wg: " << err << std::endl;
            return false;
        }
    }
    clFinish(ctx.queue);
    t1 = std::chrono::high_resolution_clock::now();
    double sqr_wg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    err = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                              host_out.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) return false;

    std::vector<uint32_t> exp_sqr(WORDS);
    fill_from_gmp(r_sqr_gmp, exp_sqr.data(), WORDS);
    bool okSqrWG = true;
    for (size_t i = 0; i < WORDS; ++i) {
        if (host_out[i] != exp_sqr[i]) {
            okSqrWG = false;
            std::cerr << "MontSqr WG mismatch at word " << i << ": got 0x" << std::hex 
                      << host_out[i] << " expected 0x" << exp_sqr[i] << std::dec << std::endl;
            break;
        }
    }

    // Results
    std::cout << "MontMul WG: time (ms)=" << mul_wg_ms << ", equal=" << (okMulWG ? "YES" : "NO") << std::endl;
    std::cout << "MontSqr WG: time (ms)=" << sqr_wg_ms << ", equal=" << (okSqrWG ? "YES" : "NO") << std::endl;

    // Throughput: ops/s = (instances * iterations) / time_in_seconds
    double ops = (double)iterations * (double)instances;
    double mul_wg_ops = ops / (mul_wg_ms / 1000.0);
    double sqr_wg_ops = ops / (sqr_wg_ms / 1000.0);

    std::cout << "Throughput (MontMul WG): " << mul_wg_ops << " ops/s ("
              << mul_wg_ops / 1e3 << " kops/s)" << std::endl;
    std::cout << "Throughput (MontSqr WG): " << sqr_wg_ops << " ops/s ("
              << sqr_wg_ops / 1e3 << " kops/s)" << std::endl;

    clReleaseKernel(kMulWG);
    clReleaseKernel(kSqrWG);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufN);
    clReleaseMemObject(bufOut);
    clReleaseProgram(program);
    cgbn::opencl::destroy_context(ctx);

    mpz_clears(n_gmp, a_gmp, b_gmp, r_mul_gmp, r_sqr_gmp, R, Rinv, tmp, nullptr);
    return okMulWG && okSqrWG;
}

} // namespace

#ifdef BUILD_OPENCL_MONT_WG_MAIN
#include <cstdlib>

int main(int argc, char **argv) {
    int iterations = 100;
    int instances = 256;
    int tpi = 4;

    if (argc >= 2) iterations = std::stoi(std::string(argv[1]));
    if (argc >= 3) instances = std::stoi(std::string(argv[2]));
    if (argc >= 4) tpi = std::stoi(std::string(argv[3]));

    bool ok_tpi4 = runOpenClMontgomeryWGBenchmark(iterations, instances, 4);
    bool ok_tpi8 = runOpenClMontgomeryWGBenchmark(iterations, instances, 8);

    return (ok_tpi4 && ok_tpi8) ? EXIT_SUCCESS : EXIT_FAILURE;
}
#endif
