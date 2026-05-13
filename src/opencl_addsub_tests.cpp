#include "opencl_addsub_tests.h"

#include "cgbn_opencl.h"

#include <CL/cl.h>
#include <gmp.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

static void fill_from_gmp(mpz_t v, uint32_t *out_words, size_t words) {
    // export little-endian 32-bit words; if v is negative, convert to
    // its unsigned representation modulo 2^(words*32) so the limb array
    // matches the OpenCL unsigned arithmetic results.
    mpz_t tmp;
    if (mpz_sgn(v) >= 0) {
        size_t count = 0;
        mpz_export(out_words, &count, -1, sizeof(uint32_t), 0, 0, v);
        for (size_t i = count; i < words; ++i) out_words[i] = 0u;
        return;
    }

    mpz_init(tmp);
    mpz_t mod;
    mpz_init(mod);
    mpz_ui_pow_ui(mod, 2, (unsigned long)(words * 32));
    mpz_add(tmp, v, mod); // tmp = v + 2^(words*32)
    size_t count = 0;
    mpz_export(out_words, &count, -1, sizeof(uint32_t), 0, 0, tmp);
    for (size_t i = count; i < words; ++i) out_words[i] = 0u;
    mpz_clear(tmp);
    mpz_clear(mod);
}

bool runOpenClAddSubBenchmark(int iterations, int instances) {
    constexpr int BITS = 1024;
    const size_t WORDS = BITS / 32; // 32

    std::cout << "OpenCL add/sub benchmark: " << BITS << "-bit, iterations=" << iterations
              << ", instances=" << instances << std::endl;

    // Prepare GMP numbers a = 2^991, b = 8218291649
    mpz_t a_gmp, b_gmp, res_gmp;
    mpz_init(a_gmp);
    mpz_init(b_gmp);
    mpz_init(res_gmp);

    mpz_ui_pow_ui(a_gmp, 2, 991);
    mpz_set_ui(b_gmp, 8218291649u);

    // Host arrays (little-endian words)
    std::vector<uint32_t> host_a((size_t)instances * WORDS);
    std::vector<uint32_t> host_b((size_t)instances * WORDS);
    std::vector<uint32_t> host_out((size_t)instances * WORDS);

    // Fill each instance with same a and b
    for (int i = 0; i < instances; ++i) {
        fill_from_gmp(a_gmp, &host_a[(size_t)i * WORDS], WORDS);
        fill_from_gmp(b_gmp, &host_b[(size_t)i * WORDS], WORDS);
    }

    // OpenCL setup
    cgbn::opencl::context_t ctx;
    cl_int err = cgbn::opencl::create_context(ctx);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL context: " << err << std::endl;
        return false;
    }

    // load kernels (try file then fallback)
    std::string src = cgbn::opencl::load_text_file("cgbn/backends/opencl/kernels/addsub.cl");
    if (src.empty()) {
        // fallback: minimal kernels
        src = R"CLC(
__kernel void cgbn_add(__global const uint *a, __global const uint *b, __global uint *out, uint limbs) {
    uint idx = get_global_id(0);
    uint base = idx * limbs;
    ulong carry = 0UL;
    for (uint i = 0; i < limbs; ++i) {
        ulong sum = (ulong)a[base + i] + (ulong)b[base + i] + carry;
        out[base + i] = (uint)sum;
        carry = sum >> 32;
    }
}

__kernel void cgbn_sub(__global const uint *a, __global const uint *b, __global uint *out, uint limbs) {
    uint idx = get_global_id(0);
    uint base = idx * limbs;
    ulong borrow = 0UL;
    for (uint i = 0; i < limbs; ++i) {
        ulong av = (ulong)a[base + i];
        ulong bv = (ulong)b[base + i];
        ulong wide = av - bv - borrow;
        out[base + i] = (uint)wide;
        borrow = ((av < bv + borrow) ? 1UL : 0UL);
    }
}
)CLC";
    }

    cl_int buildErr = CL_SUCCESS;
    cl_program program = cgbn::opencl::build_program_from_source(ctx, src.c_str(), "", buildErr);
    if (program == nullptr || buildErr != CL_SUCCESS) {
        std::cerr << "Failed to build OpenCL program: " << buildErr << std::endl;
        cgbn::opencl::destroy_context(ctx);
        return false;
    }

    // Create buffers
    size_t totalWords = (size_t)instances * WORDS;
    cl_context clCtx = ctx.ctx;
    cl_command_queue queue = ctx.queue;

    cl_mem bufA = clCreateBuffer(clCtx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(uint32_t) * totalWords, host_a.data(), &err);
    if (err != CL_SUCCESS) { std::cerr << "clCreateBuffer A failed: " << err << std::endl; return false; }

    cl_mem bufB = clCreateBuffer(clCtx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(uint32_t) * totalWords, host_b.data(), &err);
    if (err != CL_SUCCESS) { std::cerr << "clCreateBuffer B failed: " << err << std::endl; return false; }

    cl_mem bufOut = clCreateBuffer(clCtx, CL_MEM_READ_WRITE,
                                   sizeof(uint32_t) * totalWords, nullptr, &err);
    if (err != CL_SUCCESS) { std::cerr << "clCreateBuffer Out failed: " << err << std::endl; return false; }

    // Warm-up and measure OpenCL add. Prefer work-group kernel if available.
    cl_uint limbs = (cl_uint)WORDS;

    // Query device max work-group size
    size_t max_wg = 0;
    err = clGetDeviceInfo(ctx.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_wg, nullptr);
    if (err != CL_SUCCESS) max_wg = 1;

    // choose work-group size no larger than WORDS
    size_t wg = (size_t)std::min<size_t>(max_wg, (size_t)WORDS);

    size_t global = (size_t)instances;
    double cl_add_ms = 0.0;

    // Try work-group kernel
    cl_kernel kAdd = clCreateKernel(program, "cgbn_add_wg", &err);
    if (kAdd != nullptr && err == CL_SUCCESS) {
        // launch with one work-group per instance
        size_t local = wg;
        size_t gsize = (size_t)instances * local;
        clSetKernelArg(kAdd, 0, sizeof(cl_mem), &bufA);
        clSetKernelArg(kAdd, 1, sizeof(cl_mem), &bufB);
        clSetKernelArg(kAdd, 2, sizeof(cl_mem), &bufOut);
        clSetKernelArg(kAdd, 3, sizeof(cl_uint), &limbs);
        // local buffers: local_r size = WORDS uints, carry_flags stores G/P arrays (2 * local)
        clSetKernelArg(kAdd, 4, sizeof(uint32_t) * WORDS, nullptr);
        clSetKernelArg(kAdd, 5, sizeof(uint32_t) * local * 2, nullptr);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            err = clEnqueueNDRangeKernel(queue, kAdd, 1, nullptr, &gsize, &local, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) { std::cerr << "Enqueue add_wg failed: " << err << std::endl; return false; }
        }
        clFinish(queue);
        auto t1 = std::chrono::high_resolution_clock::now();
        cl_add_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    } else {
        // Fallback to per-instance scalar kernel
        kAdd = clCreateKernel(program, "cgbn_add", &err);
        if (err != CL_SUCCESS) { std::cerr << "Create kernel add failed: " << err << std::endl; return false; }
        clSetKernelArg(kAdd, 0, sizeof(cl_mem), &bufA);
        clSetKernelArg(kAdd, 1, sizeof(cl_mem), &bufB);
        clSetKernelArg(kAdd, 2, sizeof(cl_mem), &bufOut);
        clSetKernelArg(kAdd, 3, sizeof(cl_uint), &limbs);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            err = clEnqueueNDRangeKernel(queue, kAdd, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) { std::cerr << "Enqueue add failed: " << err << std::endl; return false; }
        }
        clFinish(queue);
        auto t1 = std::chrono::high_resolution_clock::now();
        cl_add_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // Read back one result (first instance) and compare with GMP
    err = clEnqueueReadBuffer(queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS, host_out.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { std::cerr << "Read buffer out failed: " << err << std::endl; return false; }

    // Compute GMP add reference with same operations: r = a + b; r = r + a
    auto g0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        mpz_add(res_gmp, a_gmp, b_gmp);
        mpz_add(res_gmp, res_gmp, a_gmp);
    }
    auto g1 = std::chrono::high_resolution_clock::now();
    double g_add_ms = std::chrono::duration<double, std::milli>(g1 - g0).count();

    // Export GMP result
    std::vector<uint32_t> expected(WORDS);
    fill_from_gmp(res_gmp, expected.data(), WORDS);

    bool match_add = true;
    for (size_t i = 0; i < WORDS; ++i) {
        if (host_out[i] != expected[i]) { match_add = false; break; }
    }

    std::cout << "Add: OpenCL time (ms)=" << cl_add_ms << ", GMP time (ms)=" << g_add_ms
              << ", equal=" << (match_add ? "YES" : "NO") << std::endl;

    // Now subtraction: try work-group kernel then fallback.
    double cl_sub_ms = 0.0;
    cl_kernel kSub = clCreateKernel(program, "cgbn_sub_wg", &err);
    if (kSub != nullptr && err == CL_SUCCESS) {
        size_t local = wg;
        size_t gsize = (size_t)instances * local;
        clSetKernelArg(kSub, 0, sizeof(cl_mem), &bufA);
        clSetKernelArg(kSub, 1, sizeof(cl_mem), &bufB);
        clSetKernelArg(kSub, 2, sizeof(cl_mem), &bufOut);
        clSetKernelArg(kSub, 3, sizeof(cl_uint), &limbs);
        clSetKernelArg(kSub, 4, sizeof(uint32_t) * WORDS, nullptr);
        clSetKernelArg(kSub, 5, sizeof(uint32_t) * local, nullptr);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            err = clEnqueueNDRangeKernel(queue, kSub, 1, nullptr, &gsize, &local, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) { std::cerr << "Enqueue sub_wg failed: " << err << std::endl; return false; }
        }
        clFinish(queue);
        auto t1 = std::chrono::high_resolution_clock::now();
        cl_sub_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    } else {
        kSub = clCreateKernel(program, "cgbn_sub", &err);
        if (err != CL_SUCCESS) { std::cerr << "Create kernel sub failed: " << err << std::endl; return false; }
        clSetKernelArg(kSub, 0, sizeof(cl_mem), &bufA);
        clSetKernelArg(kSub, 1, sizeof(cl_mem), &bufB);
        clSetKernelArg(kSub, 2, sizeof(cl_mem), &bufOut);
        clSetKernelArg(kSub, 3, sizeof(cl_uint), &limbs);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            err = clEnqueueNDRangeKernel(queue, kSub, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) { std::cerr << "Enqueue sub failed: " << err << std::endl; return false; }
        }
        clFinish(queue);
        auto t1 = std::chrono::high_resolution_clock::now();
        cl_sub_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    err = clEnqueueReadBuffer(queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS, host_out.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { std::cerr << "Read buffer out failed: " << err << std::endl; return false; }

    // GMP subtraction reference with same ops: r = a - b; r = r - a
    g0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        mpz_sub(res_gmp, a_gmp, b_gmp);
        mpz_sub(res_gmp, res_gmp, a_gmp);
    }
    g1 = std::chrono::high_resolution_clock::now();
    double g_sub_ms = std::chrono::duration<double, std::milli>(g1 - g0).count();

    fill_from_gmp(res_gmp, expected.data(), WORDS);
    bool match_sub = true;
    for (size_t i = 0; i < WORDS; ++i) {
        if (host_out[i] != expected[i]) { match_sub = false; break; }
    }

    std::cout << "Sub: OpenCL time (ms)=" << cl_sub_ms << ", GMP time (ms)=" << g_sub_ms
              << ", equal=" << (match_sub ? "YES" : "NO") << std::endl;

    // Cleanup
    clReleaseKernel(kAdd);
    clReleaseKernel(kSub);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufOut);
    clReleaseProgram(program);
    cgbn::opencl::destroy_context(ctx);

    mpz_clear(a_gmp);
    mpz_clear(b_gmp);
    mpz_clear(res_gmp);

    return (match_add && match_sub);
}

// Standalone main for this benchmark target
#ifdef BUILD_OPENCL_ADDSUB_MAIN
#include <cstdlib>
#include <string>

int main(int argc, char **argv) {
    int iterations = 1000;
    int instances = 256;
    if (argc >= 2) iterations = std::stoi(std::string(argv[1]));
    if (argc >= 3) instances = std::stoi(std::string(argv[2]));
    bool ok = runOpenClAddSubBenchmark(iterations, instances);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
#endif
