#include "opencl_ecm_selftest.h"

#include "opencl_ecm_limb24.h"
#include "opencl_ecm_log.h"
#include "opencl_ecm_mont.h"

#include <CL/cl.h>

#include <cstdio>
#include <string>

namespace {

void strip_pragma_once(std::string &src) {
    const std::string marker("#pragma once");
    for (;;) {
        const size_t pos = src.find(marker);
        if (pos == std::string::npos) {
            break;
        }
        size_t end = src.find('\n', pos);
        if (end == std::string::npos) {
            end = src.size();
        } else {
            ++end;
        }
        src.erase(pos, end - pos);
    }
}

} // namespace

int opencl_ecm_selftest_montgomery(const mpz_t N, uint32_t bits) {
    const uint32_t limbs = bits / 32;
    uint32_t buf[OPENCL_ECM_MAX_LIMBS] = {0};
    mpz_t two, mont_mpz, back;
    mpz_init(two);
    mpz_init(mont_mpz);
    mpz_init(back);
    mpz_set_ui(two, 2);
    ecm_to_montgomery(buf, two, N, bits, limbs);
    ecm_to_mpz(mont_mpz, buf, limbs);
    ecm_from_montgomery(back, mont_mpz, N, ecm_find_np0(N), limbs);
    int ok = (mpz_cmp(back, two) == 0);
    if (!ok) {
        ecm_ts_fprintf(stderr, "GPU: Montgomery self-test failed (2 -> mont -> 2)\n");
    }
    mpz_clear(two);
    mpz_clear(mont_mpz);
    mpz_clear(back);
    return ok ? 0 : -1;
}

int opencl_ecm_selftest_montgomery_limb24(const mpz_t N, uint32_t limbs) {
    const uint32_t mont_bits = ecm_limb24_mont_bits(limbs);
    uint32_t n_limbs[OPENCL_ECM_MAX_LIMBS] = {0};
    ecm_limb24_from_mpz(n_limbs, limbs, N);
    const uint32_t np0 = ecm_find_np0_limb24(n_limbs);

    static const unsigned long k_tests[] = {2UL, 3UL, 9UL, 12345UL, 1000000UL};
    mpz_t test, mont_mpz, back;
    mpz_init(test);
    mpz_init(mont_mpz);
    mpz_init(back);

    int ok = 1;
    for (unsigned long val : k_tests) {
        uint32_t buf[OPENCL_ECM_MAX_LIMBS] = {0};
        mpz_set_ui(test, val);
        if (mpz_cmp(test, N) >= 0) {
            continue;
        }
        ecm_to_montgomery_limb24(buf, test, N, mont_bits, limbs);
        ecm_limb24_to_mpz(mont_mpz, buf, limbs);
        ecm_from_montgomery_limb24(back, mont_mpz, N, np0, limbs);
        if (mpz_cmp(back, test) != 0) {
            ecm_ts_fprintf(stderr, "GPU: limb24 Montgomery self-test failed (value %lu)\n", val);
            ok = 0;
        }
    }

    mpz_clear(test);
    mpz_clear(mont_mpz);
    mpz_clear(back);
    return ok ? 0 : -1;
}

int opencl_ecm_selftest_i24_mont_mul(cgbn::opencl::context_t &ctx, const mpz_t N, uint32_t limbs,
                                     uint32_t np0, bool use_blsub) {
    const uint32_t mont_bits = ecm_limb24_mont_bits(limbs);
    std::string i24_src =
        cgbn::opencl::load_kernel_file("cgbn/backends/opencl/kernels/mont_mul_unroll_i24.cl");
    if (i24_src.empty()) {
        ecm_ts_fprintf(stderr, "GPU: mont_mul_unroll_i24.cl not found for i24 self-test\n");
        return -1;
    }
    strip_pragma_once(i24_src);
    const char *mul_body =
        use_blsub ? "mont_mul_unroll_i24_u32_blsub_priv_body"
                  : "mont_mul_unroll_i24_u32_priv_body";
    std::string tail =
        "\n__kernel void ecm_i24_mul_selftest(__global const uint *a, __global const uint *b,\n"
        "                                   __global const uint *n, __global uint *out,\n"
        "                                   uint np0_arg) {\n"
        "    uint ap[MONT_I24_LIMBS], bp[MONT_I24_LIMBS], np[MONT_I24_LIMBS];\n"
        "    uint result[MONT_I24_LIMBS];\n"
        "    for (uint i = 0u; i < MONT_I24_LIMBS; ++i) {\n"
        "        ap[i] = a[i];\n"
        "        bp[i] = b[i];\n"
        "        np[i] = n[i];\n"
        "    }\n    ";
    tail += mul_body;
    tail +=
        "(result, ap, bp, np, np0_arg);\n"
        "    for (uint i = 0u; i < MONT_I24_LIMBS; ++i) {\n"
        "        out[i] = result[i];\n"
        "    }\n"
        "}\n";
    i24_src += tail;

    char opts[64];
    snprintf(opts, sizeof(opts), "-DMAX_LIMBS=%u -DMP_LIMB_BITS=24", limbs);
    cl_int buildErr = CL_SUCCESS;
    cl_program prog =
        cgbn::opencl::build_program_from_source(ctx, i24_src.c_str(), opts, buildErr);
    if (prog == nullptr || buildErr != CL_SUCCESS) {
        if (prog != nullptr) {
            size_t log_size = 0;
            clGetProgramBuildInfo(prog, ctx.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            if (log_size > 1) {
                std::string log(log_size, '\0');
                clGetProgramBuildInfo(prog, ctx.device, CL_PROGRAM_BUILD_LOG, log_size, &log[0],
                                    nullptr);
                ecm_ts_fprintf(stderr, "GPU: i24 mont mul self-test build log:\n%s\n", log.c_str());
            }
            clReleaseProgram(prog);
        }
        ecm_ts_fprintf(stderr, "GPU: i24 mont mul self-test build failed (err=%d)\n", (int)buildErr);
        return -1;
    }

    cl_int err;
    cl_kernel kernel = clCreateKernel(prog, "ecm_i24_mul_selftest", &err);
    if (err != CL_SUCCESS) {
        clReleaseProgram(prog);
        return -1;
    }

    uint32_t a[OPENCL_ECM_MAX_LIMBS] = {0}, b[OPENCL_ECM_MAX_LIMBS] = {0},
             n[OPENCL_ECM_MAX_LIMBS] = {0}, out[OPENCL_ECM_MAX_LIMBS] = {0};
    mpz_t two, three, six, mont_mpz, back;
    mpz_init(two);
    mpz_init(three);
    mpz_init(six);
    mpz_init(mont_mpz);
    mpz_init(back);
    mpz_set_ui(two, 2);
    mpz_set_ui(three, 3);
    mpz_mul_ui(six, two, 3);
    ecm_limb24_from_mpz(n, limbs, N);
    ecm_to_montgomery_limb24(a, two, N, mont_bits, limbs);
    ecm_to_montgomery_limb24(b, three, N, mont_bits, limbs);

    cl_mem bufA = clCreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 limbs * sizeof(uint32_t), a, &err);
    cl_mem bufB = clCreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 limbs * sizeof(uint32_t), b, &err);
    cl_mem bufN = clCreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 limbs * sizeof(uint32_t), n, &err);
    cl_mem bufOut = clCreateBuffer(ctx.ctx, CL_MEM_WRITE_ONLY, limbs * sizeof(uint32_t), nullptr,
                                   &err);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufN);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufOut);
    clSetKernelArg(kernel, 4, sizeof(cl_uint), &np0);
    size_t g = 1;
    err = clEnqueueNDRangeKernel(ctx.queue, kernel, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
    if (err == CL_SUCCESS) {
        err = clFinish(ctx.queue);
    }
    if (err == CL_SUCCESS) {
        err = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, limbs * sizeof(uint32_t), out, 0,
                                  nullptr, nullptr);
    }

    int ok = 0;
    if (err == CL_SUCCESS) {
        ecm_limb24_to_mpz(mont_mpz, out, limbs);
        ecm_from_montgomery_limb24(back, mont_mpz, N, np0, limbs);
        ok = (mpz_cmp(back, six) == 0);
        if (!ok) {
            ecm_ts_fprintf(stderr, "GPU: i24 mont mul self-test failed (2*3 mod N)\n");
        }
    } else {
        ecm_ts_fprintf(stderr, "GPU: i24 mont mul self-test enqueue/read failed (%d)\n", err);
    }

    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufN);
    clReleaseMemObject(bufOut);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    mpz_clear(two);
    mpz_clear(three);
    mpz_clear(six);
    mpz_clear(mont_mpz);
    mpz_clear(back);
    return ok ? 0 : -1;
}

int opencl_ecm_selftest_mont_mul(cgbn::opencl::context_t &ctx, const mpz_t N, uint32_t bits,
                                 uint32_t np0) {
    const uint32_t limbs = bits / 32;
    std::string mont_src =
        cgbn::opencl::load_kernel_file("cgbn/backends/opencl/kernels/mont.cl");
    if (mont_src.empty()) {
        ecm_ts_fprintf(stderr,
                       "GPU: mont.cl not found (add mont.cl to APK assets for self-test)\n");
        return -1;
    }
    char opts[64];
    snprintf(opts, sizeof(opts), "-DMAX_LIMBS=%u", limbs);
    cl_int buildErr = CL_SUCCESS;
    cl_program prog =
        cgbn::opencl::build_program_from_source(ctx, mont_src.c_str(), opts, buildErr);
    if (prog == nullptr || buildErr != CL_SUCCESS) {
        ecm_ts_fprintf(stderr, "GPU: mont.cl build failed (err=%d)\n", (int)buildErr);
        return -1;
    }
    cl_int err;
    cl_kernel kMul = clCreateKernel(prog, "cgbn_mont_mul", &err);
    if (err != CL_SUCCESS) {
        clReleaseProgram(prog);
        return -1;
    }

    uint32_t a[OPENCL_ECM_MAX_LIMBS] = {0}, b[OPENCL_ECM_MAX_LIMBS] = {0},
             n[OPENCL_ECM_MAX_LIMBS] = {0}, out[OPENCL_ECM_MAX_LIMBS] = {0};
    mpz_t two, three, six, r;
    mpz_init(two);
    mpz_init(three);
    mpz_init(six);
    mpz_init(r);
    mpz_set_ui(two, 2);
    mpz_set_ui(three, 3);
    mpz_mul_ui(six, two, 3);
    ecm_to_montgomery(a, two, N, bits, limbs);
    ecm_to_montgomery(b, three, N, bits, limbs);
    ecm_from_mpz(N, n, limbs);

    cl_mem bufA = clCreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 limbs * sizeof(uint32_t), a, &err);
    cl_mem bufB = clCreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 limbs * sizeof(uint32_t), b, &err);
    cl_mem bufN = clCreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 limbs * sizeof(uint32_t), n, &err);
    cl_mem bufOut = clCreateBuffer(ctx.ctx, CL_MEM_WRITE_ONLY, limbs * sizeof(uint32_t),
                                   nullptr, &err);
    clSetKernelArg(kMul, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kMul, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kMul, 2, sizeof(cl_mem), &bufN);
    clSetKernelArg(kMul, 3, sizeof(cl_mem), &bufOut);
    clSetKernelArg(kMul, 4, sizeof(cl_uint), &np0);
    cl_uint limbs_arg = limbs;
    clSetKernelArg(kMul, 5, sizeof(cl_uint), &limbs_arg);
    size_t g = 1;
    err = clEnqueueNDRangeKernel(ctx.queue, kMul, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        ecm_ts_fprintf(stderr, "GPU: mont_mul self-test enqueue failed (err=%d)\n", err);
        clReleaseMemObject(bufA);
        clReleaseMemObject(bufB);
        clReleaseMemObject(bufN);
        clReleaseMemObject(bufOut);
        clReleaseKernel(kMul);
        clReleaseProgram(prog);
        mpz_clear(two);
        mpz_clear(three);
        mpz_clear(six);
        mpz_clear(r);
        return -1;
    }
    err = clFinish(ctx.queue);
    if (err != CL_SUCCESS) {
        ecm_ts_fprintf(stderr, "GPU: mont_mul self-test clFinish failed (err=%d)\n", err);
    }
    err = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, limbs * sizeof(uint32_t), out, 0,
                              nullptr, nullptr);
    if (err != CL_SUCCESS) {
        ecm_ts_fprintf(stderr, "GPU: mont_mul self-test readback failed (err=%d)\n", err);
        clReleaseMemObject(bufA);
        clReleaseMemObject(bufB);
        clReleaseMemObject(bufN);
        clReleaseMemObject(bufOut);
        clReleaseKernel(kMul);
        clReleaseProgram(prog);
        mpz_clear(two);
        mpz_clear(three);
        mpz_clear(six);
        mpz_clear(r);
        return -1;
    }
    ecm_to_mpz(r, out, limbs);
    ecm_from_montgomery(r, r, N, np0, limbs);
    int ok = (mpz_cmp(r, six) == 0);
    if (!ok) {
        ecm_ts_fprintf(stderr, "GPU: mont_mul self-test failed (2*3 mod N)\n");
    }
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufN);
    clReleaseMemObject(bufOut);
    clReleaseKernel(kMul);
    clReleaseProgram(prog);
    mpz_clear(two);
    mpz_clear(three);
    mpz_clear(six);
    mpz_clear(r);
    return ok ? 0 : -1;
}
