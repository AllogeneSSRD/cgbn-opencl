#include "cgbn_opencl.h"

#include <CL/cl.h>
#include <gmp.h>

#include <algorithm>
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

uint32_t inv32_odd(uint32_t x) {
    uint64_t y = 1;
    for (int i = 0; i < 5; ++i) {
        y = y * (2ull - (uint64_t)x * y);
        y &= 0xFFFFFFFFull;
    }
    return (uint32_t)y;
}

std::vector<int> build_bit_sizes(int nmin, int nmax) {
    const int candidates[] = {
        128, 256, 512, 1024,
        2048, 3072, 4096, 5120, 6144, 7168, 8192,
        9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384
    };
    std::vector<int> sizes;
    for (int bits : candidates) {
        if (bits >= nmin && bits <= nmax) {
            sizes.push_back(bits);
        }
    }
    return sizes;
}

struct DeviceInfo {
    std::string name;
    cl_uint computeUnits = 0;
    size_t maxWorkGroup = 0;
};

std::string csv_quote(const std::string &value) {
    std::string quoted;
    quoted.reserve(value.size() + 2);
    quoted.push_back('"');
    for (char ch : value) {
        if (ch == '"') {
            quoted.push_back('"');
        }
        quoted.push_back(ch);
    }
    quoted.push_back('"');
    return quoted;
}

DeviceInfo get_device_info(cl_device_id device) {
    DeviceInfo info;
    char nameBuf[256] = {0};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(nameBuf) - 1, nameBuf, nullptr);
    info.name = nameBuf;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(info.computeUnits), &info.computeUnits, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(info.maxWorkGroup), &info.maxWorkGroup, nullptr);
    return info;
}

bool runOpenClMontThroughputBenchmark(int iterations, int nmin, int nmax, int instances) {
    if (nmin > nmax) {
        std::cerr << "Invalid range: nmin > nmax" << std::endl;
        return false;
    }

    const std::vector<int> bitSizes = build_bit_sizes(nmin, nmax);
    if (bitSizes.empty()) {
        std::cerr << "No supported bit sizes in range [" << nmin << ", " << nmax << "]" << std::endl;
        return false;
    }

    std::cout << "bits,limbs,np0_hex,iterations,instances,device_name,compute_units,max_work_group_size,mul_equal,sqr_equal,cl_mul_ms,cl_mul_iter_ops_per_s,cl_mul_bit_per_s,cl_sqr_ms,cl_sqr_iter_ops_per_s,cl_sqr_bit_per_s,cpu_mul_ms,cpu_mul_iter_ops_per_s,cpu_mul_bit_per_s,cpu_sqr_ms,cpu_sqr_iter_ops_per_s,cpu_sqr_bit_per_s" << std::endl;

    cgbn::opencl::context_t ctx;
    cl_int err = cgbn::opencl::create_context(ctx);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL context: " << err << std::endl;
        return false;
    }

    DeviceInfo deviceInfo = get_device_info(ctx.device);

    std::string src = cgbn::opencl::load_text_file("cgbn/backends/opencl/kernels/mont.cl");
    if (src.empty()) {
        std::cerr << "Failed to load mont.cl" << std::endl;
        cgbn::opencl::destroy_context(ctx);
        return false;
    }

    cl_int buildErr = CL_SUCCESS;
    const char *buildOptions = "-DMAX_LIMBS=512";
    cl_program program = cgbn::opencl::build_program_from_source(ctx, src.c_str(), buildOptions, buildErr);
    if (program == nullptr || buildErr != CL_SUCCESS) {
        std::cerr << "Failed to build mont throughput program: " << buildErr << std::endl;
        cgbn::opencl::destroy_context(ctx);
        return false;
    }

    bool allOk = true;

    for (int bits : bitSizes) {
        const size_t limbs = (size_t)bits / 32u;
        const size_t totalWords = (size_t)instances * limbs;

        mpz_t n_gmp, a_gmp, b_gmp, r_mul_gmp, r_sqr_gmp, R, Rinv, tmp;
        mpz_inits(n_gmp, a_gmp, b_gmp, r_mul_gmp, r_sqr_gmp, R, Rinv, tmp, nullptr);

        mpz_set_ui(n_gmp, 1);
        mpz_mul_2exp(n_gmp, n_gmp, (unsigned long)bits);
        mpz_sub_ui(n_gmp, n_gmp, 173u);

        mpz_set_str(a_gmp, "123456789012345678901234567890123456", 10);
        mpz_set_str(b_gmp, "98765432109876543210987654321012345", 10);
        mpz_mod(a_gmp, a_gmp, n_gmp);
        mpz_mod(b_gmp, b_gmp, n_gmp);

        mpz_ui_pow_ui(R, 2, (unsigned long)bits);
        if (mpz_invert(Rinv, R, n_gmp) == 0) {
            std::cerr << "Failed to invert R modulo n at bits=" << bits << std::endl;
            mpz_clears(n_gmp, a_gmp, b_gmp, r_mul_gmp, r_sqr_gmp, R, Rinv, tmp, nullptr);
            allOk = false;
            continue;
        }

        mpz_mul(tmp, a_gmp, b_gmp);
        mpz_mul(tmp, tmp, Rinv);
        mpz_mod(r_mul_gmp, tmp, n_gmp);

        mpz_mul(tmp, a_gmp, a_gmp);
        mpz_mul(tmp, tmp, Rinv);
        mpz_mod(r_sqr_gmp, tmp, n_gmp);

        std::vector<uint32_t> a_words(limbs), b_words(limbs), n_words(limbs);
        fill_from_gmp(a_gmp, a_words.data(), limbs);
        fill_from_gmp(b_gmp, b_words.data(), limbs);
        fill_from_gmp(n_gmp, n_words.data(), limbs);

        std::vector<uint32_t> host_a(totalWords), host_b(totalWords), host_n(totalWords), host_out(totalWords);
        for (int i = 0; i < instances; ++i) {
            for (size_t j = 0; j < limbs; ++j) {
                host_a[(size_t)i * limbs + j] = a_words[j];
                host_b[(size_t)i * limbs + j] = b_words[j];
                host_n[(size_t)i * limbs + j] = n_words[j];
            }
        }

        if ((n_words[0] & 1u) == 0u) {
            std::cerr << "n must be odd for Montgomery at bits=" << bits << std::endl;
            mpz_clears(n_gmp, a_gmp, b_gmp, r_mul_gmp, r_sqr_gmp, R, Rinv, tmp, nullptr);
            allOk = false;
            continue;
        }
        const uint32_t np0 = 0u - inv32_odd(n_words[0]);

        cl_int errLocal = CL_SUCCESS;
        cl_mem bufA = clCreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(uint32_t) * totalWords, host_a.data(), &errLocal);
        if (errLocal != CL_SUCCESS) {
            std::cerr << "clCreateBuffer A failed at bits=" << bits << ": " << errLocal << std::endl;
            mpz_clears(n_gmp, a_gmp, b_gmp, r_mul_gmp, r_sqr_gmp, R, Rinv, tmp, nullptr);
            allOk = false;
            continue;
        }
        cl_mem bufB = clCreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(uint32_t) * totalWords, host_b.data(), &errLocal);
        if (errLocal != CL_SUCCESS) {
            std::cerr << "clCreateBuffer B failed at bits=" << bits << ": " << errLocal << std::endl;
            clReleaseMemObject(bufA);
            mpz_clears(n_gmp, a_gmp, b_gmp, r_mul_gmp, r_sqr_gmp, R, Rinv, tmp, nullptr);
            allOk = false;
            continue;
        }
        cl_mem bufN = clCreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(uint32_t) * totalWords, host_n.data(), &errLocal);
        if (errLocal != CL_SUCCESS) {
            std::cerr << "clCreateBuffer N failed at bits=" << bits << ": " << errLocal << std::endl;
            clReleaseMemObject(bufA);
            clReleaseMemObject(bufB);
            mpz_clears(n_gmp, a_gmp, b_gmp, r_mul_gmp, r_sqr_gmp, R, Rinv, tmp, nullptr);
            allOk = false;
            continue;
        }
        cl_mem bufOut = clCreateBuffer(ctx.ctx, CL_MEM_READ_WRITE,
                                       sizeof(uint32_t) * totalWords, nullptr, &errLocal);
        if (errLocal != CL_SUCCESS) {
            std::cerr << "clCreateBuffer Out failed at bits=" << bits << ": " << errLocal << std::endl;
            clReleaseMemObject(bufA);
            clReleaseMemObject(bufB);
            clReleaseMemObject(bufN);
            mpz_clears(n_gmp, a_gmp, b_gmp, r_mul_gmp, r_sqr_gmp, R, Rinv, tmp, nullptr);
            allOk = false;
            continue;
        }

        cl_uint limbsArg = (cl_uint)limbs;
        cl_kernel kMul = clCreateKernel(program, "cgbn_mont_mul", &errLocal);
        if (errLocal != CL_SUCCESS) {
            std::cerr << "clCreateKernel mont_mul failed at bits=" << bits << ": " << errLocal << std::endl;
            clReleaseMemObject(bufA);
            clReleaseMemObject(bufB);
            clReleaseMemObject(bufN);
            clReleaseMemObject(bufOut);
            mpz_clears(n_gmp, a_gmp, b_gmp, r_mul_gmp, r_sqr_gmp, R, Rinv, tmp, nullptr);
            allOk = false;
            continue;
        }
        cl_kernel kSqr = clCreateKernel(program, "cgbn_mont_sqr", &errLocal);
        if (errLocal != CL_SUCCESS) {
            std::cerr << "clCreateKernel mont_sqr failed at bits=" << bits << ": " << errLocal << std::endl;
            clReleaseKernel(kMul);
            clReleaseMemObject(bufA);
            clReleaseMemObject(bufB);
            clReleaseMemObject(bufN);
            clReleaseMemObject(bufOut);
            mpz_clears(n_gmp, a_gmp, b_gmp, r_mul_gmp, r_sqr_gmp, R, Rinv, tmp, nullptr);
            allOk = false;
            continue;
        }

        clSetKernelArg(kMul, 0, sizeof(cl_mem), &bufA);
        clSetKernelArg(kMul, 1, sizeof(cl_mem), &bufB);
        clSetKernelArg(kMul, 2, sizeof(cl_mem), &bufN);
        clSetKernelArg(kMul, 3, sizeof(cl_mem), &bufOut);
        clSetKernelArg(kMul, 4, sizeof(cl_uint), &np0);
        clSetKernelArg(kMul, 5, sizeof(cl_uint), &limbsArg);

        clSetKernelArg(kSqr, 0, sizeof(cl_mem), &bufA);
        clSetKernelArg(kSqr, 1, sizeof(cl_mem), &bufN);
        clSetKernelArg(kSqr, 2, sizeof(cl_mem), &bufOut);
        clSetKernelArg(kSqr, 3, sizeof(cl_uint), &np0);
        clSetKernelArg(kSqr, 4, sizeof(cl_uint), &limbsArg);

        const size_t global = (size_t)instances;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            errLocal = clEnqueueNDRangeKernel(ctx.queue, kMul, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
            if (errLocal != CL_SUCCESS) {
                std::cerr << "Enqueue mont_mul failed at bits=" << bits << ": " << errLocal << std::endl;
                allOk = false;
                break;
            }
        }
        clFinish(ctx.queue);
        auto t1 = std::chrono::high_resolution_clock::now();
        double mul_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        errLocal = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0,
                                       sizeof(uint32_t) * limbs,
                                       host_out.data(), 0, nullptr, nullptr);
        if (errLocal != CL_SUCCESS) {
            std::cerr << "Read buffer mul result failed at bits=" << bits << ": " << errLocal << std::endl;
            allOk = false;
        }

        std::vector<uint32_t> exp_mul(limbs);
        fill_from_gmp(r_mul_gmp, exp_mul.data(), limbs);
        bool okMul = true;
        for (size_t i = 0; i < limbs; ++i) {
            if (host_out[i] != exp_mul[i]) {
                okMul = false;
                break;
            }
        }

        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            errLocal = clEnqueueNDRangeKernel(ctx.queue, kSqr, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
            if (errLocal != CL_SUCCESS) {
                std::cerr << "Enqueue mont_sqr failed at bits=" << bits << ": " << errLocal << std::endl;
                allOk = false;
                break;
            }
        }
        clFinish(ctx.queue);
        t1 = std::chrono::high_resolution_clock::now();
        double sqr_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        errLocal = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0,
                                       sizeof(uint32_t) * limbs,
                                       host_out.data(), 0, nullptr, nullptr);
        if (errLocal != CL_SUCCESS) {
            std::cerr << "Read buffer sqr result failed at bits=" << bits << ": " << errLocal << std::endl;
            allOk = false;
        }

        std::vector<uint32_t> exp_sqr(limbs);
        fill_from_gmp(r_sqr_gmp, exp_sqr.data(), limbs);
        bool okSqr = true;
        for (size_t i = 0; i < limbs; ++i) {
            if (host_out[i] != exp_sqr[i]) {
                okSqr = false;
                break;
            }
        }

        double ops = (double)iterations * (double)instances;
        double mul_iter_throughput = ops / (mul_ms / 1000.0);
        double mul_bit_throughput = (double)bits * ops / (mul_ms / 1000.0);
        double sqr_iter_throughput = ops / (sqr_ms / 1000.0);
        double sqr_bit_throughput = (double)bits * ops / (sqr_ms / 1000.0);

        double cpu_mul_ms = 0.0;
        double cpu_sqr_ms = 0.0;
        {
            mpz_t cpu_tmp;
            mpz_init(cpu_tmp);
            auto cpu_t0 = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < iterations; ++it) {
                for (int ins = 0; ins < instances; ++ins) {
                    mpz_mul(cpu_tmp, a_gmp, b_gmp);
                    mpz_mul(cpu_tmp, cpu_tmp, Rinv);
                    mpz_mod(cpu_tmp, cpu_tmp, n_gmp);
                }
            }
            auto cpu_t1 = std::chrono::high_resolution_clock::now();
            cpu_mul_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();

            cpu_t0 = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < iterations; ++it) {
                for (int ins = 0; ins < instances; ++ins) {
                    mpz_mul(cpu_tmp, a_gmp, a_gmp);
                    mpz_mul(cpu_tmp, cpu_tmp, Rinv);
                    mpz_mod(cpu_tmp, cpu_tmp, n_gmp);
                }
            }
            cpu_t1 = std::chrono::high_resolution_clock::now();
            cpu_sqr_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();
            mpz_clear(cpu_tmp);
        }

        double cpu_mul_iter_throughput = ops / (cpu_mul_ms / 1000.0);
        double cpu_mul_bit_throughput = (double)bits * ops / (cpu_mul_ms / 1000.0);
        double cpu_sqr_iter_throughput = ops / (cpu_sqr_ms / 1000.0);
        double cpu_sqr_bit_throughput = (double)bits * ops / (cpu_sqr_ms / 1000.0);

        std::cout << bits << ","
              << limbs << ","
              << "0x" << std::hex << np0 << std::dec << ","
              << iterations << ","
              << instances << ","
                  << csv_quote(deviceInfo.name) << ","
                  << deviceInfo.computeUnits << ","
                  << deviceInfo.maxWorkGroup << ","
              << (okMul ? "YES" : "NO") << ","
              << (okSqr ? "YES" : "NO") << ","
              << mul_ms << ","
              << mul_iter_throughput << ","
              << mul_bit_throughput << ","
              << sqr_ms << ","
              << sqr_iter_throughput << ","
              << sqr_bit_throughput << ","
              << cpu_mul_ms << ","
              << cpu_mul_iter_throughput << ","
              << cpu_mul_bit_throughput << ","
              << cpu_sqr_ms << ","
              << cpu_sqr_iter_throughput << ","
              << cpu_sqr_bit_throughput << ","
              << std::endl;

        clReleaseKernel(kMul);
        clReleaseKernel(kSqr);
        clReleaseMemObject(bufA);
        clReleaseMemObject(bufB);
        clReleaseMemObject(bufN);
        clReleaseMemObject(bufOut);
        mpz_clears(n_gmp, a_gmp, b_gmp, r_mul_gmp, r_sqr_gmp, R, Rinv, tmp, nullptr);

        if (!(okMul && okSqr)) {
            allOk = false;
        }
    }

    clReleaseProgram(program);
    cgbn::opencl::destroy_context(ctx);
    return allOk;
}

} // namespace

#ifdef BUILD_OPENCL_MONT_THROUGHPUT_MAIN
#include <cstdlib>

int main(int argc, char **argv) {
    int iterations = 1000;
    int nmin = 128;
    int nmax = 16384;
    int instances = 256;
    if (argc >= 2) iterations = std::stoi(std::string(argv[1]));
    if (argc >= 3) instances = std::stoi(std::string(argv[2]));
    if (argc >= 4) nmin = std::stoi(std::string(argv[3]));
    if (argc >= 5) nmax = std::stoi(std::string(argv[4]));

    bool ok = runOpenClMontThroughputBenchmark(iterations, nmin, nmax, instances);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
#endif
