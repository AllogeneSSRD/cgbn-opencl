#include "cgbn_opencl.h"

#include <CL/cl.h>
#include <gmp.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <fstream>

namespace {

constexpr uint32_t BITS = 1024;
constexpr uint32_t WORDS = BITS / 32;

void fill_from_gmp(const mpz_t v, uint32_t *out_words, size_t words) {
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

bool run_kernel(cl_command_queue q, cl_kernel k, size_t global, int repeats, double &ms) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeats; ++i) {
        cl_int err = clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Enqueue failed: " << err << std::endl;
            return false;
        }
    }
    clFinish(q);
    auto t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
}

uint32_t inv32_odd(uint32_t x) {
    uint64_t y = 1;
    for (int i = 0; i < 5; ++i) {
        y = y * (2ull - (uint64_t)x * y);
        y &= 0xFFFFFFFFull;
    }
    return (uint32_t)y;
}

void query_kernel_resources(cl_kernel k, cl_device_id dev, size_t &private_bytes,
                            size_t &local_bytes, size_t &pref_wg_multiple,
                            size_t &workgroup_size) {
    private_bytes = local_bytes = pref_wg_multiple = workgroup_size = 0;
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(size_t), &private_bytes, nullptr);
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(size_t), &local_bytes, nullptr);
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t),
                             &pref_wg_multiple, nullptr);
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &workgroup_size, nullptr);
}

} // namespace

bool runOpenClEcmAddSubBenchmark(int kernel_iterations, int instances, int launch_repeats) {
    std::cout << "ECM add/sub microbench: " << BITS
              << "-bit, kernel_iterations=" << kernel_iterations
              << ", instances=" << instances
              << ", launch_repeats=" << launch_repeats << std::endl;

    mpz_t n_gmp, a_gmp, b_gmp;
    mpz_init(n_gmp);
    mpz_init(a_gmp);
    mpz_init(b_gmp);
    mpz_ui_pow_ui(n_gmp, 2, 1024);
    mpz_sub_ui(n_gmp, n_gmp, 109u);
    mpz_ui_pow_ui(a_gmp, 2, 991);
    mpz_set_ui(b_gmp, 8218291649u);
    mpz_mod(a_gmp, a_gmp, n_gmp);
    mpz_mod(b_gmp, b_gmp, n_gmp);

    std::vector<uint32_t> host_a((size_t)instances * WORDS);
    std::vector<uint32_t> host_b((size_t)instances * WORDS);
    std::vector<uint32_t> host_n((size_t)instances * WORDS);
    std::vector<uint32_t> host_out((size_t)instances * WORDS);
    std::vector<uint32_t> a_words(WORDS), b_words(WORDS), n_words(WORDS);

    fill_from_gmp(a_gmp, a_words.data(), WORDS);
    fill_from_gmp(b_gmp, b_words.data(), WORDS);
    fill_from_gmp(n_gmp, n_words.data(), WORDS);
    for (int i = 0; i < instances; ++i) {
        for (uint32_t j = 0; j < WORDS; ++j) {
            host_a[(size_t)i * WORDS + j] = a_words[j];
            host_b[(size_t)i * WORDS + j] = b_words[j];
            host_n[(size_t)i * WORDS + j] = n_words[j];
        }
    }

    cgbn::opencl::context_t ctx;
    cl_int err = cgbn::opencl::create_context(ctx);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL context: " << err << std::endl;
        return false;
    }

    std::string mont_priv = cgbn::opencl::load_text_file("cgbn/backends/opencl/kernels/mont_priv.cl");
    std::string bench_src = cgbn::opencl::load_text_file("cgbn/backends/opencl/kernels/ecm_addsub_bench.cl");
    if (mont_priv.empty() || bench_src.empty()) {
        std::cerr << "Failed to load ecm_addsub_bench.cl" << std::endl;
        return false;
    }
    std::string src = mont_priv + "\n" + bench_src;
    cl_int buildErr = CL_SUCCESS;
    cl_program program = cgbn::opencl::build_program_from_source(
        ctx, src.c_str(), "-DMAX_LIMBS=64", buildErr);
    if (program == nullptr || buildErr != CL_SUCCESS) {
        std::cerr << "Failed to build ecm_addsub_bench.cl: " << buildErr << std::endl;
        return false;
    }

    size_t totalWords = (size_t)instances * WORDS;
    cl_mem bufA = clCreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(uint32_t) * totalWords, host_a.data(), &err);
    cl_mem bufB = clCreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(uint32_t) * totalWords, host_b.data(), &err);
    cl_mem bufN = clCreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(uint32_t) * totalWords, host_n.data(), &err);
    cl_mem bufOut = clCreateBuffer(ctx.ctx, CL_MEM_READ_WRITE,
                                   sizeof(uint32_t) * totalWords, nullptr, &err);

    uint32_t inv = inv32_odd(n_words[0]);
    cl_uint np0 = 0u - inv;
    cl_uint limbs = WORDS;
    cl_uint iters = (cl_uint)kernel_iterations;
    size_t global = (size_t)instances;

    const char *csv_path = std::getenv("ECM_BENCH_CSV");
    bool csv_enabled = (csv_path && *csv_path);
    std::ofstream csv;
    if (csv_enabled) {
        csv.open(csv_path, std::ios::out | std::ios::trunc);
        if (csv.is_open()) {
            csv << "kernel,ms,ops_per_s,private_mem_bytes,local_mem_bytes,preferred_wg_multiple,max_wg_size\n";
        } else {
            std::cerr << "Warning: failed to open ECM_BENCH_CSV path: " << csv_path << std::endl;
            csv_enabled = false;
        }
    }

    auto run_named = [&](const char *kname, bool needsN, double &ms_out) -> bool {
        cl_int kerr = CL_SUCCESS;
        cl_kernel k = clCreateKernel(program, kname, &kerr);
        if (kerr != CL_SUCCESS) {
            std::cerr << "Create kernel " << kname << " failed: " << kerr << std::endl;
            return false;
        }
        clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
        clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
        if (needsN) {
            clSetKernelArg(k, 2, sizeof(cl_mem), &bufN);
            clSetKernelArg(k, 3, sizeof(cl_mem), &bufOut);
            if (std::string(kname) == "ecm_mont_mul_priv_bench") {
                clSetKernelArg(k, 4, sizeof(cl_uint), &np0);
                clSetKernelArg(k, 5, sizeof(cl_uint), &limbs);
                clSetKernelArg(k, 6, sizeof(cl_uint), &iters);
            } else {
                clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
                clSetKernelArg(k, 5, sizeof(cl_uint), &iters);
            }
        } else {
            clSetKernelArg(k, 2, sizeof(cl_mem), &bufOut);
            clSetKernelArg(k, 3, sizeof(cl_uint), &limbs);
            clSetKernelArg(k, 4, sizeof(cl_uint), &iters);
        }
        bool ok = run_kernel(ctx.queue, k, global, launch_repeats, ms_out);
        if (ok) {
            size_t priv_b = 0, loc_b = 0, pref = 0, wg = 0;
            query_kernel_resources(k, ctx.device, priv_b, loc_b, pref, wg);
            double op_count = (double)instances * (double)kernel_iterations * (double)launch_repeats;
            double ops_s = op_count / (ms_out / 1000.0);
            std::cout << "  [" << kname << "] private_mem=" << priv_b
                      << "B local_mem=" << loc_b
                      << "B pref_wg=" << pref
                      << " max_wg=" << wg << std::endl;
            if (csv_enabled) {
                csv << kname << "," << ms_out << "," << ops_s << "," << priv_b << "," << loc_b
                    << "," << pref << "," << wg << "\n";
            }
        }
        clReleaseKernel(k);
        return ok;
    };

    double t_add_n = 0.0, t_add_mod = 0.0, t_sub_mod = 0.0, t_mul_priv = 0.0, t_sqr_priv = 0.0;
    if (!run_named("ecm_mp_add_n_bench", false, t_add_n)) return false;
    if (!run_named("ecm_mp_add_mod_bench", true, t_add_mod)) return false;
    if (!run_named("ecm_mp_sub_mod_bench", true, t_sub_mod)) return false;
    if (!run_named("ecm_mont_mul_priv_bench", true, t_mul_priv)) return false;

    {
        cl_int kerr = CL_SUCCESS;
        cl_kernel ks = clCreateKernel(program, "ecm_mont_sqr_priv_bench", &kerr);
        if (kerr != CL_SUCCESS) {
            std::cerr << "Create kernel ecm_mont_sqr_priv_bench failed: " << kerr << std::endl;
            return false;
        }
        clSetKernelArg(ks, 0, sizeof(cl_mem), &bufA);
        clSetKernelArg(ks, 1, sizeof(cl_mem), &bufN);
        clSetKernelArg(ks, 2, sizeof(cl_mem), &bufOut);
        clSetKernelArg(ks, 3, sizeof(cl_uint), &np0);
        clSetKernelArg(ks, 4, sizeof(cl_uint), &limbs);
        clSetKernelArg(ks, 5, sizeof(cl_uint), &iters);
        if (!run_kernel(ctx.queue, ks, global, launch_repeats, t_sqr_priv)) {
            clReleaseKernel(ks);
            return false;
        }
        size_t priv_b = 0, loc_b = 0, pref = 0, wg = 0;
        query_kernel_resources(ks, ctx.device, priv_b, loc_b, pref, wg);
        double op_count = (double)instances * (double)kernel_iterations * (double)launch_repeats;
        double ops_s = op_count / (t_sqr_priv / 1000.0);
        std::cout << "  [ecm_mont_sqr_priv_bench] private_mem=" << priv_b
                  << "B local_mem=" << loc_b
                  << "B pref_wg=" << pref
                  << " max_wg=" << wg << std::endl;
        if (csv_enabled) {
            csv << "ecm_mont_sqr_priv_bench," << t_sqr_priv << "," << ops_s << "," << priv_b
                << "," << loc_b << "," << pref << "," << wg << "\n";
        }
        clReleaseKernel(ks);
    }

    err = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                              host_out.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Read back failed: " << err << std::endl;
        return false;
    }

    double op_count = (double)instances * (double)kernel_iterations * (double)launch_repeats;
    std::cout << "mp_add_n:   " << t_add_n << " ms, " << (op_count / (t_add_n / 1000.0)) << " ops/s" << std::endl;
    std::cout << "mp_add_mod: " << t_add_mod << " ms, " << (op_count / (t_add_mod / 1000.0)) << " ops/s" << std::endl;
    std::cout << "mp_sub_mod: " << t_sub_mod << " ms, " << (op_count / (t_sub_mod / 1000.0)) << " ops/s" << std::endl;
    std::cout << "mont_mul_priv: " << t_mul_priv << " ms, " << (op_count / (t_mul_priv / 1000.0)) << " ops/s" << std::endl;
    std::cout << "mont_sqr_priv: " << t_sqr_priv << " ms, " << (op_count / (t_sqr_priv / 1000.0)) << " ops/s" << std::endl;

    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufN);
    clReleaseMemObject(bufOut);
    clReleaseProgram(program);
    if (csv_enabled && csv.is_open()) csv.close();
    cgbn::opencl::destroy_context(ctx);
    mpz_clear(n_gmp);
    mpz_clear(a_gmp);
    mpz_clear(b_gmp);
    return true;
}

#ifdef BUILD_OPENCL_ECM_ADDSUB_MAIN
#include <cstdlib>
int main(int argc, char **argv) {
    int kernel_iterations = 1000;
    int instances = 256;
    int launch_repeats = 50;
    if (argc >= 2) kernel_iterations = std::stoi(std::string(argv[1]));
    if (argc >= 3) instances = std::stoi(std::string(argv[2]));
    if (argc >= 4) launch_repeats = std::stoi(std::string(argv[3]));
    bool ok = runOpenClEcmAddSubBenchmark(kernel_iterations, instances, launch_repeats);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
#endif
