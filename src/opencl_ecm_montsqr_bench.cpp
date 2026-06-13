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
#include <sstream>
#include <algorithm>
#include <cctype>

namespace {

constexpr uint32_t MAX_BENCH_BITS = 8192;
constexpr uint32_t MAX_BENCH_WORDS = MAX_BENCH_BITS / 32;

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

void fill_to_gmp(const uint32_t *in_words, size_t words, mpz_t out) {
    mpz_import(out, words, -1, sizeof(uint32_t), 0, 0, in_words);
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

bool run_kernel_with_local(cl_command_queue q, cl_kernel k, size_t global, size_t local,
                           int repeats, double &ms) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeats; ++i) {
        cl_int err = clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
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

std::string read_device_vendor(cl_device_id dev) {
    char vendor[256] = {0};
    clGetDeviceInfo(dev, CL_DEVICE_VENDOR, sizeof(vendor) - 1, vendor, nullptr);
    return std::string(vendor);
}

bool is_amd_gpu_device(cl_device_id dev, cl_platform_id platform) {
    char pname[256] = {0};
    char dname[256] = {0};
    if (platform) {
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(pname), pname, nullptr);
    }
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(dname), dname, nullptr);
    std::string p = pname;
    std::string d = dname;
    std::transform(p.begin(), p.end(), p.begin(),
                   [](unsigned char c) { return (char)std::toupper(c); });
    std::transform(d.begin(), d.end(), d.begin(),
                   [](unsigned char c) { return (char)std::toupper(c); });
    return p.find("AMD") != std::string::npos || d.find("AMD") != std::string::npos ||
           d.find("GFX") != std::string::npos || d.find("RADEON") != std::string::npos;
}

int resolve_impl4_unroll(cl_device_id dev) {
    if (const char *v = std::getenv("ECM_MONT_WG_IMPL4_UNROLL")) {
        int parsed = std::atoi(v);
        if (parsed == 1 || parsed == 2) {
            return parsed;
        }
        std::cerr << "Warning: invalid ECM_MONT_WG_IMPL4_UNROLL=" << parsed
                  << ", fallback to auto" << std::endl;
    }
    std::string vendor = read_device_vendor(dev);
    std::transform(vendor.begin(), vendor.end(), vendor.begin(),
                   [](unsigned char c) { return (char)std::toupper(c); });
    if (vendor.find("NVIDIA") != std::string::npos) {
        return 1;
    }
    return 2;
}

} // namespace

bool runOpenClEcmMontSqrBenchmark(int bits, int kernel_iterations, int instances, int launch_repeats,
                                  bool use_wg, int tpi) {
    if (bits <= 0 || (bits % 32) != 0 || (uint32_t)bits > MAX_BENCH_BITS) {
        std::cerr << "bits must be a positive multiple of 32 and <= " << MAX_BENCH_BITS
                  << std::endl;
        return false;
    }
    const uint32_t BITS = (uint32_t)bits;
    const uint32_t WORDS = BITS / 32;
    constexpr uint32_t FIXED_4096_WORDS = 128u;
    const bool bench_wg = use_wg && WORDS != 16u;

    std::cout << "ECM montgomery square microbench: " << BITS
              << "-bit, kernel_iterations=" << kernel_iterations
              << ", instances=" << instances
              << ", launch_repeats=" << launch_repeats
              << ", mode=" << (bench_wg ? "wg" : "priv")
              << ", tpi=" << tpi << std::endl;

    mpz_t n_gmp, a_gmp, b_gmp;
    mpz_init(n_gmp);
    mpz_init(a_gmp);
    mpz_init(b_gmp);
    mpz_ui_pow_ui(n_gmp, 2, BITS-1);
    mpz_sub_ui(n_gmp, n_gmp, 109u);
    mpz_ui_pow_ui(a_gmp, 2, BITS-1);
    mpz_sub_ui(a_gmp, a_gmp, 991u);
    mpz_ui_pow_ui(b_gmp, 2, BITS-1);
    mpz_sub_ui(b_gmp, b_gmp, 8218291649u);
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

    std::string mont_priv =
        cgbn::opencl::load_kernel_file("bench/mont_priv.cl");
    std::string mont_priv_bench_src =
        cgbn::opencl::load_kernel_file("bench/mont_priv_bench.cl");
    std::string mont_priv_opt =
        cgbn::opencl::load_kernel_file("bench/mont_priv_opt.cl");
    std::string mont_mul_manual_src = cgbn::opencl::load_kernel_file(
        "bench/mont_mul_unroll_only_512_manual_generated.cl");
    std::string mont_mul_asm_fused_src =
        cgbn::opencl::load_kernel_file("bench/mont_mul_asm_fused.cl");
    std::string mont_mul_asm_block8_src = cgbn::opencl::load_kernel_file(
        "bench/mont_mul_asm_block8_generated.cl");
    std::string mont_mul_asm_src = cgbn::opencl::load_kernel_file(
        "bench/mont_mul_asm_512_generated.cl");
    std::string mont_priv_opt_bench_src =
        cgbn::opencl::load_kernel_file("bench/mont_priv_opt_bench.cl");
    std::string mont_wg_src =
        cgbn::opencl::load_kernel_file("bench/mont_wg.cl");
    std::string mont_wg_bench_src =
        cgbn::opencl::load_kernel_file("bench/mont_wg_bench.cl");
    if (mont_priv.empty() || mont_priv_bench_src.empty() || mont_priv_opt.empty() ||
        mont_priv_opt_bench_src.empty()) {
        std::cerr << "Failed to load mont_priv / mont_priv_opt kernel sources" << std::endl;
        return false;
    }
    if (WORDS == 16u && mont_mul_manual_src.empty()) {
        std::cerr << "Warning: mont_mul_unroll_only_512_manual_generated.cl missing; run "
                     "tools/gen_mont_mul_unroll_only_512_manual.py\n";
    }
    bool mont_mul_asm_enabled = false;
    const bool amd_gpu = is_amd_gpu_device(ctx.device, ctx.platform);
    if (WORDS == 16u && amd_gpu) {
        if (mont_mul_asm_fused_src.empty() || mont_mul_asm_block8_src.empty() ||
            mont_mul_asm_src.empty()) {
            std::cerr << "Warning: mont_mul_asm*.cl missing; run tools/gen_mont_mul_asm_512.py and "
                         "tools/gen_mont_mul_asm_block8.py\n";
        } else {
            mont_mul_asm_enabled = true;
        }
    } else if (WORDS == 16u && !amd_gpu) {
        std::cout << "Note: mont_mul_asm bench skipped (AMD GPU only)\n";
    }
    if (bench_wg && (mont_wg_src.empty() || mont_wg_bench_src.empty())) {
        std::cerr << "Failed to load mont_wg sources" << std::endl;
        return false;
    }
    auto strip_include = [](std::string &s, const std::string &inc) {
        size_t pos = s.find(inc);
        if (pos != std::string::npos) {
            s.erase(pos, inc.size());
        }
    };
    strip_include(mont_priv_bench_src, "#include \"mont_priv.cl\"");
    strip_include(mont_priv_opt_bench_src, "#include \"mont_priv_opt.cl\"");
    strip_include(mont_wg_bench_src, "#include \"mont_wg.cl\"");
    const std::string mont_priv_all = mont_priv + "\n" + mont_priv_opt + "\n" + mont_mul_manual_src +
                                      (mont_mul_asm_enabled
                                           ? ("\n" + mont_mul_asm_fused_src + "\n" +
                                              mont_mul_asm_block8_src + "\n" + mont_mul_asm_src)
                                           : std::string()) +
                                      "\n" + mont_priv_bench_src + "\n" + mont_priv_opt_bench_src;
      std::string src;
    if (bench_wg) {
        src = mont_wg_src + "\n" + mont_priv_all + "\n" + mont_wg_bench_src;
    } else {
        src = mont_priv_all;
    }
    cl_int buildErr = CL_SUCCESS;
    int wg_impl = 4;
    int impl4_unroll = resolve_impl4_unroll(ctx.device);
    if (const char *v = std::getenv("ECM_MONT_WG_IMPL")) {
        wg_impl = std::atoi(v);
        if (wg_impl == 2 || wg_impl == 3) {
            std::cerr << "Warning: WG_IMPL=" << wg_impl
                      << " removed (only 0/1/4 supported), fallback to 4\n";
            wg_impl = 4;
        } else if (wg_impl != 0 && wg_impl != 1 && wg_impl != 4) {
            std::cerr << "Warning: invalid ECM_MONT_WG_IMPL=" << wg_impl
                      << ", fallback to 4\n";
            wg_impl = 4;
        }
    }
    char build_opts[256];
    if (mont_mul_asm_enabled) {
        snprintf(build_opts, sizeof(build_opts),
                 "-DMAX_LIMBS=%u -DTPI=%d -DMONT_WG_IMPL=%d -DMONT_WG_IMPL4_UNROLL=%d "
                 "-DMONT_MUL_ASM_ENABLE=1",
                 WORDS, tpi, wg_impl, impl4_unroll);
    } else {
        snprintf(build_opts, sizeof(build_opts),
                 "-DMAX_LIMBS=%u -DTPI=%d -DMONT_WG_IMPL=%d -DMONT_WG_IMPL4_UNROLL=%d",
                 WORDS, tpi, wg_impl, impl4_unroll);
    }
    std::cout << "WG build opts: impl=" << wg_impl
              << " impl4_unroll=" << impl4_unroll << std::endl;
    std::cout << "OpenCL: compiling kernels (large source; may take minutes on NVIDIA)..."
              << std::endl;
    std::cout.flush();
    cl_program program = cgbn::opencl::build_program_from_source(
        ctx, src.c_str(), build_opts, buildErr);
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
    cl_uint np0_host = np0;
    cl_uint limbs = WORDS;
    cl_uint iters = (cl_uint)kernel_iterations;
    size_t global = (size_t)instances;

    cl_mem bufN_const = clCreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(uint32_t) * WORDS, n_words.data(), &err);
    cl_mem bufNp0_const = clCreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint),
                                         &np0_host, &err);
    if (err != CL_SUCCESS || bufN_const == nullptr || bufNp0_const == nullptr) {
        std::cerr << "Failed to create constant buffers for mont_priv_opt: " << err << std::endl;
        return false;
    }

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

    auto run_priv_opt = [&](const char *kname, bool is_mul, double &ms_out) -> bool {
        cl_int kerr = CL_SUCCESS;
        cl_kernel k = clCreateKernel(program, kname, &kerr);
        if (kerr != CL_SUCCESS) {
            std::cerr << "Create kernel " << kname << " failed: " << kerr << std::endl;
            return false;
        }
        clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
        if (is_mul) {
            clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
            clSetKernelArg(k, 2, sizeof(cl_mem), &bufN_const);
            clSetKernelArg(k, 3, sizeof(cl_mem), &bufOut);
            clSetKernelArg(k, 4, sizeof(cl_mem), &bufNp0_const);
            clSetKernelArg(k, 5, sizeof(cl_uint), &limbs);
            clSetKernelArg(k, 6, sizeof(cl_uint), &iters);
        } else {
            clSetKernelArg(k, 1, sizeof(cl_mem), &bufN_const);
            clSetKernelArg(k, 2, sizeof(cl_mem), &bufOut);
            clSetKernelArg(k, 3, sizeof(cl_mem), &bufNp0_const);
            clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
            clSetKernelArg(k, 5, sizeof(cl_uint), &iters);
        }
        bool ok = run_kernel(ctx.queue, k, global, launch_repeats, ms_out);
        if (ok) {
            size_t priv_b = 0, loc_b = 0, pref = 0, wg_sz = 0;
            query_kernel_resources(k, ctx.device, priv_b, loc_b, pref, wg_sz);
            double op_count_local =
                (double)instances * (double)kernel_iterations * (double)launch_repeats;
            double ops_s = op_count_local / (ms_out / 1000.0);
            std::cout << "  [" << kname << "] private_mem=" << priv_b << "B local_mem=" << loc_b
                      << "B pref_wg=" << pref << " max_wg=" << wg_sz << std::endl;
            if (csv_enabled) {
                csv << kname << "," << ms_out << "," << ops_s << "," << priv_b << "," << loc_b << ","
                    << pref << "," << wg_sz << "\n";
            }
        }
        clReleaseKernel(k);
        return ok;
    };

    auto run_priv_local_kernel = [&](const char *kname, bool is_mul, uint32_t required_words,
                                     double &ms_out) -> bool {
        if (WORDS != required_words) {
            ms_out = 0.0;
            return true;
        }
        cl_int kerr = CL_SUCCESS;
        cl_kernel k = clCreateKernel(program, kname, &kerr);
        if (kerr != CL_SUCCESS) {
            std::cerr << "Create kernel " << kname << " failed: " << kerr << std::endl;
            return false;
        }
        clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
        if (is_mul) {
            clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
            clSetKernelArg(k, 2, sizeof(cl_mem), &bufN_const);
            clSetKernelArg(k, 3, sizeof(cl_mem), &bufOut);
            clSetKernelArg(k, 4, sizeof(cl_mem), &bufNp0_const);
            clSetKernelArg(k, 5, sizeof(cl_uint), &limbs);
            clSetKernelArg(k, 6, sizeof(cl_uint), &iters);
            size_t local_mem_size = (size_t)2u * (size_t)required_words * sizeof(uint32_t); // local_size=1 in launch
            clSetKernelArg(k, 7, local_mem_size, nullptr);
        } else {
            clSetKernelArg(k, 1, sizeof(cl_mem), &bufN_const);
            clSetKernelArg(k, 2, sizeof(cl_mem), &bufOut);
            clSetKernelArg(k, 3, sizeof(cl_mem), &bufNp0_const);
            clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
            clSetKernelArg(k, 5, sizeof(cl_uint), &iters);
            size_t local_mem_size = (size_t)2u * (size_t)required_words * sizeof(uint32_t); // local_size=1 in launch
            clSetKernelArg(k, 6, local_mem_size, nullptr);
        }

        size_t local = 1u;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < launch_repeats; ++i) {
            cl_int e = clEnqueueNDRangeKernel(ctx.queue, k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
            if (e != CL_SUCCESS) {
                std::cerr << "Enqueue " << kname << " failed: " << e << std::endl;
                clReleaseKernel(k);
                return false;
            }
        }
        clFinish(ctx.queue);
        auto t1 = std::chrono::high_resolution_clock::now();
        ms_out = std::chrono::duration<double, std::milli>(t1 - t0).count();

        size_t priv_b = 0, loc_b = 0, pref = 0, wg_sz = 0;
        query_kernel_resources(k, ctx.device, priv_b, loc_b, pref, wg_sz);
        double op_count_local =
            (double)instances * (double)kernel_iterations * (double)launch_repeats;
        double ops_s = op_count_local / (ms_out / 1000.0);
        std::cout << "  [" << kname << "] private_mem=" << priv_b << "B local_mem=" << loc_b
                  << "B pref_wg=" << pref << " max_wg=" << wg_sz << std::endl;
        if (csv_enabled) {
            csv << kname << "," << ms_out << "," << ops_s << "," << priv_b << "," << loc_b << ","
                << pref << "," << wg_sz << "\n";
        }

        clReleaseKernel(k);
        return true;
    };

    auto run_priv_unroll_kernel = [&](const char *kname, bool is_mul, uint32_t required_words,
                                      double &ms_out) -> bool {
        if (WORDS != required_words) {
            ms_out = 0.0;
            return true;
        }
        cl_int kerr = CL_SUCCESS;
        cl_kernel k = clCreateKernel(program, kname, &kerr);
        if (kerr != CL_SUCCESS) {
            std::cerr << "Create kernel " << kname << " failed: " << kerr << std::endl;
            return false;
        }
        clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
        if (is_mul) {
            clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
            clSetKernelArg(k, 2, sizeof(cl_mem), &bufN_const);
            clSetKernelArg(k, 3, sizeof(cl_mem), &bufOut);
            clSetKernelArg(k, 4, sizeof(cl_mem), &bufNp0_const);
            clSetKernelArg(k, 5, sizeof(cl_uint), &limbs);
            clSetKernelArg(k, 6, sizeof(cl_uint), &iters);
        } else {
            clSetKernelArg(k, 1, sizeof(cl_mem), &bufN_const);
            clSetKernelArg(k, 2, sizeof(cl_mem), &bufOut);
            clSetKernelArg(k, 3, sizeof(cl_mem), &bufNp0_const);
            clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
            clSetKernelArg(k, 5, sizeof(cl_uint), &iters);
        }
        bool ok = run_kernel(ctx.queue, k, global, launch_repeats, ms_out);
        if (ok) {
            size_t priv_b = 0, loc_b = 0, pref = 0, wg_sz = 0;
            query_kernel_resources(k, ctx.device, priv_b, loc_b, pref, wg_sz);
            double op_count_local =
                (double)instances * (double)kernel_iterations * (double)launch_repeats;
            double ops_s = op_count_local / (ms_out / 1000.0);
            std::cout << "  [" << kname << "] private_mem=" << priv_b << "B local_mem=" << loc_b
                      << "B pref_wg=" << pref << " max_wg=" << wg_sz << std::endl;
            if (csv_enabled) {
                csv << kname << "," << ms_out << "," << ops_s << "," << priv_b << "," << loc_b << ","
                    << pref << "," << wg_sz << "\n";
            }
        }
        clReleaseKernel(k);
        return ok;
    };

    auto run_priv_unroll_mtn_kernel = [&](const char *kname, bool is_mul, uint32_t required_words,
                                         size_t local_size, size_t meta_words,
                                         double &ms_out) -> bool {
        if (WORDS != required_words) {
            ms_out = 0.0;
            return true;
        }
        cl_int kerr = CL_SUCCESS;
        cl_kernel k = clCreateKernel(program, kname, &kerr);
        if (kerr != CL_SUCCESS) {
            std::cerr << "Create kernel " << kname << " failed: " << kerr << std::endl;
            return false;
        }
        clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
        size_t local_mem_size = ((size_t)FIXED_4096_WORDS + 2u + (size_t)FIXED_4096_WORDS +
                                 (size_t)FIXED_4096_WORDS + meta_words) * sizeof(uint32_t);
        if (is_mul) {
            clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
            clSetKernelArg(k, 2, sizeof(cl_mem), &bufN_const);
            clSetKernelArg(k, 3, sizeof(cl_mem), &bufOut);
            clSetKernelArg(k, 4, sizeof(cl_mem), &bufNp0_const);
            clSetKernelArg(k, 5, sizeof(cl_uint), &limbs);
            clSetKernelArg(k, 6, sizeof(cl_uint), &iters);
            clSetKernelArg(k, 7, local_mem_size, nullptr);
        } else {
            clSetKernelArg(k, 1, sizeof(cl_mem), &bufN_const);
            clSetKernelArg(k, 2, sizeof(cl_mem), &bufOut);
            clSetKernelArg(k, 3, sizeof(cl_mem), &bufNp0_const);
            clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
            clSetKernelArg(k, 5, sizeof(cl_uint), &iters);
            clSetKernelArg(k, 6, local_mem_size, nullptr);
        }
        size_t global_mtn = (size_t)instances * local_size;
        bool ok = run_kernel_with_local(ctx.queue, k, global_mtn, local_size, launch_repeats, ms_out);
        if (ok) {
            size_t priv_b = 0, loc_b = 0, pref = 0, wg_sz = 0;
            query_kernel_resources(k, ctx.device, priv_b, loc_b, pref, wg_sz);
            double op_count_local =
                (double)instances * (double)kernel_iterations * (double)launch_repeats;
            double ops_s = op_count_local / (ms_out / 1000.0);
            std::cout << "  [" << kname << "] private_mem=" << priv_b << "B local_mem=" << loc_b
                      << "B pref_wg=" << pref << " max_wg=" << wg_sz << std::endl;
            if (csv_enabled) {
                csv << kname << "," << ms_out << "," << ops_s << "," << priv_b << "," << loc_b << ","
                    << pref << "," << wg_sz << "\n";
            }
        }
        clReleaseKernel(k);
        return ok;
    };

    auto run_priv_unroll_mt2_kernel = [&](const char *kname, bool is_mul, uint32_t required_words,
                                          double &ms_out) -> bool {
        return run_priv_unroll_mtn_kernel(kname, is_mul, required_words, 2u, 3u, ms_out);
    };

    constexpr size_t FIPS512_MT_LOCAL_U32 = 16u + 16u + 32u * 2u + 17u;
    constexpr size_t FIPS512_CS_LOCAL_U32 = 16u + 16u + 8u * 34u + 34u;
    constexpr size_t FIPS512_CS16_LOCAL_U32 = 16u + 16u + 16u * 34u + 34u;
    constexpr size_t FIPS4096_MT_LOCAL_U32 = 128u + 128u + 256u * 2u + 129u;
    constexpr size_t FIPS4096_CS_LOCAL_U32 = 128u + 128u + 8u * 258u + 258u;
    constexpr size_t FIPS4096_CS16_LOCAL_U32 = 128u + 128u + 16u * 258u + 258u;

    auto run_priv_fips_mt_kernel = [&](const char *kname, bool is_mul, uint32_t required_words,
                                       size_t local_size, size_t local_u32, double &ms_out) -> bool {
        if (WORDS != required_words) {
            ms_out = 0.0;
            return true;
        }
        cl_int kerr = CL_SUCCESS;
        cl_kernel k = clCreateKernel(program, kname, &kerr);
        if (kerr != CL_SUCCESS) {
            std::cerr << "Create kernel " << kname << " failed: " << kerr << std::endl;
            return false;
        }
        clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
        size_t local_mem_size = local_u32 * sizeof(uint32_t);
        if (is_mul) {
            clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
            clSetKernelArg(k, 2, sizeof(cl_mem), &bufN_const);
            clSetKernelArg(k, 3, sizeof(cl_mem), &bufOut);
            clSetKernelArg(k, 4, sizeof(cl_mem), &bufNp0_const);
            clSetKernelArg(k, 5, sizeof(cl_uint), &limbs);
            clSetKernelArg(k, 6, sizeof(cl_uint), &iters);
            clSetKernelArg(k, 7, local_mem_size, nullptr);
        } else {
            clSetKernelArg(k, 1, sizeof(cl_mem), &bufN_const);
            clSetKernelArg(k, 2, sizeof(cl_mem), &bufOut);
            clSetKernelArg(k, 3, sizeof(cl_mem), &bufNp0_const);
            clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
            clSetKernelArg(k, 5, sizeof(cl_uint), &iters);
            clSetKernelArg(k, 6, local_mem_size, nullptr);
        }
        size_t global_fips = (size_t)instances * local_size;
        bool ok = run_kernel_with_local(ctx.queue, k, global_fips, local_size, launch_repeats, ms_out);
        if (ok) {
            size_t priv_b = 0, loc_b = 0, pref = 0, wg_sz = 0;
            query_kernel_resources(k, ctx.device, priv_b, loc_b, pref, wg_sz);
            double op_count_local =
                (double)instances * (double)kernel_iterations * (double)launch_repeats;
            double ops_s = op_count_local / (ms_out / 1000.0);
            std::cout << "  [" << kname << "] private_mem=" << priv_b << "B local_mem=" << loc_b
                      << "B pref_wg=" << pref << " max_wg=" << wg_sz << std::endl;
            if (csv_enabled) {
                csv << kname << "," << ms_out << "," << ops_s << "," << priv_b << "," << loc_b << ","
                    << pref << "," << wg_sz << "\n";
            }
        }
        clReleaseKernel(k);
        return ok;
    };

    auto run_named_wg = [&](const char *kname, bool is_mul, double &ms_out) -> bool {
        cl_int kerr = CL_SUCCESS;
        cl_kernel k = clCreateKernel(program, kname, &kerr);
        if (kerr != CL_SUCCESS) {
            std::cerr << "Create kernel " << kname << " failed: " << kerr << std::endl;
            return false;
        }
        clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
        if (is_mul) {
            clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
            clSetKernelArg(k, 2, sizeof(cl_mem), &bufN);
            clSetKernelArg(k, 3, sizeof(cl_mem), &bufOut);
            clSetKernelArg(k, 4, sizeof(cl_uint), &np0);
            clSetKernelArg(k, 5, sizeof(cl_uint), &limbs);
            clSetKernelArg(k, 6, sizeof(cl_uint), &iters);
        } else {
            clSetKernelArg(k, 1, sizeof(cl_mem), &bufN);
            clSetKernelArg(k, 2, sizeof(cl_mem), &bufOut);
            clSetKernelArg(k, 3, sizeof(cl_uint), &np0);
            clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
            clSetKernelArg(k, 5, sizeof(cl_uint), &iters);
        }
        // mont_wg kernel uses local arrays:
        // t[(limbs+1)] + sum_lo[limbs] + sum_hi[limbs]
        // + carry_in/out (4*TPI words) + B[limbs] + N[limbs] + A[limbs]
        size_t local_mem_size =
            ((WORDS + 1u) + WORDS + WORDS + (size_t)(4 * tpi) + WORDS + WORDS + WORDS) *
            sizeof(uint32_t);
        clSetKernelArg(k, is_mul ? 7 : 6, local_mem_size, nullptr);

        size_t local = (size_t)tpi;
        size_t global_wg = (size_t)instances * local;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < launch_repeats; ++i) {
            cl_int err2 = clEnqueueNDRangeKernel(ctx.queue, k, 1, nullptr, &global_wg, &local, 0, nullptr, nullptr);
            if (err2 != CL_SUCCESS) {
                std::cerr << "Enqueue " << kname << " failed: " << err2 << std::endl;
                clReleaseKernel(k);
                return false;
            }
        }
        clFinish(ctx.queue);
        auto t1 = std::chrono::high_resolution_clock::now();
        ms_out = std::chrono::duration<double, std::milli>(t1 - t0).count();

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
        clReleaseKernel(k);
        return true;
    };

    double t_mul_priv = 0.0, t_sqr_priv = 0.0;
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

    double t_mul_priv_opt = 0.0, t_sqr_priv_opt = 0.0;
    if (!run_priv_opt("ecm_mont_mul_priv_opt_bench", true, t_mul_priv_opt)) return false;
    if (!run_priv_opt("ecm_mont_sqr_priv_opt_bench", false, t_sqr_priv_opt)) return false;

    double t_mul_priv_unroll_only_512 = 0.0, t_sqr_priv_unroll_only_512 = 0.0;
    double t_mul_priv_unroll_only_512_manual = 0.0;
    double t_mul_priv_unroll_only_512_asm = 0.0;
    if (!run_priv_unroll_kernel("ecm_mont_mul_priv_unroll_only_512_bench", true, 16u,
                                t_mul_priv_unroll_only_512)) {
        return false;
    }
    if (!mont_mul_manual_src.empty()) {
        if (!run_priv_unroll_kernel("ecm_mont_mul_priv_unroll_only_512_manual_bench", true, 16u,
                                    t_mul_priv_unroll_only_512_manual)) {
            return false;
        }
    }
    if (mont_mul_asm_enabled) {
        if (!run_priv_unroll_kernel("ecm_mont_mul_priv_unroll_only_512_asm_bench", true, 16u,
                                    t_mul_priv_unroll_only_512_asm)) {
            return false;
        }
    }
    if (!run_priv_unroll_kernel("ecm_mont_sqr_priv_unroll_only_512_bench", false, 16u,
                                t_sqr_priv_unroll_only_512)) {
        return false;
    }

    double t_mul_priv_fips512 = 0.0, t_sqr_priv_fips512 = 0.0;
    if (!run_priv_unroll_kernel("ecm_mont_mul_priv_fips512_bench", true, 16u, t_mul_priv_fips512)) {
        return false;
    }
    if (!run_priv_unroll_kernel("ecm_mont_sqr_priv_fips512_bench", false, 16u, t_sqr_priv_fips512)) {
        return false;
    }
    // Disabled @512: fips512_mt* paths omitted from bench (see opencl_ecm_montsqr_manifest.cpp).

    double t_mul_priv_local_only_512 = 0.0, t_sqr_priv_local_only_512 = 0.0;
    if (!run_priv_local_kernel("ecm_mont_mul_priv_local_only_512_bench", true, 16u,
                               t_mul_priv_local_only_512)) {
        return false;
    }
    if (!run_priv_local_kernel("ecm_mont_sqr_priv_local_only_512_bench", false, 16u,
                               t_sqr_priv_local_only_512)) {
        return false;
    }
    double t_mul_priv_opt2_512_local = 0.0, t_sqr_priv_opt2_512_local = 0.0;
    if (!run_priv_local_kernel("ecm_mont_mul_priv_opt2_512_local_bench", true, 16u,
                               t_mul_priv_opt2_512_local)) {
        return false;
    }
    if (!run_priv_local_kernel("ecm_mont_sqr_priv_opt2_512_local_bench", false, 16u,
                               t_sqr_priv_opt2_512_local)) {
        return false;
    }

    double t_mul_priv_unroll32 = 0.0, t_sqr_priv_unroll32 = 0.0;
    double t_mul_priv_unroll64 = 0.0, t_sqr_priv_unroll64 = 0.0;
    double t_mul_priv_unroll64_4096 = 0.0, t_sqr_priv_unroll64_4096 = 0.0;
    double t_mul_priv_unroll64_4096_nod = 0.0, t_sqr_priv_unroll64_4096_nod = 0.0;
    double t_mul_priv_unroll64_4096_mt2 = 0.0, t_sqr_priv_unroll64_4096_mt2 = 0.0;
    double t_mul_priv_unroll64_4096_mt2_weak = 0.0, t_sqr_priv_unroll64_4096_mt2_weak = 0.0;
    double t_mul_priv_unroll64_4096_mt4 = 0.0, t_sqr_priv_unroll64_4096_mt4 = 0.0;
    double t_mul_priv_unroll64_4096_mt8 = 0.0, t_sqr_priv_unroll64_4096_mt8 = 0.0;
    double t_mul_priv_fips4096 = 0.0, t_sqr_priv_fips4096 = 0.0;
    double t_mul_priv_fips4096_mt4 = 0.0, t_sqr_priv_fips4096_mt4 = 0.0;
    double t_mul_priv_fips4096_mt8 = 0.0, t_sqr_priv_fips4096_mt8 = 0.0;
    double t_mul_priv_fips4096_mt16 = 0.0, t_sqr_priv_fips4096_mt16 = 0.0;
    double t_mul_priv_fips4096_mt8_cs = 0.0, t_sqr_priv_fips4096_mt8_cs = 0.0;
    double t_mul_priv_fips4096_mt16_cs = 0.0, t_sqr_priv_fips4096_mt16_cs = 0.0;
    if (!run_priv_unroll_kernel("ecm_mont_mul_priv_unroll32_bench", true, WORDS, t_mul_priv_unroll32)) {
        return false;
    }
    if (!run_priv_unroll_kernel("ecm_mont_sqr_priv_unroll32_bench", false, WORDS, t_sqr_priv_unroll32)) {
        return false;
    }
    if (!run_priv_unroll_kernel("ecm_mont_mul_priv_unroll64_bench", true, WORDS, t_mul_priv_unroll64)) {
        return false;
    }
    if (!run_priv_unroll_kernel("ecm_mont_sqr_priv_unroll64_bench", false, WORDS, t_sqr_priv_unroll64)) {
        return false;
    }
    if (WORDS == 128u) {
        if (!run_priv_unroll_kernel("ecm_mont_mul_priv_unroll64_4096_bench", true, 128u,
                                    t_mul_priv_unroll64_4096)) {
            return false;
        }
        if (!run_priv_unroll_kernel("ecm_mont_sqr_priv_unroll64_4096_bench", false, 128u,
                                    t_sqr_priv_unroll64_4096)) {
            return false;
        }
        if (!run_priv_unroll_kernel("ecm_mont_mul_priv_unroll64_4096_nod_bench", true, 128u,
                                    t_mul_priv_unroll64_4096_nod)) {
            return false;
        }
        if (!run_priv_unroll_kernel("ecm_mont_sqr_priv_unroll64_4096_nod_bench", false, 128u,
                                    t_sqr_priv_unroll64_4096_nod)) {
            return false;
        }
        if (!run_priv_unroll_mt2_kernel("ecm_mont_mul_priv_unroll64_4096_mt2_bench", true, 128u,
                                        t_mul_priv_unroll64_4096_mt2)) {
            return false;
        }
        if (!run_priv_unroll_mt2_kernel("ecm_mont_sqr_priv_unroll64_4096_mt2_bench", false, 128u,
                                        t_sqr_priv_unroll64_4096_mt2)) {
            return false;
        }
        if (!run_priv_unroll_mt2_kernel("ecm_mont_mul_priv_unroll64_4096_mt2_weak_bench", true, 128u,
                                        t_mul_priv_unroll64_4096_mt2_weak)) {
            return false;
        }
        if (!run_priv_unroll_mt2_kernel("ecm_mont_sqr_priv_unroll64_4096_mt2_weak_bench", false, 128u,
                                        t_sqr_priv_unroll64_4096_mt2_weak)) {
            return false;
        }
        if (!run_priv_unroll_mtn_kernel("ecm_mont_mul_priv_unroll64_4096_mt4_bench", true, 128u, 4u, 5u,
                                        t_mul_priv_unroll64_4096_mt4)) {
            return false;
        }
        if (!run_priv_unroll_mtn_kernel("ecm_mont_sqr_priv_unroll64_4096_mt4_bench", false, 128u, 4u, 5u,
                                        t_sqr_priv_unroll64_4096_mt4)) {
            return false;
        }
        if (!run_priv_unroll_mtn_kernel("ecm_mont_mul_priv_unroll64_4096_mt8_bench", true, 128u, 8u, 9u,
                                        t_mul_priv_unroll64_4096_mt8)) {
            return false;
        }
        if (!run_priv_unroll_mtn_kernel("ecm_mont_sqr_priv_unroll64_4096_mt8_bench", false, 128u, 8u, 9u,
                                        t_sqr_priv_unroll64_4096_mt8)) {
            return false;
        }
        if (!run_priv_unroll_kernel("ecm_mont_mul_priv_fips4096_bench", true, 128u, t_mul_priv_fips4096)) {
            return false;
        }
        if (!run_priv_unroll_kernel("ecm_mont_sqr_priv_fips4096_bench", false, 128u, t_sqr_priv_fips4096)) {
            return false;
        }
        if (!run_priv_fips_mt_kernel("ecm_mont_mul_priv_fips4096_mt4_bench", true, 128u, 4u,
                                     FIPS4096_MT_LOCAL_U32, t_mul_priv_fips4096_mt4)) {
            return false;
        }
        if (!run_priv_fips_mt_kernel("ecm_mont_sqr_priv_fips4096_mt4_bench", false, 128u, 4u,
                                     FIPS4096_MT_LOCAL_U32, t_sqr_priv_fips4096_mt4)) {
            return false;
        }
        if (!run_priv_fips_mt_kernel("ecm_mont_mul_priv_fips4096_mt8_bench", true, 128u, 8u,
                                     FIPS4096_MT_LOCAL_U32, t_mul_priv_fips4096_mt8)) {
            return false;
        }
        if (!run_priv_fips_mt_kernel("ecm_mont_sqr_priv_fips4096_mt8_bench", false, 128u, 8u,
                                     FIPS4096_MT_LOCAL_U32, t_sqr_priv_fips4096_mt8)) {
            return false;
        }
        if (!run_priv_fips_mt_kernel("ecm_mont_mul_priv_fips4096_mt16_bench", true, 128u, 16u,
                                     FIPS4096_MT_LOCAL_U32, t_mul_priv_fips4096_mt16)) {
            return false;
        }
        if (!run_priv_fips_mt_kernel("ecm_mont_sqr_priv_fips4096_mt16_bench", false, 128u, 16u,
                                     FIPS4096_MT_LOCAL_U32, t_sqr_priv_fips4096_mt16)) {
            return false;
        }
        if (!run_priv_fips_mt_kernel("ecm_mont_mul_priv_fips4096_mt8_cs_bench", true, 128u, 8u,
                                     FIPS4096_CS_LOCAL_U32, t_mul_priv_fips4096_mt8_cs)) {
            return false;
        }
        if (!run_priv_fips_mt_kernel("ecm_mont_sqr_priv_fips4096_mt8_cs_bench", false, 128u, 8u,
                                     FIPS4096_CS_LOCAL_U32, t_sqr_priv_fips4096_mt8_cs)) {
            return false;
        }
        if (!run_priv_fips_mt_kernel("ecm_mont_mul_priv_fips4096_mt16_cs_bench", true, 128u, 16u,
                                     FIPS4096_CS16_LOCAL_U32, t_mul_priv_fips4096_mt16_cs)) {
            return false;
        }
        if (!run_priv_fips_mt_kernel("ecm_mont_sqr_priv_fips4096_mt16_cs_bench", false, 128u, 16u,
                                     FIPS4096_CS16_LOCAL_U32, t_sqr_priv_fips4096_mt16_cs)) {
            return false;
        }
    }

    {
        cl_uint verify_iters = 1u;
        std::vector<uint32_t> out_base(WORDS), out_opt(WORDS);

        auto run_verify_kernel = [&](const char *kname, bool is_mul, bool use_opt,
                                     std::vector<uint32_t> &out_words) -> bool {
            cl_int kerr = CL_SUCCESS;
            cl_kernel k = clCreateKernel(program, kname, &kerr);
            if (kerr != CL_SUCCESS) {
                std::cerr << "Create verify kernel " << kname << " failed: " << kerr << std::endl;
                return false;
            }
            if (use_opt) {
                clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
                if (is_mul) {
                    clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
                    clSetKernelArg(k, 2, sizeof(cl_mem), &bufN_const);
                    clSetKernelArg(k, 3, sizeof(cl_mem), &bufOut);
                    clSetKernelArg(k, 4, sizeof(cl_mem), &bufNp0_const);
                    clSetKernelArg(k, 5, sizeof(cl_uint), &limbs);
                    clSetKernelArg(k, 6, sizeof(cl_uint), &verify_iters);
                } else {
                    clSetKernelArg(k, 1, sizeof(cl_mem), &bufN_const);
                    clSetKernelArg(k, 2, sizeof(cl_mem), &bufOut);
                    clSetKernelArg(k, 3, sizeof(cl_mem), &bufNp0_const);
                    clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
                    clSetKernelArg(k, 5, sizeof(cl_uint), &verify_iters);
                }
            } else if (is_mul) {
                clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
                clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
                clSetKernelArg(k, 2, sizeof(cl_mem), &bufN);
                clSetKernelArg(k, 3, sizeof(cl_mem), &bufOut);
                clSetKernelArg(k, 4, sizeof(cl_uint), &np0);
                clSetKernelArg(k, 5, sizeof(cl_uint), &limbs);
                clSetKernelArg(k, 6, sizeof(cl_uint), &verify_iters);
            } else {
                clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
                clSetKernelArg(k, 1, sizeof(cl_mem), &bufN);
                clSetKernelArg(k, 2, sizeof(cl_mem), &bufOut);
                clSetKernelArg(k, 3, sizeof(cl_uint), &np0);
                clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
                clSetKernelArg(k, 5, sizeof(cl_uint), &verify_iters);
            }
            size_t g1 = 1;
            cl_int e2 = clEnqueueNDRangeKernel(ctx.queue, k, 1, nullptr, &g1, nullptr, 0, nullptr, nullptr);
            clFinish(ctx.queue);
            clReleaseKernel(k);
            if (e2 != CL_SUCCESS) {
                std::cerr << "Verify enqueue " << kname << " failed: " << e2 << std::endl;
                return false;
            }
            e2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                     out_words.data(), 0, nullptr, nullptr);
            return e2 == CL_SUCCESS;
        };

        if (!run_verify_kernel("ecm_mont_mul_priv_bench", true, false, out_base)) return false;
        if (!run_verify_kernel("ecm_mont_mul_priv_opt_bench", true, true, out_opt)) return false;
        if (WORDS == 16u) {
            std::vector<uint32_t> out_sqr512(WORDS), out_sqr_mul512(WORDS);
            auto run_sqr_fixed_verify = [&](const char *kname, std::vector<uint32_t> &out_buf) -> bool {
                cl_int kerr = CL_SUCCESS;
                cl_kernel k = clCreateKernel(program, kname, &kerr);
                if (kerr != CL_SUCCESS) {
                    std::cerr << "Create verify kernel " << kname << " failed: " << kerr << std::endl;
                    return false;
                }
                clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
                clSetKernelArg(k, 1, sizeof(cl_mem), &bufN_const);
                clSetKernelArg(k, 2, sizeof(cl_mem), &bufOut);
                clSetKernelArg(k, 3, sizeof(cl_mem), &bufNp0_const);
                clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
                clSetKernelArg(k, 5, sizeof(cl_uint), &verify_iters);
                size_t g = 1;
                cl_int err2 = clEnqueueNDRangeKernel(ctx.queue, k, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
                clFinish(ctx.queue);
                clReleaseKernel(k);
                if (err2 != CL_SUCCESS) {
                    std::cerr << "Enqueue verify " << kname << " failed: " << err2 << std::endl;
                    return false;
                }
                err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                           out_buf.data(), 0, nullptr, nullptr);
                return err2 == CL_SUCCESS;
            };
            if (!run_sqr_fixed_verify("ecm_mont_sqr_priv_unroll_only_512_bench", out_sqr512)) return false;
            {
                cl_int kerr = CL_SUCCESS;
                cl_kernel k = clCreateKernel(program, "ecm_mont_mul_priv_unroll_only_512_bench", &kerr);
                if (kerr != CL_SUCCESS) return false;
                clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
                clSetKernelArg(k, 1, sizeof(cl_mem), &bufA);
                clSetKernelArg(k, 2, sizeof(cl_mem), &bufN_const);
                clSetKernelArg(k, 3, sizeof(cl_mem), &bufOut);
                clSetKernelArg(k, 4, sizeof(cl_mem), &bufNp0_const);
                clSetKernelArg(k, 5, sizeof(cl_uint), &limbs);
                clSetKernelArg(k, 6, sizeof(cl_uint), &verify_iters);
                size_t g = 1;
                cl_int err2 = clEnqueueNDRangeKernel(ctx.queue, k, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
                clFinish(ctx.queue);
                clReleaseKernel(k);
                if (err2 != CL_SUCCESS) return false;
                err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                           out_sqr_mul512.data(), 0, nullptr, nullptr);
                if (err2 != CL_SUCCESS) return false;
            }
            bool match_sqr_mul = true;
            for (size_t i = 0; i < WORDS; ++i) {
                if (out_sqr512[i] != out_sqr_mul512[i]) {
                    match_sqr_mul = false;
                    break;
                }
            }
            std::cout << "  [sqr_unroll_only_512 vs mul(a,a)] " << (match_sqr_mul ? "MATCH" : "MISMATCH")
                      << std::endl;
            if (!match_sqr_mul) return false;

            std::vector<uint32_t> out_fips512(WORDS);
            auto run_fips512_verify = [&](const char *kname, bool is_mul, size_t local_sz,
                                          size_t local_u32, std::vector<uint32_t> &out_buf) -> bool {
                cl_int kerr = CL_SUCCESS;
                cl_kernel k = clCreateKernel(program, kname, &kerr);
                if (kerr != CL_SUCCESS) return false;
                clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
                if (local_sz == 0u) {
                    if (is_mul) {
                        clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
                        clSetKernelArg(k, 2, sizeof(cl_mem), &bufN_const);
                        clSetKernelArg(k, 3, sizeof(cl_mem), &bufOut);
                        clSetKernelArg(k, 4, sizeof(cl_mem), &bufNp0_const);
                        clSetKernelArg(k, 5, sizeof(cl_uint), &limbs);
                        clSetKernelArg(k, 6, sizeof(cl_uint), &verify_iters);
                    } else {
                        clSetKernelArg(k, 1, sizeof(cl_mem), &bufN_const);
                        clSetKernelArg(k, 2, sizeof(cl_mem), &bufOut);
                        clSetKernelArg(k, 3, sizeof(cl_mem), &bufNp0_const);
                        clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
                        clSetKernelArg(k, 5, sizeof(cl_uint), &verify_iters);
                    }
                } else {
                    clSetKernelArg(k, 1, sizeof(cl_mem), &bufN_const);
                    clSetKernelArg(k, 2, sizeof(cl_mem), &bufOut);
                    clSetKernelArg(k, 3, sizeof(cl_mem), &bufNp0_const);
                    clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
                    clSetKernelArg(k, 5, sizeof(cl_uint), &verify_iters);
                    clSetKernelArg(k, 6, local_u32 * sizeof(uint32_t), nullptr);
                }
                size_t g = (local_sz == 0u) ? 1u : local_sz;
                size_t ls = local_sz;
                cl_int err2 = (local_sz == 0u)
                    ? clEnqueueNDRangeKernel(ctx.queue, k, 1, nullptr, &g, nullptr, 0, nullptr, nullptr)
                    : clEnqueueNDRangeKernel(ctx.queue, k, 1, nullptr, &g, &ls, 0, nullptr, nullptr);
                clFinish(ctx.queue);
                clReleaseKernel(k);
                if (err2 != CL_SUCCESS) return false;
                err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                           out_buf.data(), 0, nullptr, nullptr);
                return err2 == CL_SUCCESS;
            };
            if (!run_fips512_verify("ecm_mont_mul_priv_fips512_bench", true, 0u, 0u, out_fips512)) return false;
            std::vector<uint32_t> out_unroll_mul512(WORDS);
            {
                cl_int kerr = CL_SUCCESS;
                cl_kernel k = clCreateKernel(program, "ecm_mont_mul_priv_unroll_only_512_bench", &kerr);
                if (kerr != CL_SUCCESS) return false;
                clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
                clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
                clSetKernelArg(k, 2, sizeof(cl_mem), &bufN_const);
                clSetKernelArg(k, 3, sizeof(cl_mem), &bufOut);
                clSetKernelArg(k, 4, sizeof(cl_mem), &bufNp0_const);
                clSetKernelArg(k, 5, sizeof(cl_uint), &limbs);
                clSetKernelArg(k, 6, sizeof(cl_uint), &verify_iters);
                size_t g = 1;
                cl_int err2 = clEnqueueNDRangeKernel(ctx.queue, k, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
                clFinish(ctx.queue);
                clReleaseKernel(k);
                if (err2 != CL_SUCCESS) return false;
                err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                           out_unroll_mul512.data(), 0, nullptr, nullptr);
                if (err2 != CL_SUCCESS) return false;
            }
            bool match_fips512 = true;
            for (size_t i = 0; i < WORDS; ++i) {
                if (out_fips512[i] != out_unroll_mul512[i]) {
                    match_fips512 = false;
                    break;
                }
            }
            std::cout << "  [fips512 vs unroll_only_512] " << (match_fips512 ? "MATCH" : "MISMATCH")
                      << std::endl;
            if (!match_fips512) return false;

            std::vector<uint32_t> out_fips512_sqr(WORDS);
            if (!run_fips512_verify("ecm_mont_sqr_priv_fips512_bench", false, 0u, 0u, out_fips512_sqr)) {
                return false;
            }

            // fips512_mt* verify disabled (kernels omitted from bench @512).
        }
        if (WORDS == 128u) {
            std::vector<uint32_t> out_sqr4096(WORDS), out_sqr_mul4096(WORDS);
            auto run_sqr4096_verify = [&](const char *kname, std::vector<uint32_t> &out_buf) -> bool {
                cl_int kerr = CL_SUCCESS;
                cl_kernel k = clCreateKernel(program, kname, &kerr);
                if (kerr != CL_SUCCESS) return false;
                clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
                clSetKernelArg(k, 1, sizeof(cl_mem), &bufN_const);
                clSetKernelArg(k, 2, sizeof(cl_mem), &bufOut);
                clSetKernelArg(k, 3, sizeof(cl_mem), &bufNp0_const);
                clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
                clSetKernelArg(k, 5, sizeof(cl_uint), &verify_iters);
                size_t g = 1;
                cl_int err2 = clEnqueueNDRangeKernel(ctx.queue, k, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
                clFinish(ctx.queue);
                clReleaseKernel(k);
                if (err2 != CL_SUCCESS) return false;
                err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                           out_buf.data(), 0, nullptr, nullptr);
                return err2 == CL_SUCCESS;
            };
            if (!run_sqr4096_verify("ecm_mont_sqr_priv_unroll64_4096_bench", out_sqr4096)) return false;
            {
                cl_int kerr = CL_SUCCESS;
                cl_kernel k = clCreateKernel(program, "ecm_mont_mul_priv_unroll64_4096_bench", &kerr);
                if (kerr != CL_SUCCESS) return false;
                clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
                clSetKernelArg(k, 1, sizeof(cl_mem), &bufA);
                clSetKernelArg(k, 2, sizeof(cl_mem), &bufN_const);
                clSetKernelArg(k, 3, sizeof(cl_mem), &bufOut);
                clSetKernelArg(k, 4, sizeof(cl_mem), &bufNp0_const);
                clSetKernelArg(k, 5, sizeof(cl_uint), &limbs);
                clSetKernelArg(k, 6, sizeof(cl_uint), &verify_iters);
                size_t g = 1;
                cl_int err2 = clEnqueueNDRangeKernel(ctx.queue, k, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
                clFinish(ctx.queue);
                clReleaseKernel(k);
                if (err2 != CL_SUCCESS) return false;
                err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                           out_sqr_mul4096.data(), 0, nullptr, nullptr);
                if (err2 != CL_SUCCESS) return false;
            }
            bool match_sqr4096 = true;
            for (size_t i = 0; i < WORDS; ++i) {
                if (out_sqr4096[i] != out_sqr_mul4096[i]) {
                    match_sqr4096 = false;
                    break;
                }
            }
            std::cout << "  [sqr_unroll64_4096 vs mul(a,a)] " << (match_sqr4096 ? "MATCH" : "MISMATCH")
                      << std::endl;
            if (!match_sqr4096) return false;

            auto run_mt4096_verify = [&](const char *kname, size_t local_sz, size_t meta_words,
                                         std::vector<uint32_t> &out_buf) -> bool {
                cl_int kerr = CL_SUCCESS;
                cl_kernel k = clCreateKernel(program, kname, &kerr);
                if (kerr != CL_SUCCESS) return false;
                clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
                clSetKernelArg(k, 1, sizeof(cl_mem), &bufN_const);
                clSetKernelArg(k, 2, sizeof(cl_mem), &bufOut);
                clSetKernelArg(k, 3, sizeof(cl_mem), &bufNp0_const);
                clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
                clSetKernelArg(k, 5, sizeof(cl_uint), &verify_iters);
                size_t local_mem_size = ((size_t)FIXED_4096_WORDS + 2u + (size_t)FIXED_4096_WORDS +
                                         (size_t)FIXED_4096_WORDS + meta_words) * sizeof(uint32_t);
                clSetKernelArg(k, 6, local_mem_size, nullptr);
                size_t g = local_sz;
                cl_int err2 =
                    clEnqueueNDRangeKernel(ctx.queue, k, 1, nullptr, &g, &local_sz, 0, nullptr, nullptr);
                clFinish(ctx.queue);
                clReleaseKernel(k);
                if (err2 != CL_SUCCESS) return false;
                err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                           out_buf.data(), 0, nullptr, nullptr);
                return err2 == CL_SUCCESS;
            };
            std::vector<uint32_t> out_mt4(WORDS), out_mt8(WORDS);
            if (!run_mt4096_verify("ecm_mont_sqr_priv_unroll64_4096_mt4_bench", 4u, 5u, out_mt4)) {
                return false;
            }
            if (!run_mt4096_verify("ecm_mont_sqr_priv_unroll64_4096_mt8_bench", 8u, 9u, out_mt8)) {
                return false;
            }
            bool match_mt4 = true, match_mt8 = true;
            for (size_t i = 0; i < WORDS; ++i) {
                if (out_mt4[i] != out_sqr4096[i]) match_mt4 = false;
                if (out_mt8[i] != out_sqr4096[i]) match_mt8 = false;
            }
            std::cout << "  [sqr_unroll64_4096_mt4 vs baseline] "
                      << (match_mt4 ? "MATCH" : "MISMATCH") << std::endl;
            std::cout << "  [sqr_unroll64_4096_mt8 vs baseline] "
                      << (match_mt8 ? "MATCH" : "MISMATCH") << std::endl;
            if (!match_mt4 || !match_mt8) return false;

            auto run_fips4096_verify = [&](const char *kname, bool is_mul, size_t local_sz,
                                            size_t local_u32, std::vector<uint32_t> &out_buf) -> bool {
                cl_int kerr = CL_SUCCESS;
                cl_kernel k = clCreateKernel(program, kname, &kerr);
                if (kerr != CL_SUCCESS) return false;
                clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
                if (local_sz == 0u) {
                    if (is_mul) {
                        clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
                        clSetKernelArg(k, 2, sizeof(cl_mem), &bufN_const);
                        clSetKernelArg(k, 3, sizeof(cl_mem), &bufOut);
                        clSetKernelArg(k, 4, sizeof(cl_mem), &bufNp0_const);
                        clSetKernelArg(k, 5, sizeof(cl_uint), &limbs);
                        clSetKernelArg(k, 6, sizeof(cl_uint), &verify_iters);
                    } else {
                        clSetKernelArg(k, 1, sizeof(cl_mem), &bufN_const);
                        clSetKernelArg(k, 2, sizeof(cl_mem), &bufOut);
                        clSetKernelArg(k, 3, sizeof(cl_mem), &bufNp0_const);
                        clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
                        clSetKernelArg(k, 5, sizeof(cl_uint), &verify_iters);
                    }
                } else {
                    clSetKernelArg(k, 1, sizeof(cl_mem), &bufN_const);
                    clSetKernelArg(k, 2, sizeof(cl_mem), &bufOut);
                    clSetKernelArg(k, 3, sizeof(cl_mem), &bufNp0_const);
                    clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
                    clSetKernelArg(k, 5, sizeof(cl_uint), &verify_iters);
                    clSetKernelArg(k, 6, local_u32 * sizeof(uint32_t), nullptr);
                }
                size_t g = (local_sz == 0u) ? 1u : local_sz;
                size_t ls = local_sz;
                cl_int err2 = (local_sz == 0u)
                    ? clEnqueueNDRangeKernel(ctx.queue, k, 1, nullptr, &g, nullptr, 0, nullptr, nullptr)
                    : clEnqueueNDRangeKernel(ctx.queue, k, 1, nullptr, &g, &ls, 0, nullptr, nullptr);
                clFinish(ctx.queue);
                clReleaseKernel(k);
                if (err2 != CL_SUCCESS) return false;
                err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                           out_buf.data(), 0, nullptr, nullptr);
                return err2 == CL_SUCCESS;
            };

            std::vector<uint32_t> out_fips4096(WORDS);
            if (!run_fips4096_verify("ecm_mont_mul_priv_fips4096_bench", true, 0u, 0u, out_fips4096)) {
                return false;
            }
            std::vector<uint32_t> out_unroll_mul4096(WORDS);
            {
                cl_int kerr = CL_SUCCESS;
                cl_kernel k = clCreateKernel(program, "ecm_mont_mul_priv_unroll64_4096_bench", &kerr);
                if (kerr != CL_SUCCESS) return false;
                clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
                clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
                clSetKernelArg(k, 2, sizeof(cl_mem), &bufN_const);
                clSetKernelArg(k, 3, sizeof(cl_mem), &bufOut);
                clSetKernelArg(k, 4, sizeof(cl_mem), &bufNp0_const);
                clSetKernelArg(k, 5, sizeof(cl_uint), &limbs);
                clSetKernelArg(k, 6, sizeof(cl_uint), &verify_iters);
                size_t g = 1;
                cl_int err2 = clEnqueueNDRangeKernel(ctx.queue, k, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
                clFinish(ctx.queue);
                clReleaseKernel(k);
                if (err2 != CL_SUCCESS) return false;
                err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                           out_unroll_mul4096.data(), 0, nullptr, nullptr);
                if (err2 != CL_SUCCESS) return false;
            }
            bool match_fips4096 = true;
            for (size_t i = 0; i < WORDS; ++i) {
                if (out_fips4096[i] != out_unroll_mul4096[i]) {
                    match_fips4096 = false;
                    break;
                }
            }
            std::cout << "  [fips4096 vs unroll64_4096] " << (match_fips4096 ? "MATCH" : "MISMATCH")
                      << std::endl;
            if (!match_fips4096) return false;

            std::vector<uint32_t> out_fips4096_sqr(WORDS);
            if (!run_fips4096_verify("ecm_mont_sqr_priv_fips4096_bench", false, 0u, 0u, out_fips4096_sqr)) {
                return false;
            }

            std::vector<uint32_t> out_fips4096_mt4(WORDS), out_fips4096_mt8(WORDS), out_fips4096_mt16(WORDS),
                out_fips4096_cs(WORDS), out_fips4096_cs16(WORDS);
            if (!run_fips4096_verify("ecm_mont_sqr_priv_fips4096_mt4_bench", false, 4u,
                                     FIPS4096_MT_LOCAL_U32, out_fips4096_mt4)) {
                return false;
            }
            if (!run_fips4096_verify("ecm_mont_sqr_priv_fips4096_mt8_bench", false, 8u,
                                     FIPS4096_MT_LOCAL_U32, out_fips4096_mt8)) {
                return false;
            }
            if (!run_fips4096_verify("ecm_mont_sqr_priv_fips4096_mt16_bench", false, 16u,
                                     FIPS4096_MT_LOCAL_U32, out_fips4096_mt16)) {
                return false;
            }
            if (!run_fips4096_verify("ecm_mont_sqr_priv_fips4096_mt8_cs_bench", false, 8u,
                                     FIPS4096_CS_LOCAL_U32, out_fips4096_cs)) {
                return false;
            }
            if (!run_fips4096_verify("ecm_mont_sqr_priv_fips4096_mt16_cs_bench", false, 16u,
                                     FIPS4096_CS16_LOCAL_U32, out_fips4096_cs16)) {
                return false;
            }
            bool match_fips4096_mt4 = true, match_fips4096_mt8 = true, match_fips4096_mt16 = true;
            bool match_fips4096_cs = true, match_fips4096_cs16 = true;
            for (size_t i = 0; i < WORDS; ++i) {
                if (out_fips4096_mt4[i] != out_fips4096_sqr[i]) match_fips4096_mt4 = false;
                if (out_fips4096_mt8[i] != out_fips4096_sqr[i]) match_fips4096_mt8 = false;
                if (out_fips4096_mt16[i] != out_fips4096_sqr[i]) match_fips4096_mt16 = false;
                if (out_fips4096_cs[i] != out_fips4096_sqr[i]) match_fips4096_cs = false;
                if (out_fips4096_cs16[i] != out_fips4096_sqr[i]) match_fips4096_cs16 = false;
            }
            std::cout << "  [fips4096_mt4 vs fips4096] " << (match_fips4096_mt4 ? "MATCH" : "MISMATCH")
                      << std::endl;
            std::cout << "  [fips4096_mt8 vs fips4096] " << (match_fips4096_mt8 ? "MATCH" : "MISMATCH")
                      << std::endl;
            std::cout << "  [fips4096_mt16 vs fips4096] " << (match_fips4096_mt16 ? "MATCH" : "MISMATCH")
                      << std::endl;
            std::cout << "  [fips4096_mt8_cs vs fips4096] " << (match_fips4096_cs ? "MATCH" : "MISMATCH")
                      << std::endl;
            std::cout << "  [fips4096_mt16_cs vs fips4096] " << (match_fips4096_cs16 ? "MATCH" : "MISMATCH")
                      << std::endl;
            if (!match_fips4096_mt4 || !match_fips4096_mt8 || !match_fips4096_mt16 ||
                !match_fips4096_cs || !match_fips4096_cs16) {
                return false;
            }
        }
        if (WORDS == 16u && !mont_mul_manual_src.empty()) {
            std::vector<uint32_t> out_unroll512(WORDS), out_manual(WORDS), out_asm(WORDS);
            auto run_fixed_verify = [&](const char *kname, std::vector<uint32_t> &out_buf) -> bool {
                cl_int kerr = CL_SUCCESS;
                cl_kernel k = clCreateKernel(program, kname, &kerr);
                if (kerr != CL_SUCCESS) {
                    std::cerr << "Create verify kernel " << kname << " failed: " << kerr << std::endl;
                    return false;
                }
                clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
                clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
                clSetKernelArg(k, 2, sizeof(cl_mem), &bufN_const);
                clSetKernelArg(k, 3, sizeof(cl_mem), &bufOut);
                clSetKernelArg(k, 4, sizeof(cl_mem), &bufNp0_const);
                clSetKernelArg(k, 5, sizeof(cl_uint), &limbs);
                clSetKernelArg(k, 6, sizeof(cl_uint), &verify_iters);
                size_t g = 1;
                cl_int err2 = clEnqueueNDRangeKernel(ctx.queue, k, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
                clFinish(ctx.queue);
                clReleaseKernel(k);
                if (err2 != CL_SUCCESS) {
                    std::cerr << "Enqueue verify " << kname << " failed: " << err2 << std::endl;
                    return false;
                }
                err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                           out_buf.data(), 0, nullptr, nullptr);
                return err2 == CL_SUCCESS;
            };
            if (!run_fixed_verify("ecm_mont_mul_priv_unroll_only_512_bench", out_unroll512)) return false;
            if (!run_fixed_verify("ecm_mont_mul_priv_unroll_only_512_manual_bench", out_manual)) return false;
            bool match_manual = true;
            for (size_t i = 0; i < WORDS; ++i) {
                if (out_manual[i] != out_unroll512[i]) {
                    match_manual = false;
                    break;
                }
            }
            std::cout << "  [unroll_only_512 vs manual] " << (match_manual ? "MATCH" : "MISMATCH")
                      << std::endl;
            if (!match_manual) return false;
            if (mont_mul_asm_enabled) {
                if (!run_fixed_verify("ecm_mont_mul_priv_unroll_only_512_asm_bench", out_asm)) return false;
                bool match_asm = true;
                for (size_t i = 0; i < WORDS; ++i) {
                    if (out_asm[i] != out_unroll512[i]) {
                        match_asm = false;
                        break;
                    }
                }
                std::cout << "  [unroll_only_512 vs asm] " << (match_asm ? "MATCH" : "MISMATCH")
                          << std::endl;
                if (!match_asm) return false;
            }
        }
        bool mul_match = true;
        for (uint32_t i = 0; i < WORDS; ++i) {
            if (out_base[i] != out_opt[i]) {
                mul_match = false;
                break;
            }
        }
        std::cout << "  [priv vs priv_opt mul] " << (mul_match ? "MATCH" : "MISMATCH") << std::endl;
        if (!mul_match) {
            return false;
        }

        if (!run_verify_kernel("ecm_mont_sqr_priv_bench", false, false, out_base)) return false;
        if (!run_verify_kernel("ecm_mont_sqr_priv_opt_bench", false, true, out_opt)) return false;
        bool sqr_match = true;
        for (uint32_t i = 0; i < WORDS; ++i) {
            if (out_base[i] != out_opt[i]) {
                sqr_match = false;
                break;
            }
        }
        std::cout << "  [priv vs priv_opt sqr] " << (sqr_match ? "MATCH" : "MISMATCH") << std::endl;
        if (!sqr_match) {
            return false;
        }
    }

    err = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                              host_out.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Read back failed: " << err << std::endl;
        return false;
    }

    double op_count = (double)instances * (double)kernel_iterations * (double)launch_repeats;
    std::cout << "mont_mul_priv:     " << t_mul_priv << " ms, " << (op_count / (t_mul_priv / 1000.0))
              << " ops/s" << std::endl;
    std::cout << "mont_mul_priv_opt: " << t_mul_priv_opt << " ms, "
              << (op_count / (t_mul_priv_opt / 1000.0)) << " ops/s"
              << " (vs priv: " << (t_mul_priv / t_mul_priv_opt) << "x)" << std::endl;
    std::cout << "mont_sqr_priv:     " << t_sqr_priv << " ms, " << (op_count / (t_sqr_priv / 1000.0))
              << " ops/s" << std::endl;
    std::cout << "mont_sqr_priv_opt: " << t_sqr_priv_opt << " ms, "
              << (op_count / (t_sqr_priv_opt / 1000.0)) << " ops/s"
              << " (vs priv: " << (t_sqr_priv / t_sqr_priv_opt) << "x)" << std::endl;
    if (WORDS == 16u) {
        std::cout << "mont_mul_priv_unroll_only_512: " << t_mul_priv_unroll_only_512 << " ms, "
                  << (op_count / (t_mul_priv_unroll_only_512 / 1000.0)) << " ops/s"
                  << " (vs opt: " << (t_mul_priv_opt / t_mul_priv_unroll_only_512) << "x)" << std::endl;
        if (!mont_mul_manual_src.empty()) {
            std::cout << "mont_mul_priv_unroll_only_512_manual: " << t_mul_priv_unroll_only_512_manual
                      << " ms, " << (op_count / (t_mul_priv_unroll_only_512_manual / 1000.0)) << " ops/s"
                      << " (vs unroll_only_512: "
                      << (t_mul_priv_unroll_only_512 / t_mul_priv_unroll_only_512_manual) << "x)" << std::endl;
        }
        if (mont_mul_asm_enabled) {
            std::cout << "mont_mul_priv_unroll_only_512_asm: " << t_mul_priv_unroll_only_512_asm
                      << " ms, " << (op_count / (t_mul_priv_unroll_only_512_asm / 1000.0)) << " ops/s"
                      << " (vs unroll_only_512: "
                      << (t_mul_priv_unroll_only_512 / t_mul_priv_unroll_only_512_asm) << "x)" << std::endl;
        }
        std::cout << "mont_sqr_priv_unroll_only_512: " << t_sqr_priv_unroll_only_512 << " ms, "
                  << (op_count / (t_sqr_priv_unroll_only_512 / 1000.0)) << " ops/s"
                  << " (vs opt: " << (t_sqr_priv_opt / t_sqr_priv_unroll_only_512) << "x"
                  << ", vs mul_unroll_only_512: "
                  << (t_mul_priv_unroll_only_512 / t_sqr_priv_unroll_only_512) << "x)" << std::endl;
        std::cout << "mont_mul_priv_fips512: " << t_mul_priv_fips512 << " ms, "
                  << (op_count / (t_mul_priv_fips512 / 1000.0)) << " ops/s"
                  << " (vs unroll_only_512: "
                  << (t_mul_priv_unroll_only_512 / t_mul_priv_fips512) << "x)" << std::endl;
        std::cout << "mont_sqr_priv_fips512: " << t_sqr_priv_fips512 << " ms, "
                  << (op_count / (t_sqr_priv_fips512 / 1000.0)) << " ops/s"
                  << " (vs fips512_mul: "
                  << (t_mul_priv_fips512 / t_sqr_priv_fips512) << "x)" << std::endl;
        std::cout << "mont_mul_priv_local_only_512:  " << t_mul_priv_local_only_512 << " ms, "
                  << (op_count / (t_mul_priv_local_only_512 / 1000.0)) << " ops/s"
                  << " (vs opt: " << (t_mul_priv_opt / t_mul_priv_local_only_512) << "x)" << std::endl;
        std::cout << "mont_sqr_priv_local_only_512:  " << t_sqr_priv_local_only_512 << " ms, "
                  << (op_count / (t_sqr_priv_local_only_512 / 1000.0)) << " ops/s"
                  << " (vs opt: " << (t_sqr_priv_opt / t_sqr_priv_local_only_512) << "x)" << std::endl;
        std::cout << "mont_mul_priv_opt2_512_local: " << t_mul_priv_opt2_512_local << " ms, "
                  << (op_count / (t_mul_priv_opt2_512_local / 1000.0)) << " ops/s"
                  << " (vs opt: " << (t_mul_priv_opt / t_mul_priv_opt2_512_local) << "x)" << std::endl;
        std::cout << "mont_sqr_priv_opt2_512_local: " << t_sqr_priv_opt2_512_local << " ms, "
                  << (op_count / (t_sqr_priv_opt2_512_local / 1000.0)) << " ops/s"
                  << " (vs opt: " << (t_sqr_priv_opt / t_sqr_priv_opt2_512_local) << "x)" << std::endl;
    }
    std::cout << "mont_mul_priv_unroll32:  " << t_mul_priv_unroll32 << " ms, "
              << (op_count / (t_mul_priv_unroll32 / 1000.0)) << " ops/s"
              << " (vs opt: " << (t_mul_priv_opt / t_mul_priv_unroll32) << "x)" << std::endl;
    std::cout << "mont_sqr_priv_unroll32:  " << t_sqr_priv_unroll32 << " ms, "
              << (op_count / (t_sqr_priv_unroll32 / 1000.0)) << " ops/s"
              << " (vs opt: " << (t_sqr_priv_opt / t_sqr_priv_unroll32) << "x)" << std::endl;
    std::cout << "mont_mul_priv_unroll64:  " << t_mul_priv_unroll64 << " ms, "
              << (op_count / (t_mul_priv_unroll64 / 1000.0)) << " ops/s"
              << " (vs opt: " << (t_mul_priv_opt / t_mul_priv_unroll64) << "x)" << std::endl;
    std::cout << "mont_sqr_priv_unroll64:  " << t_sqr_priv_unroll64 << " ms, "
              << (op_count / (t_sqr_priv_unroll64 / 1000.0)) << " ops/s"
              << " (vs opt: " << (t_sqr_priv_opt / t_sqr_priv_unroll64) << "x)" << std::endl;
    if (WORDS == 128u) {
        std::cout << "mont_mul_priv_unroll64_4096: " << t_mul_priv_unroll64_4096 << " ms, "
                  << (op_count / (t_mul_priv_unroll64_4096 / 1000.0)) << " ops/s"
                  << " (vs generic64: " << (t_mul_priv_unroll64 / t_mul_priv_unroll64_4096) << "x)" << std::endl;
        std::cout << "mont_sqr_priv_unroll64_4096: " << t_sqr_priv_unroll64_4096 << " ms, "
                  << (op_count / (t_sqr_priv_unroll64_4096 / 1000.0)) << " ops/s"
                  << " (vs generic64: " << (t_sqr_priv_unroll64 / t_sqr_priv_unroll64_4096) << "x"
                  << ", vs mul_unroll64_4096: "
                  << (t_mul_priv_unroll64_4096 / t_sqr_priv_unroll64_4096) << "x)" << std::endl;
        std::cout << "mont_mul_priv_unroll64_4096_nod: " << t_mul_priv_unroll64_4096_nod << " ms, "
                  << (op_count / (t_mul_priv_unroll64_4096_nod / 1000.0)) << " ops/s"
                  << " (vs unroll64_4096: " << (t_mul_priv_unroll64_4096 / t_mul_priv_unroll64_4096_nod) << "x)" << std::endl;
        std::cout << "mont_sqr_priv_unroll64_4096_nod: " << t_sqr_priv_unroll64_4096_nod << " ms, "
                  << (op_count / (t_sqr_priv_unroll64_4096_nod / 1000.0)) << " ops/s"
                  << " (vs unroll64_4096: " << (t_sqr_priv_unroll64_4096 / t_sqr_priv_unroll64_4096_nod) << "x)" << std::endl;
        std::cout << "mont_mul_priv_unroll64_4096_mt2: " << t_mul_priv_unroll64_4096_mt2 << " ms, "
                  << (op_count / (t_mul_priv_unroll64_4096_mt2 / 1000.0)) << " ops/s"
                  << " (vs unroll64_4096: " << (t_mul_priv_unroll64_4096 / t_mul_priv_unroll64_4096_mt2) << "x)" << std::endl;
        std::cout << "mont_sqr_priv_unroll64_4096_mt2: " << t_sqr_priv_unroll64_4096_mt2 << " ms, "
                  << (op_count / (t_sqr_priv_unroll64_4096_mt2 / 1000.0)) << " ops/s"
                  << " (vs unroll64_4096: " << (t_sqr_priv_unroll64_4096 / t_sqr_priv_unroll64_4096_mt2) << "x)" << std::endl;
        std::cout << "mont_mul_priv_unroll64_4096_mt2_weak: " << t_mul_priv_unroll64_4096_mt2_weak << " ms, "
                  << (op_count / (t_mul_priv_unroll64_4096_mt2_weak / 1000.0)) << " ops/s"
                  << " (vs unroll64_4096: " << (t_mul_priv_unroll64_4096 / t_mul_priv_unroll64_4096_mt2_weak) << "x)" << std::endl;
        std::cout << "mont_sqr_priv_unroll64_4096_mt2_weak: " << t_sqr_priv_unroll64_4096_mt2_weak << " ms, "
                  << (op_count / (t_sqr_priv_unroll64_4096_mt2_weak / 1000.0)) << " ops/s"
                  << " (vs unroll64_4096: " << (t_sqr_priv_unroll64_4096 / t_sqr_priv_unroll64_4096_mt2_weak) << "x)" << std::endl;
        std::cout << "mont_mul_priv_unroll64_4096_mt4: " << t_mul_priv_unroll64_4096_mt4 << " ms, "
                  << (op_count / (t_mul_priv_unroll64_4096_mt4 / 1000.0)) << " ops/s"
                  << " (vs unroll64_4096: " << (t_mul_priv_unroll64_4096 / t_mul_priv_unroll64_4096_mt4) << "x)" << std::endl;
        std::cout << "mont_sqr_priv_unroll64_4096_mt4: " << t_sqr_priv_unroll64_4096_mt4 << " ms, "
                  << (op_count / (t_sqr_priv_unroll64_4096_mt4 / 1000.0)) << " ops/s"
                  << " (vs unroll64_4096: " << (t_sqr_priv_unroll64_4096 / t_sqr_priv_unroll64_4096_mt4) << "x)" << std::endl;
        std::cout << "mont_mul_priv_unroll64_4096_mt8: " << t_mul_priv_unroll64_4096_mt8 << " ms, "
                  << (op_count / (t_mul_priv_unroll64_4096_mt8 / 1000.0)) << " ops/s"
                  << " (vs unroll64_4096: " << (t_mul_priv_unroll64_4096 / t_mul_priv_unroll64_4096_mt8) << "x)" << std::endl;
        std::cout << "mont_sqr_priv_unroll64_4096_mt8: " << t_sqr_priv_unroll64_4096_mt8 << " ms, "
                  << (op_count / (t_sqr_priv_unroll64_4096_mt8 / 1000.0)) << " ops/s"
                  << " (vs unroll64_4096: " << (t_sqr_priv_unroll64_4096 / t_sqr_priv_unroll64_4096_mt8) << "x)" << std::endl;
        std::cout << "mont_mul_priv_fips4096: " << t_mul_priv_fips4096 << " ms, "
                  << (op_count / (t_mul_priv_fips4096 / 1000.0)) << " ops/s"
                  << " (vs unroll64_4096: " << (t_mul_priv_unroll64_4096 / t_mul_priv_fips4096) << "x)" << std::endl;
        std::cout << "mont_sqr_priv_fips4096: " << t_sqr_priv_fips4096 << " ms, "
                  << (op_count / (t_sqr_priv_fips4096 / 1000.0)) << " ops/s"
                  << " (vs fips4096_mul: " << (t_mul_priv_fips4096 / t_sqr_priv_fips4096) << "x)" << std::endl;
        std::cout << "mont_mul_priv_fips4096_mt4: " << t_mul_priv_fips4096_mt4 << " ms, "
                  << (op_count / (t_mul_priv_fips4096_mt4 / 1000.0)) << " ops/s"
                  << " (vs fips4096: " << (t_mul_priv_fips4096 / t_mul_priv_fips4096_mt4) << "x)" << std::endl;
        std::cout << "mont_sqr_priv_fips4096_mt4: " << t_sqr_priv_fips4096_mt4 << " ms, "
                  << (op_count / (t_sqr_priv_fips4096_mt4 / 1000.0)) << " ops/s"
                  << " (vs fips4096: " << (t_sqr_priv_fips4096 / t_sqr_priv_fips4096_mt4) << "x)" << std::endl;
        std::cout << "mont_mul_priv_fips4096_mt8: " << t_mul_priv_fips4096_mt8 << " ms, "
                  << (op_count / (t_mul_priv_fips4096_mt8 / 1000.0)) << " ops/s"
                  << " (vs fips4096: " << (t_mul_priv_fips4096 / t_mul_priv_fips4096_mt8) << "x)" << std::endl;
        std::cout << "mont_sqr_priv_fips4096_mt8: " << t_sqr_priv_fips4096_mt8 << " ms, "
                  << (op_count / (t_sqr_priv_fips4096_mt8 / 1000.0)) << " ops/s"
                  << " (vs fips4096: " << (t_sqr_priv_fips4096 / t_sqr_priv_fips4096_mt8) << "x)" << std::endl;
        std::cout << "mont_mul_priv_fips4096_mt16: " << t_mul_priv_fips4096_mt16 << " ms, "
                  << (op_count / (t_mul_priv_fips4096_mt16 / 1000.0)) << " ops/s"
                  << " (vs fips4096: " << (t_mul_priv_fips4096 / t_mul_priv_fips4096_mt16) << "x)" << std::endl;
        std::cout << "mont_sqr_priv_fips4096_mt16: " << t_sqr_priv_fips4096_mt16 << " ms, "
                  << (op_count / (t_sqr_priv_fips4096_mt16 / 1000.0)) << " ops/s"
                  << " (vs fips4096: " << (t_sqr_priv_fips4096 / t_sqr_priv_fips4096_mt16) << "x)" << std::endl;
        std::cout << "mont_mul_priv_fips4096_mt8_cs: " << t_mul_priv_fips4096_mt8_cs << " ms, "
                  << (op_count / (t_mul_priv_fips4096_mt8_cs / 1000.0)) << " ops/s"
                  << " (vs fips4096: " << (t_mul_priv_fips4096 / t_mul_priv_fips4096_mt8_cs) << "x)" << std::endl;
        std::cout << "mont_sqr_priv_fips4096_mt8_cs: " << t_sqr_priv_fips4096_mt8_cs << " ms, "
                  << (op_count / (t_sqr_priv_fips4096_mt8_cs / 1000.0)) << " ops/s"
                  << " (vs fips4096: " << (t_sqr_priv_fips4096 / t_sqr_priv_fips4096_mt8_cs) << "x)" << std::endl;
        std::cout << "mont_mul_priv_fips4096_mt16_cs: " << t_mul_priv_fips4096_mt16_cs << " ms, "
                  << (op_count / (t_mul_priv_fips4096_mt16_cs / 1000.0)) << " ops/s"
                  << " (vs fips4096: " << (t_mul_priv_fips4096 / t_mul_priv_fips4096_mt16_cs) << "x)" << std::endl;
        std::cout << "mont_sqr_priv_fips4096_mt16_cs: " << t_sqr_priv_fips4096_mt16_cs << " ms, "
                  << (op_count / (t_sqr_priv_fips4096_mt16_cs / 1000.0)) << " ops/s"
                  << " (vs fips4096: " << (t_sqr_priv_fips4096 / t_sqr_priv_fips4096_mt16_cs) << "x)" << std::endl;
    }

    if (bench_wg) {
        double t_mul_wg = 0.0, t_sqr_wg = 0.0;
        if (!run_named_wg("cgbn_mont_mul_wg_bench", true, t_mul_wg)) return false;
        if (!run_named_wg("cgbn_mont_sqr_wg_bench", false, t_sqr_wg)) return false;
        std::cout << "mont_mul_wg:   " << t_mul_wg << " ms, " << (op_count / (t_mul_wg / 1000.0))
                  << " ops/s" << std::endl;
        std::cout << "mont_sqr_wg:   " << t_sqr_wg << " ms, " << (op_count / (t_sqr_wg / 1000.0))
                  << " ops/s" << std::endl;
        if (csv_enabled) {
            csv << "summary_selected_mont_mul_wg," << t_mul_wg << ","
                << (op_count / (t_mul_wg / 1000.0)) << ",0,0,0,0\n";
            csv << "summary_selected_mont_sqr_wg," << t_sqr_wg << ","
                << (op_count / (t_sqr_wg / 1000.0)) << ",0,0,0,0\n";
        }

        // Correctness check: compare WG montgomery mul/sqr against GMP reference.
        auto verify_wg_kernel = [&](const char *kname, bool is_mul, bool square_via_mul = false) -> bool {
            cl_int kerr = CL_SUCCESS;
            cl_kernel k = clCreateKernel(program, kname, &kerr);
            if (kerr != CL_SUCCESS) {
                std::cerr << "Create verify kernel " << kname << " failed: " << kerr << std::endl;
                return false;
            }
            cl_uint verify_iters = 1u;
            clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
            if (is_mul) {
                clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
                clSetKernelArg(k, 2, sizeof(cl_mem), &bufN);
                clSetKernelArg(k, 3, sizeof(cl_mem), &bufOut);
                clSetKernelArg(k, 4, sizeof(cl_uint), &np0);
                clSetKernelArg(k, 5, sizeof(cl_uint), &limbs);
                clSetKernelArg(k, 6, sizeof(cl_uint), &verify_iters);
            } else {
                clSetKernelArg(k, 1, sizeof(cl_mem), &bufN);
                clSetKernelArg(k, 2, sizeof(cl_mem), &bufOut);
                clSetKernelArg(k, 3, sizeof(cl_uint), &np0);
                clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
                clSetKernelArg(k, 5, sizeof(cl_uint), &verify_iters);
            }
            size_t local_mem_size =
                ((WORDS + 1u) + WORDS + WORDS + (size_t)(4 * tpi) + WORDS + WORDS + WORDS) *
                sizeof(uint32_t);
            clSetKernelArg(k, is_mul ? 7 : 6, local_mem_size, nullptr);

            size_t local = (size_t)tpi;
            // Single instance for GMP check (read out[0..limbs)).
            size_t global_wg = local;
            cl_int err2 =
                clEnqueueNDRangeKernel(ctx.queue, k, 1, nullptr, &global_wg, &local, 0, nullptr, nullptr);
            if (err2 != CL_SUCCESS) {
                std::cerr << "Enqueue verify " << kname << " failed: " << err2 << std::endl;
                clReleaseKernel(k);
                return false;
            }
            clFinish(ctx.queue);
            clReleaseKernel(k);

            std::vector<uint32_t> out_words(WORDS);
            err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                       out_words.data(), 0, nullptr, nullptr);
            if (err2 != CL_SUCCESS) {
                std::cerr << "Verify readback failed: " << err2 << std::endl;
                return false;
            }

            mpz_t r, rinv, expect, got, tmp;
            mpz_init(r);
            mpz_init(rinv);
            mpz_init(expect);
            mpz_init(got);
            mpz_init(tmp);
            mpz_ui_pow_ui(r, 2u, (unsigned long)(WORDS * 32u));
            if (mpz_invert(rinv, r, n_gmp) == 0) {
                std::cerr << "GMP invert failed for R^-1 mod N" << std::endl;
                mpz_clears(r, rinv, expect, got, tmp, nullptr);
                return false;
            }
            if (is_mul) {
                if (square_via_mul) {
                    mpz_mul(tmp, a_gmp, a_gmp);
                } else {
                    mpz_mul(tmp, a_gmp, b_gmp);
                }
            } else {
                mpz_mul(tmp, a_gmp, a_gmp);
            }
            mpz_mod(tmp, tmp, n_gmp);
            mpz_mul(expect, tmp, rinv);
            mpz_mod(expect, expect, n_gmp);
            fill_to_gmp(out_words.data(), WORDS, got);

            bool ok = (mpz_cmp(expect, got) == 0);
            if (!ok) {
                char *exp_s = mpz_get_str(nullptr, 16, expect);
                char *got_s = mpz_get_str(nullptr, 16, got);
                std::cerr << "WG verify mismatch for " << kname << "\n"
                          << "  expected=0x" << exp_s << "\n"
                          << "  got=0x" << got_s << std::endl;
                free(exp_s);
                free(got_s);
            } else {
                std::cout << "  [" << kname << "] GMP verify: PASS" << std::endl;
            }
            mpz_clears(r, rinv, expect, got, tmp, nullptr);
            return ok;
        };

        if (!verify_wg_kernel("cgbn_mont_mul_wg_bench", true, false)) return false;
        err = clEnqueueCopyBuffer(ctx.queue, bufA, bufB, 0, 0, sizeof(uint32_t) * WORDS, 0, nullptr, nullptr);
        clFinish(ctx.queue);
        if (err != CL_SUCCESS) {
            std::cerr << "Verify copy A->B failed: " << err << std::endl;
            return false;
        }
        if (!verify_wg_kernel("cgbn_mont_mul_wg_bench", true, true)) return false;
        std::cout << "  [cgbn_mont_mul_wg_bench b=a copy] GMP verify: PASS" << std::endl;
        if (!verify_wg_kernel("cgbn_mont_sqr_wg_bench", false, false)) return false;
    } else {
        if (csv_enabled) {
            csv << "summary_selected_mont_mul_priv," << t_mul_priv << ","
                << (op_count / (t_mul_priv / 1000.0)) << ",0,0,0,0\n";
            csv << "summary_selected_mont_sqr_priv," << t_sqr_priv << ","
                << (op_count / (t_sqr_priv / 1000.0)) << ",0,0,0,0\n";
        }
    }

    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufN);
    clReleaseMemObject(bufN_const);
    clReleaseMemObject(bufNp0_const);
    clReleaseMemObject(bufOut);
    clReleaseProgram(program);
    if (csv_enabled && csv.is_open()) csv.close();
    cgbn::opencl::destroy_context(ctx);
    mpz_clear(n_gmp);
    mpz_clear(a_gmp);
    mpz_clear(b_gmp);
    return true;
}

#ifdef BUILD_OPENCL_ECM_MONTSQR_MAIN
#include <cstdlib>
#include <stdexcept>

namespace {

bool parse_cli_int(const char *s, const char *label, int &out) {
    if (s == nullptr || *s == '\0') {
        std::cerr << "Invalid " << label << ": (empty)" << std::endl;
        return false;
    }
    try {
        size_t consumed = 0;
        long v = std::stol(s, &consumed);
        if (consumed == 0 || s[consumed] != '\0') {
            std::cerr << "Invalid " << label << ": \"" << s << "\"" << std::endl;
            return false;
        }
        out = (int)v;
        return true;
    } catch (const std::exception &) {
        std::cerr << "Invalid " << label << ": \"" << s << "\"" << std::endl;
        return false;
    }
}

} // namespace

int main(int argc, char **argv) {
    int bits = 1024;
    int kernel_iterations = 1000;
    int instances = 256;
    int launch_repeats = 50;
    bool use_wg = true;
    int tpi = 4;
    int device_index = -1;
    auto print_usage = [&]() {
        std::cout
            << "Usage: opencl_ecm_montsqr [--bits <bits>] [--iterations <n>] [--use-wg|--no-wg] [--tpi <tpi>] [-d|--device <index>] [kernel_iterations] [instances] [launch_repeats]\n"
            << "  --bits <bits>            Benchmark bit width (multiple of 32, <= 8192)\n"
            << "  --iterations <n>         Kernel loop count (alias for 1st positional arg)\n"
            << "  --use-wg / --no-wg       Select WG or private benchmark mode\n"
            << "  --tpi <tpi>              Threads per instance for WG mode\n"
            << "  -d, --device <index>     OpenCL device index (overrides default/env)\n"
            << "  -h, --help               Show this help message\n"
            << "  Set CGBN_KERNEL_ROOT to repo root if .cl files are not found.\n";
    };
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            print_usage();
            return EXIT_SUCCESS;
        }
        if (a == "--bits" && i + 1 < argc) {
            if (!parse_cli_int(argv[++i], "--bits", bits)) {
                return EXIT_FAILURE;
            }
            continue;
        }
        if ((a == "--iterations" || a == "--iters") && i + 1 < argc) {
            if (!parse_cli_int(argv[++i], a.c_str(), kernel_iterations)) {
                return EXIT_FAILURE;
            }
            continue;
        }
        if (a == "--use-wg") {
            use_wg = true;
            continue;
        }
        if (a == "--no-wg") {
            use_wg = false;
            continue;
        }
        if (a == "--tpi" && i + 1 < argc) {
            if (!parse_cli_int(argv[++i], "--tpi", tpi)) {
                return EXIT_FAILURE;
            }
            continue;
        }
        if ((a == "-d" || a == "--device") && i + 1 < argc) {
            if (!parse_cli_int(argv[++i], "--device", device_index)) {
                return EXIT_FAILURE;
            }
            continue;
        }
        if (!a.empty() && a[0] == '-') {
            std::cerr << "Unknown option: " << a << " (use --help)" << std::endl;
            return EXIT_FAILURE;
        }
        pos.push_back(a);
    }
    if (pos.size() >= 1 && !parse_cli_int(pos[0].c_str(), "kernel_iterations", kernel_iterations)) {
        return EXIT_FAILURE;
    }
    if (pos.size() >= 2 && !parse_cli_int(pos[1].c_str(), "instances", instances)) {
        return EXIT_FAILURE;
    }
    if (pos.size() >= 3 && !parse_cli_int(pos[2].c_str(), "launch_repeats", launch_repeats)) {
        return EXIT_FAILURE;
    }
    if (device_index >= 0) {
        const std::string dev = std::to_string(device_index);
#ifdef _WIN32
        _putenv_s("CGBN_OPENCL_DEVICE_INDEX", dev.c_str());
#else
        setenv("CGBN_OPENCL_DEVICE_INDEX", dev.c_str(), 1);
#endif
        std::cout << "OpenCL device override: CGBN_OPENCL_DEVICE_INDEX=" << dev << std::endl;
    }
    bool ok = runOpenClEcmMontSqrBenchmark(bits, kernel_iterations, instances, launch_repeats,
                                          use_wg, tpi);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
#endif