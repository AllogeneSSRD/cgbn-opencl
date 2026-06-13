#include "cgbn_opencl.h"
#include "opencl_ecm_addsub_manifest.h"

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
#include <map>
#include <cstdio>
#include <cstring>

namespace {

constexpr uint32_t MAX_BENCH_BITS = 16384;
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

bool run_kernel(cl_command_queue q, cl_kernel k, size_t global, int total_enqueues, double &ms) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < total_enqueues; ++i) {
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

bool runOpenClEcmAddSubBenchmark(int bits, int kernel_iterations, int instances, int launch_repeats,
                                 bool bench_unroll_only) {
    if (bits <= 0 || (bits % 32) != 0 || (uint32_t)bits > MAX_BENCH_BITS) {
        std::cerr << "bits must be a positive multiple of 32 and <= " << MAX_BENCH_BITS
                  << std::endl;
        return false;
    }
    const uint32_t BITS = (uint32_t)bits;
    const uint32_t WORDS = BITS / 32;

    std::cout << "ECM add/sub microbench: " << BITS
              << "-bit, kernel_iterations=" << kernel_iterations
              << ", instances=" << instances
              << ", launch_repeats=" << launch_repeats
              << ", unroll_only=" << (bench_unroll_only ? "1" : "0") << std::endl;

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

    std::string bench_src = cgbn::opencl::load_kernel_file("bench/ecm_addsub_bench.cl");
    if (bench_src.empty()) {
        std::cerr << "Failed to load ecm_addsub_bench.cl" << std::endl;
        return false;
    }
    bool asm_enabled = false;
    bool asm_b64_enabled = false;
    if (const char *asm_disable = std::getenv("ECM_ADDSUB_ASM_DISABLE")) {
        if (*asm_disable == '1') {
            std::cout << "ECM_ADDSUB_ASM_DISABLE=1: skipping AMD asm kernels\n";
        }
    } else if (WORDS == 8u || WORDS == 16u || WORDS == 128u) {
        if (const char *asm_b64 = std::getenv("ECM_ADDSUB_ASM_B64")) {
            asm_b64_enabled = (*asm_b64 == '1');
        }
        asm_enabled = true;
    }
    EcmAddSubBuildManifest build =
        opencl_ecm_addsub_build_manifest(WORDS, asm_enabled, asm_b64_enabled);
    std::string src = bench_src;
    for (const std::string &rel : build.source_paths) {
        if (rel.find("ecm_addsub_bench.cl") != std::string::npos) {
            continue;
        }
        std::string part = cgbn::opencl::load_kernel_file(rel.c_str());
        if (part.empty()) {
            std::cerr << "Warning: missing " << rel << " (run python tools/mp_addsub/gen_all.py)\n";
            continue;
        }
        src += "\n" + part;
    }
    cl_int buildErr = CL_SUCCESS;
    int fused_unroll = 2;
    if (const char *v = std::getenv("ECM_MP_ADD_MOD_FUSED_UNROLL")) {
        fused_unroll = std::atoi(v);
        if (fused_unroll != 1 && fused_unroll != 2) {
            std::cerr << "Warning: invalid ECM_MP_ADD_MOD_FUSED_UNROLL=" << fused_unroll
                      << ", fallback to 2\n";
            fused_unroll = 2;
        }
    }
    char build_opts[256];
    if (asm_enabled) {
        if (asm_b64_enabled) {
            snprintf(build_opts, sizeof(build_opts),
                     "-DMAX_LIMBS=%u -DMP_ADD_MOD_FUSED_UNROLL=%d -DMP_ADDMOD_ASM_ENABLE=1 "
                     "-DMP_ADDMOD_ASM_B64=1",
                     WORDS, fused_unroll);
        } else {
            snprintf(build_opts, sizeof(build_opts),
                     "-DMAX_LIMBS=%u -DMP_ADD_MOD_FUSED_UNROLL=%d -DMP_ADDMOD_ASM_ENABLE=1", WORDS,
                     fused_unroll);
        }
    } else {
        snprintf(build_opts, sizeof(build_opts), "-DMAX_LIMBS=%u -DMP_ADD_MOD_FUSED_UNROLL=%d", WORDS,
                 fused_unroll);
    }
    std::cout << "addsub build: fused_unroll=" << fused_unroll
              << " asm=" << (asm_enabled ? "1" : "0")
              << " asm_b64=" << (asm_b64_enabled ? "1" : "0")
              << " src_kib=" << (src.size() / 1024u) << std::endl;
    std::cout << "OpenCL compiling"
              << (asm_enabled && !asm_b64_enabled
                      ? " (b64 asm off; set ECM_ADDSUB_ASM_B64=1 to enable)"
                      : "")
              << "..."
              << std::flush;
    const auto compile_t0 = std::chrono::steady_clock::now();
    cl_program program = cgbn::opencl::build_program_from_source(
        ctx, src.c_str(), build_opts, buildErr);
    const auto compile_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - compile_t0)
                                .count();
    std::cout << " done in " << compile_ms << " ms" << std::endl;
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

    cl_uint limbs = WORDS;
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

    auto run_pure = [&](const char *kname, bool needsN, double &ms_out) -> bool {
        cl_int kerr = CL_SUCCESS;
        cl_kernel k = clCreateKernel(program, kname, &kerr);
        if (kerr != CL_SUCCESS) {
            std::cerr << "Create kernel " << kname << " failed: " << kerr << std::endl;
            return false;
        }
        clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
        if (needsN) {
            clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
            clSetKernelArg(k, 2, sizeof(cl_mem), &bufN);
            clSetKernelArg(k, 3, sizeof(cl_mem), &bufOut);
            clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
        } else if (std::string(kname) == "ecm_mp_sub_n") {
            clSetKernelArg(k, 1, sizeof(cl_mem), &bufN);
            clSetKernelArg(k, 2, sizeof(cl_mem), &bufOut);
            clSetKernelArg(k, 3, sizeof(cl_uint), &limbs);
        } else {
            clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
            clSetKernelArg(k, 2, sizeof(cl_mem), &bufOut);
            clSetKernelArg(k, 3, sizeof(cl_uint), &limbs);
        }
        const int total_enqueues = launch_repeats * kernel_iterations;
        bool ok = run_kernel(ctx.queue, k, global, total_enqueues, ms_out);
        if (ok) {
            size_t priv_b = 0, loc_b = 0, pref = 0, wg = 0;
            query_kernel_resources(k, ctx.device, priv_b, loc_b, pref, wg);
            double op_count = (double)instances * (double)total_enqueues;
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

    auto run_pure_wg = [&](const char *kname, bool needsN, size_t local_size, double &ms_out) -> bool {
        cl_int kerr = CL_SUCCESS;
        cl_kernel k = clCreateKernel(program, kname, &kerr);
        if (kerr != CL_SUCCESS) {
            std::cerr << "Create kernel " << kname << " failed: " << kerr << std::endl;
            return false;
        }
        clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
        clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
        clSetKernelArg(k, 2, sizeof(cl_mem), &bufN);
        clSetKernelArg(k, 3, sizeof(cl_mem), &bufOut);
        clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
        const int total_enqueues = launch_repeats * kernel_iterations;
        size_t global_wg = (size_t)instances * local_size;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < total_enqueues; ++i) {
            cl_int err2 =
                clEnqueueNDRangeKernel(ctx.queue, k, 1, nullptr, &global_wg, &local_size, 0, nullptr, nullptr);
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
        double op_count_wg = (double)instances * (double)total_enqueues;
        double ops_s = op_count_wg / (ms_out / 1000.0);
        std::cout << "  [" << kname << "] wg=" << local_size << " private_mem=" << priv_b
                  << "B local_mem=" << loc_b << "B pref_wg=" << pref << " max_wg=" << wg << std::endl;
        if (csv_enabled) {
            csv << kname << "," << ms_out << "," << ops_s << "," << priv_b << "," << loc_b << "," << pref
                << "," << wg << "\n";
        }
        clReleaseKernel(k);
        return true;
    };

    auto verify_add_mod_kernels = [&]() -> bool {
        auto run_once = [&](const char *kname) -> bool {
            cl_int kerr = CL_SUCCESS;
            cl_kernel k = clCreateKernel(program, kname, &kerr);
            if (kerr != CL_SUCCESS) {
                std::cerr << "Create verify kernel " << kname << " failed: " << kerr << std::endl;
                return false;
            }
            clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
            clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
            clSetKernelArg(k, 2, sizeof(cl_mem), &bufN);
            clSetKernelArg(k, 3, sizeof(cl_mem), &bufOut);
            clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
            size_t g = 1u;
            cl_int err2 = clEnqueueNDRangeKernel(ctx.queue, k, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
            clFinish(ctx.queue);
            clReleaseKernel(k);
            return err2 == CL_SUCCESS;
        };

        bool ok_legacy = true;
        bool ok_mask = true;
        cl_int err2 = CL_SUCCESS;
        std::vector<uint32_t> out_legacy(WORDS);
        std::vector<uint32_t> out_mask(WORDS);
        if (!bench_unroll_only) {
            if (!run_once("ecm_mp_add_mod_legacy")) return false;
            err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                       out_legacy.data(), 0, nullptr, nullptr);
            if (err2 != CL_SUCCESS) return false;

            if (!run_once("ecm_mp_add_mod_mask")) return false;
            err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                       out_mask.data(), 0, nullptr, nullptr);
            if (err2 != CL_SUCCESS) return false;
        }

        if (!run_once("ecm_mp_add_mod_fused")) return false;
        std::vector<uint32_t> out_fused(WORDS);
        err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS, out_fused.data(),
                                  0, nullptr, nullptr);
        if (err2 != CL_SUCCESS) return false;

        bool ok_unroll = true;
        bool have_unroll = false;
        bool have_unroll_stage1 = false;
        {
            cl_int kerr = CL_SUCCESS;
            cl_kernel ku = clCreateKernel(program, "ecm_mp_add_mod_fused_unroll", &kerr);
            if (kerr == CL_SUCCESS) {
                have_unroll = true;
                clSetKernelArg(ku, 0, sizeof(cl_mem), &bufA);
                clSetKernelArg(ku, 1, sizeof(cl_mem), &bufB);
                clSetKernelArg(ku, 2, sizeof(cl_mem), &bufN);
                clSetKernelArg(ku, 3, sizeof(cl_mem), &bufOut);
                clSetKernelArg(ku, 4, sizeof(cl_uint), &limbs);
                size_t g = 1u;
                err2 = clEnqueueNDRangeKernel(ctx.queue, ku, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
                clFinish(ctx.queue);
                clReleaseKernel(ku);
                if (err2 != CL_SUCCESS) return false;
            }
            cl_kernel ks = clCreateKernel(program, "ecm_mp_add_mod_fused_unroll_auto", &kerr);
            if (kerr == CL_SUCCESS) {
                have_unroll_stage1 = true;
                clSetKernelArg(ks, 0, sizeof(cl_mem), &bufA);
                clSetKernelArg(ks, 1, sizeof(cl_mem), &bufB);
                clSetKernelArg(ks, 2, sizeof(cl_mem), &bufN);
                clSetKernelArg(ks, 3, sizeof(cl_mem), &bufOut);
                clSetKernelArg(ks, 4, sizeof(cl_uint), &limbs);
                size_t g = 1u;
                err2 = clEnqueueNDRangeKernel(ctx.queue, ks, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
                clFinish(ctx.queue);
                clReleaseKernel(ks);
                if (err2 != CL_SUCCESS) return false;
            }
        }
        mpz_t expect, got_fused, got_unroll;
        mpz_t got_unroll_stage1;
        mpz_t got_legacy, got_mask;
        mpz_init(expect);
        mpz_init(got_fused);
        mpz_init(got_unroll);
        mpz_init(got_unroll_stage1);
        if (!bench_unroll_only) {
            mpz_init(got_legacy);
            mpz_init(got_mask);
        }
        mpz_add(expect, a_gmp, b_gmp);
        mpz_mod(expect, expect, n_gmp);

        auto verify_wg = [&](const char *kname, size_t local) -> bool {
            cl_int kerr = CL_SUCCESS;
            cl_kernel k = clCreateKernel(program, kname, &kerr);
            if (kerr != CL_SUCCESS) return true;
            clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
            clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
            clSetKernelArg(k, 2, sizeof(cl_mem), &bufN);
            clSetKernelArg(k, 3, sizeof(cl_mem), &bufOut);
            clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
            size_t g = local;
            err2 = clEnqueueNDRangeKernel(ctx.queue, k, 1, nullptr, &g, &local, 0, nullptr, nullptr);
            clFinish(ctx.queue);
            clReleaseKernel(k);
            if (err2 != CL_SUCCESS) return false;
            std::vector<uint32_t> out_w(WORDS);
            err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS, out_w.data(),
                                       0, nullptr, nullptr);
            if (err2 != CL_SUCCESS) return false;
            mpz_t got;
            mpz_init(got);
            fill_to_gmp(out_w.data(), WORDS, got);
            bool ok = (mpz_cmp(expect, got) == 0);
            if (!ok) {
                std::cerr << "add_mod verify: " << kname << " FAIL" << std::endl;
            }
            mpz_clear(got);
            return ok;
        };
        bool ok_lpt_all = true;
        bool ok_asm = true;
        if (asm_enabled) {
            std::vector<const char *> asm_kernels = {
                "ecm_mp_add_mod_fused_unroll_asm",
                "ecm_mp_add_mod_fused_unroll_asm_b16",
                "ecm_mp_add_mod_fused_unroll_asm_asmfix",
                "ecm_mp_add_mod_fused_unroll_asm_soft",
                "ecm_mp_add_mod_fused_unroll_asm_soft_b16",
            };
            if (WORDS == 128u) {
                asm_kernels.push_back("ecm_mp_add_mod_fused_unroll_asm_b32");
                if (asm_b64_enabled) {
                    asm_kernels.push_back("ecm_mp_add_mod_fused_unroll_asm_b64");
                }
            }
            if (!bench_unroll_only) {
                asm_kernels.push_back("ecm_mp_add_mod_fused_asm_b16");
                asm_kernels.push_back("ecm_mp_add_mod_fused_asm_b16_vccsoft");
                asm_kernels.push_back("ecm_mp_add_mod_fused_asm8");
                asm_kernels.push_back("ecm_mp_add_mod_fused_asm8_asmfix");
                asm_kernels.push_back("ecm_mp_add_mod_fused_asm8_vccsoft");
            }
            for (const char *ak : asm_kernels) {
                cl_int kerr = CL_SUCCESS;
                cl_kernel kt = clCreateKernel(program, ak, &kerr);
                if (kerr != CL_SUCCESS) continue;
                clReleaseKernel(kt);
                if (!run_once(ak)) continue;
                std::vector<uint32_t> out_asm(WORDS);
                err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                           out_asm.data(), 0, nullptr, nullptr);
                if (err2 != CL_SUCCESS) return false;
                mpz_t got_asm;
                mpz_init(got_asm);
                fill_to_gmp(out_asm.data(), WORDS, got_asm);
                if (mpz_cmp(expect, got_asm) != 0) {
                    std::cerr << "add_mod verify: " << ak << " FAIL" << std::endl;
                    ok_asm = false;
                }
                mpz_clear(got_asm);
            }
        }
        const int lpt_chunks[] = {16, 32, 48, 64};
        if (have_unroll) {
            for (int chunk : lpt_chunks) {
                if (WORDS % (uint32_t)chunk != 0u) continue;
                if (WORDS / (uint32_t)chunk <= 1u) continue;
                char kname[64];
                std::snprintf(kname, sizeof(kname), "ecm_mp_add_mod_fused_lpt%d", chunk);
                if (!verify_wg(kname, (size_t)(WORDS / (uint32_t)chunk))) {
                    ok_lpt_all = false;
                }
            }
        }

        fill_to_gmp(out_fused.data(), WORDS, got_fused);
        if (!bench_unroll_only) {
            fill_to_gmp(out_legacy.data(), WORDS, got_legacy);
            fill_to_gmp(out_mask.data(), WORDS, got_mask);
            ok_legacy = (mpz_cmp(expect, got_legacy) == 0);
            ok_mask = (mpz_cmp(expect, got_mask) == 0);
        }
        bool ok_unroll_stage1 = true;
        if (have_unroll) {
            if (!run_once("ecm_mp_add_mod_fused_unroll")) return false;
            std::vector<uint32_t> out_unroll(WORDS);
            err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                       out_unroll.data(), 0, nullptr, nullptr);
            if (err2 != CL_SUCCESS) return false;
            fill_to_gmp(out_unroll.data(), WORDS, got_unroll);
            ok_unroll = (mpz_cmp(expect, got_unroll) == 0);
        }
        if (have_unroll_stage1) {
            if (!run_once("ecm_mp_add_mod_fused_unroll_auto")) return false;
            std::vector<uint32_t> out_s(WORDS);
            err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                       out_s.data(), 0, nullptr, nullptr);
            if (err2 != CL_SUCCESS) return false;
            fill_to_gmp(out_s.data(), WORDS, got_unroll_stage1);
            ok_unroll_stage1 = (mpz_cmp(expect, got_unroll_stage1) == 0);
        }
        bool ok_fused = (mpz_cmp(expect, got_fused) == 0);
        if ((!bench_unroll_only && (!ok_legacy || !ok_mask)) || !ok_fused || !ok_unroll ||
            !ok_unroll_stage1 || !ok_lpt_all || !ok_asm) {
            std::cerr << "add_mod verify:";
            if (!bench_unroll_only) {
                std::cerr << " legacy=" << (ok_legacy ? "PASS" : "FAIL")
                          << " mask=" << (ok_mask ? "PASS" : "FAIL");
            }
            std::cerr << " fused=" << (ok_fused ? "PASS" : "FAIL")
                      << " unroll=" << (ok_unroll ? "PASS" : "FAIL")
                      << " unroll_stage1=" << (ok_unroll_stage1 ? "PASS" : "FAIL")
                      << " lpt=" << (ok_lpt_all ? "PASS" : "FAIL")
                      << " asm=" << (ok_asm ? "PASS" : "FAIL") << std::endl;
            mpz_clears(expect, got_unroll_stage1, nullptr);
            if (!bench_unroll_only) {
                mpz_clears(got_legacy, got_mask, got_fused, got_unroll, nullptr);
            } else {
                mpz_clears(got_fused, got_unroll, nullptr);
            }
            return false;
        }
        std::cout << "  [ecm_mp_add_mod] GMP verify: PASS (";
        if (!bench_unroll_only) {
            std::cout << "legacy, mask, ";
        }
        std::cout << "fused";
        if (have_unroll) {
            std::cout << ", fused_unroll";
        }
        if (have_unroll_stage1) {
            std::cout << ", fused_unroll_auto";
        }
        if (have_unroll) {
            std::cout << ", lpt{16,32,48,64}";
        }
        if (asm_enabled) {
            std::cout << ", unroll_asm";
            if (WORDS == 128u) {
                std::cout << "+b32";
                if (asm_b64_enabled) {
                    std::cout << "+b64";
                }
            }
        }
        std::cout << ")" << std::endl;
        mpz_clears(expect, got_unroll_stage1, nullptr);
        if (!bench_unroll_only) {
            mpz_clears(got_legacy, got_mask, got_fused, got_unroll, nullptr);
        } else {
            mpz_clears(got_fused, got_unroll, nullptr);
        }
        return true;
    };

    auto verify_sub_mod_kernels = [&]() -> bool {
        auto run_once = [&](const char *kname) -> bool {
            cl_int kerr = CL_SUCCESS;
            cl_kernel k = clCreateKernel(program, kname, &kerr);
            if (kerr != CL_SUCCESS) {
                std::cerr << "Create verify kernel " << kname << " failed: " << kerr << std::endl;
                return false;
            }
            clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
            clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
            clSetKernelArg(k, 2, sizeof(cl_mem), &bufN);
            clSetKernelArg(k, 3, sizeof(cl_mem), &bufOut);
            clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
            size_t g = 1u;
            cl_int err2 =
                clEnqueueNDRangeKernel(ctx.queue, k, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
            clFinish(ctx.queue);
            clReleaseKernel(k);
            return err2 == CL_SUCCESS;
        };

        if (!run_once("ecm_mp_sub_mod")) return false;
        std::vector<uint32_t> out_base(WORDS);
        cl_int err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                          out_base.data(), 0, nullptr, nullptr);
        if (err2 != CL_SUCCESS) return false;

        bool ok_unroll = true;
        bool have_unroll = false;
        bool have_unroll_stage1 = false;
        {
            cl_int kerr = CL_SUCCESS;
            cl_kernel ku = clCreateKernel(program, "ecm_mp_sub_mod_fused_unroll", &kerr);
            if (kerr == CL_SUCCESS) {
                have_unroll = true;
                clSetKernelArg(ku, 0, sizeof(cl_mem), &bufA);
                clSetKernelArg(ku, 1, sizeof(cl_mem), &bufB);
                clSetKernelArg(ku, 2, sizeof(cl_mem), &bufN);
                clSetKernelArg(ku, 3, sizeof(cl_mem), &bufOut);
                clSetKernelArg(ku, 4, sizeof(cl_uint), &limbs);
                size_t g = 1u;
                err2 = clEnqueueNDRangeKernel(ctx.queue, ku, 1, nullptr, &g, nullptr, 0, nullptr,
                                              nullptr);
                clFinish(ctx.queue);
                clReleaseKernel(ku);
                if (err2 != CL_SUCCESS) return false;
            }
            cl_kernel ks = clCreateKernel(program, "ecm_mp_sub_mod_fused_unroll_auto", &kerr);
            if (kerr == CL_SUCCESS) {
                have_unroll_stage1 = true;
                clSetKernelArg(ks, 0, sizeof(cl_mem), &bufA);
                clSetKernelArg(ks, 1, sizeof(cl_mem), &bufB);
                clSetKernelArg(ks, 2, sizeof(cl_mem), &bufN);
                clSetKernelArg(ks, 3, sizeof(cl_mem), &bufOut);
                clSetKernelArg(ks, 4, sizeof(cl_uint), &limbs);
                size_t g = 1u;
                err2 = clEnqueueNDRangeKernel(ctx.queue, ks, 1, nullptr, &g, nullptr, 0, nullptr,
                                              nullptr);
                clFinish(ctx.queue);
                clReleaseKernel(ks);
                if (err2 != CL_SUCCESS) return false;
            }
        }

        mpz_t expect, got_base, got_unroll;
        mpz_t got_unroll_stage1;
        mpz_init(expect);
        mpz_init(got_base);
        mpz_init(got_unroll);
        mpz_init(got_unroll_stage1);
        mpz_sub(expect, a_gmp, b_gmp);
        mpz_mod(expect, expect, n_gmp);

        fill_to_gmp(out_base.data(), WORDS, got_base);
        bool ok_base = (mpz_cmp(expect, got_base) == 0);
        bool ok_unroll_stage1 = true;
        if (have_unroll) {
            if (!run_once("ecm_mp_sub_mod_fused_unroll")) return false;
            std::vector<uint32_t> out_unroll(WORDS);
            err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                       out_unroll.data(), 0, nullptr, nullptr);
            if (err2 != CL_SUCCESS) return false;
            fill_to_gmp(out_unroll.data(), WORDS, got_unroll);
            ok_unroll = (mpz_cmp(expect, got_unroll) == 0);
        }
        if (have_unroll_stage1) {
            if (!run_once("ecm_mp_sub_mod_fused_unroll_auto")) return false;
            std::vector<uint32_t> out_s(WORDS);
            err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                       out_s.data(), 0, nullptr, nullptr);
            if (err2 != CL_SUCCESS) return false;
            fill_to_gmp(out_s.data(), WORDS, got_unroll_stage1);
            ok_unroll_stage1 = (mpz_cmp(expect, got_unroll_stage1) == 0);
        }
        bool ok_asm = true;
        if (asm_enabled) {
            std::vector<const char *> asm_kernels;
            if (WORDS == 128u) {
                asm_kernels = {
                    "ecm_mp_sub_mod_fused_unroll_asm_b32",
                };
                if (asm_b64_enabled) {
                    asm_kernels.push_back("ecm_mp_sub_mod_fused_unroll_asm_b64");
                }
            }
            for (const char *ak : asm_kernels) {
                cl_int kerr = CL_SUCCESS;
                cl_kernel kt = clCreateKernel(program, ak, &kerr);
                if (kerr != CL_SUCCESS) continue;
                clReleaseKernel(kt);
                if (!run_once(ak)) continue;
                std::vector<uint32_t> out_asm(WORDS);
                err2 = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                                           out_asm.data(), 0, nullptr, nullptr);
                if (err2 != CL_SUCCESS) return false;
                mpz_t got_asm;
                mpz_init(got_asm);
                fill_to_gmp(out_asm.data(), WORDS, got_asm);
                if (mpz_cmp(expect, got_asm) != 0) {
                    std::cerr << "sub_mod verify: " << ak << " FAIL" << std::endl;
                    ok_asm = false;
                }
                mpz_clear(got_asm);
            }
        }

        if (!ok_base || !ok_unroll || !ok_unroll_stage1 || !ok_asm) {
            std::cerr << "sub_mod verify: base=" << (ok_base ? "PASS" : "FAIL")
                      << " unroll=" << (ok_unroll ? "PASS" : "FAIL")
                      << " unroll_stage1=" << (ok_unroll_stage1 ? "PASS" : "FAIL")
                      << " asm=" << (ok_asm ? "PASS" : "FAIL") << std::endl;
            mpz_clears(expect, got_base, got_unroll, got_unroll_stage1, nullptr);
            return false;
        }
        std::cout << "  [ecm_mp_sub_mod] GMP verify: PASS (base";
        if (have_unroll) {
            std::cout << ", fused_unroll";
        }
        if (have_unroll_stage1) {
            std::cout << ", fused_unroll_auto";
        }
        if (asm_enabled && WORDS == 128u) {
            std::cout << ", unroll_asm_b32";
            if (asm_b64_enabled) {
                std::cout << "+b64";
            }
        }
        std::cout << ")" << std::endl;
        mpz_clears(expect, got_base, got_unroll, got_unroll_stage1, nullptr);
        return true;
    };

    double t_add_n = 0.0, t_sub_n = 0.0, t_add_mod = 0.0, t_add_mod_legacy = 0.0, t_add_mod_mask = 0.0,
           t_add_mod_unroll = 0.0, t_add_mod_unroll_priv = 0.0, t_add_mod_unroll_auto = 0.0,
           t_add_mod_unroll_asm = 0.0,
           t_add_mod_unroll_asm_b16 = 0.0, t_add_mod_unroll_asm_asmfix = 0.0,
           t_add_mod_unroll_asm_soft = 0.0, t_add_mod_unroll_asm_soft_b16 = 0.0,
           t_add_mod_unroll_asm_b32 = 0.0, t_add_mod_unroll_asm_b64 = 0.0,
           t_add_mod_asm_b16 = 0.0, t_add_mod_asm_b16_vccsoft = 0.0, t_add_mod_asm8 = 0.0,
           t_add_mod_asm8_asmfix = 0.0, t_add_mod_asm8_vccsoft = 0.0,
           t_sub_mod = 0.0, t_sub_mod_unroll = 0.0, t_sub_mod_unroll_priv = 0.0,
           t_sub_mod_unroll_auto = 0.0,
           t_sub_mod_unroll_asm_b32 = 0.0, t_sub_mod_unroll_asm_b64 = 0.0;
    std::map<int, double> t_lpt_ms;
    if (!bench_unroll_only) {
        if (!run_pure("ecm_mp_add_n", false, t_add_n)) return false;
        if (!run_pure("ecm_mp_sub_n", false, t_sub_n)) return false;
    }
    if (!verify_add_mod_kernels()) return false;
    if (!verify_sub_mod_kernels()) return false;
    if (!bench_unroll_only) {
        if (!run_pure("ecm_mp_add_mod_legacy", true, t_add_mod_legacy)) return false;
        if (!run_pure("ecm_mp_add_mod_mask", true, t_add_mod_mask)) return false;
    }
    if (!run_pure("ecm_mp_add_mod_fused", true, t_add_mod)) return false;
    {
        cl_int kerr = CL_SUCCESS;
        cl_kernel ku = clCreateKernel(program, "ecm_mp_add_mod_fused_unroll", &kerr);
        if (kerr == CL_SUCCESS) {
            clReleaseKernel(ku);
            if (!run_pure("ecm_mp_add_mod_fused_unroll", true, t_add_mod_unroll)) return false;
        } else {
            std::cout << "mp_add_mod_fused_unroll: (no kernel for MAX_LIMBS=" << WORDS << ")" << std::endl;
        }
        cl_kernel kp = clCreateKernel(program, "ecm_mp_add_mod_fused_unroll_priv", &kerr);
        if (kerr == CL_SUCCESS) {
            clReleaseKernel(kp);
            if (!run_pure("ecm_mp_add_mod_fused_unroll_priv", true, t_add_mod_unroll_priv)) return false;
        }
        {
            auto try_bench_stage1 = [&](const char *kname, double &t_out) {
                cl_int kerr2 = CL_SUCCESS;
                cl_kernel ka = clCreateKernel(program, kname, &kerr2);
                if (kerr2 != CL_SUCCESS) return;
                clReleaseKernel(ka);
                (void)run_pure(kname, true, t_out);
            };
            try_bench_stage1("ecm_mp_add_mod_fused_unroll_auto", t_add_mod_unroll_auto);
        }
        if (asm_enabled) {
            auto try_bench_asm = [&](const char *kname, double &t_out) {
                cl_int kerr = CL_SUCCESS;
                cl_kernel ka = clCreateKernel(program, kname, &kerr);
                if (kerr != CL_SUCCESS) return;
                clReleaseKernel(ka);
                (void)run_pure(kname, true, t_out);
            };
            try_bench_asm("ecm_mp_add_mod_fused_unroll_asm", t_add_mod_unroll_asm);
            try_bench_asm("ecm_mp_add_mod_fused_unroll_asm_b16", t_add_mod_unroll_asm_b16);
            try_bench_asm("ecm_mp_add_mod_fused_unroll_asm_asmfix", t_add_mod_unroll_asm_asmfix);
            try_bench_asm("ecm_mp_add_mod_fused_unroll_asm_soft", t_add_mod_unroll_asm_soft);
            try_bench_asm("ecm_mp_add_mod_fused_unroll_asm_soft_b16", t_add_mod_unroll_asm_soft_b16);
            if (WORDS == 128u) {
                try_bench_asm("ecm_mp_add_mod_fused_unroll_asm_b32", t_add_mod_unroll_asm_b32);
                if (asm_b64_enabled) {
                    try_bench_asm("ecm_mp_add_mod_fused_unroll_asm_b64", t_add_mod_unroll_asm_b64);
                }
            }
            if (!bench_unroll_only) {
                try_bench_asm("ecm_mp_add_mod_fused_asm_b16", t_add_mod_asm_b16);
                try_bench_asm("ecm_mp_add_mod_fused_asm_b16_vccsoft", t_add_mod_asm_b16_vccsoft);
                try_bench_asm("ecm_mp_add_mod_fused_asm8", t_add_mod_asm8);
                try_bench_asm("ecm_mp_add_mod_fused_asm8_asmfix", t_add_mod_asm8_asmfix);
                try_bench_asm("ecm_mp_add_mod_fused_asm8_vccsoft", t_add_mod_asm8_vccsoft);
            }
        }
        const int lpt_chunks[] = {16, 32, 48, 64};
        for (int chunk : lpt_chunks) {
            if (WORDS % (uint32_t)chunk != 0u) continue;
            uint32_t threads = WORDS / (uint32_t)chunk;
            if (threads <= 1u) continue;
            char kname[64];
            std::snprintf(kname, sizeof(kname), "ecm_mp_add_mod_fused_lpt%d", chunk);
            cl_kernel kl = clCreateKernel(program, kname, &kerr);
            if (kerr != CL_SUCCESS) continue;
            clReleaseKernel(kl);
            double t_lpt = 0.0;
            if (!run_pure_wg(kname, true, (size_t)threads, t_lpt)) return false;
            t_lpt_ms[chunk] = t_lpt;
        }
    }
    if (!run_pure("ecm_mp_sub_mod", true, t_sub_mod)) return false;
    {
        cl_int kerr = CL_SUCCESS;
        cl_kernel ku = clCreateKernel(program, "ecm_mp_sub_mod_fused_unroll", &kerr);
        if (kerr == CL_SUCCESS) {
            clReleaseKernel(ku);
            if (!run_pure("ecm_mp_sub_mod_fused_unroll", true, t_sub_mod_unroll)) return false;
        } else {
            std::cout << "mp_sub_mod_fused_unroll: (no kernel for MAX_LIMBS=" << WORDS << ")"
                      << std::endl;
        }
        cl_kernel kp = clCreateKernel(program, "ecm_mp_sub_mod_fused_unroll_priv", &kerr);
        if (kerr == CL_SUCCESS) {
            clReleaseKernel(kp);
            if (!run_pure("ecm_mp_sub_mod_fused_unroll_priv", true, t_sub_mod_unroll_priv))
                return false;
        }
        {
            auto try_bench_sub_stage1 = [&](const char *kname, double &t_out) {
                cl_int kerr2 = CL_SUCCESS;
                cl_kernel ka = clCreateKernel(program, kname, &kerr2);
                if (kerr2 != CL_SUCCESS) return;
                clReleaseKernel(ka);
                (void)run_pure(kname, true, t_out);
            };
            try_bench_sub_stage1("ecm_mp_sub_mod_fused_unroll_auto", t_sub_mod_unroll_auto);
        }
        if (asm_enabled && WORDS == 128u) {
            auto try_bench_sub_asm = [&](const char *kname, double &t_out) {
                cl_int kerr2 = CL_SUCCESS;
                cl_kernel ka = clCreateKernel(program, kname, &kerr2);
                if (kerr2 != CL_SUCCESS) return;
                clReleaseKernel(ka);
                (void)run_pure(kname, true, t_out);
            };
            try_bench_sub_asm("ecm_mp_sub_mod_fused_unroll_asm_b32", t_sub_mod_unroll_asm_b32);
            if (asm_b64_enabled) {
                try_bench_sub_asm("ecm_mp_sub_mod_fused_unroll_asm_b64", t_sub_mod_unroll_asm_b64);
            }
        }
    }

    err = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, sizeof(uint32_t) * WORDS,
                              host_out.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Read back failed: " << err << std::endl;
        return false;
    }

    double op_count = (double)instances * (double)kernel_iterations * (double)launch_repeats;
    {
        struct Row {
            std::string path;
            double ms;
        };
        std::vector<Row> rows;
        auto push = [&](const char *path, double ms) {
            if (ms > 0.0) {
                rows.push_back({path, ms});
            }
        };
        auto push_lpt = [&](int chunk, double ms) {
            if (ms > 0.0) {
                rows.push_back({"fused_lpt" + std::to_string(chunk), ms});
            }
        };
        if (asm_enabled) {
            push("fused_unroll_asm_b64", t_add_mod_unroll_asm_b64);
            push("fused_unroll_asm_b32", t_add_mod_unroll_asm_b32);
            push("fused_unroll_asm_soft_b16", t_add_mod_unroll_asm_soft_b16);
            push("fused_unroll_asm_b16", t_add_mod_unroll_asm_b16);
            push("fused_unroll_asm_soft_b8", t_add_mod_unroll_asm_soft);
            push("fused_unroll_asm_asmfix_b8", t_add_mod_unroll_asm_asmfix);
            push("fused_unroll_asm_b8", t_add_mod_unroll_asm);
            if (!bench_unroll_only) {
                push("fused_asm_b16_vccsoft", t_add_mod_asm_b16_vccsoft);
                push("fused_asm_b16", t_add_mod_asm_b16);
            }
        }
        const int lpt_chunks[] = {64, 48, 32, 16};
        for (int chunk : lpt_chunks) {
            auto it = t_lpt_ms.find(chunk);
            if (it != t_lpt_ms.end()) {
                push_lpt(chunk, it->second);
            }
        }
        push("fused_unroll", t_add_mod_unroll);
        push("fused_unroll_priv", t_add_mod_unroll_priv);
        push("fused_unroll_auto", t_add_mod_unroll_auto);
        if (!bench_unroll_only) {
            push("fused", t_add_mod);
            push("mask", t_add_mod_mask);
            push("legacy", t_add_mod_legacy);
        }
        std::cout << "\n--- mp_add_mod (priority high -> low) ---\n";
        for (size_t i = 0; i < rows.size(); ++i) {
            const double ops = op_count / (rows[i].ms / 1000.0);
            std::cout << "  [" << (i + 1) << "] " << rows[i].path << ": " << rows[i].ms << " ms, "
                      << ops << " ops/s";
            if (i + 1 < rows.size()) {
                std::cout << " (" << (rows[i + 1].ms / rows[i].ms) << "x vs next tier)";
            } else if (t_add_mod > 0.0 && rows[i].path != "fused") {
                std::cout << " (" << (t_add_mod / rows[i].ms) << "x vs fused)";
            }
            std::cout << std::endl;
        }
        auto print_sub = [&](const char *path, double ms) {
            if (ms > 0.0) {
                std::cout << "  " << path << ": " << ms << " ms, "
                          << (op_count / (ms / 1000.0)) << " ops/s\n";
            }
        };
        std::cout << "--- mp_sub_mod (priority high -> low) ---\n";
        if (asm_enabled && WORDS == 128u) {
            print_sub("fused_unroll_asm_b64", t_sub_mod_unroll_asm_b64);
            print_sub("fused_unroll_asm_b32", t_sub_mod_unroll_asm_b32);
        }
        print_sub("fused_unroll", t_sub_mod_unroll);
        print_sub("fused_unroll_priv", t_sub_mod_unroll_priv);
        print_sub("fused_unroll_auto", t_sub_mod_unroll_auto);
        print_sub("fused_loop", t_sub_mod);
        std::cout << std::endl;
    }
    if (!bench_unroll_only) {
        std::cout << "mp_add_n:   " << t_add_n << " ms, " << (op_count / (t_add_n / 1000.0)) << " ops/s"
                  << std::endl;
        std::cout << "mp_sub_n:   " << t_sub_n << " ms, " << (op_count / (t_sub_n / 1000.0)) << " ops/s"
                  << std::endl;
        std::cout << "mp_add_mod_legacy: " << t_add_mod_legacy << " ms, "
                  << (op_count / (t_add_mod_legacy / 1000.0)) << " ops/s" << std::endl;
        std::cout << "mp_add_mod_mask:   " << t_add_mod_mask << " ms, "
                  << (op_count / (t_add_mod_mask / 1000.0)) << " ops/s"
                  << " (vs legacy: " << (t_add_mod_legacy / t_add_mod_mask) << "x)" << std::endl;
    }
    std::cout << "mp_add_mod_fused:  " << t_add_mod << " ms, " << (op_count / (t_add_mod / 1000.0))
              << " ops/s";
    if (!bench_unroll_only) {
        std::cout << " (vs legacy: " << (t_add_mod_legacy / t_add_mod) << "x)";
    }
    std::cout << std::endl;
    if (t_add_mod_unroll > 0.0) {
        std::cout << "mp_add_mod_fused_unroll:               " << t_add_mod_unroll << " ms, "
                  << (op_count / (t_add_mod_unroll / 1000.0)) << " ops/s (vs fused: "
                  << (t_add_mod / t_add_mod_unroll) << "x";
        if (!bench_unroll_only) {
            std::cout << ", vs legacy: " << (t_add_mod_legacy / t_add_mod_unroll) << "x";
        }
        std::cout << ")" << std::endl;
    }
    if (t_add_mod_unroll_priv > 0.0) {
        std::cout << "mp_add_mod_fused_unroll_priv:          " << t_add_mod_unroll_priv << " ms, "
                  << (op_count / (t_add_mod_unroll_priv / 1000.0)) << " ops/s (vs fused: "
                  << (t_add_mod / t_add_mod_unroll_priv) << "x, vs unroll: "
                  << (t_add_mod_unroll / t_add_mod_unroll_priv) << "x)" << std::endl;
    }
    if (t_add_mod_unroll_auto > 0.0) {
        std::cout << "mp_add_mod_fused_unroll_auto:        " << t_add_mod_unroll_auto << " ms, "
                  << (op_count / (t_add_mod_unroll_auto / 1000.0)) << " ops/s (ECM fused_unroll, "
                  << "vs fused: " << (t_add_mod / t_add_mod_unroll_auto) << "x";
        if (t_add_mod_unroll > 0.0) {
            std::cout << ", vs scalar unroll: " << (t_add_mod_unroll / t_add_mod_unroll_auto) << "x";
        }
        std::cout << ")" << std::endl;
    }
    if (t_add_mod_unroll_asm > 0.0) {
        std::cout << "mp_add_mod_fused_unroll_asm_b8:        " << t_add_mod_unroll_asm << " ms, "
                  << (op_count / (t_add_mod_unroll_asm / 1000.0)) << " ops/s (vs fused: "
                  << (t_add_mod / t_add_mod_unroll_asm) << "x, vs unroll: "
                  << (t_add_mod_unroll / t_add_mod_unroll_asm) << "x)" << std::endl;
    }
    if (t_add_mod_unroll_asm_asmfix > 0.0) {
        std::cout << "mp_add_mod_fused_unroll_asm_fix_b8:    "
                  << t_add_mod_unroll_asm_asmfix << " ms, "
                  << (op_count / (t_add_mod_unroll_asm_asmfix / 1000.0)) << " ops/s (vs c fix: "
                  << (t_add_mod_unroll_asm / t_add_mod_unroll_asm_asmfix) << "x, vs unroll: "
                  << (t_add_mod_unroll / t_add_mod_unroll_asm_asmfix) << "x)" << std::endl;
    }
    if (t_add_mod_unroll_asm_soft > 0.0) {
        std::cout << "mp_add_mod_fused_unroll_asm_soft_b8:   " << t_add_mod_unroll_asm_soft
                  << " ms, " << (op_count / (t_add_mod_unroll_asm_soft / 1000.0)) << " ops/s (vs b8: "
                  << (t_add_mod_unroll_asm / t_add_mod_unroll_asm_soft) << "x, vs unroll: "
                  << (t_add_mod_unroll / t_add_mod_unroll_asm_soft) << "x)" << std::endl;
    }
    if (t_add_mod_unroll_asm_b16 > 0.0) {
        std::cout << "mp_add_mod_fused_unroll_asm_b16:       " << t_add_mod_unroll_asm_b16
                  << " ms, " << (op_count / (t_add_mod_unroll_asm_b16 / 1000.0)) << " ops/s (vs b8: "
                  << (t_add_mod_unroll_asm / t_add_mod_unroll_asm_b16) << "x, vs unroll: "
                  << (t_add_mod_unroll / t_add_mod_unroll_asm_b16) << "x)" << std::endl;
    }
    if (t_add_mod_unroll_asm_soft_b16 > 0.0) {
        std::cout << "mp_add_mod_fused_unroll_asm_soft_b16:  "
                  << t_add_mod_unroll_asm_soft_b16 << " ms, "
                  << (op_count / (t_add_mod_unroll_asm_soft_b16 / 1000.0)) << " ops/s (vs b16: "
                  << (t_add_mod_unroll_asm_b16 / t_add_mod_unroll_asm_soft_b16) << "x, vs soft b8: "
                  << (t_add_mod_unroll_asm_soft / t_add_mod_unroll_asm_soft_b16) << "x, vs unroll: "
                  << (t_add_mod_unroll / t_add_mod_unroll_asm_soft_b16) << "x)" << std::endl;
    }
    if (t_add_mod_unroll_asm_b32 > 0.0) {
        std::cout << "mp_add_mod_fused_unroll_asm_b32:       "
                  << t_add_mod_unroll_asm_b32 << " ms, "
                  << (op_count / (t_add_mod_unroll_asm_b32 / 1000.0)) << " ops/s (vs b16: "
                  << (t_add_mod_unroll_asm_b16 / t_add_mod_unroll_asm_b32) << "x, vs b8: "
                  << (t_add_mod_unroll_asm / t_add_mod_unroll_asm_b32) << "x, vs unroll: "
                  << (t_add_mod_unroll / t_add_mod_unroll_asm_b32) << "x)" << std::endl;
    }
    if (t_add_mod_unroll_asm_b64 > 0.0) {
        std::cout << "mp_add_mod_fused_unroll_asm_b64:       "
                  << t_add_mod_unroll_asm_b64 << " ms, "
                  << (op_count / (t_add_mod_unroll_asm_b64 / 1000.0)) << " ops/s (vs b32: "
                  << (t_add_mod_unroll_asm_b32 / t_add_mod_unroll_asm_b64) << "x, vs b16: "
                  << (t_add_mod_unroll_asm_b16 / t_add_mod_unroll_asm_b64) << "x, vs unroll: "
                  << (t_add_mod_unroll / t_add_mod_unroll_asm_b64) << "x)" << std::endl;
    }
    if (!bench_unroll_only && t_add_mod_asm_b16 > 0.0) {
        std::cout << "mp_add_mod_fused_asm_b16 (512b block16 chain): " << t_add_mod_asm_b16 << " ms, "
                  << (op_count / (t_add_mod_asm_b16 / 1000.0)) << " ops/s" << std::endl;
    }
    if (!bench_unroll_only && t_add_mod_asm_b16_vccsoft > 0.0) {
        std::cout << "mp_add_mod_fused_asm_b16_vccsoft (512b VCC switch): " << t_add_mod_asm_b16_vccsoft
                  << " ms, " << (op_count / (t_add_mod_asm_b16_vccsoft / 1000.0)) << " ops/s (vs b16 chain: "
                  << (t_add_mod_asm_b16 / t_add_mod_asm_b16_vccsoft) << "x)" << std::endl;
    }
    if (!bench_unroll_only && t_add_mod_asm8 > 0.0) {
        std::cout << "mp_add_mod_fused_asm8 (fix=c ulong): " << t_add_mod_asm8 << " ms, "
                  << (op_count / (t_add_mod_asm8 / 1000.0)) << " ops/s (vs unroll: "
                  << (t_add_mod_unroll / t_add_mod_asm8) << "x)" << std::endl;
    }
    if (!bench_unroll_only && t_add_mod_asm8_asmfix > 0.0) {
        std::cout << "mp_add_mod_fused_asm8_asmfix (fix=v_add_co asm): " << t_add_mod_asm8_asmfix
                  << " ms, " << (op_count / (t_add_mod_asm8_asmfix / 1000.0)) << " ops/s (vs c fix: "
                  << (t_add_mod_asm8 / t_add_mod_asm8_asmfix) << "x)" << std::endl;
    }
    if (!bench_unroll_only && t_add_mod_asm8_vccsoft > 0.0) {
        std::cout << "mp_add_mod_fused_asm8_vccsoft (VCC switch): " << t_add_mod_asm8_vccsoft << " ms, "
                  << (op_count / (t_add_mod_asm8_vccsoft / 1000.0)) << " ops/s (vs asm8: "
                  << (t_add_mod_asm8 / t_add_mod_asm8_vccsoft) << "x)" << std::endl;
    }
    if (!t_lpt_ms.empty()) {
        std::cout << "mp_add_mod_fused_lpt (limbs/thread):" << std::endl;
        for (int chunk : {16, 32, 48, 64}) {
            auto it = t_lpt_ms.find(chunk);
            if (it == t_lpt_ms.end()) {
                if (WORDS % (uint32_t)chunk != 0u) {
                    std::cout << "  lpt" << chunk << ": (n/a, " << WORDS << " limbs not divisible by " << chunk
                              << ")" << std::endl;
                }
                continue;
            }
            double ms = it->second;
            uint32_t threads = WORDS / (uint32_t)chunk;
            double ops = op_count / (ms / 1000.0);
            std::cout << "  lpt" << chunk << " (" << threads << " thr): " << ms << " ms, " << ops
                      << " ops/s (vs fused: " << (t_add_mod / ms) << "x";
            if (t_add_mod_unroll > 0.0) {
                std::cout << ", vs unroll: " << (t_add_mod_unroll / ms) << "x";
            }
            std::cout << ")" << std::endl;
        }
    }
    if (!bench_unroll_only) {
        std::cout << "mp_sub_mod: " << t_sub_mod << " ms, " << (op_count / (t_sub_mod / 1000.0))
                  << " ops/s" << std::endl;
    }
    if (t_sub_mod_unroll > 0.0) {
        std::cout << "mp_sub_mod_fused_unroll:               " << t_sub_mod_unroll << " ms, "
                  << (op_count / (t_sub_mod_unroll / 1000.0)) << " ops/s (vs base: "
                  << (t_sub_mod / t_sub_mod_unroll) << "x)" << std::endl;
    }
    if (t_sub_mod_unroll_priv > 0.0) {
        std::cout << "mp_sub_mod_fused_unroll_priv:          " << t_sub_mod_unroll_priv << " ms, "
                  << (op_count / (t_sub_mod_unroll_priv / 1000.0)) << " ops/s (vs base: "
                  << (t_sub_mod / t_sub_mod_unroll_priv) << "x, vs unroll: "
                  << (t_sub_mod_unroll / t_sub_mod_unroll_priv) << "x)" << std::endl;
    }
    if (t_sub_mod_unroll_auto > 0.0) {
        std::cout << "mp_sub_mod_fused_unroll_auto:        " << t_sub_mod_unroll_auto << " ms, "
                  << (op_count / (t_sub_mod_unroll_auto / 1000.0)) << " ops/s (ECM fused_unroll, "
                  << "vs base: " << (t_sub_mod / t_sub_mod_unroll_auto) << "x";
        if (t_sub_mod_unroll > 0.0) {
            std::cout << ", vs scalar unroll: " << (t_sub_mod_unroll / t_sub_mod_unroll_auto) << "x";
        }
        std::cout << ")" << std::endl;
    }
    if (t_sub_mod_unroll_asm_b32 > 0.0) {
        std::cout << "mp_sub_mod_fused_unroll_asm_b32:       " << t_sub_mod_unroll_asm_b32 << " ms, "
                  << (op_count / (t_sub_mod_unroll_asm_b32 / 1000.0)) << " ops/s (vs unroll: "
                  << (t_sub_mod_unroll / t_sub_mod_unroll_asm_b32) << "x";
        if (t_sub_mod_unroll_asm_b64 > 0.0) {
            std::cout << ", vs b64: " << (t_sub_mod_unroll_asm_b64 / t_sub_mod_unroll_asm_b32) << "x";
        }
        std::cout << ")" << std::endl;
    }
    if (t_sub_mod_unroll_asm_b64 > 0.0) {
        std::cout << "mp_sub_mod_fused_unroll_asm_b64:       " << t_sub_mod_unroll_asm_b64 << " ms, "
                  << (op_count / (t_sub_mod_unroll_asm_b64 / 1000.0)) << " ops/s (vs unroll: "
                  << (t_sub_mod_unroll / t_sub_mod_unroll_asm_b64) << "x";
        if (t_sub_mod_unroll_asm_b32 > 0.0) {
            std::cout << ", vs b32: " << (t_sub_mod_unroll_asm_b32 / t_sub_mod_unroll_asm_b64) << "x";
        }
        std::cout << ")" << std::endl;
    }
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
    int bits = 1024;
    int kernel_iterations = 1000;
    int instances = 128;
    int launch_repeats = 10;
    bool bench_unroll_only = false;
    int device_index = -1;
    auto print_usage = [&]() {
        std::cout
            << "Usage: opencl_ecm_addsub [--bits <bits>] [-d|--device <index>] "
               "[--unroll] [kernel_iterations] [instances] [launch_repeats]\n"
            << "  --bits <bits>            Benchmark bit width (multiple of 32, <= 8192)\n"
            << "  --unroll                 Only benchmark fused_unroll / asm unroll / lpt paths\n"
            << "  -d, --device <index>     OpenCL device index (overrides default/env)\n"
            << "  -h, --help               Show this help message\n";
    };
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            print_usage();
            return EXIT_SUCCESS;
        }
        if (a == "--bits" && i + 1 < argc) {
            bits = std::stoi(std::string(argv[++i]));
            continue;
        }
        if (a == "--unroll") {
            bench_unroll_only = true;
            continue;
        }
        if ((a == "-d" || a == "--device") && i + 1 < argc) {
            device_index = std::stoi(std::string(argv[++i]));
            continue;
        }
        pos.push_back(a);
    }
    if (pos.size() >= 1) kernel_iterations = std::stoi(pos[0]);
    if (pos.size() >= 2) instances = std::stoi(pos[1]);
    if (pos.size() >= 3) launch_repeats = std::stoi(pos[2]);
    if (device_index >= 0) {
        const std::string dev = std::to_string(device_index);
#ifdef _WIN32
        _putenv_s("CGBN_OPENCL_DEVICE_INDEX", dev.c_str());
#else
        setenv("CGBN_OPENCL_DEVICE_INDEX", dev.c_str(), 1);
#endif
        std::cout << "OpenCL device override: CGBN_OPENCL_DEVICE_INDEX=" << dev << std::endl;
    }
    bool ok = runOpenClEcmAddSubBenchmark(bits, kernel_iterations, instances, launch_repeats,
                                          bench_unroll_only);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
#endif
