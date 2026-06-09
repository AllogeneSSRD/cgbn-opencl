#include "opencl_runtime.h"

#include "kernel_assets.h"
#include "opencl_ecm_addsub_manifest.h"
#include "opencl_loader.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <vector>

namespace {

constexpr uint32_t kMaxBenchBits = 8192;
constexpr uint32_t kLimb24Mask = (1u << 24) - 1u;

struct Limb24AddBenchKernel {
    const char* kernel_name;
    const char* path_label;
    bool hot_inner_loop;
};

constexpr Limb24AddBenchKernel kLimb24AddKernels[] = {
    {"ecm_mp_add_mod_fused", "fused", false},
    {"ecm_mp_add_mod_fused_u2", "fused_u2", false},
    {"ecm_mp_add_mod_fused_unroll", "fused_unroll", false},
};

constexpr Limb24AddBenchKernel kLimb24AddHotKernels[] = {
    {"ecm_mp_add_mod_fused_hot", "fused_hot", true},
    {"ecm_mp_add_mod_fused_unroll_hot", "fused_unroll_hot", true},
};

std::string program_build_log(OpenCLApi& api, cl_program prog, cl_device_id dev) {
    size_t need = 0;
    if (api.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &need) != CL_SUCCESS ||
        need == 0) {
        return {};
    }
    std::vector<char> buf(need);
    api.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, need, buf.data(), nullptr);
    return std::string(buf.data());
}

void set_pow2_minus_ui_limb24(uint32_t* out, uint32_t limbs, uint32_t bits, uint64_t k) {
    std::memset(out, 0, sizeof(uint32_t) * limbs);
    const uint32_t hi = (bits - 1u) / 24u;
    const uint32_t bit = (bits - 1u) % 24u;
    out[hi] = 1u << bit;
    uint64_t borrow = k;
    for (uint32_t i = 0; i < limbs && borrow != 0; ++i) {
        const uint64_t v = static_cast<uint64_t>(out[i] & kLimb24Mask);
        if (v >= borrow) {
            out[i] = static_cast<uint32_t>((v - borrow) & kLimb24Mask);
            borrow = 0;
        } else {
            out[i] = static_cast<uint32_t>((v + (1ull << 24) - borrow) & kLimb24Mask);
            borrow = 1;
        }
    }
}

void mp_add_mod_fused_host_limb24(uint32_t* r, const uint32_t* a, const uint32_t* b,
                                  const uint32_t* n, uint32_t limbs) {
    uint64_t carry_add = 0;
    uint64_t carry_sub = 1;
    for (uint32_t i = 0; i < limbs; ++i) {
        const uint64_t sum = static_cast<uint64_t>(a[i]) + b[i] + carry_add;
        carry_add = sum >> 24;
        const uint64_t temp = static_cast<uint64_t>(static_cast<uint32_t>(sum)) + (~n[i]) + carry_sub;
        carry_sub = temp >> 24;
        r[i] = static_cast<uint32_t>(temp);
    }
    if ((carry_add | carry_sub) != 0) {
        return;
    }
    uint64_t c = 0;
    for (uint32_t i = 0; i < limbs; ++i) {
        const uint64_t s = static_cast<uint64_t>(r[i]) + n[i] + c;
        r[i] = static_cast<uint32_t>(s);
        c = s >> 24;
    }
}

void print_limb24_analysis(std::ostringstream& out, int bits, uint32_t words) {
    const int fair_bits = static_cast<int>(words) * 32;
    out << "\n--- limb24 analysis ---\n";
    out << "limbs=" << words << " global_words/instance=" << words << " bytes/instance="
        << (sizeof(uint32_t) * words * 4u) << " (a,b,n,out)\n";
    out << "384@32 uses 12 limbs; 384@24 uses 16 limbs => +33% limb ops per add_mod.\n";
    out << "Fair limb-count compare: " << fair_bits << "-bit@32 (" << words
        << " limbs) vs " << bits << "-bit@24 (" << words << " limbs).\n";
    out << "Each enqueue reloads global; CLPeak 24b tests in-register hot loops.\n";
    out << "See fused_hot (1 enqueue, inner=kernel_iterations) for ALU-only estimate.\n";
}

bool buffers_equal_limb24(const uint32_t* a, const uint32_t* b, uint32_t limbs) {
    for (uint32_t i = 0; i < limbs; ++i) {
        if ((a[i] & kLimb24Mask) != (b[i] & kLimb24Mask)) {
            return false;
        }
    }
    return true;
}

void set_pow2_minus_ui(uint32_t* out, uint32_t limbs, uint32_t bits, uint64_t k) {
    std::memset(out, 0, sizeof(uint32_t) * limbs);
    const uint32_t hi = (bits - 1u) / 32u;
    const uint32_t bit = (bits - 1u) % 32u;
    out[hi] = 1u << bit;
    uint64_t borrow = k;
    for (uint32_t i = 0; i < limbs && borrow != 0; ++i) {
        const uint64_t v = static_cast<uint64_t>(out[i]);
        if (v >= borrow) {
            out[i] = static_cast<uint32_t>(v - borrow);
            borrow = 0;
        } else {
            out[i] = static_cast<uint32_t>(v + (1ull << 32) - borrow);
            borrow = 1;
        }
    }
}

int mp_ge_host(const uint32_t* a, const uint32_t* n, uint32_t limbs) {
    for (int i = static_cast<int>(limbs) - 1; i >= 0; --i) {
        if (a[static_cast<uint32_t>(i)] > n[static_cast<uint32_t>(i)]) {
            return 1;
        }
        if (a[static_cast<uint32_t>(i)] < n[static_cast<uint32_t>(i)]) {
            return 0;
        }
    }
    return 1;
}

void mp_sub_n_host(uint32_t* r, const uint32_t* a, const uint32_t* b, uint32_t limbs) {
    uint64_t borrow = 0;
    for (uint32_t i = 0; i < limbs; ++i) {
        const uint64_t av = a[i];
        const uint64_t bv = b[i];
        const uint64_t w = av - bv - borrow;
        r[i] = static_cast<uint32_t>(w);
        borrow = (av < bv + borrow) ? 1ull : 0ull;
    }
}

void mp_add_mod_legacy_host(uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* n,
                            uint32_t limbs) {
    uint64_t carry = 0;
    for (uint32_t i = 0; i < limbs; ++i) {
        const uint64_t sum = static_cast<uint64_t>(a[i]) + b[i] + carry;
        r[i] = static_cast<uint32_t>(sum);
        carry = sum >> 32;
    }
    if (carry != 0 || mp_ge_host(r, n, limbs)) {
        mp_sub_n_host(r, r, n, limbs);
    }
}

void mp_sub_mod_host(uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* n, uint32_t limbs) {
    uint64_t borrow = 0;
    for (uint32_t i = 0; i < limbs; ++i) {
        const uint64_t av = a[i];
        const uint64_t bv = b[i];
        const uint64_t w = av - bv - borrow;
        r[i] = static_cast<uint32_t>(w);
        borrow = (av < bv + borrow) ? 1ull : 0ull;
    }
    if (borrow) {
        uint64_t carry = 0;
        for (uint32_t i = 0; i < limbs; ++i) {
            const uint64_t sum = static_cast<uint64_t>(r[i]) + n[i] + carry;
            r[i] = static_cast<uint32_t>(sum);
            carry = sum >> 32;
        }
    }
}

bool buffers_equal(const uint32_t* a, const uint32_t* b, uint32_t limbs) {
    return std::memcmp(a, b, sizeof(uint32_t) * limbs) == 0;
}

std::string build_addsub_source(uint32_t words, int limb_bits, std::ostringstream& log) {
    std::string src;
    if (limb_bits == 24) {
        const char* rel = "mp_addsub/limb24_addsub.cl";
        src = load_kernel_asset(rel);
        if (src.empty()) {
            log << "missing kernel asset: " << rel << "\n";
            log << "run Gradle syncAddsubKernels or rebuild the app\n";
        }
        return src;
    }
    const EcmAddSubBuildManifest manifest = opencl_ecm_addsub_build_manifest(words, false, false);
    for (const std::string& rel : manifest.source_paths) {
        const std::string part = load_kernel_asset(rel.c_str());
        if (part.empty()) {
            log << "missing kernel asset: " << rel << "\n";
            log << "run Gradle syncAddsubKernels or rebuild the app\n";
            return {};
        }
        if (!src.empty()) {
            src += "\n";
        }
        src += part;
    }
    return src;
}

std::string format_ops_per_s(double ops) {
    std::ostringstream oss;
    if (ops >= 1e6) {
        oss << std::scientific << std::setprecision(4) << ops;
    } else {
        oss << std::fixed << std::setprecision(3) << ops;
    }
    return oss.str();
}

bool run_kernel_timed(
        OpenCLApi& api,
        cl_command_queue q,
        cl_kernel k,
        size_t global,
        const size_t* local,
        int total_enqueues,
        double& ms_out) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < total_enqueues; ++i) {
        const cl_int err = api.clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, local, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            return false;
        }
    }
    api.clFinish(q);
    const auto t1 = std::chrono::steady_clock::now();
    ms_out = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
}

} // namespace

std::string run_addsub_bench(int bits, int kernel_iterations, int instances, int launch_repeats,
                             int limb_bits) {
    std::ostringstream out;
    if (limb_bits != 24 && limb_bits != 32) {
        out << "limb_bits must be 24 or 32\n";
        return out.str();
    }
    const uint32_t limb_divisor = static_cast<uint32_t>(limb_bits);
    if (bits <= 0 || (bits % static_cast<int>(limb_divisor)) != 0 ||
        static_cast<uint32_t>(bits) > kMaxBenchBits) {
        out << "bits must be a positive multiple of " << limb_bits << " and <= " << kMaxBenchBits
            << "\n";
        return out.str();
    }
    if (kernel_iterations <= 0 || instances <= 0 || launch_repeats <= 0) {
        out << "kernel_iterations, instances, launch_repeats must be > 0\n";
        return out.str();
    }

    const uint32_t words = static_cast<uint32_t>(bits) / limb_divisor;
    const bool limb24 = limb_bits == 24;
    out << "=== ECM add/sub microbench ===\n";
    out << bits << "-bit, limb_bits=" << limb_bits << ", limbs=" << words
        << ", kernel_iterations=" << kernel_iterations << ", instances=" << instances
        << ", launch_repeats=" << launch_repeats << "\n";

    OpenCLApi api{};
    bool own_lib = false;
    if (!load_opencl_api(api, own_lib, out)) {
        out << "FAIL: OpenCL not loaded\n";
        return out.str();
    }

    cl_device_id dev = nullptr;
    if (!acquire_gpu_device(api, dev, out)) {
        unload_opencl_api(api, own_lib);
        out << "FAIL: no GPU\n";
        return out.str();
    }
    out << "device: " << query_device_string(api, dev, CL_DEVICE_NAME) << "\n";

    const std::string src = build_addsub_source(words, limb_bits, out);
    if (src.empty()) {
        unload_opencl_api(api, own_lib);
        out << "FAIL: kernel sources\n";
        return out.str();
    }

    cl_context ctx = nullptr;
    cl_command_queue q = nullptr;
    if (!create_context_queue(api, dev, ctx, q, out)) {
        unload_opencl_api(api, own_lib);
        return out.str();
    }

    char build_opts[128];
    if (limb24) {
        std::snprintf(build_opts, sizeof(build_opts),
                      "-DMAX_LIMBS=%u -DMP_LIMB_BITS=24 -cl-fast-relaxed-math", words);
    } else {
        std::snprintf(build_opts, sizeof(build_opts),
                      "-DMAX_LIMBS=%u -DMP_ADD_MOD_FUSED_UNROLL=2 -cl-fast-relaxed-math", words);
    }
    out << "build: MAX_LIMBS=" << words << " limb_bits=" << limb_bits
        << " src_kib=" << (src.size() / 1024u) << "\n";

    cl_int err = 0;
    const char* src_ptr = src.c_str();
    const size_t src_len = src.size();
    cl_program program = api.clCreateProgramWithSource(ctx, 1, &src_ptr, &src_len, &err);
    if (!program || err != CL_SUCCESS) {
        out << "clCreateProgramWithSource err=" << err << "\n";
        api.clReleaseContext(ctx);
        unload_opencl_api(api, own_lib);
        return out.str();
    }

    const auto compile_t0 = std::chrono::steady_clock::now();
    err = api.clBuildProgram(program, 1, &dev, build_opts, nullptr, nullptr);
    const double compile_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - compile_t0).count();
    if (err != CL_SUCCESS) {
        out << "clBuildProgram err=" << err << " (" << cl_err_str(err) << ")\n";
        out << program_build_log(api, program, dev) << "\n";
        api.clReleaseProgram(program);
        api.clReleaseContext(ctx);
        unload_opencl_api(api, own_lib);
        return out.str();
    }
    out << "compile: " << compile_ms << " ms\n";

    std::vector<uint32_t> n_words(words), a_words(words), b_words(words);
    if (limb24) {
        set_pow2_minus_ui_limb24(n_words.data(), words, static_cast<uint32_t>(bits), 109ull);
        set_pow2_minus_ui_limb24(a_words.data(), words, static_cast<uint32_t>(bits), 991ull);
        set_pow2_minus_ui_limb24(b_words.data(), words, static_cast<uint32_t>(bits), 8218291649ull);
    } else {
        set_pow2_minus_ui(n_words.data(), words, static_cast<uint32_t>(bits), 109ull);
        set_pow2_minus_ui(a_words.data(), words, static_cast<uint32_t>(bits), 991ull);
        set_pow2_minus_ui(b_words.data(), words, static_cast<uint32_t>(bits), 8218291649ull);
    }

    const size_t total_words = static_cast<size_t>(instances) * words;
    std::vector<uint32_t> host_a(total_words), host_b(total_words), host_n(total_words);
    for (int i = 0; i < instances; ++i) {
        std::memcpy(host_a.data() + static_cast<size_t>(i) * words, a_words.data(), sizeof(uint32_t) * words);
        std::memcpy(host_b.data() + static_cast<size_t>(i) * words, b_words.data(), sizeof(uint32_t) * words);
        std::memcpy(host_n.data() + static_cast<size_t>(i) * words, n_words.data(), sizeof(uint32_t) * words);
    }

    const size_t bytes = sizeof(uint32_t) * total_words;
    cl_mem buf_a = api.clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, &err);
    cl_mem buf_b = api.clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, &err);
    cl_mem buf_n = api.clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, &err);
    cl_mem buf_out = api.clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (!buf_a || !buf_b || !buf_n || !buf_out) {
        out << "buffer alloc failed\n";
        api.clReleaseProgram(program);
        api.clReleaseContext(ctx);
        unload_opencl_api(api, own_lib);
        return out.str();
    }
    api.clEnqueueWriteBuffer(q, buf_a, 1, 0, bytes, host_a.data(), 0, nullptr, nullptr);
    api.clEnqueueWriteBuffer(q, buf_b, 1, 0, bytes, host_b.data(), 0, nullptr, nullptr);
    api.clEnqueueWriteBuffer(q, buf_n, 1, 0, bytes, host_n.data(), 0, nullptr, nullptr);

    const cl_uint limbs = words;
    const size_t global = static_cast<size_t>(instances);
    const int total_enqueues = launch_repeats * kernel_iterations;
    const double op_count =
        static_cast<double>(instances) * static_cast<double>(kernel_iterations) * launch_repeats;

    auto run_once = [&](const char* kname, const size_t* local, size_t gws) -> bool {
        cl_int kerr = CL_SUCCESS;
        cl_kernel k = api.clCreateKernel(program, kname, &kerr);
        if (kerr != CL_SUCCESS) {
            return false;
        }
        api.clSetKernelArg(k, 0, sizeof(cl_mem), &buf_a);
        api.clSetKernelArg(k, 1, sizeof(cl_mem), &buf_b);
        api.clSetKernelArg(k, 2, sizeof(cl_mem), &buf_n);
        api.clSetKernelArg(k, 3, sizeof(cl_mem), &buf_out);
        api.clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
        const cl_int e = api.clEnqueueNDRangeKernel(q, k, 1, nullptr, &gws, local, 0, nullptr, nullptr);
        api.clFinish(q);
        api.clReleaseKernel(k);
        return e == CL_SUCCESS;
    };

    std::vector<uint32_t> expect(words), got(words);
    if (limb24) {
        mp_add_mod_fused_host_limb24(expect.data(), a_words.data(), b_words.data(), n_words.data(), words);
    } else {
        mp_add_mod_legacy_host(expect.data(), a_words.data(), b_words.data(), n_words.data(), words);
    }
    if (run_once("ecm_mp_add_mod_fused", nullptr, 1u)) {
        api.clEnqueueReadBuffer(q, buf_out, 1, 0, sizeof(uint32_t) * words, got.data(), 0, nullptr, nullptr);
        const bool ok_verify = limb24 ? buffers_equal_limb24(expect.data(), got.data(), words)
                                      : buffers_equal(expect.data(), got.data(), words);
        out << "verify ecm_mp_add_mod_fused: " << (ok_verify ? "PASS" : "FAIL") << "\n";
    }

    const cl_uint inner_iters = static_cast<cl_uint>(kernel_iterations);
    const double hot_op_count =
        static_cast<double>(instances) * static_cast<double>(kernel_iterations) * launch_repeats;

    auto run_add_kernel = [&](const char* kernel_name, const char* path_label, bool hot_inner_loop,
                              bool use_wg, int lpt_chunk) -> bool {
        cl_int kerr = CL_SUCCESS;
        cl_kernel k = api.clCreateKernel(program, kernel_name, &kerr);
        if (kerr != CL_SUCCESS) {
            if (hot_inner_loop) {
                out << path_label << ": kernel missing\n";
            }
            return false;
        }
        api.clSetKernelArg(k, 0, sizeof(cl_mem), &buf_a);
        api.clSetKernelArg(k, 1, sizeof(cl_mem), &buf_b);
        api.clSetKernelArg(k, 2, sizeof(cl_mem), &buf_n);
        api.clSetKernelArg(k, 3, sizeof(cl_mem), &buf_out);
        api.clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
        double ms = 0.0;
        bool ok = false;
        double measured_ops = op_count;
        size_t gws = global;
        const size_t* local = nullptr;
        size_t local_sz = 0;
        if (use_wg) {
            local_sz = static_cast<size_t>(words / static_cast<uint32_t>(lpt_chunk));
            gws = global * local_sz;
            local = &local_sz;
        }
        if (hot_inner_loop) {
            api.clSetKernelArg(k, 5, sizeof(cl_uint), &inner_iters);
            measured_ops = hot_op_count;
            ok = run_kernel_timed(api, q, k, gws, local, launch_repeats, ms);
        } else {
            ok = run_kernel_timed(api, q, k, gws, local, total_enqueues, ms);
        }
        api.clReleaseKernel(k);
        if (!ok) {
            out << path_label << ": enqueue failed\n";
            return false;
        }
        const double ops_s = measured_ops / (ms / 1000.0);
        out << path_label << ": " << ms << " ms, " << format_ops_per_s(ops_s) << " ops/s\n";
        return true;
    };

    out << std::fixed << std::setprecision(3);
    if (limb24) {
        print_limb24_analysis(out, bits, words);
    }
    out << "\n--- mp_add_mod ---\n";
    if (limb24) {
        for (const Limb24AddBenchKernel& spec : kLimb24AddKernels) {
            run_add_kernel(spec.kernel_name, spec.path_label, false, false, 0);
        }
        out << "\n--- mp_add_mod (hot, inner=" << kernel_iterations << ") ---\n";
        for (const Limb24AddBenchKernel& spec : kLimb24AddHotKernels) {
            run_add_kernel(spec.kernel_name, spec.path_label, true, false, 0);
        }
        out << "\nRESULT: PASS\n";
        api.clReleaseMemObject(buf_a);
        api.clReleaseMemObject(buf_b);
        api.clReleaseMemObject(buf_n);
        api.clReleaseMemObject(buf_out);
        api.clReleaseProgram(program);
        api.clReleaseContext(ctx);
        unload_opencl_api(api, own_lib);
        return out.str();
    }
    for (const EcmAddSubBenchKernel& spec :
         opencl_ecm_addsub_add_kernels(words, false, false, false)) {
        if (spec.hot_inner_loop) {
            continue;
        }
        run_add_kernel(spec.kernel_name, spec.path_label, false, spec.use_wg, spec.lpt_chunk);
    }
    out << "\n--- mp_add_mod (hot, inner=" << kernel_iterations << ") ---\n";
    for (const EcmAddSubBenchKernel& spec :
         opencl_ecm_addsub_add_kernels(words, false, false, false)) {
        if (!spec.hot_inner_loop) {
            continue;
        }
        run_add_kernel(spec.kernel_name, spec.path_label, true, spec.use_wg, spec.lpt_chunk);
    }

    out << "\n--- mp_sub_mod ---\n";
    for (const EcmAddSubBenchKernel& spec : opencl_ecm_addsub_sub_kernels(words, false, false)) {
        if (spec.hot_inner_loop) {
            continue;
        }
        run_add_kernel(spec.kernel_name, spec.path_label, false, spec.use_wg, spec.lpt_chunk);
    }
    out << "\n--- mp_sub_mod (hot, inner=" << kernel_iterations << ") ---\n";
    for (const EcmAddSubBenchKernel& spec : opencl_ecm_addsub_sub_kernels(words, false, false)) {
        if (!spec.hot_inner_loop) {
            continue;
        }
        run_add_kernel(spec.kernel_name, spec.path_label, true, spec.use_wg, spec.lpt_chunk);
    }

    out << "\nRESULT: PASS\n";

    api.clReleaseMemObject(buf_a);
    api.clReleaseMemObject(buf_b);
    api.clReleaseMemObject(buf_n);
    api.clReleaseMemObject(buf_out);
    api.clReleaseProgram(program);
    api.clReleaseContext(ctx);
    unload_opencl_api(api, own_lib);
    return out.str();
}
