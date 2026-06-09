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

std::string build_addsub_source(uint32_t words, std::ostringstream& log) {
    const EcmAddSubBuildManifest manifest = opencl_ecm_addsub_build_manifest(words, false, false);
    std::string src;
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

std::string run_addsub_bench(int bits, int kernel_iterations, int instances, int launch_repeats) {
    std::ostringstream out;
    if (bits <= 0 || (bits % 32) != 0 || static_cast<uint32_t>(bits) > kMaxBenchBits) {
        out << "bits must be a positive multiple of 32 and <= " << kMaxBenchBits << "\n";
        return out.str();
    }
    if (kernel_iterations <= 0 || instances <= 0 || launch_repeats <= 0) {
        out << "kernel_iterations, instances, launch_repeats must be > 0\n";
        return out.str();
    }

    const uint32_t words = static_cast<uint32_t>(bits) / 32u;
    out << "=== ECM add/sub microbench ===\n";
    out << bits << "-bit, kernel_iterations=" << kernel_iterations << ", instances=" << instances
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

    const std::string src = build_addsub_source(words, out);
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

    char build_opts[96];
    std::snprintf(build_opts, sizeof(build_opts), "-DMAX_LIMBS=%u -DMP_ADD_MOD_FUSED_UNROLL=2", words);
    out << "build: MAX_LIMBS=" << words << " src_kib=" << (src.size() / 1024u) << "\n";

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
    set_pow2_minus_ui(n_words.data(), words, static_cast<uint32_t>(bits), 109ull);
    set_pow2_minus_ui(a_words.data(), words, static_cast<uint32_t>(bits), 991ull);
    set_pow2_minus_ui(b_words.data(), words, static_cast<uint32_t>(bits), 8218291649ull);

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
    mp_add_mod_legacy_host(expect.data(), a_words.data(), b_words.data(), n_words.data(), words);
    if (run_once("ecm_mp_add_mod_fused", nullptr, 1u)) {
        api.clEnqueueReadBuffer(q, buf_out, 1, 0, sizeof(uint32_t) * words, got.data(), 0, nullptr, nullptr);
        out << "verify ecm_mp_add_mod_fused: " << (buffers_equal(expect.data(), got.data(), words) ? "PASS" : "FAIL")
            << "\n";
    }

    out << std::fixed << std::setprecision(3);
    out << "\n--- mp_add_mod ---\n";
    for (const EcmAddSubBenchKernel& spec :
         opencl_ecm_addsub_add_kernels(words, false, false, false)) {
        cl_int kerr = CL_SUCCESS;
        cl_kernel k = api.clCreateKernel(program, spec.kernel_name, &kerr);
        if (kerr != CL_SUCCESS) {
            continue;
        }
        api.clSetKernelArg(k, 0, sizeof(cl_mem), &buf_a);
        api.clSetKernelArg(k, 1, sizeof(cl_mem), &buf_b);
        api.clSetKernelArg(k, 2, sizeof(cl_mem), &buf_n);
        api.clSetKernelArg(k, 3, sizeof(cl_mem), &buf_out);
        api.clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
        double ms = 0.0;
        size_t gws = global;
        const size_t* local = nullptr;
        size_t local_sz = 0;
        if (spec.use_wg) {
            local_sz = static_cast<size_t>(words / static_cast<uint32_t>(spec.lpt_chunk));
            gws = global * local_sz;
            local = &local_sz;
        }
        const bool ok = run_kernel_timed(api, q, k, gws, local, total_enqueues, ms);
        api.clReleaseKernel(k);
        if (!ok) {
            out << spec.path_label << ": enqueue failed\n";
            continue;
        }
        const double ops_s = op_count / (ms / 1000.0);
        out << spec.path_label << ": " << ms << " ms, " << format_ops_per_s(ops_s) << " ops/s\n";
    }

    out << "\n--- mp_sub_mod ---\n";
    for (const EcmAddSubBenchKernel& spec : opencl_ecm_addsub_sub_kernels(words, false, false)) {
        cl_int kerr = CL_SUCCESS;
        cl_kernel k = api.clCreateKernel(program, spec.kernel_name, &kerr);
        if (kerr != CL_SUCCESS) {
            continue;
        }
        api.clSetKernelArg(k, 0, sizeof(cl_mem), &buf_a);
        api.clSetKernelArg(k, 1, sizeof(cl_mem), &buf_b);
        api.clSetKernelArg(k, 2, sizeof(cl_mem), &buf_n);
        api.clSetKernelArg(k, 3, sizeof(cl_mem), &buf_out);
        api.clSetKernelArg(k, 4, sizeof(cl_uint), &limbs);
        double ms = 0.0;
        const bool ok = run_kernel_timed(api, q, k, global, nullptr, total_enqueues, ms);
        api.clReleaseKernel(k);
        if (!ok) {
            out << spec.path_label << ": enqueue failed\n";
            continue;
        }
        const double ops_s = op_count / (ms / 1000.0);
        out << spec.path_label << ": " << ms << " ms, " << format_ops_per_s(ops_s) << " ops/s\n";
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
