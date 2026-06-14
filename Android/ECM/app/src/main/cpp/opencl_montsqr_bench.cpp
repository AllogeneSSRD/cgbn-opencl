#include "opencl_runtime.h"

#include "kernel_assets.h"
#include "opencl_ecm_montsqr_manifest.h"
#include "opencl_loader.h"
#include "opencl_program_cache.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace {

#if defined(__ANDROID__)
constexpr uint32_t kMaxBenchBits = 2048;
#else
constexpr uint32_t kMaxBenchBits = 8192;
#endif

constexpr uint32_t kLimb24Mask = (1u << 24) - 1u;
constexpr uint32_t kMontI24Limbs512 = 22u;
constexpr const char* kKernelAssetRoot = "kernels/opencl/bench/";

struct MontI24BenchKernel {
    const char* kernel_name;
    const char* path_label;
    bool is_mul;
};
constexpr size_t kFips512MtLocalU32 = 16u + 16u + 32u * 2u + 17u;
constexpr size_t kFips512Cs8LocalU32 = 16u + 16u + 8u * 34u + 34u;
constexpr size_t kFips512Cs16LocalU32 = 16u + 16u + 16u * 34u + 34u;

void strip_include(std::string& src, const char* inc) {
    const size_t pos = src.find(inc);
    if (pos != std::string::npos) {
        src.erase(pos, std::strlen(inc));
    }
}

void strip_pragma_once(std::string& src) {
    const std::string tag = "#pragma once";
    for (size_t pos = src.find(tag); pos != std::string::npos; pos = src.find(tag)) {
        const size_t end = src.find('\n', pos);
        if (end == std::string::npos) {
            src.erase(pos);
            break;
        }
        src.erase(pos, end - pos + 1);
    }
}

void sanitize_mont_part(std::string& part, const std::string& rel) {
    strip_pragma_once(part);
    if (rel == "mont_priv_bench.cl") {
        strip_include(part, "#include \"mont_priv.cl\"");
    } else if (rel == "mont_priv_opt_bench.cl") {
        strip_include(part, "#include \"mont_priv_opt.cl\"");
    } else if (rel == "mont_wg_bench.cl") {
        strip_include(part, "#include \"mont_wg.cl\"");
    }
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

uint32_t inv32_odd(uint32_t x) {
    uint64_t y = 1;
    for (int i = 0; i < 5; ++i) {
        y = y * (2ull - static_cast<uint64_t>(x) * y);
        y &= 0xFFFFFFFFull;
    }
    return static_cast<uint32_t>(y);
}

uint32_t inv24_odd(uint32_t x) {
    uint64_t y = 1;
    for (int i = 0; i < 4; ++i) {
        y = y * (2ull - static_cast<uint64_t>(x & kLimb24Mask) * y);
        y &= 0xFFFFFFull;
    }
    return static_cast<uint32_t>(y) & kLimb24Mask;
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

uint32_t mont_i24_words_for_bits(int bits) {
    if (bits == 512) {
        return kMontI24Limbs512;
    }
    if (bits <= 0 || (bits % 24) != 0) {
        return 0u;
    }
    return static_cast<uint32_t>(bits) / 24u;
}

std::string build_montsqr_source_i24(std::ostringstream& log) {
    const std::string mul_rel = std::string(kKernelAssetRoot) + "mont_mul_unroll_i24.cl";
    const std::string manual_rel =
        std::string(kKernelAssetRoot) + "mont_mul_unroll_i24_384_manual_generated.cl";
    const std::string bench_rel = std::string(kKernelAssetRoot) + "mont_mul_unroll_i24_bench.cl";
    std::string mul = load_kernel_asset(mul_rel.c_str());
    std::string manual = load_kernel_asset(manual_rel.c_str());
    std::string bench = load_kernel_asset(bench_rel.c_str());
    if (mul.empty() || bench.empty()) {
        log << "missing mont_mul_unroll_i24.cl or mont_mul_unroll_i24_bench.cl\n";
        log << "run Gradle syncMontsqrKernels or rebuild the app\n";
        return {};
    }
    if (manual.empty()) {
        log << "missing mont_mul_unroll_i24_384_manual_generated.cl\n";
        log << "run tools/gen_mont_mul_unroll_i24_384_manual.py then Gradle syncMontsqrKernels\n";
        return {};
    }
    strip_pragma_once(mul);
    strip_pragma_once(manual);
    strip_pragma_once(bench);
    strip_include(bench, "#include \"mont_mul_unroll_i24.cl\"");
    strip_include(bench, "#include \"mont_mul_unroll_i24_384_manual_generated.cl\"");
    return mul + "\n" + manual + "\n" + bench;
}

std::string build_montsqr_source(uint32_t words, bool use_wg, std::ostringstream& log) {
    const EcmMontSqrBuildManifest manifest = opencl_ecm_montsqr_build_manifest(words, use_wg);
    std::string src;
    for (const std::string& rel : manifest.source_paths) {
        const std::string asset_rel = std::string(kKernelAssetRoot) + rel;
        std::string part = load_kernel_asset(asset_rel.c_str());
        if (part.empty()) {
            log << "missing kernel asset: " << rel << "\n";
            log << "run Gradle syncEcmStage1Kernels or rebuild the app\n";
            return {};
        }
        sanitize_mont_part(part, rel);
        if (!src.empty()) {
            src += "\n";
        }
        src += part;
    }
    return src;
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

struct MontBenchCtx {
    OpenCLApi& api;
    cl_command_queue q;
    cl_program program;
    cl_mem buf_a;
    cl_mem buf_b;
    cl_mem buf_n;
    cl_mem buf_out;
    cl_mem buf_n_const;
    cl_mem buf_np0_const;
    cl_uint limbs;
    cl_uint np0;
    cl_uint inner_iters;
    size_t global;
    int launch_repeats;
    int tpi;
    uint32_t words;
    double op_count;
    std::ostringstream& out;
};

size_t fips_local_u32(MontDispatch dispatch, uint32_t mt_local) {
    if (dispatch == MontDispatch::PrivFipsMtCs) {
        return (mt_local == 16u) ? kFips512Cs16LocalU32 : kFips512Cs8LocalU32;
    }
    return kFips512MtLocalU32;
}

void print_mont_kernel_plan(
        std::ostringstream& out,
        const char* section,
        const std::vector<EcmMontSqrBenchKernel>& specs,
        uint32_t words) {
    out << section << " (" << specs.size() << " paths, limbs=" << words << "):\n";
    for (const EcmMontSqrBenchKernel& spec : specs) {
        out << "  " << spec.path_label << " [" << spec.kernel_name << "]";
        if (spec.required_words != 0u && spec.required_words != words) {
            out << " (needs " << spec.required_words << " limbs)";
        }
        if (spec.mt_local_size != 0u) {
            out << " local=" << spec.mt_local_size;
        }
        out << "\n";
    }
}

bool run_mont_kernel(MontBenchCtx& ctx, const EcmMontSqrBenchKernel& spec, double& ms_out) {
    if (spec.required_words != 0u && spec.required_words != ctx.words) {
        ctx.out << spec.path_label << ": skipped (requires " << spec.required_words
                << " limbs, have " << ctx.words << ")\n";
        return false;
    }
    cl_int kerr = CL_SUCCESS;
    cl_kernel k = ctx.api.clCreateKernel(ctx.program, spec.kernel_name.c_str(), &kerr);
    if (kerr != CL_SUCCESS || k == nullptr) {
        ctx.out << spec.path_label << ": skipped (clCreateKernel \"" << spec.kernel_name << "\" err="
                << kerr << " " << cl_err_str(kerr) << ")\n";
        return false;
    }

    size_t gws = ctx.global;
    const size_t* local = nullptr;
    size_t local_sz = 0;
    bool ok = false;

    switch (spec.dispatch) {
    case MontDispatch::PrivLegacy:
        ctx.api.clSetKernelArg(k, 0, sizeof(cl_mem), &ctx.buf_a);
        if (spec.is_mul) {
            ctx.api.clSetKernelArg(k, 1, sizeof(cl_mem), &ctx.buf_b);
            ctx.api.clSetKernelArg(k, 2, sizeof(cl_mem), &ctx.buf_n);
            ctx.api.clSetKernelArg(k, 3, sizeof(cl_mem), &ctx.buf_out);
            ctx.api.clSetKernelArg(k, 4, sizeof(cl_uint), &ctx.np0);
            ctx.api.clSetKernelArg(k, 5, sizeof(cl_uint), &ctx.limbs);
            ctx.api.clSetKernelArg(k, 6, sizeof(cl_uint), &ctx.inner_iters);
        } else {
            ctx.api.clSetKernelArg(k, 1, sizeof(cl_mem), &ctx.buf_n);
            ctx.api.clSetKernelArg(k, 2, sizeof(cl_mem), &ctx.buf_out);
            ctx.api.clSetKernelArg(k, 3, sizeof(cl_uint), &ctx.np0);
            ctx.api.clSetKernelArg(k, 4, sizeof(cl_uint), &ctx.limbs);
            ctx.api.clSetKernelArg(k, 5, sizeof(cl_uint), &ctx.inner_iters);
        }
        ok = run_kernel_timed(ctx.api, ctx.q, k, gws, nullptr, ctx.launch_repeats, ms_out);
        break;
    case MontDispatch::PrivOpt:
    case MontDispatch::PrivUnroll:
        ctx.api.clSetKernelArg(k, 0, sizeof(cl_mem), &ctx.buf_a);
        if (spec.is_mul) {
            ctx.api.clSetKernelArg(k, 1, sizeof(cl_mem), &ctx.buf_b);
            ctx.api.clSetKernelArg(k, 2, sizeof(cl_mem), &ctx.buf_n_const);
            ctx.api.clSetKernelArg(k, 3, sizeof(cl_mem), &ctx.buf_out);
            ctx.api.clSetKernelArg(k, 4, sizeof(cl_mem), &ctx.buf_np0_const);
            ctx.api.clSetKernelArg(k, 5, sizeof(cl_uint), &ctx.limbs);
            ctx.api.clSetKernelArg(k, 6, sizeof(cl_uint), &ctx.inner_iters);
        } else {
            ctx.api.clSetKernelArg(k, 1, sizeof(cl_mem), &ctx.buf_n_const);
            ctx.api.clSetKernelArg(k, 2, sizeof(cl_mem), &ctx.buf_out);
            ctx.api.clSetKernelArg(k, 3, sizeof(cl_mem), &ctx.buf_np0_const);
            ctx.api.clSetKernelArg(k, 4, sizeof(cl_uint), &ctx.limbs);
            ctx.api.clSetKernelArg(k, 5, sizeof(cl_uint), &ctx.inner_iters);
        }
        ok = run_kernel_timed(ctx.api, ctx.q, k, gws, nullptr, ctx.launch_repeats, ms_out);
        break;
    case MontDispatch::PrivLocal512:
    case MontDispatch::PrivOpt2Local512: {
        const size_t local_mem_size = static_cast<size_t>(2u) * static_cast<size_t>(ctx.words) *
                                      sizeof(uint32_t);
        ctx.api.clSetKernelArg(k, 0, sizeof(cl_mem), &ctx.buf_a);
        if (spec.is_mul) {
            ctx.api.clSetKernelArg(k, 1, sizeof(cl_mem), &ctx.buf_b);
            ctx.api.clSetKernelArg(k, 2, sizeof(cl_mem), &ctx.buf_n_const);
            ctx.api.clSetKernelArg(k, 3, sizeof(cl_mem), &ctx.buf_out);
            ctx.api.clSetKernelArg(k, 4, sizeof(cl_mem), &ctx.buf_np0_const);
            ctx.api.clSetKernelArg(k, 5, sizeof(cl_uint), &ctx.limbs);
            ctx.api.clSetKernelArg(k, 6, sizeof(cl_uint), &ctx.inner_iters);
            ctx.api.clSetKernelArg(k, 7, local_mem_size, nullptr);
        } else {
            ctx.api.clSetKernelArg(k, 1, sizeof(cl_mem), &ctx.buf_n_const);
            ctx.api.clSetKernelArg(k, 2, sizeof(cl_mem), &ctx.buf_out);
            ctx.api.clSetKernelArg(k, 3, sizeof(cl_mem), &ctx.buf_np0_const);
            ctx.api.clSetKernelArg(k, 4, sizeof(cl_uint), &ctx.limbs);
            ctx.api.clSetKernelArg(k, 5, sizeof(cl_uint), &ctx.inner_iters);
            ctx.api.clSetKernelArg(k, 6, local_mem_size, nullptr);
        }
        local_sz = 1u;
        local = &local_sz;
        ok = run_kernel_timed(ctx.api, ctx.q, k, gws, local, ctx.launch_repeats, ms_out);
        break;
    }
    case MontDispatch::PrivFipsMt:
    case MontDispatch::PrivFipsMtCs: {
        const size_t local_mem_size = fips_local_u32(spec.dispatch, spec.mt_local_size) * sizeof(uint32_t);
        ctx.api.clSetKernelArg(k, 0, sizeof(cl_mem), &ctx.buf_a);
        if (spec.is_mul) {
            ctx.api.clSetKernelArg(k, 1, sizeof(cl_mem), &ctx.buf_b);
            ctx.api.clSetKernelArg(k, 2, sizeof(cl_mem), &ctx.buf_n_const);
            ctx.api.clSetKernelArg(k, 3, sizeof(cl_mem), &ctx.buf_out);
            ctx.api.clSetKernelArg(k, 4, sizeof(cl_mem), &ctx.buf_np0_const);
            ctx.api.clSetKernelArg(k, 5, sizeof(cl_uint), &ctx.limbs);
            ctx.api.clSetKernelArg(k, 6, sizeof(cl_uint), &ctx.inner_iters);
            ctx.api.clSetKernelArg(k, 7, local_mem_size, nullptr);
        } else {
            ctx.api.clSetKernelArg(k, 1, sizeof(cl_mem), &ctx.buf_n_const);
            ctx.api.clSetKernelArg(k, 2, sizeof(cl_mem), &ctx.buf_out);
            ctx.api.clSetKernelArg(k, 3, sizeof(cl_mem), &ctx.buf_np0_const);
            ctx.api.clSetKernelArg(k, 4, sizeof(cl_uint), &ctx.limbs);
            ctx.api.clSetKernelArg(k, 5, sizeof(cl_uint), &ctx.inner_iters);
            ctx.api.clSetKernelArg(k, 6, local_mem_size, nullptr);
        }
        local_sz = static_cast<size_t>(spec.mt_local_size);
        gws = ctx.global * local_sz;
        local = &local_sz;
        ok = run_kernel_timed(ctx.api, ctx.q, k, gws, local, ctx.launch_repeats, ms_out);
        break;
    }
    case MontDispatch::Wg: {
        const size_t local_mem_size =
            ((ctx.words + 1u) + ctx.words + ctx.words + static_cast<size_t>(4 * ctx.tpi) + ctx.words +
             ctx.words + ctx.words) *
            sizeof(uint32_t);
        ctx.api.clSetKernelArg(k, 0, sizeof(cl_mem), &ctx.buf_a);
        if (spec.is_mul) {
            ctx.api.clSetKernelArg(k, 1, sizeof(cl_mem), &ctx.buf_b);
            ctx.api.clSetKernelArg(k, 2, sizeof(cl_mem), &ctx.buf_n);
            ctx.api.clSetKernelArg(k, 3, sizeof(cl_mem), &ctx.buf_out);
            ctx.api.clSetKernelArg(k, 4, sizeof(cl_uint), &ctx.np0);
            ctx.api.clSetKernelArg(k, 5, sizeof(cl_uint), &ctx.limbs);
            ctx.api.clSetKernelArg(k, 6, sizeof(cl_uint), &ctx.inner_iters);
            ctx.api.clSetKernelArg(k, 7, local_mem_size, nullptr);
        } else {
            ctx.api.clSetKernelArg(k, 1, sizeof(cl_mem), &ctx.buf_n);
            ctx.api.clSetKernelArg(k, 2, sizeof(cl_mem), &ctx.buf_out);
            ctx.api.clSetKernelArg(k, 3, sizeof(cl_uint), &ctx.np0);
            ctx.api.clSetKernelArg(k, 4, sizeof(cl_uint), &ctx.limbs);
            ctx.api.clSetKernelArg(k, 5, sizeof(cl_uint), &ctx.inner_iters);
            ctx.api.clSetKernelArg(k, 6, local_mem_size, nullptr);
        }
        local_sz = static_cast<size_t>(ctx.tpi);
        gws = ctx.global * local_sz;
        local = &local_sz;
        ok = run_kernel_timed(ctx.api, ctx.q, k, gws, local, ctx.launch_repeats, ms_out);
        break;
    }
    }

    ctx.api.clReleaseKernel(k);
    if (!ok) {
        ctx.out << spec.path_label << ": skipped (enqueue failed)\n";
        return false;
    }
    const double ops_s = ctx.op_count / (ms_out / 1000.0);
    ctx.out << spec.path_label << ": " << ms_out << " ms, " << format_ops_per_s(ops_s) << " ops/s\n";
    return true;
}

struct MontSectionStats {
    int ran = 0;
    int skipped = 0;
};

void run_mont_section(
        MontBenchCtx& ctx,
        const std::vector<EcmMontSqrBenchKernel>& specs,
        MontSectionStats& stats) {
    stats.ran = 0;
    stats.skipped = 0;
    for (const EcmMontSqrBenchKernel& spec : specs) {
        double ms = 0.0;
        if (run_mont_kernel(ctx, spec, ms)) {
            ++stats.ran;
        } else {
            ++stats.skipped;
        }
    }
}

std::string montsqr_bench_i24(int bits, int kernel_iterations, int instances, int launch_repeats) {
    std::ostringstream out;
    const uint32_t words = mont_i24_words_for_bits(bits);
    if (words == 0u || static_cast<uint32_t>(bits) > kMaxBenchBits) {
        out << "mont unroll_i24: bits must be 512 (22 limbs) or a positive multiple of 24 and <= "
            << kMaxBenchBits << "\n";
        return out.str();
    }
    if (kernel_iterations <= 0 || instances <= 0 || launch_repeats <= 0) {
        out << "kernel_iterations, instances, launch_repeats must be > 0\n";
        return out.str();
    }

    out << "=== ECM mont mul/sqr microbench (unroll_i24) ===\n";
    out << bits << "-bit, limbs=" << words << ", kernel_iterations=" << kernel_iterations
        << ", instances=" << instances << ", launch_repeats=" << launch_repeats << "\n";
    out << "path: L1 ulong | L2 u32 MAC | L4 blsub";
    if (words == 16u) {
        out << " | 384 manual u32_blsub";
    }
    out << "\n";

    OpenCLApi api{};
    bool own_lib = false;
    if (!load_opencl_api(api, own_lib, out)) {
        out << "FAIL: OpenCL not loaded\n";
        return out.str();
    }

    cl_device_id dev = nullptr;
    if (!acquire_gpu_device(api, dev, out)) {
        maybe_unload_opencl_api(api, own_lib);
        out << "FAIL: no GPU\n";
        return out.str();
    }
    out << "device: " << query_device_string(api, dev, CL_DEVICE_NAME) << "\n";

    const std::string src = build_montsqr_source_i24(out);
    if (src.empty()) {
        maybe_unload_opencl_api(api, own_lib);
        out << "FAIL: kernel sources\n";
        return out.str();
    }

    cl_context ctx = nullptr;
    cl_command_queue q = nullptr;
    if (!acquire_opencl_cache_session(api, dev, ctx, q, out)) {
        maybe_unload_opencl_api(api, own_lib);
        return out.str();
    }

    char build_opts[128];
    std::snprintf(build_opts, sizeof(build_opts),
                  "-DMAX_LIMBS=%u -DMP_LIMB_BITS=24 -cl-fast-relaxed-math", words);
    out << "build: MAX_LIMBS=" << words << " MP_LIMB_BITS=24 src_kib=" << (src.size() / 1024u)
        << "\n";

    double compile_ms = 0.0;
    bool cache_hit = false;
    cl_program program = build_opencl_program_cached(
        api, ctx, dev, src.c_str(), src.size(), build_opts, out, compile_ms, cache_hit);
    if (!program) {
        maybe_unload_opencl_api(api, own_lib);
        out << "FAIL: program build\n";
        return out.str();
    }

    std::vector<uint32_t> n_words(words), a_words(words), b_words(words);
    set_pow2_minus_ui_limb24(n_words.data(), words, static_cast<uint32_t>(bits), 109ull);
    set_pow2_minus_ui_limb24(a_words.data(), words, static_cast<uint32_t>(bits), 991ull);
    set_pow2_minus_ui_limb24(b_words.data(), words, static_cast<uint32_t>(bits), 8218291649ull);

    const size_t total_words = static_cast<size_t>(instances) * words;
    std::vector<uint32_t> host_a(total_words), host_b(total_words), host_n(total_words);
    for (int i = 0; i < instances; ++i) {
        std::memcpy(host_a.data() + static_cast<size_t>(i) * words, a_words.data(),
                    sizeof(uint32_t) * words);
        std::memcpy(host_b.data() + static_cast<size_t>(i) * words, b_words.data(),
                    sizeof(uint32_t) * words);
        std::memcpy(host_n.data() + static_cast<size_t>(i) * words, n_words.data(),
                    sizeof(uint32_t) * words);
    }

    const size_t bytes = sizeof(uint32_t) * total_words;
    cl_int err = CL_SUCCESS;
    cl_mem buf_a = api.clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, &err);
    cl_mem buf_b = api.clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, &err);
    cl_mem buf_n = api.clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, &err);
    cl_mem buf_out = api.clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    cl_mem buf_n_const = api.clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(uint32_t) * words, nullptr, &err);
    const cl_uint np0_host = 0u - inv24_odd(n_words[0] | 1u);
    cl_mem buf_np0_const = api.clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(cl_uint), nullptr, &err);
    if (!buf_a || !buf_b || !buf_n || !buf_out || !buf_n_const || !buf_np0_const) {
        out << "buffer alloc failed\n";
        api.clReleaseProgram(program);
        maybe_unload_opencl_api(api, own_lib);
        return out.str();
    }
    api.clEnqueueWriteBuffer(q, buf_a, 1, 0, bytes, host_a.data(), 0, nullptr, nullptr);
    api.clEnqueueWriteBuffer(q, buf_b, 1, 0, bytes, host_b.data(), 0, nullptr, nullptr);
    api.clEnqueueWriteBuffer(q, buf_n, 1, 0, bytes, host_n.data(), 0, nullptr, nullptr);
    api.clEnqueueWriteBuffer(q, buf_n_const, 1, 0, sizeof(uint32_t) * words, n_words.data(), 0,
                             nullptr, nullptr);
    api.clEnqueueWriteBuffer(q, buf_np0_const, 1, 0, sizeof(cl_uint), &np0_host, 0, nullptr,
                             nullptr);

    MontBenchCtx mctx{api,
                      q,
                      program,
                      buf_a,
                      buf_b,
                      buf_n,
                      buf_out,
                      buf_n_const,
                      buf_np0_const,
                      words,
                      np0_host,
                      static_cast<cl_uint>(kernel_iterations),
                      static_cast<size_t>(instances),
                      launch_repeats,
                      4,
                      words,
                      static_cast<double>(instances) * static_cast<double>(kernel_iterations) *
                          launch_repeats,
                      out};

    constexpr MontI24BenchKernel kBaseSpecs[] = {
        {"ecm_mont_mul_unroll_i24_bench", "mont_mul_unroll_i24", true},
        {"ecm_mont_mul_unroll_i24_u32_bench", "mont_mul_unroll_i24_u32", true},
        {"ecm_mont_mul_unroll_i24_blsub_bench", "mont_mul_unroll_i24_blsub", true},
        {"ecm_mont_mul_unroll_i24_u32_blsub_bench", "mont_mul_unroll_i24_u32_blsub", true},
        {"ecm_mont_sqr_unroll_i24_bench", "mont_sqr_unroll_i24", false},
        {"ecm_mont_sqr_unroll_i24_u32_bench", "mont_sqr_unroll_i24_u32", false},
        {"ecm_mont_sqr_unroll_i24_blsub_bench", "mont_sqr_unroll_i24_blsub", false},
        {"ecm_mont_sqr_unroll_i24_u32_blsub_bench", "mont_sqr_unroll_i24_u32_blsub", false},
    };
    constexpr MontI24BenchKernel kManual384Specs[] = {
        {"ecm_mont_mul_unroll_i24_384_manual_bench", "mont_mul_unroll_i24_384_manual", true},
        {"ecm_mont_sqr_unroll_i24_384_manual_bench", "mont_sqr_unroll_i24_384_manual", false},
    };

    std::vector<MontI24BenchKernel> specs(std::begin(kBaseSpecs), std::end(kBaseSpecs));
    if (words == 16u) {
        specs.insert(specs.end(), std::begin(kManual384Specs), std::end(kManual384Specs));
    }
    const int kSpecCount = static_cast<int>(specs.size());

    out << std::fixed << std::setprecision(3);
    int ran = 0;
    out << "\n--- mont_mul ---\n";
    for (const MontI24BenchKernel& entry : specs) {
        if (!entry.is_mul) {
            continue;
        }
        EcmMontSqrBenchKernel spec{
            entry.kernel_name, entry.path_label, entry.is_mul,
            MontDispatch::PrivUnroll, words, 0};
        double ms = 0.0;
        if (run_mont_kernel(mctx, spec, ms)) {
            ++ran;
        }
    }

    out << "\n--- mont_sqr ---\n";
    for (const MontI24BenchKernel& entry : specs) {
        if (entry.is_mul) {
            continue;
        }
        EcmMontSqrBenchKernel spec{
            entry.kernel_name, entry.path_label, entry.is_mul,
            MontDispatch::PrivUnroll, words, 0};
        double ms = 0.0;
        if (run_mont_kernel(mctx, spec, ms)) {
            ++ran;
        }
    }

    out << "\n--- summary ---\n";
    out << "kernels ran: " << ran << " / " << kSpecCount << "\n";
    out << "note: L1=ulong+branchy sub; L2=u32 MAC+branchy; L4=branchless final sub\n";
    out << "note: 384@i24=16 limbs (manual u32_blsub when bits=384); 512@i24=22 limbs\n";
    out << "\nRESULT: " << (ran == kSpecCount ? "PASS" : "FAIL") << "\n";

    api.clReleaseMemObject(buf_a);
    api.clReleaseMemObject(buf_b);
    api.clReleaseMemObject(buf_n);
    api.clReleaseMemObject(buf_out);
    api.clReleaseMemObject(buf_n_const);
    api.clReleaseMemObject(buf_np0_const);
    api.clReleaseProgram(program);
    maybe_unload_opencl_api(api, own_lib);
    return out.str();
}

} // namespace

std::string run_montsqr_bench(int bits, int kernel_iterations, int instances, int launch_repeats,
                              bool use_wg, int tpi, int limb_bits) {
    if (limb_bits == 24) {
        return montsqr_bench_i24(bits, kernel_iterations, instances, launch_repeats);
    }
    if (limb_bits != 32) {
        std::ostringstream out;
        out << "limb_bits must be 24 or 32\n";
        return out.str();
    }

    std::ostringstream out;
    if (bits <= 0 || (bits % 32) != 0 || static_cast<uint32_t>(bits) > kMaxBenchBits) {
        out << "bits must be a positive multiple of 32 and <= " << kMaxBenchBits << "\n";
        return out.str();
    }
    if (kernel_iterations <= 0 || instances <= 0 || launch_repeats <= 0) {
        out << "kernel_iterations, instances, launch_repeats must be > 0\n";
        return out.str();
    }
    if (tpi <= 0) {
        out << "tpi must be > 0\n";
        return out.str();
    }

    const uint32_t words = static_cast<uint32_t>(bits) / 32u;
    out << "=== ECM mont mul/sqr microbench ===\n";
    out << bits << "-bit, kernel_iterations=" << kernel_iterations << ", instances=" << instances
        << ", launch_repeats=" << launch_repeats << ", mode=" << (use_wg ? "wg" : "priv")
        << ", tpi=" << tpi << "\n";
#if defined(__ANDROID__)
    out << "note: Android u32 bench capped at " << kMaxBenchBits
        << " bits; 4096 mont paths omitted when MAX_LIMBS<128\n";
#endif
    out << "note: AMD asm paths omitted on Android\n";

    OpenCLApi api{};
    bool own_lib = false;
    if (!load_opencl_api(api, own_lib, out)) {
        out << "FAIL: OpenCL not loaded\n";
        return out.str();
    }

    cl_device_id dev = nullptr;
    if (!acquire_gpu_device(api, dev, out)) {
        maybe_unload_opencl_api(api, own_lib);
        out << "FAIL: no GPU\n";
        return out.str();
    }
    out << "device: " << query_device_string(api, dev, CL_DEVICE_NAME) << "\n";

    const std::string src = build_montsqr_source(words, use_wg, out);
    if (src.empty()) {
        maybe_unload_opencl_api(api, own_lib);
        out << "FAIL: kernel sources\n";
        return out.str();
    }

    cl_context ctx = nullptr;
    cl_command_queue q = nullptr;
    if (!acquire_opencl_cache_session(api, dev, ctx, q, out)) {
        maybe_unload_opencl_api(api, own_lib);
        return out.str();
    }

    char build_opts[160];
    std::snprintf(build_opts, sizeof(build_opts),
                  "-DMAX_LIMBS=%u -DTPI=%d -DMONT_WG_IMPL=4 -DMONT_WG_IMPL4_UNROLL=2 "
                  "-cl-fast-relaxed-math",
                  words, tpi);
    out << "build: MAX_LIMBS=" << words << " src_kib=" << (src.size() / 1024u) << "\n";

    double compile_ms = 0.0;
    bool cache_hit = false;
    cl_program program = build_opencl_program_cached(
        api, ctx, dev, src.c_str(), src.size(), build_opts, out, compile_ms, cache_hit);
    if (!program) {
        maybe_unload_opencl_api(api, own_lib);
        out << "FAIL: program build\n";
        return out.str();
    }
    if (!cache_hit) {
        out << "note: first compile of mont kernels may take 1-3 min\n";
    }

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
    cl_int err = CL_SUCCESS;
    cl_mem buf_a = api.clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, &err);
    cl_mem buf_b = api.clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, &err);
    cl_mem buf_n = api.clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, &err);
    cl_mem buf_out = api.clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    cl_mem buf_n_const = api.clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(uint32_t) * words, nullptr, &err);
    cl_uint np0_host = 0u - inv32_odd(n_words[0]);
    cl_mem buf_np0_const = api.clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(cl_uint), nullptr, &err);
    if (!buf_a || !buf_b || !buf_n || !buf_out || !buf_n_const || !buf_np0_const) {
        out << "buffer alloc failed\n";
        api.clReleaseProgram(program);
        maybe_unload_opencl_api(api, own_lib);
        return out.str();
    }
    api.clEnqueueWriteBuffer(q, buf_a, 1, 0, bytes, host_a.data(), 0, nullptr, nullptr);
    api.clEnqueueWriteBuffer(q, buf_b, 1, 0, bytes, host_b.data(), 0, nullptr, nullptr);
    api.clEnqueueWriteBuffer(q, buf_n, 1, 0, bytes, host_n.data(), 0, nullptr, nullptr);
    api.clEnqueueWriteBuffer(q, buf_n_const, 1, 0, sizeof(uint32_t) * words, n_words.data(), 0, nullptr, nullptr);
    api.clEnqueueWriteBuffer(q, buf_np0_const, 1, 0, sizeof(cl_uint), &np0_host, 0, nullptr, nullptr);

    MontBenchCtx mctx{api,
                      q,
                      program,
                      buf_a,
                      buf_b,
                      buf_n,
                      buf_out,
                      buf_n_const,
                      buf_np0_const,
                      words,
                      np0_host,
                      static_cast<cl_uint>(kernel_iterations),
                      static_cast<size_t>(instances),
                      launch_repeats,
                      tpi,
                      words,
                      static_cast<double>(instances) * static_cast<double>(kernel_iterations) *
                          launch_repeats,
                      out};

    const std::vector<EcmMontSqrBenchKernel> mul_specs =
        opencl_ecm_montsqr_mul_kernels(words, use_wg);
    const std::vector<EcmMontSqrBenchKernel> sqr_specs =
        opencl_ecm_montsqr_sqr_kernels(words, use_wg);

    if (words == 16u) {
        out << "\n--- planned 512-bit mont paths ---\n";
        print_mont_kernel_plan(out, "mont_mul", mul_specs, words);
        print_mont_kernel_plan(out, "mont_sqr", sqr_specs, words);
    } else {
        out << "\n--- planned mont paths ---\n";
        print_mont_kernel_plan(out, "mont_mul", mul_specs, words);
        print_mont_kernel_plan(out, "mont_sqr", sqr_specs, words);
    }

    out << std::fixed << std::setprecision(3);
    out << "\n--- mont_mul ---\n";
    MontSectionStats mul_stats{};
    run_mont_section(mctx, mul_specs, mul_stats);
    out << "\n--- mont_sqr ---\n";
    MontSectionStats sqr_stats{};
    run_mont_section(mctx, sqr_specs, sqr_stats);
    out << "\n--- summary ---\n";
    out << "mont_mul: " << mul_stats.ran << " ran, " << mul_stats.skipped << " skipped (of "
        << mul_specs.size() << ")\n";
    out << "mont_sqr: " << sqr_stats.ran << " ran, " << sqr_stats.skipped << " skipped (of "
        << sqr_specs.size() << ")\n";
    if (mul_stats.skipped > 0 || sqr_stats.skipped > 0) {
        out << "hint: skipped lines above show clCreateKernel / limb / enqueue reasons\n";
    }
    out << "\nRESULT: PASS\n";

    api.clReleaseMemObject(buf_a);
    api.clReleaseMemObject(buf_b);
    api.clReleaseMemObject(buf_n);
    api.clReleaseMemObject(buf_out);
    api.clReleaseMemObject(buf_n_const);
    api.clReleaseMemObject(buf_np0_const);
    api.clReleaseProgram(program);
    maybe_unload_opencl_api(api, own_lib);
    return out.str();
}
