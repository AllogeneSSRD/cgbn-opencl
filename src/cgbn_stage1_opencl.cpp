// OpenCL ECM Stage 1 host (mirrors test/cgbn_stage1.cu)

#include "cgbn_stage1.h"
#include "ecm.h"
#include "cgbn_opencl.h"
#include "opencl_ecm_debug_utils.h"
#include "opencl_ecm_log.h"

#include <CL/cl.h>
#include <gmp.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <process.h>
#ifndef getpid
#define getpid _getpid
#endif
#else
#include <unistd.h>
#endif
#include <ctime>

#define CARRY_BITS 6
#ifndef MAX_LIMBS
#define MAX_LIMBS 320
#endif

static cgbn::opencl::context_t g_ctx;
static bool g_ctx_ready = false;
static cl_program g_ecm_program = nullptr;
static cl_kernel g_ecm_kernel = nullptr;
static cl_kernel g_ecm_kernel_wg = nullptr;
static uint32_t g_kernel_limbs = 0;
static uint32_t g_kernel_tpi = 0;
static bool g_device_info_printed = false;
static int g_kernel_impl4_unroll = -1;

static int selected_device_index_from_env() {
    const char *v = std::getenv("CGBN_OPENCL_DEVICE_INDEX");
    if (!v || !*v) {
        return 0;
    }
    try {
        return std::stoi(v);
    } catch (...) {
        return 0;
    }
}

static uint32_t requested_tpi_from_env() {
    const char *v = std::getenv("ECM_OPENCL_TPI");
    if (!v || !*v) {
        return 8u;
    }
    char *endp = nullptr;
    unsigned long parsed = std::strtoul(v, &endp, 10);
    if (endp == v || *endp != '\0' || parsed == 0ul || parsed > 32ul) {
        ecm_ts_fprintf(stderr, "OpenCL: invalid ECM_OPENCL_TPI=%s, fallback to 8\n", v);
        return 8u;
    }
    return (uint32_t)parsed;
}

static bool is_power_of_two_u32(uint32_t x) {
    return x != 0u && (x & (x - 1u)) == 0u;
}

static int selected_wg_impl_from_env() {
    constexpr int kDefaultWgImpl = 4;
    int wg_impl = kDefaultWgImpl;
    if (const char *v = std::getenv("ECM_MONT_WG_IMPL")) {
        wg_impl = std::atoi(v);
    }
    if (wg_impl == 2 || wg_impl == 3) {
        ecm_ts_fprintf(stderr,
                       "OpenCL: WG_IMPL=%d removed (only 0/1/4 supported), fallback to WG_IMPL=%d\n",
                       wg_impl, kDefaultWgImpl);
        wg_impl = kDefaultWgImpl;
    } else if (wg_impl != 0 && wg_impl != 1 && wg_impl != 4) {
        ecm_ts_fprintf(stderr, "OpenCL: invalid ECM_MONT_WG_IMPL=%d, fallback to %d\n", wg_impl,
                       kDefaultWgImpl);
        wg_impl = kDefaultWgImpl;
    }
    return wg_impl;
}

static uint32_t choose_effective_tpi(uint32_t limbs) {
    static const uint32_t kChoices[] = {32u, 16u, 8u, 4u, 2u, 1u};
    uint32_t requested = requested_tpi_from_env();
    if (!is_power_of_two_u32(requested)) {
        ecm_ts_fprintf(stderr, "OpenCL: ECM_OPENCL_TPI=%u is not power-of-two, fallback to 8\n",
                       requested);
        requested = 8u;
    }

    uint32_t chosen = 0u;
    for (uint32_t d : kChoices) {
        if (d <= requested && (limbs % d) == 0u) {
            chosen = d;
            break;
        }
    }
    if (chosen == 0u) {
        for (uint32_t d : kChoices) {
            if ((limbs % d) == 0u) {
                chosen = d;
                break;
            }
        }
    }
    if (chosen == 0u) {
        chosen = 1u;
    }
    if (chosen != requested) {
        ecm_ts_fprintf(stderr,
                       "OpenCL: requested TPI=%u incompatible with limbs=%u, using TPI=%u\n",
                       requested, limbs, chosen);
    }
    return chosen;
}

static opencl_dump_ctx_t g_dump_ctx;

struct ecm_ops_profile_counts_t {
    uint64_t kernel_bits_processed = 0;   // bits processed by kernel loop only
    uint64_t double_add_calls = 0;        // per curve, per processed bit
    uint64_t mp_add_mod = 0;
    uint64_t mp_sub_mod = 0;
    uint64_t mont_mul_priv = 0;
    uint64_t mont_sqr_priv = 0;
    uint64_t mont_normalize = 0;
    uint64_t special_mult_ui32 = 0;
    uint64_t mp_shift_left_1_mod = 0;
};

static ecm_ops_profile_counts_t compute_ops_profile_counts(uint64_t kernel_bits_processed,
                                                           uint32_t curves) {
    const uint64_t calls = kernel_bits_processed * (uint64_t)curves;
    ecm_ops_profile_counts_t c;
    c.kernel_bits_processed = kernel_bits_processed;
    c.double_add_calls = calls;
    c.mp_add_mod = calls * 4ull;
    c.mp_sub_mod = calls * 4ull;
    c.mont_mul_priv = calls * 4ull;
    c.mont_sqr_priv = calls * 4ull;
    c.mont_normalize = calls * 8ull;
    c.special_mult_ui32 = calls;
    c.mp_shift_left_1_mod = calls;
    return c;
}

static void emit_ops_profile(const ecm_ops_profile_counts_t &c, uint32_t curves, uint64_t s_num_bits,
                             int batches_complete, float gputime_ms, int verbose) {
    (void)verbose;
    ecm_ts_fprintf(stdout,
            "ECM_PROFILE_OPS: curves=%u s_bits=%llu kernel_bits=%llu batches=%d gputime_ms=%.3f\n",
            curves, (unsigned long long)s_num_bits, (unsigned long long)c.kernel_bits_processed,
            batches_complete, gputime_ms);
    ecm_ts_fprintf(stdout,
            "ECM_PROFILE_OPS: double_add_v2=%llu, mp_add_mod=%llu, mp_sub_mod=%llu, "
            "mont_mul_priv=%llu, mont_sqr_priv=%llu, mont_normalize=%llu, "
            "special_mult_ui32=%llu, mp_shift_left_1_mod=%llu\n",
            (unsigned long long)c.double_add_calls, (unsigned long long)c.mp_add_mod,
            (unsigned long long)c.mp_sub_mod, (unsigned long long)c.mont_mul_priv,
            (unsigned long long)c.mont_sqr_priv, (unsigned long long)c.mont_normalize,
            (unsigned long long)c.special_mult_ui32,
            (unsigned long long)c.mp_shift_left_1_mod);
    fflush(stdout);

    if (!env_flag_enabled("ECM_PROFILE_OPS")) {
        return;
    }
    const char *csv_path = env_string_or_default("ECM_PROFILE_OPS_FILE", "ecm_ops_profile.csv");
    std::ofstream out(csv_path, std::ios::out | std::ios::app);
    if (!out.is_open()) {
        ecm_ts_fprintf(stderr, "ECM_PROFILE_OPS: failed to open %s for append\n", csv_path);
        return;
    }
    if (out.tellp() == std::streampos(0)) {
        out << "curves,s_num_bits,kernel_bits_processed,batches,gputime_ms,double_add_calls,"
               "mp_add_mod,mp_sub_mod,mont_mul_priv,mont_sqr_priv,mont_normalize,"
               "special_mult_ui32,mp_shift_left_1_mod\n";
    }
    out << curves << "," << s_num_bits << "," << c.kernel_bits_processed << "," << batches_complete
        << "," << gputime_ms << "," << c.double_add_calls << "," << c.mp_add_mod << ","
        << c.mp_sub_mod << "," << c.mont_mul_priv << "," << c.mont_sqr_priv << ","
        << c.mont_normalize << "," << c.special_mult_ui32 << "," << c.mp_shift_left_1_mod
        << "\n";
}


static int print_nth_batch(int n) {
    return ((n < 3) || (n < 30 && n % 10 == 0) || (n < 500 && n % 100 == 0) ||
            (n < 5000 && n % 1000 == 0) || (n % 10000 == 0));
}

static void print_opencl_device_info(int device_index, double init_ms) {
    if (!g_ctx_ready) {
        return;
    }

    char dev_name[512] = {0};
    char dev_version[256] = {0};
    char driver_version[256] = {0};
    cl_uint compute_units = 0;
    cl_ulong local_mem = 0;
    size_t max_wg = 0;
    cl_ulong max_mem_alloc = 0;

    clGetDeviceInfo(g_ctx.device, CL_DEVICE_NAME, sizeof(dev_name) - 1, dev_name, nullptr);
    clGetDeviceInfo(g_ctx.device, CL_DEVICE_VERSION, sizeof(dev_version) - 1, dev_version,
                    nullptr);
    clGetDeviceInfo(g_ctx.device, CL_DRIVER_VERSION, sizeof(driver_version) - 1,
                    driver_version, nullptr);
    clGetDeviceInfo(g_ctx.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units),
                    &compute_units, nullptr);
    clGetDeviceInfo(g_ctx.device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem), &local_mem,
                    nullptr);
    clGetDeviceInfo(g_ctx.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg,
                    nullptr);
    clGetDeviceInfo(g_ctx.device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc),
                    &max_mem_alloc, nullptr);

    ecm_ts_fprintf(stdout, "GPU: will use device %d: %s, %s, %u compute units.\n", device_index,
            dev_name, dev_version, compute_units);
    ecm_ts_fprintf(stdout, "GPU: driver %s\n", driver_version);
    ecm_ts_fprintf(stdout,
            "GPU: maxSharedPerBlock = %lu maxThreadsPerBlock = %zu maxMemAllocPerBuffer = %lu\n",
            (unsigned long)local_mem, max_wg, (unsigned long)max_mem_alloc);
    ecm_ts_fprintf(stdout, "GPU: Selection and initialization of the device took %.0fms\n", init_ms);
    fflush(stdout);
}

static std::string opencl_device_vendor_string(cl_device_id device) {
    char vendor[256] = {0};
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor) - 1, vendor, nullptr);
    return std::string(vendor);
}

static int selected_impl4_unroll_for_device(cl_device_id device) {
    if (const char *v = std::getenv("ECM_MONT_WG_IMPL4_UNROLL")) {
        int parsed = std::atoi(v);
        if (parsed == 1 || parsed == 2) {
            return parsed;
        }
        ecm_ts_fprintf(stderr, "OpenCL: invalid ECM_MONT_WG_IMPL4_UNROLL=%d, fallback to auto\n", parsed);
    }
    std::string vendor = opencl_device_vendor_string(device);
    std::string upper = vendor;
    std::transform(upper.begin(), upper.end(), upper.begin(),
                   [](unsigned char c) { return (char)std::toupper(c); });
    if (upper.find("NVIDIA") != std::string::npos) {
        return 1;
    }
    return 2;
}

// Uniform sigma in [1, UINT32_MAX - curves] for ECM_PARAM_BATCH_32BITS_D.
extern "C" uint32_t gpu_pick_random_sigma(uint32_t curves) {
    if (curves == 0 || (uint64_t)curves >= (uint64_t)UINT32_MAX) {
        return 2u;
    }

    gmp_randstate_t rng;
    gmp_randinit_default(rng);
    unsigned long seed = (unsigned long)time(nullptr);
    seed ^= (unsigned long)getpid() * 0x9e3779b9ul;
    seed ^= (unsigned long)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    gmp_randseed_ui(rng, seed);

    mpz_t range, r;
    mpz_init(range);
    mpz_init(r);
    mpz_set_ui(range, 0);
    mpz_setbit(range, 32);
    mpz_sub_ui(range, range, (unsigned long)curves);
    mpz_urandomm(r, rng, range);
    uint32_t sigma = (uint32_t)mpz_get_ui(r) + 1u;

    mpz_clear(range);
    mpz_clear(r);
    gmp_randclear(rng);
    return sigma;
}

extern "C" void gpu_compute_batch_d(mpz_t d_out, uint32_t sigma, const mpz_t N) {
    mpz_t pow2_32, inv;
    mpz_init(pow2_32);
    mpz_init(inv);
    mpz_ui_pow_ui(pow2_32, 2, 32);
    mpz_invert(inv, pow2_32, N);
    mpz_set_ui(d_out, sigma);
    mpz_mul(d_out, d_out, inv);
    mpz_mod(d_out, d_out, N);
    mpz_clear(pow2_32);
    mpz_clear(inv);
}

static void from_mpz(const mpz_t s, uint32_t *x, uint32_t count) {
    size_t words;
    if (mpz_sizeinbase(s, 2) > (size_t)count * 32) {
        ecm_ts_fprintf(stderr, "from_mpz: value does not fit in %u limbs\n", count);
        exit(EXIT_FAILURE);
    }
    mpz_export(x, &words, -1, sizeof(uint32_t), 0, 0, s);
    while (words < count) {
        x[words++] = 0;
    }
}

static void to_mpz(mpz_t r, const uint32_t *x, uint32_t count) {
    mpz_import(r, count, -1, sizeof(uint32_t), 0, 0, x);
}

static uint32_t find_np0(const mpz_t N) {
    mpz_t temp;
    mpz_init(temp);
    mpz_ui_pow_ui(temp, 2, 32);
    if (!mpz_invert(temp, N, temp)) {
        mpz_clear(temp);
        return 0;
    }
    uint32_t np0 = 0u - (uint32_t)mpz_get_ui(temp);
    mpz_clear(temp);
    return np0;
}

static void to_montgomery(uint32_t *out, const mpz_t bn, const mpz_t N, uint32_t bits, uint32_t limbs) {
    mpz_t t;
    mpz_init(t);
    mpz_mul_2exp(t, bn, bits);
    mpz_fdiv_r(t, t, N);
    from_mpz(t, out, limbs);
    mpz_clear(t);
}

// CGBN mont2bn (CIOS reduction), matches cgbn/impl_mpz.cc
static void from_montgomery(mpz_t out, const mpz_t mont, const mpz_t N, uint32_t np0,
                            uint32_t limbs) {
    mpz_t prod, add;
    mpz_init(prod);
    mpz_init(add);
    mpz_set(prod, mont);
    for (uint32_t index = 0; index < limbs; index++) {
        uint32_t low = np0 * (uint32_t)mpz_get_ui(prod);
        mpz_mul_ui(add, N, low);
        mpz_add(prod, prod, add);
        mpz_fdiv_q_2exp(prod, prod, 32);
    }
    if (mpz_cmp(prod, N) < 0) {
        mpz_set(out, prod);
    } else {
        mpz_sub(out, prod, N);
    }
    mpz_clear(prod);
    mpz_clear(add);
}

static void curves_to_montgomery(uint32_t *data, uint32_t curves, uint32_t limbs, const mpz_t N,
                                 uint32_t bits) {
    const uint32_t stride = 5u * limbs;
    mpz_t t;
    mpz_init(t);
    for (uint32_t c = 0; c < curves; c++) {
        uint32_t *datum = data + c * stride;
        for (uint32_t slot = 1; slot <= 4; slot++) {
            to_mpz(t, datum + slot * limbs, limbs);
            to_montgomery(datum + slot * limbs, t, N, bits, limbs);
        }
    }
    mpz_clear(t);
}

static void curves_from_montgomery(uint32_t *data, uint32_t curves, uint32_t limbs, const mpz_t N,
                                   uint32_t np0) {
    const uint32_t stride = 5u * limbs;
    mpz_t t, mont;
    mpz_init(t);
    mpz_init(mont);
    for (uint32_t c = 0; c < curves; c++) {
        uint32_t *datum = data + c * stride;
        for (uint32_t slot = 1; slot <= 4; slot++) {
            to_mpz(mont, datum + slot * limbs, limbs);
            from_montgomery(t, mont, N, np0, limbs);
            from_mpz(t, datum + slot * limbs, limbs);
        }
    }
    mpz_clear(t);
    mpz_clear(mont);
}

static int selftest_opencl_mont_mul(const mpz_t N, uint32_t bits, uint32_t np0) {
    const uint32_t limbs = bits / 32;
    std::string mont_src = cgbn::opencl::load_text_file("cgbn/backends/opencl/kernels/mont.cl");
    if (mont_src.empty()) {
        return -1;
    }
    char opts[64];
    snprintf(opts, sizeof(opts), "-DMAX_LIMBS=%u", limbs);
    cl_int buildErr = CL_SUCCESS;
    cl_program prog =
        cgbn::opencl::build_program_from_source(g_ctx, mont_src.c_str(), opts, buildErr);
    if (prog == nullptr || buildErr != CL_SUCCESS) {
        return -1;
    }
    cl_int err;
    cl_kernel kMul = clCreateKernel(prog, "cgbn_mont_mul", &err);
    if (err != CL_SUCCESS) {
        clReleaseProgram(prog);
        return -1;
    }

    uint32_t a[MAX_LIMBS] = {0}, b[MAX_LIMBS] = {0}, n[MAX_LIMBS] = {0}, out[MAX_LIMBS] = {0};
    mpz_t two, three, six, r;
    mpz_init(two);
    mpz_init(three);
    mpz_init(six);
    mpz_init(r);
    mpz_set_ui(two, 2);
    mpz_set_ui(three, 3);
    mpz_mul_ui(six, two, 3);
    to_montgomery(a, two, N, bits, limbs);
    to_montgomery(b, three, N, bits, limbs);
    from_mpz(N, n, limbs);

    cl_mem bufA = clCreateBuffer(g_ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 limbs * sizeof(uint32_t), a, &err);
    cl_mem bufB = clCreateBuffer(g_ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 limbs * sizeof(uint32_t), b, &err);
    cl_mem bufN = clCreateBuffer(g_ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 limbs * sizeof(uint32_t), n, &err);
    cl_mem bufOut = clCreateBuffer(g_ctx.ctx, CL_MEM_WRITE_ONLY, limbs * sizeof(uint32_t),
                                   nullptr, &err);
    clSetKernelArg(kMul, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kMul, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kMul, 2, sizeof(cl_mem), &bufN);
    clSetKernelArg(kMul, 3, sizeof(cl_mem), &bufOut);
    clSetKernelArg(kMul, 4, sizeof(cl_uint), &np0);
    cl_uint limbs_arg = limbs;
    clSetKernelArg(kMul, 5, sizeof(cl_uint), &limbs_arg);
    size_t g = 1;
    clEnqueueNDRangeKernel(g_ctx.queue, kMul, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(g_ctx.queue, bufOut, CL_TRUE, 0, limbs * sizeof(uint32_t), out, 0,
                        nullptr, nullptr);
    to_mpz(r, out, limbs);
    from_montgomery(r, r, N, np0, limbs);
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

static int selftest_montgomery(const mpz_t N, uint32_t bits) {
    const uint32_t limbs = bits / 32;
    uint32_t buf[MAX_LIMBS] = {0};
    mpz_t two, mont_mpz, back;
    mpz_init(two);
    mpz_init(mont_mpz);
    mpz_init(back);
    mpz_set_ui(two, 2);
    to_montgomery(buf, two, N, bits, limbs);
    to_mpz(mont_mpz, buf, limbs);
    from_montgomery(back, mont_mpz, N, find_np0(N), limbs);
    int ok = (mpz_cmp(back, two) == 0);
    if (!ok) {
        ecm_ts_fprintf(stderr, "GPU: Montgomery self-test failed (2 -> mont -> 2)\n");
    }
    mpz_clear(two);
    mpz_clear(mont_mpz);
    mpz_clear(back);
    return ok ? 0 : -1;
}

static uint32_t *allocate_and_set_s_bits(const mpz_t s, uint64_t *nbits) {
    uint64_t num_bits = *nbits = mpz_sizeinbase(s, 2);
    uint64_t allocated = (num_bits + 31) / 32;
    uint32_t *s_bits = (uint32_t *)malloc(sizeof(uint32_t) * allocated);
    uint64_t countp;
    mpz_export(s_bits, &countp, -1, sizeof(uint32_t), 0, 0, s);
    for (uint64_t i = countp; i < allocated; ++i) {
        s_bits[i] = 0;
    }
    return s_bits;
}

// CUDA set_p_2p: N, P=(2,1), 2P=(9, 64*d+8) in standard form; bn2mont before each GPU batch.
static uint32_t *set_p_2p(const mpz_t N, uint32_t curves, uint32_t sigma, uint32_t BITS,
                          size_t *data_size) {
    const uint32_t limbs_per = BITS / 32;
    *data_size = 5 * curves * limbs_per * sizeof(uint32_t);
    uint32_t *data = (uint32_t *)malloc(*data_size);
    uint32_t *datum = data;

    mpz_t x, t;
    mpz_init(x);
    mpz_init(t);

    for (uint32_t index = 0; index < curves; index++) {
        uint32_t d = sigma + index;

        from_mpz(N, datum + 0 * limbs_per, limbs_per);

        mpz_set_ui(x, 2);
        from_mpz(x, datum + 1 * limbs_per, limbs_per);
        mpz_set_ui(x, 1);
        from_mpz(x, datum + 2 * limbs_per, limbs_per);

        mpz_set_ui(x, 9);
        from_mpz(x, datum + 3 * limbs_per, limbs_per);

        mpz_ui_pow_ui(t, 2, 32);
        mpz_invert(t, t, N);
        mpz_mul_ui(t, t, d);
        mpz_mul_ui(t, t, 64);
        mpz_add_ui(t, t, 8);
        mpz_mod(t, t, N);
        from_mpz(t, datum + 4 * limbs_per, limbs_per);

        datum += 5 * limbs_per;
    }

    mpz_clear(x);
    mpz_clear(t);
    return data;
}

static int findfactor(mpz_t factor, const mpz_t N, const mpz_t x_final, const mpz_t z_final) {
    if (mpz_invert(factor, z_final, N)) {
        mpz_mul(factor, x_final, factor);
        mpz_mod(factor, factor, N);
        return ECM_NO_FACTOR_FOUND;
    }
    mpz_gcd(factor, z_final, N);
    return ECM_FACTOR_FOUND_STEP1;
}

static int process_results(mpz_t *factors, int *array_found, const mpz_t N,
                           const uint32_t *data, uint32_t cgbn_bits, int curves,
                           uint32_t sigma, int verbose) {
    mpz_t modulo, x_std, z_std;
    mpz_init(modulo);
    mpz_init(x_std);
    mpz_init(z_std);

    const bool verify_results = env_flag_enabled("ECM_VERIFY_GPU_RESULTS");
    const bool verify_strict = env_flag_enabled("ECM_VERIFY_GPU_STRICT");

    const uint32_t limbs_per = cgbn_bits / 32;
    const uint32_t np0 = find_np0(N);
    int youpi = ECM_NO_FACTOR_FOUND;
    int errors = 0;
    int verify_errors = 0;

    for (int i = 0; i < curves; i++) {
        const uint32_t *datum = data + (5 * i * limbs_per);

        to_mpz(modulo, datum + 0 * limbs_per, limbs_per);
        if (mpz_cmp(modulo, N) != 0) {
            ecm_ts_fprintf(stderr, "GPU: curve %d modulus mismatch\n", i);
        }

        mpz_t x_mont, z_mont;
        mpz_init(x_mont);
        mpz_init(z_mont);
        to_mpz(x_mont, datum + 1 * limbs_per, limbs_per);
        to_mpz(z_mont, datum + 2 * limbs_per, limbs_per);
        mpz_set(x_std, x_mont);
        mpz_set(z_std, z_mont);

        mpz_clear(x_mont);
        mpz_clear(z_mont);

        if (mpz_cmp_ui(x_std, 2) == 0 && mpz_cmp_ui(z_std, 1) == 0) {
            errors++;
            if (errors < 10) {
                ecm_ts_fprintf(stderr, "GPU: curve %d may not have computed (still at initial point)\n", i);
            }
        }

        array_found[i] = findfactor(factors[i], N, x_std, z_std);
        if (array_found[i] != ECM_NO_FACTOR_FOUND) {
            if (verify_results) {
                mpz_t rem, gcdz;
                mpz_init(rem);
                mpz_init(gcdz);

                mpz_mod(rem, N, factors[i]);
                if (mpz_cmp_ui(rem, 0) != 0) {
                    verify_errors++;
                    ecm_ts_fprintf(stderr,
                                   "GPU verify: curve %d reported invalid factor (N %% factor != 0), sigma=%u\n",
                                   i, sigma + (uint32_t)i);
                    if (verify_strict) {
                        mpz_clear(rem);
                        mpz_clear(gcdz);
                        mpz_clear(modulo);
                        mpz_clear(x_std);
                        mpz_clear(z_std);
                        return ECM_ERROR;
                    }
                }

                mpz_gcd(gcdz, z_std, N);
                if (mpz_cmp(gcdz, factors[i]) != 0) {
                    verify_errors++;
                    ecm_ts_fprintf(stderr,
                                   "GPU verify: curve %d gcd(z,N) mismatch reported factor, sigma=%u\n",
                                   i, sigma + (uint32_t)i);
                    if (verify_strict) {
                        mpz_clear(rem);
                        mpz_clear(gcdz);
                        mpz_clear(modulo);
                        mpz_clear(x_std);
                        mpz_clear(z_std);
                        return ECM_ERROR;
                    }
                }
                mpz_clear(rem);
                mpz_clear(gcdz);
            }
            youpi = array_found[i];
            ecm_ts_fprintf(stderr, "GPU: factor found in Step 1 with curve %d (-sigma %d:%u)\n",
                    i, ECM_PARAM_BATCH_32BITS_D, sigma + (uint32_t)i);
        }
    }

    mpz_clear(modulo);
    mpz_clear(x_std);
    mpz_clear(z_std);

    if (errors > 2) {
        return ECM_ERROR;
    }
    if (verify_results && verify_errors > 0) {
        ecm_ts_fprintf(stderr, "GPU verify: detected %d invalid result(s)\n", verify_errors);
        return ECM_ERROR;
    }
    return youpi;
}

static uint32_t select_bits(size_t n_log2) {
    static const uint32_t candidates[] = {
        512, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096,
        4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192, 8704, 9216};
    for (uint32_t b : candidates) {
        if (n_log2 + CARRY_BITS <= b) {
            return b;
        }
    }
    return 0;
}

static int ensure_ecm_kernel(uint32_t limbs, uint32_t tpi, int verbose, double *device_init_ms) {
    if (!g_ctx_ready) {
        cl_int err = cgbn::opencl::create_context(g_ctx);
        if (err != CL_SUCCESS) {
            ecm_ts_fprintf(stderr, "OpenCL: failed to create context (%d)\n", err);
            return -1;
        }
        g_ctx_ready = true;
    }
    const int impl4_unroll = selected_impl4_unroll_for_device(g_ctx.device);
    if (g_ecm_kernel && g_kernel_limbs == limbs && g_kernel_tpi == tpi &&
        g_kernel_impl4_unroll == impl4_unroll) {
        if (device_init_ms) {
            *device_init_ms = 0.0;
        }
        return 0;
    }

    auto t_init0 = std::chrono::high_resolution_clock::now();
    if (g_ecm_kernel) {
        clReleaseKernel(g_ecm_kernel);
        g_ecm_kernel = nullptr;
    }
    if (g_ecm_kernel_wg) {
        clReleaseKernel(g_ecm_kernel_wg);
        g_ecm_kernel_wg = nullptr;
    }
    if (g_ecm_program) {
        clReleaseProgram(g_ecm_program);
        g_ecm_program = nullptr;
    }
    g_device_info_printed = false;

    std::string mont_priv =
        cgbn::opencl::load_text_file("cgbn/backends/opencl/kernels/mont_priv.cl");
    std::string mont_wg =
        cgbn::opencl::load_text_file("cgbn/backends/opencl/kernels/mont_wg.cl");
    std::string ecm_src =
        cgbn::opencl::load_text_file("cgbn/backends/opencl/kernels/ecm_stage1.cl");
    if (mont_priv.empty() || mont_wg.empty() || ecm_src.empty()) {
        ecm_ts_fprintf(stderr, "OpenCL: failed to load kernel sources (run from project root)\n");
        return -1;
    }

    // Some OpenCL drivers compile from a temporary location and do not resolve
    // relative #include paths reliably. Inline mont_wg.cl explicitly.
    const std::string include_line = "#include \"mont_wg.cl\"";
    size_t inc_pos = ecm_src.find(include_line);
    if (inc_pos != std::string::npos) {
        size_t erase_len = include_line.size();
        if (inc_pos + erase_len < ecm_src.size() &&
            (ecm_src[inc_pos + erase_len] == '\n' || ecm_src[inc_pos + erase_len] == '\r')) {
            while (inc_pos + erase_len < ecm_src.size() &&
                   (ecm_src[inc_pos + erase_len] == '\n' || ecm_src[inc_pos + erase_len] == '\r')) {
                ++erase_len;
            }
        }
        ecm_src.erase(inc_pos, erase_len);
    }
    std::string src = mont_wg + "\n" + mont_priv + "\n" + ecm_src;

    int wg_impl = selected_wg_impl_from_env();
    int stage1_force_normalize = 1;
    int add_mod_fused_unroll = 2;
    if (const char *v = std::getenv("ECM_STAGE1_FORCE_NORMALIZE")) stage1_force_normalize = std::atoi(v);
    if (const char *v = std::getenv("ECM_MP_ADD_MOD_FUSED_UNROLL")) {
        add_mod_fused_unroll = std::atoi(v);
        if (add_mod_fused_unroll != 1 && add_mod_fused_unroll != 2) {
            add_mod_fused_unroll = 2;
        }
    }
    char opts[256];
    snprintf(opts, sizeof(opts),
             "-DMAX_LIMBS=%u -DTPI=%u -DMONT_WG_IMPL=%d -DMONT_WG_IMPL4_UNROLL=%d "
             "-DECM_STAGE1_FORCE_NORMALIZE=%d -DMP_ADD_MOD_FUSED_UNROLL=%d",
             limbs, tpi, wg_impl, impl4_unroll, stage1_force_normalize, add_mod_fused_unroll);

    cl_int buildErr = CL_SUCCESS;
    g_ecm_program = cgbn::opencl::build_program_from_source(g_ctx, src.c_str(), opts, buildErr);
    if (g_ecm_program == nullptr || buildErr != CL_SUCCESS) {
        ecm_ts_fprintf(stderr, "OpenCL: failed to build ecm_stage1.cl\n");
        return -1;
    }

    cl_int err;
    g_ecm_kernel = clCreateKernel(g_ecm_program, "kernel_double_add", &err);
    if (err != CL_SUCCESS) {
        ecm_ts_fprintf(stderr, "OpenCL: kernel_double_add not found (%d)\n", err);
        return -1;
    }
    g_ecm_kernel_wg = clCreateKernel(g_ecm_program, "kernel_double_add_wg", &err);
    if (err != CL_SUCCESS) {
        ecm_ts_fprintf(stderr, "OpenCL: kernel_double_add_wg not found (%d)\n", err);
        return -1;
    }
    g_kernel_limbs = limbs;
    g_kernel_tpi = tpi;
    g_kernel_impl4_unroll = impl4_unroll;
    auto t_init1 = std::chrono::high_resolution_clock::now();
    double init_ms =
        std::chrono::duration<double, std::milli>(t_init1 - t_init0).count();
    if (device_init_ms) {
        *device_init_ms = init_ms;
    }
    if (!g_device_info_printed) {
        print_opencl_device_info(selected_device_index_from_env(), init_ms);
        g_device_info_printed = true;
    }
    ocl_log_verbose(verbose,
                    "OpenCL: built kernel MAX_LIMBS=%u TPI=%u WG_IMPL=%d IMPL4_UNROLL=%d ADDMOD_UNROLL=%d NORM=%d (%.0fms)\n",
                    limbs, tpi, wg_impl, impl4_unroll, add_mod_fused_unroll, stage1_force_normalize, init_ms);
    return 0;
}

extern "C" int gpu_prepare_opencl(size_t n_log2, int verbose) {
    uint32_t BITS = select_bits(n_log2);
    if (BITS == 0) {
        return ECM_ERROR;
    }
    double init_ms = 0.0;
    const uint32_t limbs = BITS / 32;
    const uint32_t tpi = choose_effective_tpi(limbs);
    return ensure_ecm_kernel(limbs, tpi, verbose, &init_ms);
}

extern "C" int cgbn_ecm_stage1(mpz_t *factors, int *array_found, const mpz_t N, const mpz_t s,
                    uint32_t curves, uint32_t *sigma_ptr,
                    unsigned long checkpoint_interval_ms, float *gputime, int verbose) {
    (void)checkpoint_interval_ms;

    uint32_t sigma = *sigma_ptr;
    if (sigma == 0 || (uint64_t)sigma + curves > 0xFFFFFFFFull) {
        ecm_ts_fprintf(stderr, "Invalid sigma/curves range\n");
        return ECM_ERROR;
    }

    uint64_t s_num_bits;
    uint32_t *s_bits = allocate_and_set_s_bits(s, &s_num_bits);
    if (!s_bits) {
        return ECM_ERROR;
    }

    size_t n_log2 = mpz_sizeinbase(N, 2);
    uint32_t BITS = select_bits(n_log2);
    if (BITS == 0) {
        ecm_ts_fprintf(stderr, "No OpenCL kernel large enough for N (%zu bits)\n", n_log2);
        free(s_bits);
        return ECM_ERROR;
    }
    const uint32_t limbs = BITS / 32;
    const uint32_t tpi = choose_effective_tpi(limbs);

    double device_init_ms = 0.0;
    if (ensure_ecm_kernel(limbs, tpi, verbose, &device_init_ms) != 0) {
        free(s_bits);
        return ECM_ERROR;
    }

    const uint32_t np0 = find_np0(N);
    if (selftest_montgomery(N, BITS) != 0) {
        free(s_bits);
        return ECM_ERROR;
    }
    if (selftest_opencl_mont_mul(N, BITS, np0) != 0) {
        ecm_ts_fprintf(stderr, "GPU: warning: mont.cl mul self-test failed\n");
    }
    size_t data_size = 0;
    uint32_t *data = set_p_2p(N, curves, sigma, BITS, &data_size);

    uint64_t s_partial = 1; // first bit handled by initial P / 2P setup (matches CUDA)
    uint64_t batch_size = 200;

    cl_int err;
    uint32_t s_words = (uint32_t)((s_num_bits + 31) / 32);
    cl_mem gpu_s_bits = clCreateBuffer(g_ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(uint32_t) * s_words, s_bits, &err);
    cl_mem gpu_data = clCreateBuffer(g_ctx.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                     data_size, data, &err);

    auto t_global_start = std::chrono::high_resolution_clock::now();

    ecm_ts_fprintf(stdout,
            "GPU: CGBN<%u,%u> kernel, %zu-bit N, %u curves, sigma=%u-%u, s=%llu bits, np0=0x%08x\n",
            BITS, tpi, n_log2, curves, sigma, sigma + curves - 1,
            (unsigned long long)s_num_bits, np0);
    if (device_init_ms > 0.0) {
        ecm_ts_fprintf(stdout, "GPU: kernel compile/build for this limb size took %.0fms\n",
                device_init_ms);
    }
    fflush(stdout);

    curves_to_montgomery(data, curves, limbs, N, BITS);
    err = clEnqueueWriteBuffer(g_ctx.queue, gpu_data, CL_TRUE, 0, data_size, data, 0, nullptr,
                               nullptr);
    if (err != CL_SUCCESS) {
        ecm_ts_fprintf(stderr, "initial GPU upload failed\n");
        free(s_bits);
        free(data);
        clReleaseMemObject(gpu_s_bits);
        clReleaseMemObject(gpu_data);
        return ECM_ERROR;
    }

    const bool use_mont_wg = !env_flag_enabled("ECM_DISABLE_MONT_WG");
    opencl_dump_begin(g_dump_ctx, verbose);

    const bool sync_each_batch = g_dump_ctx.enabled || env_flag_enabled("ECM_SYNC_EACH_BATCH");

    int batches_complete = 0;
    while (s_partial < s_num_bits) {
        uint64_t this_batch = std::min(batch_size, s_num_bits - s_partial);
        if (g_dump_ctx.enabled) {
            size_t words_total = data_size / sizeof(uint32_t);
            std::vector<uint32_t> dump_rows(data, data + words_total);
            curves_from_montgomery(dump_rows.data(), curves, limbs, N, np0);
            dump_opencl_state_rows(g_dump_ctx, "begin", batches_complete, s_partial, this_batch,
                                   sigma, curves, BITS, tpi, dump_rows.data(), limbs);
        }

        if (verbose >= 1 && print_nth_batch(batches_complete)) {
            ecm_ts_fprintf(stderr, "GPU: Computing %llu bits/call, %llu/%llu (%.1f%%)\n",
                    (unsigned long long)this_batch, (unsigned long long)s_partial,
                    (unsigned long long)s_num_bits,
                    100.0 * (double)s_partial / (double)s_num_bits);
        }

        cl_ulong s_num_bits_arg = (cl_ulong)s_num_bits;
        cl_ulong s_start_arg = (cl_ulong)s_partial;
        cl_ulong s_interval_arg = (cl_ulong)this_batch;
        cl_uint count_arg = curves;
        cl_uint sigma_arg = sigma;
        cl_uint np0_arg = np0;
        cl_uint limbs_arg = limbs;

        cl_kernel active_kernel = use_mont_wg ? g_ecm_kernel_wg : g_ecm_kernel;
        err = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), &gpu_s_bits);
        err |= clSetKernelArg(active_kernel, 1, sizeof(cl_ulong), &s_num_bits_arg);
        err |= clSetKernelArg(active_kernel, 2, sizeof(cl_ulong), &s_start_arg);
        err |= clSetKernelArg(active_kernel, 3, sizeof(cl_ulong), &s_interval_arg);
        err |= clSetKernelArg(active_kernel, 4, sizeof(cl_mem), &gpu_data);
        err |= clSetKernelArg(active_kernel, 5, sizeof(cl_uint), &count_arg);
        err |= clSetKernelArg(active_kernel, 6, sizeof(cl_uint), &sigma_arg);
        err |= clSetKernelArg(active_kernel, 7, sizeof(cl_uint), &np0_arg);
        err |= clSetKernelArg(active_kernel, 8, sizeof(cl_uint), &limbs_arg);
        if (use_mont_wg) {
            // kernel_double_add_wg local words:
            // 12*limbs state + MONT_WG_SCRATCH_WORDS + swapped flag
            int wg_impl_runtime = selected_wg_impl_from_env();
            size_t mont_scratch_words =
                (wg_impl_runtime == 0) ? (size_t)(limbs + 1u) : (size_t)(3u * limbs + 1u);
            size_t wg_local_words = (size_t)(12u * limbs) + mont_scratch_words + 1u;
            size_t wg_local_bytes = wg_local_words * sizeof(uint32_t);
            err |= clSetKernelArg(active_kernel, 9, wg_local_bytes, nullptr);
        }
        if (err != CL_SUCCESS) {
            ecm_ts_fprintf(stderr, "clSetKernelArg failed\n");
            break;
        }

        size_t global = use_mont_wg ? (size_t)curves * (size_t)tpi : (size_t)curves;
        size_t local = use_mont_wg ? (size_t)tpi : 0u;
        auto t0 = std::chrono::high_resolution_clock::now();
        err = clEnqueueNDRangeKernel(g_ctx.queue, active_kernel, 1, nullptr, &global,
                                     use_mont_wg ? &local : nullptr, 0, nullptr, nullptr);
        if (err == CL_SUCCESS && sync_each_batch) {
            clFinish(g_ctx.queue);
            err = clEnqueueReadBuffer(g_ctx.queue, gpu_data, CL_TRUE, 0, data_size, data, 0,
                                      nullptr, nullptr);
            if (err == CL_SUCCESS) {
                curves_from_montgomery(data, curves, limbs, N, np0);
                dump_opencl_state_rows(g_dump_ctx, "end", batches_complete + 1,
                                       s_partial + this_batch, this_batch, sigma, curves, BITS,
                                       tpi, data, limbs);
                if (s_partial + this_batch < s_num_bits) {
                    curves_to_montgomery(data, curves, limbs, N, BITS);
                    err = clEnqueueWriteBuffer(g_ctx.queue, gpu_data, CL_TRUE, 0, data_size, data,
                                               0, nullptr, nullptr);
                }
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        if (err != CL_SUCCESS) {
            ecm_ts_fprintf(stderr, "kernel enqueue failed (%d)\n", err);
            break;
        }

        if (sync_each_batch) {
            double batch_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (batch_ms < 80.0) {
                batch_size = 11 * batch_size / 10;
            } else if (batch_ms > 120.0) {
                batch_size = std::max<uint64_t>(100, 9 * batch_size / 10);
            }
        }

        s_partial += this_batch;
        batches_complete++;
    }

    if (err == CL_SUCCESS && !sync_each_batch) {
        err = clEnqueueReadBuffer(g_ctx.queue, gpu_data, CL_TRUE, 0, data_size, data, 0,
                                  nullptr, nullptr);
        if (err == CL_SUCCESS) {
            curves_from_montgomery(data, curves, limbs, N, np0);
        } else {
            ecm_ts_fprintf(stderr, "final GPU readback failed (%d)\n", err);
        }
    }

    auto t_global_end = std::chrono::high_resolution_clock::now();
    if (gputime) {
        *gputime = (float)std::chrono::duration<double, std::milli>(t_global_end - t_global_start)
                       .count();
    }
    const uint64_t kernel_bits_processed = (s_partial > 0u) ? (s_partial - 1u) : 0u;
    if (env_flag_enabled("ECM_PROFILE_OPS")) {
        const float gputime_local = gputime ? *gputime : 0.0f;
        const ecm_ops_profile_counts_t counts =
            compute_ops_profile_counts(kernel_bits_processed, curves);
        emit_ops_profile(counts, curves, s_num_bits, batches_complete, gputime_local, verbose);
    }

    int youpi = ECM_NO_FACTOR_FOUND;
    if (err == CL_SUCCESS) {
        youpi = process_results(factors, array_found, N, data, BITS, (int)curves, sigma, verbose);
    } else {
        youpi = ECM_ERROR;
    }

    clReleaseMemObject(gpu_s_bits);
    clReleaseMemObject(gpu_data);
    opencl_dump_end(g_dump_ctx);
    free(s_bits);
    free(data);

    *sigma_ptr = sigma;
    return youpi;
}
