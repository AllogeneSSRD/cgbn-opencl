// OpenCL ECM Stage 1 host (mirrors test/cgbn_stage1.cu)

#include "cgbn_stage1.h"
#include "ecm.h"
#include "cgbn_opencl.h"
#include "opencl_ecm_checkpoint.h"
#include "opencl_ecm_debug_utils.h"
#include "opencl_ecm_log.h"
#include "opencl_ecm_mont.h"
#include "opencl_ecm_path_registry.h"
#include "opencl_ecm_selftest.h"

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

#define CARRY_BITS ECM_STAGE1_MONT_CARRY_BITS
constexpr uint32_t kStage1Container512Limbs = 16u;

static cgbn::opencl::context_t g_ctx;
static bool g_ctx_ready = false;
static cl_program g_ecm_program = nullptr;
static cl_kernel g_ecm_kernel = nullptr;
static uint32_t g_kernel_limbs = 0;
static uint32_t g_kernel_tpi = 0;
static int g_kernel_coop_wg = 1;
static bool g_kernel_use_coop_wg = false;
static EcmStage1KernelBuildPlan g_kernel_build_plan{};
static bool g_device_info_printed = false;

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

static int ensure_opencl_context_ready() {
    if (!g_ctx_ready) {
        cl_int err = cgbn::opencl::create_context(g_ctx);
        if (err != CL_SUCCESS) {
            ecm_ts_fprintf(stderr, "OpenCL: failed to create context (%d)\n", err);
            return -1;
        }
        g_ctx_ready = true;
    }
    return 0;
}

static EcmPathContext make_path_context(uint32_t limbs, cl_device_id dev) {
    EcmPathContext ctx{};
    ctx.limbs = limbs;
    ctx.n_bit_size = 0;
    ctx.container_limbs = limbs;
    ctx.os_mask = ecm_path_host_os_mask();
    ctx.gpu_vendor_mask = 0;
    if (dev != nullptr) {
        char vendor[256] = {};
        clGetDeviceInfo(dev, CL_DEVICE_VENDOR, sizeof(vendor), vendor, nullptr);
        for (char *p = vendor; *p; ++p) {
            *p = (char)std::tolower((unsigned char)*p);
        }
        ctx.gpu_vendor_mask = ecm_path_gpu_vendor_from_cl_vendor_string(vendor);
    }
    return ctx;
}

static bool stage1_need_512_container(const EcmMontPathDescriptor *mul_probe,
                                      const EcmMontPathDescriptor *sqr_probe, size_t n_bit_size,
                                      const char *gpu_add_path, const char *gpu_sub_path,
                                      uint32_t limbs32) {
    const uint32_t probe_limbs = std::max(limbs32, kStage1Container512Limbs);
    EcmPathContext ctx = make_path_context(probe_limbs, g_ctx.device);
    ctx.n_bit_size = n_bit_size;
    const EcmAddSubPathDescriptor *add = opencl_ecm_resolve_addmod_path(gpu_add_path, ctx);
    const EcmAddSubPathDescriptor *sub = opencl_ecm_resolve_submod_path(gpu_sub_path, ctx);
    return (add != nullptr && add->max_container_limbs >= kStage1Container512Limbs) ||
           (sub != nullptr && sub->max_container_limbs >= kStage1Container512Limbs) ||
           (mul_probe != nullptr && mul_probe->dedicated && mul_probe->max_limbs >= 12u) ||
           (sqr_probe != nullptr && sqr_probe->dedicated && sqr_probe->max_limbs >= 12u);
}

static uint32_t stage1_container_limbs(size_t n_bit_size, uint32_t limbs32,
                                       const EcmMontPathDescriptor *mul_probe,
                                       const EcmMontPathDescriptor *sqr_probe,
                                       const char *gpu_add_path, const char *gpu_sub_path) {
    uint32_t limbs = limbs32;
    if (stage1_need_512_container(mul_probe, sqr_probe, n_bit_size, gpu_add_path, gpu_sub_path,
                                  limbs32) &&
        limbs < kStage1Container512Limbs) {
        limbs = kStage1Container512Limbs;
    }
    return limbs;
}

static int resolve_addsub_paths(const char *gpu_add_path, const char *gpu_sub_path, size_t n_bit_size,
                                uint32_t limbs, const EcmAddSubPathDescriptor **add_out,
                                const EcmAddSubPathDescriptor **sub_out) {
    if (ensure_opencl_context_ready() != 0) {
        return -1;
    }
    EcmPathContext ctx = make_path_context(limbs, g_ctx.device);
    ctx.n_bit_size = n_bit_size;
    const EcmAddSubPathDescriptor *add = opencl_ecm_resolve_addmod_path(gpu_add_path, ctx);
    if (add == nullptr) {
        ecm_ts_fprintf(stderr,
                       "OpenCL: unknown --add path '%s' (fused, fused_unroll, fused_unroll_b16, "
                       "fused_unroll_b32, asm_b16, asm_b32, default)\n",
                       gpu_add_path ? gpu_add_path : "");
        return -1;
    }
    const EcmAddSubPathDescriptor *sub = opencl_ecm_resolve_submod_path(gpu_sub_path, ctx);
    if (sub == nullptr) {
        ecm_ts_fprintf(stderr,
                       "OpenCL: unknown --sub path '%s' (fused, fused_unroll, fused_unroll_b16, "
                       "fused_unroll_b32, asm_b32, default)\n",
                       gpu_sub_path ? gpu_sub_path : "");
        return -1;
    }
    if (add->max_container_limbs > 0u && limbs < add->max_container_limbs) {
        ecm_ts_fprintf(stderr, "OpenCL: add path '%s' requires >=%u limbs, got %u limbs\n",
                       add->id, add->max_container_limbs, limbs);
        return -1;
    }
    if (sub->max_container_limbs > 0u && limbs < sub->max_container_limbs) {
        ecm_ts_fprintf(stderr, "OpenCL: sub path '%s' requires >=%u limbs, got %u limbs\n",
                       sub->id, sub->max_container_limbs, limbs);
        return -1;
    }
    *add_out = add;
    *sub_out = sub;
    return 0;
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
    return ((n < 3) ||
            (n < 30 && n % 10 == 0) ||
            (n < 500 && n % 100 == 0) ||
            (n < 5000 && n % 1000 == 0) ||
            (n % 10000 == 0));
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

static void stage1_curves_to_montgomery(uint32_t *data, uint32_t curves, uint32_t limbs,
                                        const mpz_t N, uint32_t bits) {
    const uint32_t stride = 5u * limbs;
    mpz_t t;
    mpz_init(t);
    for (uint32_t c = 0; c < curves; ++c) {
        uint32_t *datum = data + c * stride;
        for (uint32_t slot = 1; slot <= 4; ++slot) {
            ecm_to_mpz(t, datum + slot * limbs, limbs);
            ecm_to_montgomery(datum + slot * limbs, t, N, bits, limbs);
        }
    }
    mpz_clear(t);
}

// CUDA set_p_2p: N, P=(2,1), 2P=(9, 64*d+8) in standard form; bn2mont before each GPU batch.
static uint32_t *set_p_2p(const mpz_t N, uint32_t curves, uint32_t sigma, uint32_t limbs,
                          size_t *data_size) {
    *data_size = 5 * curves * limbs * sizeof(uint32_t);
    uint32_t *data = (uint32_t *)malloc(*data_size);
    uint32_t *datum = data;

    mpz_t x, t;
    mpz_init(x);
    mpz_init(t);

    for (uint32_t index = 0; index < curves; index++) {
        uint32_t d = sigma + index;

        ecm_from_mpz(N, datum + 0 * limbs, limbs);

        mpz_set_ui(x, 2);
        ecm_from_mpz(x, datum + 1 * limbs, limbs);
        mpz_set_ui(x, 1);
        ecm_from_mpz(x, datum + 2 * limbs, limbs);

        mpz_set_ui(x, 9);
        ecm_from_mpz(x, datum + 3 * limbs, limbs);

        mpz_ui_pow_ui(t, 2, 32);
        mpz_invert(t, t, N);
        mpz_mul_ui(t, t, d);
        mpz_mul_ui(t, t, 64);
        mpz_add_ui(t, t, 8);
        mpz_mod(t, t, N);
        ecm_from_mpz(t, datum + 4 * limbs, limbs);

        datum += 5 * limbs;
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
                           const uint32_t *data, uint32_t limbs, int curves, uint32_t sigma,
                           int verbose) {
    mpz_t x_std, z_std;
    mpz_init(x_std);
    mpz_init(z_std);

    const bool verify_results = env_flag_enabled("ECM_VERIFY_GPU_RESULTS");
    const bool verify_strict = env_flag_enabled("ECM_VERIFY_GPU_STRICT");

    int youpi = ECM_NO_FACTOR_FOUND;
    int errors = 0;
    int verify_errors = 0;

    for (int i = 0; i < curves; i++) {
        const uint32_t *datum = data + (5 * i * limbs);

        mpz_t modulo;
        mpz_init(modulo);
        ecm_to_mpz(modulo, datum + 0 * limbs, limbs);
        if (mpz_cmp(modulo, N) != 0) {
            ecm_ts_fprintf(stderr, "GPU: curve %d modulus mismatch\n", i);
        }
        mpz_clear(modulo);

        mpz_t x_mont, z_mont;
        mpz_init(x_mont);
        mpz_init(z_mont);
        ecm_to_mpz(x_mont, datum + 1 * limbs, limbs);
        ecm_to_mpz(z_mont, datum + 2 * limbs, limbs);
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

static void stage1_clamp_mont_desc(const EcmMontPathDescriptor *&mul,
                                 const EcmMontPathDescriptor *&sqr, size_t n_bit_size,
                                 uint32_t limbs) {
    EcmPathContext ctx = make_path_context(limbs, g_ctx.device);
    ctx.n_bit_size = n_bit_size;
    if (mul != nullptr && !ecm_mont_path_fits(mul, ctx.limbs, ctx.os_mask | ctx.gpu_vendor_mask)) {
        if (!ecm_path_limbs_fits(mul->min_limbs, mul->max_limbs, limbs)) {
            ecm_ts_fprintf(stderr,
                           "Warning: mul %s requires limbs [%u,%u], got %u; using %s\n", mul->id,
                           mul->min_limbs, mul->max_limbs, limbs,
                           opencl_ecm_mont_mul_cl_name(
                               opencl_ecm_mont_mul_descriptor(ECM_STAGE1_MONT_UNROLL512)));
            mul = opencl_ecm_mont_mul_descriptor(ECM_STAGE1_MONT_UNROLL512);
        } else if (!ecm_mont_path_fits(mul, ctx.limbs, ctx.os_mask | ctx.gpu_vendor_mask)) {
            ecm_ts_fprintf(stderr,
                           "Warning: mul %s does not fit %u-limb container; using %s\n",
                           mul->id, limbs,
                           opencl_ecm_mont_mul_cl_name(
                               opencl_ecm_mont_mul_descriptor(ECM_STAGE1_MONT_PRIV_OPT)));
            mul = opencl_ecm_mont_mul_descriptor(ECM_STAGE1_MONT_PRIV_OPT);
        }
    }
    if (sqr != nullptr && !ecm_mont_path_fits(sqr, ctx.limbs, ctx.os_mask | ctx.gpu_vendor_mask)) {
        if (!ecm_path_limbs_fits(sqr->min_limbs, sqr->max_limbs, limbs)) {
            ecm_ts_fprintf(stderr,
                           "Warning: sqr %s requires limbs [%u,%u], got %u; using %s\n", sqr->id,
                           sqr->min_limbs, sqr->max_limbs, limbs,
                           opencl_ecm_mont_sqr_cl_name(
                               opencl_ecm_mont_sqr_descriptor(ECM_STAGE1_MONT_UNROLL512)));
            sqr = opencl_ecm_mont_sqr_descriptor(ECM_STAGE1_MONT_UNROLL512);
        } else if (!ecm_mont_path_fits(sqr, ctx.limbs, ctx.os_mask | ctx.gpu_vendor_mask)) {
            ecm_ts_fprintf(stderr,
                           "Warning: sqr %s does not fit %u-limb container; using %s\n",
                           sqr->id, limbs,
                           opencl_ecm_mont_sqr_cl_name(
                               opencl_ecm_mont_sqr_descriptor(ECM_STAGE1_MONT_PRIV_OPT)));
            sqr = opencl_ecm_mont_sqr_descriptor(ECM_STAGE1_MONT_PRIV_OPT);
        }
    }
}

static uint32_t select_bits(size_t n_log2) {
    static const uint32_t candidates[] = {
        256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096,
        4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192, 8704, 9216};
    for (uint32_t b : candidates) {
        if (n_log2 + CARRY_BITS <= b) {
            return b;
        }
    }
    return 0;
}

static int ensure_ecm_kernel(const EcmStage1KernelBuildPlan &plan, int verbose,
                             double *device_init_ms) {
    const uint32_t limbs = plan.limbs;
    const uint32_t tpi = plan.tpi;
    int coop_wg = 1;
    if (limbs == 128u) {
        if (plan.mul != nullptr) {
            coop_wg = std::max(coop_wg, static_cast<int>(plan.mul->coop_work_group_size));
        }
        if (plan.sqr != nullptr) {
            coop_wg = std::max(coop_wg, static_cast<int>(plan.sqr->coop_work_group_size));
        }
    }
    const bool use_coop_wg = (limbs == 128u) && (coop_wg > 1);
    if (!g_ctx_ready) {
        cl_int err = cgbn::opencl::create_context(g_ctx);
        if (err != CL_SUCCESS) {
            ecm_ts_fprintf(stderr, "OpenCL: failed to create context (%d)\n", err);
            return -1;
        }
        g_ctx_ready = true;
    }
    if (g_ecm_kernel && opencl_ecm_stage1_build_plan_equal(g_kernel_build_plan, plan)) {
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
    if (g_ecm_program) {
        clReleaseProgram(g_ecm_program);
        g_ecm_program = nullptr;
    }
    g_device_info_printed = false;

    const std::vector<const char *> kernel_paths = opencl_ecm_stage1_kernel_source_paths(plan);
    std::string src = opencl_ecm_stage1_assemble_kernel_source(
        plan, [](const char *rel_path) { return cgbn::opencl::load_ecm_stage1_kernel_file(rel_path); });
    if (src.empty()) {
        for (const char *rel_path : kernel_paths) {
            if (cgbn::opencl::load_ecm_stage1_kernel_file(rel_path).empty()) {
                ecm_ts_fprintf(stderr,
                               "OpenCL: failed to load %s (set ECM_KERNEL_ROOT or run from project root)\n",
                               rel_path);
                return -1;
            }
        }
        ecm_ts_fprintf(stderr, "OpenCL: failed to assemble stage1 kernel source\n");
        return -1;
    }
    EcmStage1KernelBuildPlan build_plan = plan;
    if (const char *v = std::getenv("ECM_STAGE1_FORCE_NORMALIZE")) {
        build_plan.stage1_force_normalize = std::atoi(v);
    }
    if (const char *v = std::getenv("ECM_MP_ADD_MOD_FUSED_UNROLL")) {
        int fused = std::atoi(v);
        build_plan.add_mod_fused_unroll = (fused == 1 || fused == 2) ? fused : 2;
    }
    const std::string opts = opencl_ecm_stage1_generate_build_options(build_plan);
    if (opts.empty()) {
        ecm_ts_fprintf(stderr, "OpenCL: failed to generate build options\n");
        return -1;
    }

    cl_int buildErr = CL_SUCCESS;
    g_ecm_program = cgbn::opencl::build_program_from_source(g_ctx, src.c_str(), opts.c_str(), buildErr);
    if (g_ecm_program == nullptr || buildErr != CL_SUCCESS) {
        if (g_ecm_program != nullptr) {
            size_t log_size = 0;
            clGetProgramBuildInfo(g_ecm_program, g_ctx.device, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                                  &log_size);
            if (log_size > 1) {
                std::string log(log_size, '\0');
                clGetProgramBuildInfo(g_ecm_program, g_ctx.device, CL_PROGRAM_BUILD_LOG, log_size,
                                      &log[0], nullptr);
                ecm_ts_fprintf(stderr, "OpenCL build log:\n%s\n", log.c_str());
            }
            clReleaseProgram(g_ecm_program);
            g_ecm_program = nullptr;
        }
        ecm_ts_fprintf(stderr, "OpenCL: failed to build ecm_stage1.cl\n");
        return -1;
    }

    cl_int err;
    g_ecm_kernel = clCreateKernel(g_ecm_program, "kernel_double_add", &err);
    if (err != CL_SUCCESS) {
        ecm_ts_fprintf(stderr, "OpenCL: kernel_double_add not found (%d)\n", err);
        return -1;
    }
    g_kernel_limbs = limbs;
    g_kernel_tpi = tpi;
    g_kernel_coop_wg = coop_wg;
    g_kernel_use_coop_wg = use_coop_wg;
    g_kernel_build_plan = build_plan;
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
                    "OpenCL: built kernel MAX_LIMBS=%u TPI=%u ADDMOD_UNROLL=%d NORM=%d "
                    "mul=%s sqr=%s add=%s sub=%s special_mult=%s asm_u32=%d asm_u16=%d coop_wg=%d (%.0fms)\n",
                    limbs, tpi, build_plan.add_mod_fused_unroll, build_plan.stage1_force_normalize,
                    plan.mul != nullptr ? plan.mul->id : "unknown",
                    plan.sqr != nullptr ? plan.sqr->id : "unknown",
                    plan.add != nullptr ? plan.add->id : "unknown",
                    plan.sub != nullptr ? plan.sub->id : "unknown",
                    plan.special_mult != nullptr ? plan.special_mult->id : "unknown",
                    ((plan.add != nullptr && plan.add->kernel_path != nullptr &&
                      strcmp(plan.add->kernel_path, "add_mod/asm_4096b.cl") == 0) ||
                     (plan.sub != nullptr && plan.sub->kernel_path != nullptr &&
                      strcmp(plan.sub->kernel_path, "sub_mod/asm_4096b.cl") == 0))
                        ? 1
                        : 0,
                    ((plan.add != nullptr && plan.add->kernel_path != nullptr &&
                      strcmp(plan.add->kernel_path, "add_mod/asm_512b.cl") == 0) ||
                     (plan.sub != nullptr && plan.sub->kernel_path != nullptr &&
                      strcmp(plan.sub->kernel_path, "sub_mod/asm_512b.cl") == 0))
                        ? 1
                        : 0,
                    use_coop_wg ? coop_wg : 1, init_ms);
    return 0;
}

static int resolve_mont_paths(const char *gpu_mul_path, const char *gpu_sqr_path, size_t n_bit_size,
                            uint32_t container_limbs, const EcmMontPathDescriptor **mul_out,
                            const EcmMontPathDescriptor **sqr_out) {
    EcmPathContext ctx = make_path_context(container_limbs, g_ctx.device);
    ctx.n_bit_size = n_bit_size;
    bool unknown = false;
    const EcmMontPathDescriptor *mul = opencl_ecm_resolve_mont_mul(gpu_mul_path, ctx, &unknown);
    if (unknown) {
        ecm_ts_fprintf(stderr, "OpenCL: unknown --mul path '%s'\n",
                       gpu_mul_path ? gpu_mul_path : "");
        return -1;
    }
    unknown = false;
    const EcmMontPathDescriptor *sqr = opencl_ecm_resolve_mont_sqr(gpu_sqr_path, ctx, &unknown);
    if (unknown) {
        ecm_ts_fprintf(stderr, "OpenCL: unknown --sqr path '%s'\n",
                       gpu_sqr_path ? gpu_sqr_path : "");
        return -1;
    }
    *mul_out = mul;
    *sqr_out = sqr;
    return 0;
}

static uint32_t stage1_ckpt_limbs(size_t data_size, uint32_t curves) {
    return static_cast<uint32_t>(data_size / (5ull * curves * sizeof(uint32_t)));
}

extern "C" int gpu_prepare_opencl(size_t n_log2, int verbose, const char *gpu_mul_path,
                                  const char *gpu_sqr_path, const char *gpu_add_path,
                                  const char *gpu_sub_path) {
    if (ensure_opencl_context_ready() != 0) {
        return ECM_ERROR;
    }
    const uint32_t bits32 = select_bits(n_log2);
    if (bits32 == 0u) {
        return ECM_ERROR;
    }
    const uint32_t limbs32 = bits32 / 32u;

    const EcmMontPathDescriptor *mul_probe = nullptr;
    const EcmMontPathDescriptor *sqr_probe = nullptr;
    if (resolve_mont_paths(gpu_mul_path, gpu_sqr_path, n_log2, 0, &mul_probe, &sqr_probe) != 0) {
        return ECM_ERROR;
    }
    const uint32_t limbs =
        stage1_container_limbs(n_log2, limbs32, mul_probe, sqr_probe, gpu_add_path, gpu_sub_path);

    const EcmMontPathDescriptor *mul = nullptr;
    const EcmMontPathDescriptor *sqr = nullptr;
    if (resolve_mont_paths(gpu_mul_path, gpu_sqr_path, n_log2, limbs, &mul, &sqr) != 0) {
        return ECM_ERROR;
    }
    const EcmAddSubPathDescriptor *add = nullptr;
    const EcmAddSubPathDescriptor *sub = nullptr;
    if (resolve_addsub_paths(gpu_add_path, gpu_sub_path, n_log2, limbs, &add, &sub) != 0) {
        return ECM_ERROR;
    }
    double init_ms = 0.0;
    const uint32_t tpi = choose_effective_tpi(limbs);
    stage1_clamp_mont_desc(mul, sqr, n_log2, limbs);
    EcmPathContext ctx_sm = make_path_context(limbs, g_ctx.device);
    ctx_sm.n_bit_size = n_log2;
    const EcmSpecialMultPathDescriptor *special_mult =
        opencl_ecm_resolve_special_mult(nullptr, ctx_sm);
    const EcmStage1KernelBuildPlan plan =
        opencl_ecm_stage1_make_build_plan(limbs, tpi, mul, sqr, add, sub, special_mult, 1, 2);
    return ensure_ecm_kernel(plan, verbose, &init_ms);
}

extern "C" int cgbn_ecm_stage1(mpz_t *factors, int *array_found, const mpz_t N, const mpz_t s,
                    uint32_t curves, uint32_t *sigma_ptr,
                    unsigned long checkpoint_interval_ms, float *gputime, int verbose,
                    const char *gpu_mul_path, const char *gpu_sqr_path, const char *gpu_add_path,
                    const char *gpu_sub_path) {
    uint32_t sigma = *sigma_ptr;
    if (sigma == 0 || (uint64_t)sigma + curves > 0xFFFFFFFFull) {
        ecm_ts_fprintf(stderr, "Invalid sigma/curves range\n");
        return ECM_ERROR;
    }

    const size_t n_log2 = mpz_sizeinbase(N, 2);

    uint64_t s_num_bits;
    uint32_t *s_bits = allocate_and_set_s_bits(s, &s_num_bits);
    if (!s_bits) {
        return ECM_ERROR;
    }

    const char *ckpt_filename = opencl_ecm_checkpoint_filename(N);
    opencl_ecm_checkpoint_header_t ckpt_header{};
    uint32_t *ckpt_data = nullptr;
    size_t ckpt_data_size = 0;
    int ckpt_loaded = 0;

    if (opencl_ecm_checkpoint_load(ckpt_filename, &ckpt_header, &ckpt_data, &ckpt_data_size) ==
        ECM_NO_FACTOR_FOUND) {
        if (ckpt_header.curves == curves && ckpt_header.s_num_bits == s_num_bits) {
            ecm_ts_fprintf(stderr,
                           "Resuming from checkpoint: %.1f%% complete (s_partial=%llu/%llu)\n",
                           ckpt_header.s_num_bits
                               ? (100.0 * ckpt_header.s_partial / ckpt_header.s_num_bits)
                               : 0.0,
                           (unsigned long long)ckpt_header.s_partial,
                           (unsigned long long)ckpt_header.s_num_bits);
            if (sigma != ckpt_header.sigma) {
                ocl_log_verbose(verbose, "Checkpoint sigma overrides current sigma: %u -> %u\n",
                                sigma, ckpt_header.sigma);
            }
            ckpt_loaded = 1;
            sigma = ckpt_header.sigma;
        } else {
            ecm_ts_fprintf(stderr,
                           "Checkpoint parameters mismatch (curves or s_num_bits differ), "
                           "starting fresh\n");
            free(ckpt_data);
            ckpt_data = nullptr;
        }
    }

    const EcmMontPathDescriptor *mul_probe = nullptr;
    const EcmMontPathDescriptor *sqr_probe = nullptr;
    if (resolve_mont_paths(gpu_mul_path, gpu_sqr_path, n_log2, 0, &mul_probe, &sqr_probe) != 0) {
        free(s_bits);
        return ECM_ERROR;
    }
    if (ensure_opencl_context_ready() != 0) {
        free(s_bits);
        return ECM_ERROR;
    }
    const uint32_t bits32_plan = select_bits(n_log2);
    if (bits32_plan == 0u && !ckpt_loaded) {
        ecm_ts_fprintf(stderr, "No OpenCL kernel large enough for N (%zu bits)\n", n_log2);
        free(s_bits);
        return ECM_ERROR;
    }
    const uint32_t limbs32_plan = bits32_plan / 32u;
    const uint32_t expect_limbs =
        stage1_container_limbs(n_log2, limbs32_plan, mul_probe, sqr_probe, gpu_add_path,
                               gpu_sub_path);
    const uint32_t expect_bits = expect_limbs * 32u;

    uint32_t limbs = 0;
    uint32_t BITS = 0;
    if (ckpt_loaded) {
        limbs = stage1_ckpt_limbs(ckpt_data_size, curves);
        BITS = ckpt_header.BITS;
        if (limbs != expect_limbs || BITS != expect_bits) {
            ecm_ts_fprintf(stderr,
                           "Checkpoint container mismatch (limbs=%u bits=%u, expected limbs=%u "
                           "bits=%u), starting fresh\n",
                           limbs, BITS, expect_limbs, expect_bits);
            ckpt_loaded = 0;
            free(ckpt_data);
            ckpt_data = nullptr;
        }
    }
    if (!ckpt_loaded) {
        limbs = expect_limbs;
        BITS = expect_bits;
    }

    const uint32_t tpi = ckpt_loaded ? ckpt_header.TPI : choose_effective_tpi(limbs);

    const EcmAddSubPathDescriptor *add = nullptr;
    const EcmAddSubPathDescriptor *sub = nullptr;
    if (resolve_addsub_paths(gpu_add_path, gpu_sub_path, n_log2, limbs, &add, &sub) != 0) {
        free(s_bits);
        if (ckpt_data) {
            free(ckpt_data);
        }
        return ECM_ERROR;
    }

    const EcmMontPathDescriptor *mul = nullptr;
    const EcmMontPathDescriptor *sqr = nullptr;
    if (resolve_mont_paths(gpu_mul_path, gpu_sqr_path, n_log2, limbs, &mul, &sqr) != 0) {
        free(s_bits);
        if (ckpt_data) {
            free(ckpt_data);
        }
        return ECM_ERROR;
    }

    stage1_clamp_mont_desc(mul, sqr, n_log2, limbs);
    EcmPathContext ctx_sm2 = make_path_context(limbs, g_ctx.device);
    ctx_sm2.n_bit_size = n_log2;
    const EcmSpecialMultPathDescriptor *special_mult2 =
        opencl_ecm_resolve_special_mult(nullptr, ctx_sm2);
    const EcmStage1KernelBuildPlan build_plan =
        opencl_ecm_stage1_make_build_plan(limbs, tpi, mul, sqr, add, sub, special_mult2, 1, 2);

    double device_init_ms = 0.0;
    if (ensure_ecm_kernel(build_plan, verbose, &device_init_ms) != 0) {
        free(s_bits);
        if (ckpt_data) {
            free(ckpt_data);
        }
        return ECM_ERROR;
    }

    const uint32_t np0 = ecm_find_np0(N);
    if (!ckpt_loaded && opencl_ecm_selftest_montgomery(N, BITS) != 0) {
        free(s_bits);
        return ECM_ERROR;
    }
    if (!ckpt_loaded && opencl_ecm_selftest_mont_mul(g_ctx, N, BITS, np0) != 0) {
        ecm_ts_fprintf(stderr, "GPU: warning: mont.cl mul self-test failed\n");
    }

    size_t data_size = 0;
    uint32_t *data = nullptr;
    uint64_t s_partial = 1;
    int batches_complete = 0;
    if (ckpt_loaded) {
        data = ckpt_data;
        data_size = ckpt_data_size;
        s_partial = ckpt_header.s_partial;
        batches_complete = ckpt_header.batches_complete;
        ckpt_data = nullptr;
        ocl_log_verbose(verbose, "Checkpoint: restored BITS=%u, TPI=%u, limbs=%u\n", BITS, tpi,
                        limbs);
    } else {
        data = set_p_2p(N, curves, sigma, limbs, &data_size);
        s_partial = 1;
        batches_complete = 0;
    }

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
    const char *mont_mul_op = opencl_ecm_mont_mul_cl_name(mul);
    const char *mont_sqr_op = opencl_ecm_mont_sqr_cl_name(sqr);
    const char *add_op = add != nullptr ? add->cl_name : "unknown";
    const char *sub_op = sub != nullptr ? sub->cl_name : "unknown";
    const char *special_op = special_mult2 != nullptr ? special_mult2->cl_name : "special_mult_ui32_generic";
    ecm_ts_fprintf(stdout, "GPU: stage1 operators: mul=%s, sqr=%s, add=%s, sub=%s, special_mult=%s\n",
                   mont_mul_op, mont_sqr_op, add_op, sub_op, special_op);
    if (device_init_ms > 0.0) {
        ecm_ts_fprintf(stdout, "GPU: kernel compile/build for this limb size took %.0fms\n",
                device_init_ms);
    }
    if (checkpoint_interval_ms > 0) {
        ecm_ts_fprintf(stderr, "Checkpoint autosave interval: %.0f seconds\n",
                       checkpoint_interval_ms / 1000.0);
    } else {
        ecm_ts_fprintf(stderr, "Checkpoint autosave disabled\n");
    }
    fflush(stdout);

    if (!ckpt_loaded) {
        stage1_curves_to_montgomery(data, curves, limbs, N, BITS);
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
    } else {
        ocl_log_verbose(verbose, "Checkpoint: GPU curve data restored (%zu bytes, Montgomery form)\n",
                        data_size);
    }

    opencl_dump_begin(g_dump_ctx, verbose);

    const bool sync_each_batch = g_dump_ctx.enabled || env_flag_enabled("ECM_SYNC_EACH_BATCH");

    using steady_clock = std::chrono::steady_clock;
    const auto checkpoint_epoch = steady_clock::now();
    auto last_checkpoint_wall = checkpoint_epoch;
    while (s_partial < s_num_bits) {
        uint64_t this_batch = std::min(batch_size, s_num_bits - s_partial);
        if (g_dump_ctx.enabled) {
            size_t words_total = data_size / sizeof(uint32_t);
            std::vector<uint32_t> dump_rows(data, data + words_total);
            ecm_curves_from_montgomery(dump_rows.data(), curves, limbs, N, np0);
            dump_opencl_state_rows(g_dump_ctx, "begin", batches_complete, s_partial, this_batch,
                                   sigma, curves, BITS, tpi, dump_rows.data(), limbs);
        }

        const bool should_log_batch = (verbose >= 1 && print_nth_batch(batches_complete));

        cl_ulong s_num_bits_arg = (cl_ulong)s_num_bits;
        cl_ulong s_start_arg = (cl_ulong)s_partial;
        cl_ulong s_interval_arg = (cl_ulong)this_batch;
        cl_uint count_arg = curves;
        cl_uint sigma_arg = sigma;
        cl_uint np0_arg = np0;
        cl_uint limbs_arg = limbs;

        cl_kernel active_kernel = g_ecm_kernel;
        err = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), &gpu_s_bits);
        err |= clSetKernelArg(active_kernel, 1, sizeof(cl_ulong), &s_num_bits_arg);
        err |= clSetKernelArg(active_kernel, 2, sizeof(cl_ulong), &s_start_arg);
        err |= clSetKernelArg(active_kernel, 3, sizeof(cl_ulong), &s_interval_arg);
        err |= clSetKernelArg(active_kernel, 4, sizeof(cl_mem), &gpu_data);
        err |= clSetKernelArg(active_kernel, 5, sizeof(cl_uint), &count_arg);
        err |= clSetKernelArg(active_kernel, 6, sizeof(cl_uint), &sigma_arg);
        err |= clSetKernelArg(active_kernel, 7, sizeof(cl_uint), &np0_arg);
        err |= clSetKernelArg(active_kernel, 8, sizeof(cl_uint), &limbs_arg);
        if (err != CL_SUCCESS) {
            ecm_ts_fprintf(stderr, "clSetKernelArg failed\n");
            break;
        }

        size_t global =
            g_kernel_use_coop_wg ? (size_t)curves * (size_t)g_kernel_coop_wg : (size_t)curves;
        size_t local_wg = (size_t)g_kernel_coop_wg;
        const size_t *local_ptr = g_kernel_use_coop_wg ? &local_wg : nullptr;
        auto t0 = std::chrono::high_resolution_clock::now();
        err = clEnqueueNDRangeKernel(g_ctx.queue, active_kernel, 1, nullptr, &global, local_ptr, 0,
                                     nullptr, nullptr);
        if (err != CL_SUCCESS) {
            ecm_ts_fprintf(stderr, "kernel enqueue failed (%d)\n", err);
            break;
        }
        err = clFinish(g_ctx.queue);
        if (err != CL_SUCCESS) {
            ecm_ts_fprintf(stderr, "kernel wait failed (%d)\n", err);
            break;
        }

        if (sync_each_batch) {
            err = clEnqueueReadBuffer(g_ctx.queue, gpu_data, CL_TRUE, 0, data_size, data, 0,
                                      nullptr, nullptr);
            if (err == CL_SUCCESS) {
                ecm_curves_from_montgomery(data, curves, limbs, N, np0);
                dump_opencl_state_rows(g_dump_ctx, "end", batches_complete + 1,
                                       s_partial + this_batch, this_batch, sigma, curves, BITS,
                                       tpi, data, limbs);
                if (s_partial + this_batch < s_num_bits) {
                    stage1_curves_to_montgomery(data, curves, limbs, N, BITS);
                    err = clEnqueueWriteBuffer(g_ctx.queue, gpu_data, CL_TRUE, 0, data_size, data,
                                               0, nullptr, nullptr);
                }
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        const double batch_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const float current_gputime =
            (float)std::chrono::duration<double, std::milli>(t1 - t_global_start).count();
        if (gputime) {
            *gputime = current_gputime;
        }

        if (batch_ms < 80.0) {
            batch_size = 11 * batch_size / 10;
        } else if (batch_ms > 120.0) {
            batch_size = std::max<uint64_t>(100, 9 * batch_size / 10);
        }

        s_partial += this_batch;
        batches_complete++;

        if (checkpoint_interval_ms > 0) {
            const auto now_wall = steady_clock::now();
            const auto since_last_ckpt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                                now_wall - last_checkpoint_wall)
                                                .count();
            if (since_last_ckpt_ms >= (long long)checkpoint_interval_ms) {
                std::vector<uint32_t> ckpt_buf(data_size / sizeof(uint32_t));
                cl_int ckpt_err = clEnqueueReadBuffer(g_ctx.queue, gpu_data, CL_TRUE, 0, data_size,
                                                      ckpt_buf.data(), 0, nullptr, nullptr);
                if (ckpt_err == CL_SUCCESS) {
                    opencl_ecm_checkpoint_header_t header{};
                    header.magic = OPENCL_ECM_CHECKPOINT_MAGIC;
                    header.version = OPENCL_ECM_CHECKPOINT_VERSION;
                    header.s_partial = s_partial;
                    header.s_num_bits = s_num_bits;
                    header.batches_complete = batches_complete;
                    header.curves = curves;
                    header.sigma = sigma;
                    header.BITS = BITS;
                    header.TPI = tpi;
                    header.data_size = (uint64_t)data_size;
                    header.timestamp = (int64_t)time(nullptr);
                    opencl_ecm_checkpoint_save(ckpt_filename, &header, ckpt_buf.data(), data_size);
                    last_checkpoint_wall = now_wall;
                } else {
                    ecm_ts_fprintf(stderr, "Warning: checkpoint GPU readback failed (%d)\n",
                                   ckpt_err);
                }
            }
        }

        if (should_log_batch) {
            double elapsed_s =
                std::chrono::duration<double>(std::chrono::high_resolution_clock::now() -
                                              t_global_start)
                    .count();
            double progress =
                (s_num_bits > 0u) ? ((double)s_partial / (double)s_num_bits) : 0.0;
            const double progress_pct = 100.0 * progress;
            if (progress > 1e-9) {
                double total_s = elapsed_s / progress;
                double remain_s = std::max(0.0, total_s - elapsed_s);
                double total_ms = total_s * 1000.0;
                double per_curve_ms = (curves > 0u) ? (total_ms / (double)curves) : 0.0;
                ecm_ts_fprintf(stderr,
                               "GPU: Computing %llu bits/call, %llu/%llu (%.1f%%), "
                               "ETA %.0f + %.0f = %.0f seconds (~%.0f ms/curves)\n",
                               (unsigned long long)this_batch, (unsigned long long)s_partial,
                               (unsigned long long)s_num_bits, progress_pct,
                               remain_s, elapsed_s, total_s, per_curve_ms);
            } else {
                ecm_ts_fprintf(stderr, "GPU: Computing %llu bits/call, %llu/%llu (%.1f%%)\n",
                               (unsigned long long)this_batch, (unsigned long long)s_partial,
                               (unsigned long long)s_num_bits, progress_pct);
            }
        }
    }

    if (err == CL_SUCCESS && !sync_each_batch) {
        err = clEnqueueReadBuffer(g_ctx.queue, gpu_data, CL_TRUE, 0, data_size, data, 0,
                                  nullptr, nullptr);
        if (err == CL_SUCCESS) {
            ecm_curves_from_montgomery(data, curves, limbs, N, np0);
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
        youpi = process_results(factors, array_found, N, data, limbs, (int)curves, sigma, verbose);
    } else {
        youpi = ECM_ERROR;
    }

    clReleaseMemObject(gpu_s_bits);
    clReleaseMemObject(gpu_data);
    opencl_dump_end(g_dump_ctx);
    free(s_bits);
    free(data);

    if (youpi != ECM_ERROR && opencl_ecm_checkpoint_remove(ckpt_filename) == 0) {
        ocl_log_verbose(verbose, "Checkpoint file removed\n");
    }

    *sigma_ptr = sigma;
    return youpi;
}
