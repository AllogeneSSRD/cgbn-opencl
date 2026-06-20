/* ──────────────────────────────────────────────────────────────────────────
 * cpu_mont_bench — standalone CPU Montgomery multiplication benchmark
 *
 * Build: g++ -O3 -mavx512f -mavx512dq -mavx2 -mfma -mf16c src/cpu_mont_bench.cpp
 *              src/cpu_mont_avx.cpp src/cpu_mont_scalar.cpp
 *              -Iinclude -Isrc -lgmp -o cpu_mont_bench
 *
 * Run:   cpu_mont_bench --bits 512 --iterations 10000
 * ──────────────────────────────────────────────────────────────────────── */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>

#include "cpu_mont_avx.h"
#include "cpu_mont_scalar.h"

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#endif

/* ══════════════════════════════════════════════════════════════════════════
 *  Basic CLI helpers
 * ════════════════════════════════════════════════════════════════════════ */

struct BenchConfig {
    uint32_t bits       = 512;
    uint32_t iterations = 1000;
    uint32_t instances  = 0;       /* 0 = auto (16 AVX512, 8 AVX2) */
    uint32_t repeats    = 1;       /* launch repeats per timing sample */
    bool     force_avx2 = false;
    bool     verify     = true;
    bool     verbose    = false;
};

static void print_usage()
{
    std::printf(
        "Usage: cpu_mont_bench --bits <512|1024> [options]\n"
        "Options:\n"
        "  --iterations N       Montgomery multiplications per timing loop (default: 1000)\n"
        "  --instances N        Batched curves (auto by default: 16 AVX512 / 8 AVX2)\n"
        "  --repeats N          Launch repeats per timing sample (default: 1)\n"
        "  --avx2               Force AVX2 code path (even on AVX512-capable CPU)\n"
        "  --no-verify          Skip correctness self-test\n"
        "  -v, --verbose\n");
}

static bool parse_cli(int argc, char *argv[], BenchConfig &cfg)
{
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bits" && i + 1 < argc) { cfg.bits = (uint32_t)std::atoi(argv[++i]); }
        else if (a == "--iterations" && i + 1 < argc) { cfg.iterations = (uint32_t)std::atoi(argv[++i]); }
        else if (a == "--instances" && i + 1 < argc) { cfg.instances = (uint32_t)std::atoi(argv[++i]); }
        else if (a == "--repeats" && i + 1 < argc) { cfg.repeats = (uint32_t)std::atoi(argv[++i]); }
        else if (a == "--avx2") { cfg.force_avx2 = true; }
        else if (a == "--no-verify") { cfg.verify = false; }
        else if (a == "-v" || a == "--verbose") { cfg.verbose = true; }
        else if (a == "-h" || a == "--help") { print_usage(); return false; }
        else { std::printf("Unknown option: %s\n", a.c_str()); print_usage(); return false; }
    }
    if (cfg.bits != 512 && cfg.bits != 1024) {
        std::printf("Error: --bits must be 512 or 1024 (got %u)\n", cfg.bits);
        print_usage();
        return false;
    }
    return true;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  High-res timer
 * ════════════════════════════════════════════════════════════════════════ */

static double now_sec()
{
#ifdef _WIN32
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
#else
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
#endif
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Test vector generation (simple LCG for benchmarking)
 * ════════════════════════════════════════════════════════════════════════ */

/* Seeded LCG for reproducible random fill */
static uint32_t lcg_state[4] = { 0x12345678u, 0x9abcdef0u, 0xdeadbeefu, 0xcafebabeu };

static inline uint32_t lcg_rand_u32()
{
    /* simple xorshift128+ */
    uint64_t s1 = ((uint64_t)lcg_state[0] << 32) | lcg_state[1];
    uint64_t s0 = ((uint64_t)lcg_state[2] << 32) | lcg_state[3];
    uint64_t result = s0 + s0;
    s1 ^= s1 << 23;
    uint64_t ns = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    lcg_state[0] = (uint32_t)(ns >> 32);
    lcg_state[1] = (uint32_t)(ns);
    lcg_state[2] = (uint32_t)(s0 >> 32);
    lcg_state[3] = (uint32_t)(s0);
    return (uint32_t)(result >> 32);
}

static void fill_random_limbs(uint32_t *dst, uint32_t limbs)
{
    for (uint32_t i = 0; i < limbs; ++i)
        dst[i] = lcg_rand_u32();
}

/* Generate a random odd modulus and its montgomery constant */
static void generate_modulus(uint32_t *N, uint32_t *np0_out, uint32_t limbs)
{
    fill_random_limbs(N, limbs);
    N[0] |= 1u;                    /* make odd */
    N[limbs - 1] |= (1u << 31);   /* ensure top bit set: full-width modulus */
    cpu_mont_np0_compute(np0_out, N, limbs);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Self-test: compare AVX batch against scalar reference
 * ════════════════════════════════════════════════════════════════════════ */

static bool selftest_one(const BenchConfig &cfg, bool use_avx512)
{
    const uint32_t LIM    = cfg.bits / 32;
    const uint32_t K      = (cfg.instances > 0) ? cfg.instances
                          : (use_avx512 ? 16u : 8u);
    const uint32_t stride = LIM;

    /* Allocate SoA buffers */
    std::vector<uint32_t> a(K * stride);
    std::vector<uint32_t> b(K * stride);
    std::vector<uint32_t> N(LIM);
    uint32_t np0;

    /* Generate random data */
    lcg_state[0] = 0x11111111u; /* deterministic seed for test */
    lcg_state[1] = 0x22222222u;
    lcg_state[2] = 0x33333333u;
    lcg_state[3] = 0x44444444u;

    generate_modulus(N.data(), &np0, LIM);
    for (uint32_t k = 0; k < K; ++k) {
        fill_random_limbs(&a[k * stride], LIM);
        fill_random_limbs(&b[k * stride], LIM);
        /* reduce a,b mod N */
        for (uint32_t j = 0; j < LIM; ++j) {
            a[k * stride + j] &= N[j];
            b[k * stride + j] &= N[j];
        }
    }

    /* Compute scalar reference */
    std::vector<uint32_t> ref(K * stride);
    for (uint32_t k = 0; k < K; ++k) {
        cpu_mont_scalar_cios(&ref[k * stride], &a[k * stride], &b[k * stride],
                             N.data(), np0, LIM);
    }

    /* Compute AVX batch */
    std::vector<uint32_t> out(K * stride);
    if (use_avx512) {
        avx512_mont_cios_batch(out.data(), a.data(), b.data(),
                               N.data(), np0, LIM, stride, K);
    } else {
        avx2_mont_cios_batch(out.data(), a.data(), b.data(),
                             N.data(), np0, LIM, stride, K);
    }

    /* Compare */
    for (uint32_t k = 0; k < K; ++k) {
        for (uint32_t j = 0; j < LIM; ++j) {
            if (out[k * stride + j] != ref[k * stride + j]) {
                std::printf("  FAIL: instance %u limb %u: got 0x%08x expected 0x%08x\n",
                            k, j, out[k * stride + j], ref[k * stride + j]);
                return false;
            }
        }
    }
    return true;
}

static bool run_selftest(const BenchConfig &cfg, bool use_avx512)
{
    std::printf("  Self-test (%s, %u curves, %u-bit)... ",
                use_avx512 ? "AVX512" : "AVX2",
                (cfg.instances > 0) ? cfg.instances : (use_avx512 ? 16u : 8u),
                cfg.bits);
    fflush(stdout);

    const int NUM_TESTS = 10;
    for (int t = 0; t < NUM_TESTS; ++t) {
        if (!selftest_one(cfg, use_avx512)) {
            std::printf("FAILED at test %d\n", t);
            return false;
        }
    }
    std::printf("PASSED (%d tests)\n", NUM_TESTS);
    return true;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Benchmark harness
 * ════════════════════════════════════════════════════════════════════════ */

static double benchmark_batch(const BenchConfig &cfg, bool use_avx512)
{
    const uint32_t LIM    = cfg.bits / 32;
    const uint32_t K      = (cfg.instances > 0) ? cfg.instances
                          : (use_avx512 ? 16u : 8u);
    const uint32_t stride = LIM;

    /* Allocate buffers */
    std::vector<uint32_t> a(K * stride);
    std::vector<uint32_t> b(K * stride);
    std::vector<uint32_t> out(K * stride);
    std::vector<uint32_t> N(LIM);
    uint32_t np0;

    /* Generate deterministic data for benchmarking */
    lcg_state[0] = 0xaaaaaaaa;
    lcg_state[1] = 0xbbbbbbbb;
    lcg_state[2] = 0xcccccccc;
    lcg_state[3] = 0xdddddddd;

    generate_modulus(N.data(), &np0, LIM);
    for (uint32_t k = 0; k < K; ++k) {
        fill_random_limbs(&a[k * stride], LIM);
        fill_random_limbs(&b[k * stride], LIM);
        for (uint32_t j = 0; j < LIM; ++j) {
            a[k * stride + j] &= N[j];
            b[k * stride + j] &= N[j];
        }
    }

    /* Warm-up */
    const uint32_t WARMUP = 10;
    for (uint32_t w = 0; w < WARMUP; ++w) {
        if (use_avx512) {
            avx512_mont_cios_batch(out.data(), a.data(), b.data(),
                                   N.data(), np0, LIM, stride, K);
        } else {
            avx2_mont_cios_batch(out.data(), a.data(), b.data(),
                                 N.data(), np0, LIM, stride, K);
        }
    }

    /* Timed run */
    double t0 = now_sec();
    for (uint32_t r = 0; r < cfg.repeats; ++r) {
        for (uint32_t i = 0; i < cfg.iterations; ++i) {
            if (use_avx512) {
                avx512_mont_cios_batch(out.data(), a.data(), b.data(),
                                       N.data(), np0, LIM, stride, K);
            } else {
                avx2_mont_cios_batch(out.data(), a.data(), b.data(),
                                     N.data(), np0, LIM, stride, K);
            }
        }
    }
    double t1 = now_sec();

    return t1 - t0;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Main
 * ════════════════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[])
{
    BenchConfig cfg;
    if (!parse_cli(argc, argv, cfg)) return 1;

    /* CPU feature detection */
    bool have_avx512 = cpu_has_avx512f();
    bool have_avx2   = cpu_has_avx2();
    bool use_avx512  = have_avx512 && !cfg.force_avx2;
    bool use_avx2    = have_avx2;

    if (cfg.verbose) {
        std::printf("CPU features: AVX512F=%d  AVX2=%d\n", (int)have_avx512, (int)have_avx2);
    }

    if (!use_avx512 && !use_avx2) {
        std::printf("Error: CPU does not support AVX2 or AVX512. Cannot run.\n");
        return 1;
    }

    const uint32_t LIM = cfg.bits / 32;
    if (LIM > CPU_MONT_MAX_LIMBS) {
        std::printf("Error: bits=%u exceeds CPU_MONT_MAX_LIMBS=%u (max %u bits)\n",
                    cfg.bits, CPU_MONT_MAX_LIMBS, CPU_MONT_MAX_LIMBS * 32);
        return 1;
    }

    /* Print header */
    const char *isa_name = use_avx512 ? "AVX512F" : "AVX2";
    uint32_t K = (cfg.instances > 0) ? cfg.instances : (use_avx512 ? 16u : 8u);
    std::printf("=== CPU Montgomery Mul Benchmark ===\n");
    std::printf("  ISA:          %s\n", isa_name);
    std::printf("  Bit-width:    %u\n", cfg.bits);
    std::printf("  Limbs:        %u\n", LIM);
    std::printf("  Instances:    %u\n", K);
    std::printf("  Iterations:   %u\n", cfg.iterations);
    std::printf("  Repeats:      %u\n", cfg.repeats);
    std::printf("  Total batches: %u\n", cfg.iterations * cfg.repeats);
    std::printf("\n");

    /* Self-test */
    if (cfg.verify) {
        if (!run_selftest(cfg, use_avx512))
            return 1;
        std::printf("\n");
    }

    /* Benchmark */
    std::printf("  Benchmark running..."); fflush(stdout);
    double elapsed = benchmark_batch(cfg, use_avx512);
    std::printf("\r                      \r"); /* clear line */

    uint64_t total_ops = (uint64_t)cfg.iterations * cfg.repeats * K;
    double ops_per_sec  = (double)total_ops / elapsed;
    double ns_per_batch = elapsed / (double)(cfg.iterations * cfg.repeats) * 1e9;

    std::printf("  === Results ===\n");
    std::printf("  Elapsed:       %.3f s\n", elapsed);
    std::printf("  Total ops:     %llu mont_mul\n", (unsigned long long)total_ops);
    std::printf("  Throughput:    %.3f M mont_mul/s\n", ops_per_sec / 1e6);
    // std::printf("  Latency:       %.1f ns/batch (%u curves)\n", ns_per_batch, K);
    // std::printf("  Latency/curve: %.1f ns\n", ns_per_batch / K);
    std::printf("\n");

    return 0;
}