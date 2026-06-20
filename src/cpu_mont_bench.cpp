/* ──────────────────────────────────────────────────────────────────────────
 *  cpu_mont_bench — standalone CPU Montgomery multiplication benchmark
 *
 *  Build: g++ -O3 -mavx512f -mavx512dq -mavx2 -mfma -mf16c src/cpu_mont_bench.cpp
 *              src/cpu_mont_avx.cpp src/cpu_mont_scalar.cpp
 *              -Iinclude -Isrc -lgmp -o cpu_mont_bench
 *
 *  Run:   cpu_mont_bench --bits 512 --iterations 10000 --instances 64
 * ──────────────────────────────────────────────────────────────────────── */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>

#include "cpu_mont_avx.h"
#include "cpu_mont_scalar.h"

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#else
    #include <pthread.h>
    #include <sched.h>
    #include <unistd.h>
#endif

/* ══════════════════════════════════════════════════════════════════════════
 *  Basic CLI helpers
 * ════════════════════════════════════════════════════════════════════════ */

struct BenchConfig {
    uint32_t bits         = 512;
    uint32_t iterations   = 1000;
    uint32_t instances    = 0;          /* 0 = auto (16 AVX512, 8 AVX2) */
    uint32_t repeats      = 1;
    bool     force_avx2   = false;
    bool     verify       = true;
    bool     verbose      = false;
    bool     sync_barrier = false;      /* barrier-sync every iteration */
    std::string affinity = "auto";      /* "auto" | "none" | "0,2,4,6" */
};

static void print_usage()
{
    std::printf(
        "Usage: cpu_mont_bench --bits <512|1024> [options]\n"
        "Options:\n"
        "  --iterations N       Montgomery multiplications per thread (default: 1000)\n"
        "  --instances N        Batched curves (auto=16 AVX512 / 8 AVX2; rounds up to K multiple)\n"
        "  --repeats N          Launch repeats per timing sample (default: 1)\n"
        "  --affinity MODE      CPU affinity: auto (default), none, or comma-sep logical CPUs\n"
        "  --sync-barrier       Barrier-sync every iteration across threads\n"
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
        else if (a == "--affinity" && i + 1 < argc) { cfg.affinity = argv[++i]; }
        else if (a == "--avx2") { cfg.force_avx2 = true; }
        else if (a == "--sync-barrier") { cfg.sync_barrier = true; }
        else if (a == "--no-verify") { cfg.verify = false; }
        else if (a == "-v" || a == "--verbose") { cfg.verbose = true; }
        else if (a == "-h" || a == "--help") { print_usage(); return false; }
        else { std::printf("Unknown option: %s\n", a.c_str()); print_usage(); return false; }
    }
    // if (cfg.bits != 512 && cfg.bits != 1024) {
    //     std::printf("Error: --bits must be 512 or 1024 (got %u)\n", cfg.bits);
    //     print_usage();
    //     return false;
    // }
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
 *  CPU topology detection
 * ════════════════════════════════════════════════════════════════════════ */

static uint32_t get_logical_cpu_count()
{
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors;
#else
    return (uint32_t)sysconf(_SC_NPROCESSORS_CONF);
#endif
}

#ifdef _WIN32
/* Detect physical core count on Windows by using GetLogicalProcessorInformation.
 * Returns number of physical cores (packages × cores per package, ignoring HT),
 * or falls back to logical count. */
static uint32_t get_physical_core_count()
{
    DWORD len = 0;
    GetLogicalProcessorInformation(nullptr, &len);
    if (len == 0) return get_logical_cpu_count();

    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buf(len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    if (!GetLogicalProcessorInformation(buf.data(), &len))
        return get_logical_cpu_count();

    uint32_t phys_cores = 0;
    for (auto &info : buf) {
        if (info.Relationship == RelationProcessorCore) {
            phys_cores++;
        }
    }
    return phys_cores > 0 ? phys_cores : get_logical_cpu_count();
}
#else
static uint32_t get_physical_core_count()
{
    /* Linux: count unique physical core ids via /sys/devices/system/cpu/cpuN/topology/core_id */
    FILE *f = fopen("/sys/devices/system/cpu/present", "r");
    if (!f) return get_logical_cpu_count();
    uint32_t start, end;
    if (fscanf(f, "%u-%u", &start, &end) != 2) { fclose(f); return get_logical_cpu_count(); }
    fclose(f);

    /* count unique physical core ids */
    std::vector<uint32_t> core_ids;
    char path[256];
    for (uint32_t cpu = start; cpu <= end; cpu++) {
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u/topology/core_id", cpu);
        f = fopen(path, "r");
        if (!f) continue;
        uint32_t core_id;
        if (fscanf(f, "%u", &core_id) == 1) {
            bool found = false;
            for (auto &id : core_ids) if (id == core_id) { found = true; break; }
            if (!found) core_ids.push_back(core_id);
        }
        fclose(f);
    }
    return core_ids.empty() ? get_logical_cpu_count() : (uint32_t)core_ids.size();
}
#endif

/* ══════════════════════════════════════════════════════════════════════════
 *  CPU affinity
 * ════════════════════════════════════════════════════════════════════════ */

#ifdef _WIN32
static bool apply_affinity_any()
{
    return SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)-1) != 0;
}

static bool apply_affinity_core(uint32_t core_idx)
{
    /* core_idx is zero-based physical core index.
     * On Windows, we map it to the first logical processor of that physical core.
     * For simplicity, if core_idx < physical_cores, bind to logical CPU = core_idx
     * (first thread of each physical core). */
    DWORD_PTR mask = (DWORD_PTR)1 << core_idx;
    return SetThreadAffinityMask(GetCurrentThread(), mask) != 0;
}

static bool apply_affinity_list(const std::vector<uint32_t> &cpus, uint32_t thread_idx)
{
    if (thread_idx >= cpus.size()) return false;
    DWORD_PTR mask = (DWORD_PTR)1 << cpus[thread_idx];
    return SetThreadAffinityMask(GetCurrentThread(), mask) != 0;
}

#else
static bool apply_affinity_any()
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int i = 0; i < CPU_SETSIZE; i++)
        CPU_SET(i, &cpuset);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) == 0;
}

static bool apply_affinity_core(uint32_t core_idx)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_idx, &cpuset);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) == 0;
}

static bool apply_affinity_list(const std::vector<uint32_t> &cpus, uint32_t thread_idx)
{
    if (thread_idx >= cpus.size()) return false;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpus[thread_idx], &cpuset);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) == 0;
}
#endif

static std::vector<uint32_t> parse_affinity_list(const std::string &s)
{
    std::vector<uint32_t> result;
    std::string token;
    for (size_t i = 0; i <= s.size(); ++i) {
        if (i == s.size() || s[i] == ',') {
            if (!token.empty()) {
                result.push_back((uint32_t)std::atoi(token.c_str()));
                token.clear();
            }
        } else if (s[i] >= '0' && s[i] <= '9') {
            token += s[i];
        }
    }
    return result;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Seeded LCG for reproducible random fill
 * ════════════════════════════════════════════════════════════════════════ */

static uint32_t lcg_state[4] = { 0x12345678u, 0x9abcdef0u, 0xdeadbeefu, 0xcafebabeu };

static inline uint32_t lcg_rand_u32()
{
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

static void generate_modulus(uint32_t *N, uint32_t *np0_out, uint32_t limbs)
{
    fill_random_limbs(N, limbs);
    N[0] |= 1u;
    N[limbs - 1] |= (1u << 31);
    cpu_mont_np0_compute(np0_out, N, limbs);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Self-test helpers
 * ════════════════════════════════════════════════════════════════════════ */

static bool selftest_one(uint32_t LIM, uint32_t K, uint32_t stride,
                          const uint32_t *N_data, uint32_t np0, bool use_avx512)
{
    std::vector<uint32_t> a(K * stride);
    std::vector<uint32_t> b(K * stride);

    for (uint32_t k = 0; k < K; ++k) {
        fill_random_limbs(&a[k * stride], LIM);
        fill_random_limbs(&b[k * stride], LIM);
        for (uint32_t j = 0; j < LIM; ++j) {
            a[k * stride + j] &= N_data[j];
            b[k * stride + j] &= N_data[j];
        }
    }

    /* scalar reference */
    std::vector<uint32_t> ref(K * stride);
    for (uint32_t k = 0; k < K; ++k) {
        cpu_mont_scalar_cios(&ref[k * stride], &a[k * stride], &b[k * stride],
                             N_data, np0, LIM);
    }

    /* AVX batch */
    std::vector<uint32_t> out(K * stride);
    if (use_avx512) {
        avx512_mont_cios_batch(out.data(), a.data(), b.data(),
                                N_data, np0, LIM, stride, K);
    } else {
        avx2_mont_cios_batch(out.data(), a.data(), b.data(),
                              N_data, np0, LIM, stride, K);
    }

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

static bool run_selftest(uint32_t LIM, uint32_t K, uint32_t stride,
                          const uint32_t *N_data, uint32_t np0,
                          bool use_avx512, int thread_idx, int num_tests)
{
    char label[64];
    // if (thread_idx >= 0)
    //     snprintf(label, sizeof(label), "  Self-test thread %d (%s, %u curves, %u-bit)... ",
    //              thread_idx, use_avx512 ? "AVX512" : "AVX2", K, LIM * 32);
    // else
    //     snprintf(label, sizeof(label), "  Self-test (%s, %u curves, %u-bit)... ",
    //              use_avx512 ? "AVX512" : "AVX2", K, LIM * 32);

    std::printf("%s", label);
    fflush(stdout);

    lcg_state[0] = 0x11111111u + (thread_idx >= 0 ? (uint32_t)thread_idx : 0u);
    lcg_state[1] = 0x22222222u;
    lcg_state[2] = 0x33333333u;
    lcg_state[3] = 0x44444444u;

    for (int t = 0; t < num_tests; ++t) {
        if (!selftest_one(LIM, K, stride, N_data, np0, use_avx512)) {
            std::printf("FAILED at test %d\n", t);
            return false;
        }
    }
    // std::printf("PASSED (%d tests)\n", num_tests);
    return true;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Benchmarking data structures
 * ════════════════════════════════════════════════════════════════════════ */

struct ThreadResult {
    double elapsed_sec = 0.0;
    uint64_t ops = 0;
    bool selftest_passed = true;
};

/* Context passed to each worker thread */
struct ThreadCtx {
    uint32_t thread_idx;
    uint32_t K;            /* instances per thread */
    uint32_t LIM;          /* limbs */
    uint32_t stride;       /* = LIM */
    uint32_t iterations;
    uint32_t repeats;
    bool use_avx512;
    bool verify;
    bool sync_barrier;
    const uint32_t *shared_N;
    uint32_t np0;

    /* worker local buffers */
    std::vector<uint32_t> a;
    std::vector<uint32_t> b;
    std::vector<uint32_t> out;

    /* shared barrier */
    std::atomic<int> *barrier_count;
    int num_threads;
    std::mutex *barrier_mtx;
#ifdef _WIN32
    CONDITION_VARIABLE *barrier_cv;
#else
    pthread_cond_t *barrier_cv;
    pthread_mutex_t *barrier_pmtx;
#endif

    ThreadResult result;
};

static void thread_benchmark(ThreadCtx &ctx)
{
    const uint32_t K   = ctx.K;
    const uint32_t LIM = ctx.LIM;
    const uint32_t stride = ctx.stride;
    const uint32_t *N  = ctx.shared_N;
    uint32_t np0       = ctx.np0;

    /* self-test (each thread uses own seed offset) */
    if (ctx.verify) {
        ctx.result.selftest_passed = run_selftest(LIM, K, stride, N, np0,
                                                   ctx.use_avx512, (int)ctx.thread_idx, 10);
        if (!ctx.result.selftest_passed) return;
    }

    /* warm-up */
    const uint32_t WARMUP = 10;
    for (uint32_t w = 0; w < WARMUP; ++w) {
        if (ctx.use_avx512) {
            avx512_mont_cios_batch(ctx.out.data(), ctx.a.data(), ctx.b.data(),
                                   N, np0, LIM, stride, K);
        } else {
            avx2_mont_cios_batch(ctx.out.data(), ctx.a.data(), ctx.b.data(),
                                  N, np0, LIM, stride, K);
        }
    }

    /* timed run */
    double t0 = now_sec();
    for (uint32_t r = 0; r < ctx.repeats; ++r) {
        for (uint32_t i = 0; i < ctx.iterations; ++i) {
            if (ctx.use_avx512) {
                avx512_mont_cios_batch(ctx.out.data(), ctx.a.data(), ctx.b.data(),
                                       N, np0, LIM, stride, K);
            } else {
                avx2_mont_cios_batch(ctx.out.data(), ctx.a.data(), ctx.b.data(),
                                      N, np0, LIM, stride, K);
            }

            if (ctx.sync_barrier && ctx.num_threads > 1) {
                /* barrier: all threads wait here after each iteration */
                int old_val = (*ctx.barrier_count)++;
                if (old_val + 1 < ctx.num_threads) {
#ifdef _WIN32
                    std::unique_lock<std::mutex> lock(*ctx.barrier_mtx);
                    ctx.barrier_cv; // placeholder for cond var wait
                    /* Wait for last thread to arrive */
                    while (*ctx.barrier_count < ctx.num_threads) {
                        // spin-wait is simpler but less efficient
                        // For brief waits this is acceptable
                    }
#else
                    // Simple spin-barrier for cross-platform
                    while (*ctx.barrier_count < ctx.num_threads) {
                        /* spin */
                    }
#endif
                } else {
                    /* last thread resets barrier */
                    *ctx.barrier_count = 0;
                }
            }
        }
    }
    double t1 = now_sec();

    ctx.result.elapsed_sec = t1 - t0;
    ctx.result.ops = (uint64_t)ctx.iterations * ctx.repeats * K;
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
        std::printf("Physical cores: %u  Logical CPUs: %u\n",
                    get_physical_core_count(), get_logical_cpu_count());
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

    /* Determine instances per thread and total threads */
    const uint32_t K_PER_THREAD = use_avx512 ? 16u : 8u;
    uint32_t total_instances;
    uint32_t num_threads;

    if (cfg.instances == 0) {
        total_instances = K_PER_THREAD;
        num_threads = 1;
    } else {
        total_instances = cfg.instances;
        /* round up to K_PER_THREAD multiple */
        if (total_instances % K_PER_THREAD != 0) {
            uint32_t rounded = ((total_instances + K_PER_THREAD - 1) / K_PER_THREAD) * K_PER_THREAD;
            std::printf("  Warning: --instances %u not a multiple of %u, rounding up to %u\n",
                        total_instances, K_PER_THREAD, rounded);
            total_instances = rounded;
        }
        num_threads = total_instances / K_PER_THREAD;
    }

    /* Validate thread count against physical cores */
    uint32_t phys_cores = get_physical_core_count();
    if (num_threads > phys_cores && cfg.affinity == "auto") {
        std::printf("  Warning: %u threads requested but only %u physical cores detected. "
                    "Performance may degrade.\n", num_threads, phys_cores);
    }

    /* Parse affinity */
    std::vector<uint32_t> affinity_cpus;
    bool affinity_auto = (cfg.affinity == "auto");
    bool affinity_none = (cfg.affinity == "none");
    if (!affinity_auto && !affinity_none) {
        affinity_cpus = parse_affinity_list(cfg.affinity);
        if (affinity_cpus.size() < num_threads) {
            std::printf("Error: --affinity specified %zu CPUs but %u threads needed\n",
                        affinity_cpus.size(), num_threads);
            return 1;
        }
    }

    /* Generate shared modulus */
    lcg_state[0] = 0xaaaaaaaa;
    lcg_state[1] = 0xbbbbbbbb;
    lcg_state[2] = 0xcccccccc;
    lcg_state[3] = 0xdddddddd;

    std::vector<uint32_t> shared_N(LIM);
    uint32_t np0;
    generate_modulus(shared_N.data(), &np0, LIM);

    /* Print header */
    const char *isa_name = use_avx512 ? "AVX512F" : "AVX2";
    std::printf("=== CPU Montgomery Mul Benchmark ===\n");
    std::printf("  ISA:          %s\n", isa_name);
    std::printf("  Bit-width:    %u\n", cfg.bits);
    std::printf("  Limbs:        %u\n", LIM);
    std::printf("  Instances:    %u\n", total_instances);
    std::printf("  Threads:      %u\n", num_threads);
    std::printf("  K/thread:     %u\n", K_PER_THREAD);
    std::printf("  Iterations:   %u\n", cfg.iterations);
    std::printf("  Repeats:      %u\n", cfg.repeats);
    std::printf("  Total batches: %u\n", cfg.iterations * cfg.repeats);
    std::printf("  Affinity:     %s\n", cfg.affinity.c_str());
    if (cfg.sync_barrier) std::printf("  Barrier sync: enabled\n");
    std::printf("\n");

    /* Shared barrier state */
    std::atomic<int> barrier_count(0);
    std::mutex barrier_mtx;
#ifdef _WIN32
    CONDITION_VARIABLE barrier_cv;
#else
    pthread_cond_t barrier_cv = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t barrier_pmtx = PTHREAD_MUTEX_INITIALIZER;
#endif

    /* Prepare thread contexts */
    std::vector<ThreadCtx> thread_ctxs(num_threads);
    std::vector<std::thread> threads;

    for (uint32_t t = 0; t < num_threads; ++t) {
        ThreadCtx &ctx = thread_ctxs[t];
        ctx.thread_idx = t;
        ctx.K = K_PER_THREAD;
        ctx.LIM = LIM;
        ctx.stride = LIM;
        ctx.iterations = cfg.iterations;
        ctx.repeats = cfg.repeats;
        ctx.use_avx512 = use_avx512;
        ctx.verify = cfg.verify;
        ctx.sync_barrier = cfg.sync_barrier;
        ctx.shared_N = shared_N.data();
        ctx.np0 = np0;
        ctx.num_threads = (int)num_threads;
        ctx.barrier_count = &barrier_count;
        ctx.barrier_mtx = &barrier_mtx;
#ifdef _WIN32
        ctx.barrier_cv = &barrier_cv;
#else
        ctx.barrier_cv = &barrier_cv;
        ctx.barrier_pmtx = &barrier_pmtx;
#endif

        /* allocate per-thread buffers */
        ctx.a.resize(K_PER_THREAD * LIM);
        ctx.b.resize(K_PER_THREAD * LIM);
        ctx.out.resize(K_PER_THREAD * LIM);

        /* fill with thread-specific random data */
        lcg_state[0] = 0xaaaaaaaa + t;
        lcg_state[1] = 0xbbbbbbbb;
        lcg_state[2] = 0xcccccccc;
        lcg_state[3] = 0xdddddddd;
        for (uint32_t k = 0; k < K_PER_THREAD; ++k) {
            fill_random_limbs(&ctx.a[k * LIM], LIM);
            fill_random_limbs(&ctx.b[k * LIM], LIM);
            for (uint32_t j = 0; j < LIM; ++j) {
                ctx.a[k * LIM + j] &= shared_N[j];
                ctx.b[k * LIM + j] &= shared_N[j];
            }
        }
    }

    /* Apply main thread affinity */
    if (affinity_auto) {
        if (apply_affinity_core(0)) {
            if (cfg.verbose) std::printf("  Main thread affinity: core 0\n");
        }
    } else if (affinity_none) {
        apply_affinity_any();
    } else {
        apply_affinity_list(affinity_cpus, 0);
    }

    /* self-test (single-threaded, then multi-threaded) */
    if (cfg.verify) {
        /* Single-threaded self-test: one test to verify basic correctness */
        if (!run_selftest(LIM, K_PER_THREAD, LIM, shared_N.data(), np0, use_avx512, -1, 10))
            return 1;
        std::printf("\n");
    }

    /* Launch worker threads (thread 0 is main thread; threads 1..N-1 are spawned) */
    double t_wall_start = now_sec();

    for (uint32_t t = 1; t < num_threads; ++t) {
        threads.emplace_back([&thread_ctxs, &cfg, affinity_auto, affinity_none, &affinity_cpus](uint32_t idx) {
            /* Apply affinity */
            if (affinity_auto) {
                apply_affinity_core(idx);
            } else if (affinity_none) {
                apply_affinity_any();
            } else {
                apply_affinity_list(affinity_cpus, idx);
            }
            /* Priority boost */
#ifdef _WIN32
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
#endif
            thread_benchmark(thread_ctxs[idx]);
        }, t);
    }

    /* Thread 0 runs in main thread */
    {
        ThreadCtx &ctx = thread_ctxs[0];
#ifdef _WIN32
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
#endif
        thread_benchmark(ctx);
    }

    /* Join all threads */
    for (auto &th : threads) {
        th.join();
    }

    double t_wall_end = now_sec();
    double wall_elapsed = t_wall_end - t_wall_start;

    /* Collect results */
    bool all_passed = true;
    double total_thread_elapsed = 0.0;
    uint64_t total_ops = 0;

    for (uint32_t t = 0; t < num_threads; ++t) {
        ThreadResult &res = thread_ctxs[t].result;
        if (!res.selftest_passed) all_passed = false;
        total_ops += res.ops;
        if (res.elapsed_sec > total_thread_elapsed)
            total_thread_elapsed = res.elapsed_sec;
    }

    if (cfg.verify && !all_passed) {
        std::printf("  Self-test FAILED in one or more threads.\n");
        return 1;
    }

    /* Print results */
    std::printf("  === Results ===\n");
    std::printf("  Wall-clock:    %.3f s\n", wall_elapsed);
    std::printf("  Max thread:    %.3f s\n", total_thread_elapsed);
    std::printf("  Total ops:     %llu mont_mul\n", (unsigned long long)total_ops);
    std::printf("  Throughput:    %.3f M mont_mul/s\n", (double)total_ops / total_thread_elapsed / 1e6);

    /* Per-thread stats */
    if (num_threads > 1) {
        std::printf("\n  Per-thread:\n");
        for (uint32_t t = 0; t < num_threads; ++t) {
            ThreadResult &res = thread_ctxs[t].result;
            double thr_mops = (double)res.ops / res.elapsed_sec / 1e6;
            std::printf("    Thread %2u: %7.3f s, %7.3f M/s, %6llu ops",
                        t, res.elapsed_sec, thr_mops, (unsigned long long)res.ops);
            if (cfg.verify) std::printf(" [%s]", res.selftest_passed ? "PASS" : "FAIL");
            std::printf("\n");
        }
    }

    /* Scaling efficiency */
    if (num_threads > 1 && cfg.instances > 0) {
        double efficiency = (total_thread_elapsed > 0)
            ? (double)total_ops / total_thread_elapsed / 1e6 / (num_threads * (double)K_PER_THREAD) * 100.0
            : 0.0;
        /* Actually compute efficiency as: actual / (single_thread_rate * num_threads) */
        /* For simplicity, just show relative scaling info */
        (void)efficiency;
    }

    std::printf("\n");
    return 0;
}