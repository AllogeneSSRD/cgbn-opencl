// ============================================================================
// cpu_addsub_bench.cpp — portable CPU modular add/sub micro-benchmark.
//
// Runs `cpu_add_mod_unroll_<W>b` and `cpu_sub_mod_unroll_<W>b` for the
// width selected via --bits, plus `cpu_add_mod` (fused fallback) for all
// widths.  Reports ops/s per variant.
//
// Multi-threading & affinity:
//   -t, --threads N          Number of threads (default: 1, or affinity count)
//   -a, --affinity 1,3,5,7   Pin thread t to core c_t
//
// Instances-per-thread (IPT):
//   -i, --ipt N              Instances per thread (default: 16, avx512 lane width)
//   total_instances = ipt * threads
//
// Kernel iterations support scientific notation: -k 1e6
//
// No OpenCL dependency.  Requires GMP for operand generation.
// ============================================================================

#include "cpu_addsub_impl.h"

#include "opencl_ecm_runtime_config.h"

#include <gmp.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#endif

namespace {

constexpr uint32_t MAX_BENCH_BITS  = 16384;
constexpr uint32_t MAX_BENCH_WORDS = MAX_BENCH_BITS / 32;
constexpr int DEFAULT_IPT          = 16; // matches avx2/avx512 lane width

void fill_from_gmp(const mpz_t v, uint32_t *out, size_t words) {
    mpz_t tmp, mod;
    mpz_init(tmp);
    mpz_init(mod);
    mpz_ui_pow_ui(mod, 2, (unsigned long)(words * 32));
    mpz_mod(tmp, v, mod);
    size_t count = 0;
    mpz_export(out, &count, -1, sizeof(uint32_t), 0, 0, tmp);
    for (size_t i = count; i < words; ++i) out[i] = 0u;
    mpz_clear(tmp);
    mpz_clear(mod);
}

struct RunResult {
    std::string label;
    double ms;
    int ipt;
    int threads;
};

// ── Affinity helpers ──────────────────────────────────────────────────

std::vector<int> parse_affinity(const std::string &s) {
    std::vector<int> cores;
    std::istringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try { cores.push_back(std::stoi(token)); }
        catch (...) { return {}; }
    }
    return cores;
}

bool pin_thread_to_core(int core) {
#ifdef _WIN32
    DWORD_PTR mask = 1ull << core;
    return SetThreadAffinityMask(GetCurrentThread(), mask) != 0;
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) == 0;
#endif
}

// ── Function pointer types ───────────────────────────────────────────

typedef void (*add_fn_t)(uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, uint32_t);
typedef int  (*sub_fn_t)(uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, uint32_t);

struct AddSubVariant {
    const char *name;
    add_fn_t add_fn;
    sub_fn_t sub_fn;
};

// ── Multi-threaded benchmark runner ───────────────────────────────────

struct ThreadRange {
    int inst_start; // inclusive
    int inst_end;   // exclusive
    int core;       // -1 if no affinity
};

void thread_addsub_run(bool is_add,
                       add_fn_t add_fn, sub_fn_t sub_fn,
                       uint32_t *r, const uint32_t *a, const uint32_t *b,
                       const uint32_t *N, int iters, int width_words,
                       ThreadRange range, double &thread_ms) {
    if (range.core >= 0) pin_thread_to_core(range.core);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int inst_idx = range.inst_start; inst_idx < range.inst_end; ++inst_idx) {
        uint32_t *r_i = r + inst_idx * width_words;
        const uint32_t *a_i = a + inst_idx * width_words;
        const uint32_t *b_i = b + inst_idx * width_words;
        for (int it = 0; it < iters; ++it) {
            if (is_add) {
                add_fn(r_i, (it == 0) ? a_i : r_i, b_i, N, (uint32_t)width_words);
            } else {
                sub_fn(r_i, (it == 0) ? a_i : r_i, b_i, N, (uint32_t)width_words);
            }
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    thread_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

double run_addsub_mt(const char *label, bool is_add,
                     add_fn_t add_fn, sub_fn_t sub_fn,
                     uint32_t *r_buf, const uint32_t *a_buf, const uint32_t *b_buf,
                     const uint32_t *N, int iters, int ipt, int width_words,
                     int repeats, int num_threads, const std::vector<int> &affinity,
                     std::vector<RunResult> &results) {
    int total_inst = ipt * num_threads;
    double best_ms = 1e12;
    for (int rep = 0; rep < repeats; ++rep) {
        // Reset r to a for each repeat
        std::memcpy(r_buf, a_buf, (size_t)width_words * (size_t)total_inst * sizeof(uint32_t));

        std::vector<ThreadRange> ranges(num_threads);
        int base = 0;
        for (int t = 0; t < num_threads; ++t) {
            int share = (total_inst - base + num_threads - t - 1) / (num_threads - t);
            ranges[t].inst_start = base;
            ranges[t].inst_end   = base + share;
            ranges[t].core       = (t < (int)affinity.size()) ? affinity[t] : -1;
            base += share;
        }

        std::vector<std::thread> threads;
        std::vector<double> thread_times(num_threads, 0.0);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(thread_addsub_run, is_add,
                                 add_fn, sub_fn,
                                 r_buf, a_buf, b_buf, N, iters, width_words,
                                 ranges[t], std::ref(thread_times[t]));
        }
        for (auto &th : threads) th.join();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best_ms) best_ms = ms;
    }
    results.push_back({label, best_ms, ipt, num_threads});
    return best_ms;
}

} // namespace

bool runCpuAddSubBenchmark(int bits, int kernel_iterations, int ipt, int launch_repeats,
                           bool bench_unroll_only,
                           int num_threads, const std::vector<int> &affinity,
                           bool no_overflow = false) {
    if (bits <= 0 || (bits % 32) != 0 || (uint32_t)bits > MAX_BENCH_BITS) {
        std::cerr << "bits must be a positive multiple of 32 and <= " << MAX_BENCH_BITS << std::endl;
        return false;
    }
    const uint32_t WORDS = (uint32_t)bits / 32u;
    const int effective_threads = num_threads > 0 ? num_threads : 1;
    const int total_inst = ipt * effective_threads;
    const bool use_affinity = !affinity.empty();

    std::cout << "CPU add/sub microbench: " << bits << " bits (" << WORDS << " words), "
              << kernel_iterations << " kernel_iters, "
              << ipt << " ipt * " << effective_threads << "t = " << total_inst << " instances, "
              << launch_repeats << " repeats";
    if (use_affinity) {
        std::cout << ", affinity=";
        for (size_t i = 0; i < affinity.size(); ++i) {
            if (i > 0) std::cout << ",";
            std::cout << affinity[i];
        }
    }
    std::cout << "\n\n";

    // Two test cases per bit-width (same seed scheme as OpenCL):
    //   0 = a+b >= N (overflow — fused speculative subtract triggers)
    //   1 = a+b <  N (no-overflow — sum stays below N)
    struct BenchCase {
        const char *label;
        mpz_t a_gmp, b_gmp, n_gmp;
    };
    BenchCase cases[2] = {};
    {
        gmp_randstate_t rng;
        gmp_randinit_default(rng);
        for (int ic = 0; ic < 2; ++ic) {
            gmp_randseed_ui(rng, (unsigned long)((uint32_t)bits * 31337u + (unsigned)ic * 0x9e3779b9u));
            cases[ic].label = (ic == 0) ? "overflow (a+b>=N)" : "no-overflow (a+b<N)";
            mpz_init(cases[ic].a_gmp); mpz_init(cases[ic].b_gmp); mpz_init(cases[ic].n_gmp);
            mpz_t &N = cases[ic].n_gmp, &a = cases[ic].a_gmp, &b = cases[ic].b_gmp;
            mpz_urandomb(N, rng, (unsigned long)bits); mpz_setbit(N, bits-1); mpz_setbit(N, 0);
            if (ic == 0) {  // overflow
                mpz_t half; mpz_init(half); mpz_tdiv_q_ui(half, N, 2u);
                mpz_urandomm(a, rng, half); mpz_add(a, a, half);
                mpz_urandomm(b, rng, N);
                mpz_t sum; mpz_init(sum); mpz_add(sum, a, b);
                if (mpz_cmp(sum, N) < 0) { mpz_sub(b, N, a); mpz_sub_ui(b, b, 1u); }
                mpz_clear(sum); mpz_clear(half);
            } else {  // no-overflow
                mpz_t quar; mpz_init(quar); mpz_tdiv_q_ui(quar, N, 4u);
                mpz_urandomm(a, rng, quar); mpz_urandomm(b, rng, quar);
                mpz_clear(quar);
            }
        }
        gmp_randclear(rng);
    }

    const int case_idx = no_overflow ? 1 : 0;
    std::cout << "  [" << cases[case_idx].label << "]\n";
    mpz_t &a_gmp = cases[case_idx].a_gmp;
    mpz_t &b_gmp = cases[case_idx].b_gmp;
    mpz_t &n_gmp = cases[case_idx].n_gmp;

    size_t buf_words = (size_t)total_inst * WORDS;
    uint32_t *a_buf = new uint32_t[buf_words];
    uint32_t *b_buf = new uint32_t[buf_words];
    uint32_t *n_buf = new uint32_t[WORDS];
    uint32_t *r_buf = new uint32_t[buf_words];
    fill_from_gmp(n_gmp, n_buf, WORDS);
    for (int i = 0; i < total_inst; ++i) {
        fill_from_gmp(a_gmp, a_buf + i * WORDS, WORDS);
        fill_from_gmp(b_gmp, b_buf + i * WORDS, WORDS);
        std::memcpy(r_buf + i * WORDS, a_buf + i * WORDS, WORDS * sizeof(uint32_t));
    }

    std::vector<RunResult> results;

    // ── Build variant registry ───────────────────────────────────────
    std::vector<AddSubVariant> variants;
    variants.push_back({"scalar", cpu_add_fused_scalar, cpu_sub_fused_scalar});
#ifdef CPU_ADDSUB_AVX2
    variants.push_back({"avx2_manual", cpu_add_fused_avx2_manual, cpu_sub_fused_avx2_manual});
    variants.push_back({"avx2_lookahead", cpu_add_fused_avx2_lookahead, cpu_sub_fused_avx2_lookahead});
#endif
#ifdef CPU_ADDSUB_AVX512
    variants.push_back({"avx512_manual", cpu_add_fused_avx512_manual, cpu_sub_fused_avx512_manual});
#endif

    // ── Fused variants ───────────────────────────────────────────────
    if (!bench_unroll_only) {
        for (const auto &v : variants) {
            std::string add_label = std::string("cpu_add_") + v.name;
            std::string sub_label = std::string("cpu_sub_") + v.name;
            run_addsub_mt(add_label.c_str(), true,  v.add_fn, v.sub_fn,
                          r_buf, a_buf, b_buf, n_buf,
                          kernel_iterations, ipt, (int)WORDS,
                          launch_repeats, effective_threads, affinity, results);
            run_addsub_mt(sub_label.c_str(), false, v.add_fn, v.sub_fn,
                          r_buf, a_buf, b_buf, n_buf,
                          kernel_iterations, ipt, (int)WORDS,
                          launch_repeats, effective_threads, affinity, results);
        }
    }

    // ── Width-specific unroll ───────────────────────────────────────
    static const int kWidths[] = {
        192,256,384,512,768,1024,1536,2048,2560,3072,3584,4096
    };
    bool width_matched = false;
    for (int w : kWidths) { if (bits == w) { width_matched = true; break; } }

    if (width_matched) {
        char buf[128];
        snprintf(buf, sizeof(buf), "cpu_add_unroll_%db", bits);
        run_addsub_mt(buf, true, cpu_add_fused_scalar, cpu_sub_fused_scalar,
                      r_buf, a_buf, b_buf, n_buf,
                      kernel_iterations, ipt, (int)WORDS,
                      launch_repeats, effective_threads, affinity, results);
        snprintf(buf, sizeof(buf), "cpu_sub_unroll_%db", bits);
        run_addsub_mt(buf, false, cpu_add_fused_scalar, cpu_sub_fused_scalar,
                      r_buf, a_buf, b_buf, n_buf,
                      kernel_iterations, ipt, (int)WORDS,
                      launch_repeats, effective_threads, affinity, results);
    }

    // ── Output ───────────────────────────────────────────────────────
    const double op_count = (double)kernel_iterations * (double)total_inst * (double)launch_repeats;

    std::cout << "\n  ["
#ifdef CPU_ADDSUB_AVX512
              << "AVX512"
#elif defined(CPU_ADDSUB_AVX2)
              << "AVX2"
#else
              << "scalar"
#endif
              << "]\n";

    for (const auto &r : results) {
        double ops_s = op_count / (r.ms / 1000.0);
        std::cout << "  " << r.label << ": " << r.ms << " ms, " << ops_s << " ops/s";
        if (r.threads > 1) std::cout << " (" << r.threads << "t)";
        std::cout << "\n";
    }

    // ── Optional CSV ─────────────────────────────────────────────────
    const std::string &csv_path = ecm_runtime_config().bench_csv;
    if (!csv_path.empty()) {
        std::ofstream csv(csv_path, std::ios::app);
        if (csv.is_open()) {
            for (const auto &r : results) {
                double ops_s = op_count / (r.ms / 1000.0);
                csv << bits << "," << r.label << "," << r.ms << "," << ops_s << ","
                    << kernel_iterations << "," << r.ipt << "," << launch_repeats
                    << "," << r.threads << "\n";
            }
        }
    }

    delete[] a_buf;
    delete[] b_buf;
    delete[] n_buf;
    delete[] r_buf;
    for (int ic = 0; ic < 2; ++ic) {
        mpz_clear(cases[ic].a_gmp);
        mpz_clear(cases[ic].b_gmp);
        mpz_clear(cases[ic].n_gmp);
    }
    return true;
}

// ── main ────────────────────────────────────────────────────────────────
#ifdef BUILD_CPU_ADDSUB_MAIN

namespace {

/// Parse a CLI value that may be in scientific notation (e.g. 1e6, 5e5).
/// Returns the rounded integer, or the parsed int when no exponent.
bool parse_cli_count(const char *s, const char *label, int &out) {
    if (s == nullptr || *s == '\0') return true;
    try {
        double d = std::stod(s); // handles 1e6, 5e5, 1000
        if (!std::isfinite(d) || d < 0.0) {
            std::cerr << "Invalid " << label << ": " << s << std::endl;
            return false;
        }
        out = (int)(d + 0.5);
        return true;
    } catch (...) {
        std::cerr << "Invalid " << label << ": " << s << std::endl;
        return false;
    }
}

bool parse_cli_int(const char *s, const char *label, int &out) {
    return parse_cli_count(s, label, out);
}

} // namespace

int main(int argc, char **argv) {
    int bits = 1024;
    int kernel_iterations = 1000;
    int ipt = DEFAULT_IPT;
    int launch_repeats = 10;
    bool bench_unroll_only = false;
    bool no_overflow = false;
    int num_threads = 0; // 0 = auto (1 if no affinity, else aff.size())
    std::string affinity_str;

    auto print_usage = [&]() {
        std::cout
            << "Usage: cpu_addsub_bench [options] [bits] [kernel_iterations] [ipt] [repeats]\n"
            << "  Positional args:\n"
            << "    bits                    Benchmark bit width (mult of 32, <= " << MAX_BENCH_BITS << ", default: 1024)\n"
            << "    kernel_iterations       Kernel inner-loop count; supports 1e6 notation (default: 1000)\n"
            << "    ipt                     Instances per thread (default: " << DEFAULT_IPT << ")\n"
            << "    repeats                 Measurement repeats for averaging (default: 10)\n"
            << "  Options:\n"
            << "  -b, --bits <bits>        Alias for 1st positional\n"
            << "  -k, --kernel-iters <N>   Alias for 2nd positional; supports 1e6\n"
            << "  -i, --ipt <N>            Alias for 3rd positional\n"
            << "  -t, --threads <N>        Number of threads (default: 1, or affinity count)\n"
            << "  -r, --repeats <N>        Alias for 4th positional\n"
            << "  -a, --affinity c1,c2,... Pin thread t to core c_t\n"
            << "  --unroll                 Only benchmark unroll_*b width-specific paths\n"
            << "  --no-overflow            Use a+b < N test data (default: a+b >= N)\n"
            << "  --csv <file>             Append results CSV\n"
            << "  -h, --help               Show this help\n"
            << "\nExamples:\n"
            << "  cpu_addsub_bench 512 1e6 16                          # latency + overflow case\n"
            << "  cpu_addsub_bench 512 1e6 16 5 --no-overflow          # latency + no-overflow case\n"
            << "  cpu_addsub_bench 512 5e4 16 5 -t 12 -a 1,3,5,7,9,11,13,15,17,19,21,23\n"
            << "                                                        # throughput: 12 threads, 192 total instances\n";
    };

    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { print_usage(); return EXIT_SUCCESS; }
        if (a == "-b" || a == "--bits") {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << a << "\n"; return EXIT_FAILURE; }
            if (!parse_cli_int(argv[++i], "bits", bits)) return EXIT_FAILURE;
            continue;
        }
        if (a == "-k" || a == "--kernel-iters") {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << a << "\n"; return EXIT_FAILURE; }
            if (!parse_cli_count(argv[++i], "--kernel-iters", kernel_iterations)) return EXIT_FAILURE;
            continue;
        }
        if (a == "-i" || a == "--ipt") {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << a << "\n"; return EXIT_FAILURE; }
            if (!parse_cli_int(argv[++i], "--ipt", ipt)) return EXIT_FAILURE;
            continue;
        }
        if (a == "-t" || a == "--threads") {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << a << "\n"; return EXIT_FAILURE; }
            if (!parse_cli_int(argv[++i], "--threads", num_threads)) return EXIT_FAILURE;
            continue;
        }
        if (a == "-r" || a == "--repeats") {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << a << "\n"; return EXIT_FAILURE; }
            if (!parse_cli_int(argv[++i], "--repeats", launch_repeats)) return EXIT_FAILURE;
            continue;
        }
        if (a == "-a" || a == "--affinity" || a == "--aff") {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << a << "\n"; return EXIT_FAILURE; }
            affinity_str = argv[++i];
            continue;
        }
        if (a == "--unroll") { bench_unroll_only = true; continue; }
        if (a == "--no-overflow") { no_overflow = true; continue; }
        if (a == "--csv") { if (i + 1 < argc) ecm_runtime_config().bench_csv = argv[++i]; continue; }
        if (!a.empty() && a[0] == '-') { std::cerr << "Unknown option: " << a << " (use --help)\n"; return EXIT_FAILURE; }
        pos.push_back(a);
    }
    if (pos.size() >= 1 && !parse_cli_count(pos[0].c_str(), "bits", bits)) return EXIT_FAILURE;
    if (pos.size() >= 2 && !parse_cli_count(pos[1].c_str(), "kernel_iterations", kernel_iterations)) return EXIT_FAILURE;
    if (pos.size() >= 3 && !parse_cli_count(pos[2].c_str(), "ipt", ipt)) return EXIT_FAILURE;
    if (pos.size() >= 4 && !parse_cli_count(pos[3].c_str(), "launch_repeats", launch_repeats)) return EXIT_FAILURE;

    // Resolve threads/affinity
    std::vector<int> affinity_cores = affinity_str.empty() ? std::vector<int>() : parse_affinity(affinity_str);
    if (!affinity_str.empty() && affinity_cores.empty()) {
        std::cerr << "Invalid --affinity format (expect comma-separated ints)\n";
        return EXIT_FAILURE;
    }
    if (num_threads <= 0) num_threads = affinity_cores.empty() ? 1 : (int)affinity_cores.size();

    bool ok = runCpuAddSubBenchmark(bits, kernel_iterations, ipt, launch_repeats,
                                    bench_unroll_only, num_threads, affinity_cores,
                                    no_overflow);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
#endif // BUILD_CPU_ADDSUB_MAIN
