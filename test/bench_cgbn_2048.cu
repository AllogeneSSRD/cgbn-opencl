
// #ifndef __CUDACC__
// #error "This file should only be compiled with nvcc"
// #endif

// GMP import must proceed cgbn.h
#include <stdint.h>
// #include <cuda_runtime.h>
#include <gmp.h>
#include <cgbn.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include <vector>

#define TPI 8
#define BITS 1024

typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;
typedef typename env_t::cgbn_t bn_t;
typedef cgbn_mem_t<BITS> mem_t;

__global__ void bench_add_sub(
    cgbn_error_report_t *report,
    mem_t *a_data,
    mem_t *b_data,
    mem_t *out_add_data,
    mem_t *out_sub_data,
    int iterations);

// Disabled - not needed for add/sub benchmark
/*
uint32_t compute_np0(const mpz_t N) {
    mpz_t mod, inv;
    mpz_init(mod);
    mpz_init(inv);

    // mod = 2^32
    mpz_ui_pow_ui(mod, 2, 32);

    // inv = N^{-1} mod 2^32
    if (mpz_invert(inv, N, mod) == 0) {
        printf("Error: N not invertible mod 2^32\n");
        exit(1);
    }

    uint32_t np0 = (uint32_t)(-mpz_get_ui(inv));

    mpz_clear(mod);
    mpz_clear(inv);

    return np0;
}
*/

extern "C"
void bench_cgbn_2048_wapper(
    int iterations,
    int blocks,
    int threads
    )
{
    int instances = blocks * (threads / TPI);

    printf("CGBN<%d, %d> CUDA add/sub benchmark: %d-bit, iterations=%d, instances=%d\n",
            BITS, TPI, BITS, iterations, instances);

    const int WORDS = BITS / 32;

    // Device memory for a, b, out_add, out_sub
    mem_t *d_a, *d_b, *d_out_add, *d_out_sub;
    cudaMalloc(&d_a, sizeof(mem_t) * instances);
    cudaMalloc(&d_b, sizeof(mem_t) * instances);
    cudaMalloc(&d_out_add, sizeof(mem_t) * instances);
    cudaMalloc(&d_out_sub, sizeof(mem_t) * instances);

    // Host memory
    mem_t *h_a = (mem_t*) malloc(sizeof(mem_t) * instances);
    mem_t *h_b = (mem_t*) malloc(sizeof(mem_t) * instances);
    mem_t *h_out_add = (mem_t*) malloc(sizeof(mem_t) * instances);
    mem_t *h_out_sub = (mem_t*) malloc(sizeof(mem_t) * instances);
    memset(h_a, 0, sizeof(mem_t) * instances);
    memset(h_b, 0, sizeof(mem_t) * instances);

    // Initialize a = floor(2^991 / 8218291649), b = 8218291649 using GMP
    mpz_t a_gmp, b_gmp;
    mpz_init(a_gmp);
    mpz_init(b_gmp);
    mpz_t temp_mod;
    mpz_t temp_div;
    mpz_init(temp_mod);
    mpz_init_set_str(temp_div, "6721885469050382920298612421830968178768028093133627640714502537112053019071272802490029644416444090606146146238871876925423959", 10);
    // 8218291649
    // 6721885469050382920298612421830968178768028093133627640714502537112053019071272802490029644416444090606146146238871876925423959
    mpz_ui_pow_ui(temp_mod, 2, 991);
    mpz_tdiv_q(a_gmp, temp_mod, temp_div);
    mpz_set(b_gmp, temp_div);
    mpz_clear(temp_mod);
    mpz_clear(temp_div);

    // Fill all instances with same a and b
    uint32_t *a_words = (uint32_t*)h_a;
    uint32_t *b_words = (uint32_t*)h_b;
    
    // Fill first instance with GMP values, then copy to others
    size_t count_a = 0, count_b = 0;
    mpz_export(a_words, &count_a, -1, sizeof(uint32_t), 0, 0, a_gmp);
    mpz_export(b_words, &count_b, -1, sizeof(uint32_t), 0, 0, b_gmp);
    
    // Zero remaining words
    for (size_t i = count_a; i < (size_t)WORDS; ++i) a_words[i] = 0u;
    for (size_t i = count_b; i < (size_t)WORDS; ++i) b_words[i] = 0u;
    
    // Copy first instance to all others
    for (int i = 1; i < instances; ++i) {
        memcpy(&h_a[i], &h_a[0], sizeof(mem_t));
        memcpy(&h_b[i], &h_b[0], sizeof(mem_t));
    }

    // Helper to print hex
    auto print_hex = [&](const mem_t *arr, int idx, const char *name) {
        const uint32_t *w = (const uint32_t*)&arr[idx];
        printf("%s: 0x", name);
        bool leading = true;
        for (int i = WORDS - 1; i >= 0; --i) {
            if (leading && w[i] == 0) continue;
            leading = false;
            printf("%08x", w[i]);
        }
        if (leading) printf("0");
        printf("\n");
    };

    // Print input values
    printf("--- Input values ---\n");
    print_hex(h_a, 0, "a");
    print_hex(h_b, 0, "b");

    // Copy to device
    cudaMemcpy(d_a, h_a, sizeof(mem_t) * instances, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(mem_t) * instances, cudaMemcpyHostToDevice);

    cgbn_error_report_t *report;
    cgbn_error_report_alloc(&report);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel
    cudaEventRecord(start);
    bench_add_sub<<<blocks, threads>>>(
        report,
        d_a,
        d_b,
        d_out_add,
        d_out_sub,
        iterations);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy results back
    cudaMemcpy(h_out_add, d_out_add, sizeof(mem_t) * instances, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_sub, d_out_sub, sizeof(mem_t) * instances, cudaMemcpyDeviceToHost);

    // Compute GMP reference (chained iterations)
    mpz_t res_add, res_sub, add_state, sub_state;
    mpz_init(res_add);
    mpz_init(res_sub);
    mpz_init(add_state);
    mpz_init(sub_state);
    mpz_set(add_state, a_gmp);
    mpz_set(sub_state, a_gmp);
    
    // Chained iterations:
    // add: r = a + b; r = r + a; a = r
    // sub: r = a - b; r = r - a; a = r
    for (int i = 0; i < iterations; ++i) {
        mpz_add(res_add, add_state, b_gmp);
        // mpz_add(res_add, res_add, add_state);
        mpz_set(add_state, res_add);

        mpz_sub(res_sub, sub_state, b_gmp);
        // mpz_sub(res_sub, res_sub, sub_state);
        mpz_set(sub_state, res_sub);
    }

    // Convert GMP results to words (mod 2^BITS to match fixed-size limb arrays)
    std::vector<uint32_t> expected_add(WORDS), expected_sub(WORDS);
    mpz_t mod_ref, tmp_ref;
    mpz_init(mod_ref);
    mpz_init(tmp_ref);
    mpz_ui_pow_ui(mod_ref, 2, BITS);
    
    // Handle add result
    memset(expected_add.data(), 0, sizeof(uint32_t) * WORDS);
    mpz_mod(tmp_ref, res_add, mod_ref);
    size_t count = 0;
    mpz_export(expected_add.data(), &count, -1, sizeof(uint32_t), 0, 0, tmp_ref);
    
    // Handle sub result (convert to unsigned mod 2^BITS)
    memset(expected_sub.data(), 0, sizeof(uint32_t) * WORDS);
    mpz_mod(tmp_ref, res_sub, mod_ref);
    count = 0;
    mpz_export(expected_sub.data(), &count, -1, sizeof(uint32_t), 0, 0, tmp_ref);
    mpz_clear(mod_ref);
    mpz_clear(tmp_ref);

    // Compare and print results
    bool match_add = true, match_sub = true;
    const uint32_t *out_add_words = (const uint32_t*)&h_out_add[0];
    const uint32_t *out_sub_words = (const uint32_t*)&h_out_sub[0];
    
    for (int i = 0; i < WORDS; ++i) {
        if (out_add_words[i] != expected_add[i]) match_add = false;
        if (out_sub_words[i] != expected_sub[i]) match_sub = false;
    }

    printf("--- CUDA results (instance 0) ---\n");
    print_hex(h_out_add, 0, "Add result");
    print_hex(h_out_sub, 0, "Sub result");
    
    printf("\n--- GMP reference ---\n");
    printf("Add result: 0x");
    bool leading = true;
    for (int i = WORDS - 1; i >= 0; --i) {
        if (leading && expected_add[i] == 0) continue;
        leading = false;
        printf("%08x", expected_add[i]);
    }
    if (leading) printf("0");
    printf("\n");
    
    printf("Sub result: 0x");
    leading = true;
    for (int i = WORDS - 1; i >= 0; --i) {
        if (leading && expected_sub[i] == 0) continue;
        leading = false;
        printf("%08x", expected_sub[i]);
    }
    if (leading) printf("0");
    printf("\n");

    printf("\nCUDA add/sub benchmark: %d-bit, iterations=%d, instances=%d\n", BITS, iterations, instances);
    printf("Add: CUDA time (ms)=%.4f, equal=%s\n", ms, match_add ? "YES" : "NO");
    printf("Sub: CUDA time (ms)=%.4f, equal=%s\n", ms, match_sub ? "YES" : "NO");

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out_add);
    cudaFree(d_out_sub);
    free(h_a);
    free(h_b);
    free(h_out_add);
    free(h_out_sub);
    cgbn_error_report_free(report);
    
    mpz_clear(a_gmp);
    mpz_clear(b_gmp);
    mpz_clear(res_add);
    mpz_clear(res_sub);
    mpz_clear(add_state);
    mpz_clear(sub_state);
}


__global__ void bench_add_sub(
    cgbn_error_report_t *report,
    mem_t *a_data,
    mem_t *b_data,
    mem_t *out_add_data,
    mem_t *out_sub_data,
    int iterations)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int instance = tid / TPI;

    context_t context(cgbn_report_monitor, report, instance);
    env_t env(context);

    bn_t a_add, a_sub, b, r_add, r_sub;

    // Load a and b
    cgbn_load(env, a_add, &a_data[instance]);
    cgbn_set(env, a_sub, a_add);
    cgbn_load(env, b, &b_data[instance]);

    // Add benchmark (chained): r = a + b; r = r + a; a = r
    #pragma unroll 1
    for (int i = 0; i < iterations; i++) {
        cgbn_add(env, r_add, a_add, b);
        // cgbn_add(env, r_add, r_add, a_add);
        cgbn_set(env, a_add, r_add);
    }

    // Sub benchmark (chained): r = a - b; r = r - a; a = r
    #pragma unroll 1
    for (int i = 0; i < iterations; i++) {
        cgbn_sub(env, r_sub, a_sub, b);
        // cgbn_sub(env, r_sub, r_sub, a_sub);
        cgbn_set(env, a_sub, r_sub);
    }

    // Store results
    cgbn_store(env, &out_add_data[instance], r_add);
    cgbn_store(env, &out_sub_data[instance], r_sub);
}


__global__ void bench_mont_mul(
    cgbn_error_report_t *report,
    mem_t *data,
    mem_t *modulus_data,
    uint32_t np0,
    int iterations)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int instance = tid / TPI;

    context_t context(cgbn_report_monitor, report, instance);
    env_t env(context);

    bn_t a, b, r, modulus;

    // ✅ 每个 instance 读自己的数据
    cgbn_load(env, a, &data[instance*2 + 0]);
    cgbn_load(env, b, &data[instance*2 + 1]);
    cgbn_load(env, modulus, &modulus_data[instance]);

    // Verify np0 on device (before conversion, cgbn_bn2mont returns np0)
    uint32_t np0_test = cgbn_bn2mont(env, a, a, modulus);
    if (instance == 0 && threadIdx.x == 0) {
        printf("[Device] np0 from host: 0x%08x, np0_test: 0x%08x, match: %s\n",
               np0, np0_test, (np0 == np0_test) ? "YES" : "NO");
    }
    // Verify they match
    assert(np0 == np0_test);
    
    cgbn_bn2mont(env, b, b, modulus);

    #pragma unroll 1
    for (int i = 0; i < iterations; i++) {
        cgbn_mont_mul(env, r, a, b, modulus, np0);

        // 防止优化
        cgbn_add(env, a, r, b);
    }

    // ✅ 写回
    cgbn_store(env, &data[instance*2], a);
}

__global__ void bench_mont_sqr(
    cgbn_error_report_t *report,
    mem_t *data,
    mem_t *mod,
    uint32_t np0,
    int iterations)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int instance = tid / TPI;

    context_t context(cgbn_report_monitor, report, instance);
    env_t env(context);

    bn_t a, r, modulus;

    cgbn_load(env, a, &data[instance]);
    cgbn_load(env, modulus, &mod[instance]);

    cgbn_bn2mont(env, a, a, modulus);

    for (int i = 0; i < iterations; i++) {
        cgbn_mont_sqr(env, r, a, modulus, np0);

        cgbn_add(env, a, r, r);  // 防优化
    }

    cgbn_store(env, &data[instance], a);
}

