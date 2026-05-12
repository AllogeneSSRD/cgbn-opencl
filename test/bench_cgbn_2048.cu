
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

#define TPI 8
#define BITS 2048

typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;
typedef typename env_t::cgbn_t bn_t;
typedef cgbn_mem_t<BITS> mem_t;

__global__ void bench_mont_mul(
    cgbn_error_report_t *report,
    mem_t *data,
    mem_t *modulus_data,
    uint32_t np0,
    int iterations);

__global__ void bench_mont_sqr(
    cgbn_error_report_t *report,
    mem_t *data,
    mem_t *modulus_data,
    uint32_t np0,
    int iterations);

__global__ void bench_add_sub(
    cgbn_error_report_t *report,
    mem_t *data,
    int iterations);

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

extern "C"
void bench_cgbn_2048_wapper(
    int iterations,
    int blocks,
    int threads
    )
{
    int instances = blocks * (threads / TPI);


    printf("CGBN<%d, %d> running kernel<%d block x %d threads> input number is %d bits\n",
            BITS, TPI, blocks, threads, BITS);
    printf("  instances: %d\n", instances);


    typedef cgbn_mem_t<BITS> mem_t;

    mem_t *d_data, *d_mod;

    cudaMalloc(&d_data, sizeof(mem_t) * instances * 2);
    cudaMalloc(&d_mod,  sizeof(mem_t) * instances);

    // Host-side init for a, b and modulus so we can print before/after
    const int WORDS = BITS / 32;
    mem_t *h_data = (mem_t*) malloc(sizeof(mem_t) * instances * 2);
    mem_t *h_mod  = (mem_t*) malloc(sizeof(mem_t) * instances);
    memset(h_data, 0, sizeof(mem_t) * instances * 2);
    memset(h_mod,  0, sizeof(mem_t) * instances);

    // Provide specific initial values for the first instance
    // a = 0x12345678, b = 0x9abcdef0, modulus = 2^991 - 1 (Mersenne prime)
    uint32_t *a_words = (uint32_t*)&h_data[0];
    uint32_t *b_words = (uint32_t*)&h_data[1];
    uint32_t *mod_words = (uint32_t*)&h_mod[0];
    a_words[0] = 0x12345678u;
    b_words[0] = 0x9abcdef0u;

    // Initialize modulus to 2^991 / 8218291649 using GMP
    mpz_t temp_mod;
    mpz_t temp_div;
    mpz_init(temp_mod);
    mpz_init_set_str(temp_div, "8218291649", 10);
    mpz_ui_pow_ui(temp_mod, 2, 991);
    mpz_tdiv_q(temp_mod, temp_mod, temp_div);
    mpz_export(mod_words, NULL, -1, sizeof(uint32_t), 0, 0, temp_mod);

    // Compute np0 correctly from modulus using GMP
    // np0 is -(N^-1 mod 2**32), used for montgomery representation
    mpz_t modulus_mpz;
    mpz_init(modulus_mpz);
    mpz_import(modulus_mpz, WORDS, -1, sizeof(uint32_t), 0, 0, mod_words);
    uint32_t np0 = compute_np0(modulus_mpz);
    mpz_clear(modulus_mpz);
    mpz_clear(temp_div);
    mpz_clear(temp_mod);

    // Copy the initialized modulus to all instances
    for (int i = 1; i < instances; i++) {
        memcpy(&h_mod[i], &h_mod[0], sizeof(mem_t));
    }

    // Helper to print a mem_t value (hex, big-endian word order)
    auto print_mem = [&](const mem_t *arr, int idx, const char *name) {
        const uint32_t *w = (const uint32_t*)&arr[idx];
        printf("%s: 0x", name);
        for (int i = WORDS - 1; i >= 0; --i)
            printf("%08x", w[i]);
        printf("\n");
    };

    // Helper to calculate bit length of a number
    auto get_bit_length = [&](const mem_t *arr, int idx) -> int {
        const uint32_t *w = (const uint32_t*)&arr[idx];
        for (int i = WORDS - 1; i >= 0; --i) {
            if (w[i] != 0) {
                return i * 32 + 32 - __builtin_clz(w[i]);
            }
        }
        return 0;
    };

    // Print values before kernel
    printf("--- Before kernel (host) ---\n");
    print_mem(h_data, 0, "a");
    int a_bits = get_bit_length(h_data, 0);
    printf("a bit length: %d bits\n", a_bits);
    
    print_mem(h_data, 1, "b");
    int b_bits = get_bit_length(h_data, 1);
    printf("b bit length: %d bits\n", b_bits);
    
    print_mem(h_mod, 0, "modulus");
    int mod_bits = get_bit_length(h_mod, 0);
    printf("modulus bit length: %d bits (expected 991)\n", mod_bits);
    
    printf("np0: 0x%08x\n", np0);
    
    // Verify np0 correctness:
    // For N = 2^991 - 1, we have N ≡ 2^31 - 1 (mod 2^32) = 0x7fffffff
    // np0 = -(N^-1 mod 2^32)
    printf("\nAnalysis:\n");
    printf("modulus[0] (lowest 32 bits): 0x%08x\n", mod_words[0]);
    printf("Expected modulus[0] for 2^991-1: 0x%08x (which is 2^31-1)\n", 0x7fffffffu);
    
    // Verify: N^-1 mod 2^32 should be such that np0 is correct
    mpz_t N_low, mod_2_32, inv_test;
    mpz_init(N_low);
    mpz_init(mod_2_32);
    mpz_init(inv_test);
    
    mpz_set_ui(N_low, mod_words[0]);  // lowest word of modulus
    mpz_ui_pow_ui(mod_2_32, 2, 32);   // 2^32
    mpz_invert(inv_test, N_low, mod_2_32);
    uint32_t computed_np0_inv = mpz_get_ui(inv_test);
    printf("N^-1 mod 2^32: 0x%08x\n", computed_np0_inv);
    printf("-(N^-1 mod 2^32): 0x%08x\n", (uint32_t)(-computed_np0_inv));
    printf("Match with host np0: %s\n", (uint32_t)(-computed_np0_inv) == np0 ? "YES" : "NO");
    
    mpz_clear(N_low);
    mpz_clear(mod_2_32);
    mpz_clear(inv_test);

    // Copy initialized host data to device
    cudaMemcpy(d_data, h_data, sizeof(mem_t) * instances * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mod,  h_mod,  sizeof(mem_t) * instances, cudaMemcpyHostToDevice);

    cgbn_error_report_t *report;
    cgbn_error_report_alloc(&report);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    bench_mont_mul<<<blocks, threads>>>(
        report,
        d_data,
        d_mod,
        np0,
        iterations);

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Time: %f ms\n", ms);

    double ops = (double)instances * iterations;
    printf("Throughput: %.3e ops/sec\n", ops / (ms/1000.0));

    // Copy back results for the first instance and print
    cudaMemcpy(h_data, d_data, sizeof(mem_t) * instances * 2, cudaMemcpyDeviceToHost);

    printf("--- After kernel (host) ---\n");
    print_mem(h_data, 0, "a");
    print_mem(h_data, 1, "b");
    printf("np0: 0x%08x\n", np0);

    cudaFree(d_data);
    cudaFree(d_mod);
    free(h_data);
    free(h_mod);
    cgbn_error_report_free(report);
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

__global__ void bench_add_sub(
    cgbn_error_report_t *report,
    mem_t *data,
    int iterations)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int instance = tid / TPI;

    context_t context(cgbn_report_monitor, report, instance);
    env_t env(context);

    bn_t a, b, r;

    cgbn_load(env, a, &data[instance*2 + 0]);
    cgbn_load(env, b, &data[instance*2 + 1]);

    for (int i = 0; i < iterations; i++) {
        cgbn_add(env, r, a, b);
        cgbn_sub(env, a, r, b);
    }

    cgbn_store(env, &data[instance], a);
}
