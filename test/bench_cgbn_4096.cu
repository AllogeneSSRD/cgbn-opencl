// Adapted from bench_cgbn_2048.cu -> 4096-bit tests

#include <stdint.h>
#include <gmp.h>
#include <cgbn.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include <vector>

#define TPI 16
#define BITS 4096

typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;
typedef typename env_t::cgbn_t bn_t;
typedef cgbn_mem_t<BITS> mem_t;

__global__ void bench_mont_mul(
    cgbn_error_report_t *report,
    mem_t *data,         // pairs (a,b) per instance
    mem_t *modulus_data,
    uint32_t np0,
    int iterations);

__global__ void bench_mont_sqr(
    cgbn_error_report_t *report,
    mem_t *data,
    mem_t *mod,
    uint32_t np0,
    int iterations);

extern "C"
void bench_cgbn_4096_wapper(
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

    // Initialize values using GMP
    mpz_t a_gmp, b_gmp;
    mpz_init(a_gmp);
    mpz_init(b_gmp);
    mpz_t temp_mod;
    mpz_t temp_div;
    mpz_init(temp_mod);
    mpz_init_set_str(temp_div, "6721885469050382920298612421830968178768028093133627640714502537112053019071272802490029644416444090606146146238871876925423959", 10);

    mpz_ui_pow_ui(temp_mod, 2, 4095); // use 2^4095 to get a large number (still <2^4096)
    mpz_tdiv_q(a_gmp, temp_mod, temp_div);
    mpz_set(b_gmp, temp_div);
    mpz_clear(temp_mod);
    mpz_clear(temp_div);

    uint32_t *a_words = (uint32_t*)h_a;
    uint32_t *b_words = (uint32_t*)h_b;

    size_t count_a = 0, count_b = 0;
    mpz_export(a_words, &count_a, -1, sizeof(uint32_t), 0, 0, a_gmp);
    mpz_export(b_words, &count_b, -1, sizeof(uint32_t), 0, 0, b_gmp);

    for (size_t i = count_a; i < (size_t)WORDS; ++i) a_words[i] = 0u;
    for (size_t i = count_b; i < (size_t)WORDS; ++i) b_words[i] = 0u;

    for (int i = 1; i < instances; ++i) {
        memcpy(&h_a[i], &h_a[0], sizeof(mem_t));
        memcpy(&h_b[i], &h_b[0], sizeof(mem_t));
    }

    // auto print_hex = [&](const mem_t *arr, int idx, const char *name) {
    //     const uint32_t *w = (const uint32_t*)&arr[idx];
    //     printf("%s: 0x", name);
    //     bool leading = true;
    //     for (int i = WORDS - 1; i >= 0; --i) {
    //         if (leading && w[i] == 0) continue;
    //         leading = false;
    //         printf("%08x", w[i]);
    //     }
    //     if (leading) printf("0");
    //     printf("\n");
    // };

    // printf("--- Input values ---\n");
    // print_hex(h_a, 0, "a");
    // print_hex(h_b, 0, "b");

    cudaMemcpy(d_a, h_a, sizeof(mem_t) * instances, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(mem_t) * instances, cudaMemcpyHostToDevice);

    cgbn_error_report_t *report;
    cgbn_error_report_alloc(&report);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Prepare data layout for mont tests: data array holds pairs (a,b) per instance
    mem_t *d_data; // size instances * 2
    mem_t *d_mod;
    mem_t *h_data = (mem_t*) malloc(sizeof(mem_t) * instances * 2);
    mem_t *h_mod = (mem_t*) malloc(sizeof(mem_t) * instances);

    // copy a and b into h_data[0*2], h_data[0*2+1]
    memcpy(&h_data[0], &h_a[0], sizeof(mem_t));
    memcpy(&h_data[1], &h_b[0], sizeof(mem_t));
    // construct an odd modulus based on 'a' for the test (ensure bn < modulus and invertible mod 2^32)
    mpz_t modulus_gmp;
    mpz_init_set(modulus_gmp, a_gmp);
    if (mpz_even_p(modulus_gmp))
        mpz_add_ui(modulus_gmp, modulus_gmp, 1);

    uint32_t *mod_words = (uint32_t*)h_mod;
    size_t count_mod = 0;
    mpz_export(mod_words, &count_mod, -1, sizeof(uint32_t), 0, 0, modulus_gmp);
    for (size_t i = count_mod; i < (size_t)WORDS; ++i) mod_words[i] = 0u;

    for (int i = 1; i < instances; ++i) {
        memcpy(&h_data[i*2 + 0], &h_data[0], sizeof(mem_t));
        memcpy(&h_data[i*2 + 1], &h_data[1], sizeof(mem_t));
        memcpy(&h_mod[i], &h_mod[0], sizeof(mem_t));
    }

    cudaMalloc(&d_data, sizeof(mem_t) * instances * 2);
    cudaMalloc(&d_mod, sizeof(mem_t) * instances);
    cudaMemcpy(d_data, h_data, sizeof(mem_t) * instances * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mod, h_mod, sizeof(mem_t) * instances, cudaMemcpyHostToDevice);

    // compute np0 from modulus using GMP (np0 = -N^{-1} mod 2^32)
    auto compute_np0 = [&](const mpz_t N)->uint32_t{
        mpz_t mod, inv;
        mpz_init(mod); mpz_init(inv);
        mpz_ui_pow_ui(mod, 2, 32);
        if (mpz_invert(inv, N, mod) == 0) {
            printf("Error: modulus not invertible mod 2^32\n");
            exit(1);
        }
        uint32_t inv32 = (uint32_t) mpz_get_ui(inv);
        mpz_clear(mod); mpz_clear(inv);
        return (uint32_t)(-inv32);
    };

    // compute np0 from the odd modulus we constructed
    uint32_t np0 = compute_np0(modulus_gmp);

    mpz_clear(modulus_gmp);

    // Timing: mont mul
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
    float ms_mul; cudaEventElapsedTime(&ms_mul, start, stop);

    // Timing: mont sqr
    cudaEventRecord(start);
    bench_mont_sqr<<<blocks, threads>>>(
        report,
        d_data,
        d_mod,
        np0,
        iterations);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_sqr; cudaEventElapsedTime(&ms_sqr, start, stop);

    printf("\nCUDA mont benchmark: %d-bit, iterations=%d, instances=%d\n", BITS, iterations, instances);
    printf("MontMul: CUDA time (ms)=%.4f\n", ms_mul);
    printf("MontSqr: CUDA time (ms)=%.4f\n", ms_sqr);

    // Cleanup
    cudaFree(d_a); cudaFree(d_b);
    cudaFree(d_data); cudaFree(d_mod);
    free(h_a); free(h_b); free(h_data); free(h_mod);
    cgbn_error_report_free(report);

    mpz_clear(a_gmp); mpz_clear(b_gmp);
}

// Montgomery multiply kernel
__global__ void bench_mont_mul(
    cgbn_error_report_t *report,
    mem_t *data,         // pairs (a,b) per instance
    mem_t *modulus_data,
    uint32_t np0,
    int iterations)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int instance = tid / TPI;

    context_t context(cgbn_report_monitor, report, instance);
    env_t env(context);

    bn_t a, b, r, modulus;

    // load a and b from data[instance*2 + 0/1]
    cgbn_load(env, a, &data[instance*2 + 0]);
    cgbn_load(env, b, &data[instance*2 + 1]);
    cgbn_load(env, modulus, &modulus_data[instance]);

    // Verify np0 on device
    uint32_t np0_test = cgbn_bn2mont(env, a, a, modulus);
    if (instance == 0 && threadIdx.x == 0) {
        printf("[Device] np0 from host: 0x%08x, np0_test: 0x%08x, match: %s\n",
               np0, np0_test, (np0 == np0_test) ? "YES" : "NO");
    }
    assert(np0 == np0_test);

    cgbn_bn2mont(env, b, b, modulus);

    #pragma unroll 1
    for (int i = 0; i < iterations; i++) {
        cgbn_mont_mul(env, r, a, b, modulus, np0);
        cgbn_add(env, a, r, b); // prevent optimization
    }

    cgbn_store(env, &data[instance*2], a);
}

// Montgomery square kernel
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

    cgbn_load(env, a, &data[instance*2 + 0]);
    cgbn_load(env, modulus, &mod[instance]);

    cgbn_bn2mont(env, a, a, modulus);

    for (int i = 0; i < iterations; i++) {
        cgbn_mont_sqr(env, r, a, modulus, np0);
        cgbn_add(env, a, r, r); // prevent optimization
    }

    cgbn_store(env, &data[instance*2], a);
}
