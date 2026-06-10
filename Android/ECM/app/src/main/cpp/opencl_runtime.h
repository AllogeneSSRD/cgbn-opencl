#pragma once

#include <string>

// Full device probe (platforms, devices, buffer R/W).
std::string probe_opencl();

// Quick sanity: GPU name + buffer ping.
std::string run_short_test();

// Modular add-mod microbench at limb_bits (16 / 24 / 32).
std::string run_bit_bench(int limb_bits, int elements, int kernel_iters, int launch_repeats);

// ECM mp_add_mod / mp_sub_mod microbench (limb_bits 24 or 32).
std::string run_addsub_bench(int bits, int kernel_iterations, int instances, int launch_repeats,
                             int limb_bits);

// ECM Montgomery mul/sqr microbench (use_wg default true, tpi=4 like desktop).
// limb_bits: 32 = full desktop path set; 24 = unroll_only_512_limb24 (mul24, 22 limbs @ 512-bit).
std::string run_montsqr_bench(int bits, int kernel_iterations, int instances, int launch_repeats,
                              bool use_wg, int tpi, int limb_bits = 32);
