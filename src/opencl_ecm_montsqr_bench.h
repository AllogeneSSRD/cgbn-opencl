#ifndef OPENCL_ECM_MONTSQR_BENCH_H
#define OPENCL_ECM_MONTSQR_BENCH_H

// ECM-oriented montgomery square microbench using kernels from ecm_montsqr_bench.cl
bool runOpenClEcmMontSqrBenchmark(int bits, int kernel_iterations, int instances, int launch_repeats,
                                  bool use_wg, int tpi);

#endif // OPENCL_ECM_MONTSQR_BENCH_H
