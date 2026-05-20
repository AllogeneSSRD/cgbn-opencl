#ifndef OPENCL_ECM_ADDSUB_BENCH_H
#define OPENCL_ECM_ADDSUB_BENCH_H

// ECM-oriented add/sub microbench using kernels from ecm_addsub_bench.cl
bool runOpenClEcmAddSubBenchmark(int bits, int kernel_iterations, int instances, int launch_repeats);

#endif // OPENCL_ECM_ADDSUB_BENCH_H
