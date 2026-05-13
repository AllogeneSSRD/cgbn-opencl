#ifndef OPENCL_ADDSUB_TESTS_H
#define OPENCL_ADDSUB_TESTS_H

// Run the 1024-bit add/sub benchmark comparing GMP (CPU) and OpenCL.
// Returns true on success.
bool runOpenClAddSubBenchmark(int iterations, int instances);

#endif // OPENCL_ADDSUB_TESTS_H
